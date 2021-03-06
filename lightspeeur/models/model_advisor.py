import os
import inspect
import logging
import shutil

import numpy as np
import tensorflow as tf

from enum import Enum
from tqdm import tqdm
from tensorflow.keras import Model, backend as K
from tensorflow.keras.metrics import Metric
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import BatchNormalization, Layer, Dense
from lightspeeur.layers import Conv2D, Conv2DTranspose, DepthwiseConv2D, ReLU
from lightspeeur.drivers import Specification
from lightspeeur.models.model_reorganizer import reorganize_layers, get_inbound_layers, organize_layer
from typing import List


class LearningStage(Enum):
    FULL_FLOATING_POINT_TRAINING = 0
    QUANTIZED_CONVOLUTION_TRAINING = 1
    CALIBRATE_RELU_CAPS = 2
    QUANTIZED_ACTIVATION_TRAINING = 3
    MODEL_FUSION = 4


def stage_name(stage: LearningStage):
    if stage == LearningStage.FULL_FLOATING_POINT_TRAINING:
        return 'Full Floating-Point Training'
    elif stage == LearningStage.QUANTIZED_CONVOLUTION_TRAINING:
        return 'Quantized Convolutional Layer Training'
    elif stage == LearningStage.CALIBRATE_RELU_CAPS:
        return 'Calibrating ReLU Caps'
    elif stage == LearningStage.QUANTIZED_ACTIVATION_TRAINING:
        return 'Quantized Activation Layer Training'
    elif stage == LearningStage.MODEL_FUSION:
        return 'Model Fusion'
    else:
        raise ValueError('Invalid stage: {}'.format(stage))


QUANTIZABLE_CONVOLUTION = [Conv2D, Conv2DTranspose, DepthwiseConv2D]
QUANTIZABLE_ACTIVATION = [ReLU]
MODEL_FUSION_GROUPS = [[Conv2D, ReLU],
                       [Conv2D, BatchNormalization, ReLU],
                       [Conv2DTranspose, ReLU],
                       [Conv2DTranspose, BatchNormalization, ReLU],
                       [DepthwiseConv2D, ReLU],
                       [DepthwiseConv2D, BatchNormalization, ReLU]]

logger = logging.getLogger('lightspeeur')
logger.setLevel(logging.INFO)
logging.basicConfig()


def is_eligible(layer, instance_types):
    for instance_type in instance_types:
        if isinstance(layer, instance_type):
            return True
    return False


class ModelStageAdvisor:

    def __init__(self,
                 chip_id,
                 model: Model,
                 compile_options,
                 folded_shape_model=None,
                 checkpoints_dir=None,
                 cleanup_checkpoints=False):
        if checkpoints_dir is None:
            checkpoints_dir = '/tmp/advisor_checkpoints/{}'.format(model.name)

        self.specification = Specification(chip_id=chip_id)
        self.model = model
        self.folded_shape_model = folded_shape_model
        self.current_stage = None
        self.compile_options = compile_options
        self.checkpoints_dir = checkpoints_dir
        self.cleanup_checkpoints = cleanup_checkpoints
        self.probes = {stage: None for stage in LearningStage.__members__.values()}

        if self.cleanup_checkpoints and os.path.exists(self.checkpoints_dir):
            shutil.rmtree(self.checkpoints_dir)

        messages = []
        messages += self.validate_model_bias()
        messages += self.validate_folded_shape_model()
        if len(messages) > 0:
            messages.insert(0, '{} problems have been found in the analog and folded model:'.format(len(messages)))
            raise ValueError('\n'.join(messages))

    def validate_model_bias(self) -> List[str]:
        analog_convs_has_bias = []
        quantized_convs_has_no_bias = []
        for layer in self.model.layers:
            if is_eligible(layer, QUANTIZABLE_CONVOLUTION) and layer.use_bias:
                analog_convs_has_bias.append(layer.name)

        if self.folded_shape_model:
            for layer in self.folded_shape_model.layers:
                if is_eligible(layer, QUANTIZABLE_CONVOLUTION) and not layer.use_bias:
                    quantized_convs_has_no_bias.append(layer.name)

        messages = []
        if len(analog_convs_has_bias) > 0:
            messages.append('{} analog convolutional layers have bias. layer names: {}'
                            .format(len(analog_convs_has_bias), ', '.join(analog_convs_has_bias)))
        if len(quantized_convs_has_no_bias) > 0:
            messages.append('{} quantized convolutional layers have no bias. layer names: {}'
                            .format(len(quantized_convs_has_no_bias), ', '.join(quantized_convs_has_no_bias)))
        return messages

    def validate_folded_shape_model(self) -> List[str]:
        if self.folded_shape_model is None:
            return []

        analog_conv = []
        not_folded = []
        analog_relu = []
        for layer in self.folded_shape_model.layers:
            if is_eligible(layer, QUANTIZABLE_CONVOLUTION):
                if not layer.quantize:
                    analog_conv.append(layer.name)
            elif is_eligible(layer, QUANTIZABLE_ACTIVATION):
                if not layer.quantize:
                    analog_relu.append(layer.name)
            elif isinstance(layer, BatchNormalization):
                not_folded.append(layer.name)

        messages = []
        if len(analog_conv) > 0:
            messages.append('{} convolutional layers are not quantized. layer names: {}'
                            .format(len(analog_conv), ', '.join(analog_conv)))
        if len(analog_relu) > 0:
            messages.append('{} activation layers are not quantized. layer names: {}'
                            .format(len(analog_relu), ', '.join(analog_relu)))
        if len(not_folded) > 0:
            messages.append('{} batch normalization layer have been found. layer names: {}'
                            .format(len(not_folded), ', '.join(not_folded)))
        if len(messages) > 0:
            messages.insert(0, '{} problems have been found in the folded shape model:'.format(len(messages)))
        return messages

    def invalidate_list_compile_option_recursively(self, array):
        for v in array:
            if isinstance(v, Metric):
                v.reset_state()
            elif isinstance(v, dict):
                self.invalidate_dict_compile_option_recursively(v)
            elif isinstance(v, (list, tuple)):
                self.invalidate_list_compile_option_recursively(v)

    def invalidate_dict_compile_option_recursively(self, dicts):
        for k, v in dicts.items():
            if isinstance(v, Metric):
                v.reset_state()
            elif isinstance(v, dict):
                self.invalidate_dict_compile_option_recursively(v)
            elif isinstance(v, (list, tuple)):
                self.invalidate_list_compile_option_recursively(v)

    def compile(self, model=None, options=None):
        model = self.model if model is None else model

        compile_options = self.compile_options
        if options is not None:
            for k, v in options.items():
                compile_options[k] = v

        self.invalidate_dict_compile_option_recursively(compile_options)
        model.compile(**compile_options)

    def fit(self,
            x=None,
            y=None,
            batch_size=None,
            epochs=1,
            verbose='auto',
            callbacks=None,
            validation_split=0.,
            validation_data=None,
            shuffle=True,
            class_weight=None,
            sample_weight=None,
            initial_epoch=0,
            steps_per_epoch=None,
            validation_steps=None,
            validation_batch_size=None,
            validation_freq=1,
            max_queue_size=10,
            workers=1,
            use_multiprocessing=False):
        if self.current_stage is None:
            raise ValueError('Current stage is not valid')
        self.probes[self.current_stage] = self.model.fit(x, y,
                                                         batch_size, epochs,
                                                         verbose, callbacks,
                                                         validation_split, validation_data,
                                                         shuffle,
                                                         class_weight, sample_weight,
                                                         initial_epoch, steps_per_epoch,
                                                         validation_steps, validation_batch_size, validation_freq,
                                                         max_queue_size, workers, use_multiprocessing)

    def advance_stage(self):
        previous_stage = self.current_stage
        if self.current_stage is None:
            self.current_stage = LearningStage.FULL_FLOATING_POINT_TRAINING
        elif self.current_stage == LearningStage.MODEL_FUSION:
            self.load_weights_from_checkpoints(LearningStage.MODEL_FUSION)
            logger.info('All stages have finished')
            return False
        else:
            self.current_stage = LearningStage(self.current_stage.value + 1)

        self.load_weights_from_checkpoints(previous_stage)
        logger.info('Next stage is {}'.format(stage_name(self.current_stage)))
        return True

    def load_weights_from_checkpoints(self, previous_stage):
        if previous_stage is not None:
            logger.info('Previous stage was {}'.format(stage_name(previous_stage)))
            stage_dir = self.get_checkpoint_stage_dir(previous_stage)
            checkpoint = tf.train.latest_checkpoint(stage_dir)
            if checkpoint is not None:
                self.model.load_weights(checkpoint)
                logger.info('Loaded best checkpoints from previous stage')
            else:
                logger.info('Checkpoints from previous stage is not available. Skipped.')

    def propose(self,
                x=None,
                y=None,
                batch_size=None,
                epochs=1,
                verbose='auto',
                callbacks=None,
                validation_split=0.,
                validation_data=None,
                shuffle=True,
                class_weight=None,
                sample_weight=None,
                initial_epoch=0,
                steps_per_epoch=None,
                validation_steps=None,
                validation_batch_size=None,
                validation_freq=1,
                max_queue_size=10,
                workers=1,
                use_multiprocessing=False,
                clip_bias=False,
                relu_calibration_sample_steps=None,
                fine_tune_folded_shape_model=False,
                monitor='val_loss'):

        # Checkpoints
        stage_dir = self.get_checkpoint_stage_dir(self.current_stage)
        os.makedirs(stage_dir, exist_ok=True)

        checkpoint_path = os.path.join(stage_dir, '%s-{epoch:04d}.ckpt' % self.model.name)
        callback_checkpoint = ModelCheckpoint(filepath=checkpoint_path,
                                              monitor=monitor,
                                              verbose=True,
                                              save_best_only=True,
                                              save_weights_only=True)
        if callbacks is None:
            callbacks = []

        if type(callbacks) is list:
            callbacks.append(callback_checkpoint)

        if self.current_stage == LearningStage.CALIBRATE_RELU_CAPS:
            self.compile()
            self.calibrate_relu(x, batch_size, steps_per_epoch, relu_calibration_sample_steps, checkpoint_path)
        else:
            is_model_fusion = self.current_stage == LearningStage.MODEL_FUSION
            if is_model_fusion:
                self.fold(x, y, batch_size, steps_per_epoch, clip_bias)
                if not fine_tune_folded_shape_model:
                    return
            else:
                is_conv_quantization = self.current_stage == LearningStage.QUANTIZED_CONVOLUTION_TRAINING
                is_activation_quantization = self.current_stage == LearningStage.QUANTIZED_ACTIVATION_TRAINING

                if is_conv_quantization or is_activation_quantization:
                    self.quantize(is_activation_quantization)

            self.compile()
            self.fit(x, y,
                     batch_size, epochs,
                     verbose, callbacks,
                     validation_split, validation_data,
                     shuffle,
                     class_weight, sample_weight,
                     initial_epoch, steps_per_epoch,
                     validation_steps, validation_batch_size, validation_freq,
                     max_queue_size, workers, use_multiprocessing)

    def quantize(self, quantize_activation=False):
        eligible_types = QUANTIZABLE_CONVOLUTION
        if quantize_activation:
            eligible_types += QUANTIZABLE_ACTIVATION

        # Convolutional and Activation Layer Quantization
        if eligible_types is not None:
            layers = [layer for layer in self.model.layers if is_eligible(layer, eligible_types)]
            for layer in layers:
                layer.quantize = True

    def fold(self, x, y, batch_size, steps_per_epoch, clip_bias):
        folded = self.fold_batch_normalization(self.model, clip_bias=clip_bias)
        if self.folded_shape_model is not None:
            logger.info('Folded-architecture model have been provided.')
            logger.info('Projects all weights from automatically folded model to provided fused model.')
            logger.info('All lightspeeur layers in folded-architecture model must turn on quantize option')
            for index, layer in enumerate(self.folded_shape_model.layers):
                fused_layer = folded.get_layer(index=index)
                layer.set_weights(fused_layer.get_weights())
            self.model = self.folded_shape_model
        else:
            logger.info('No folded-architecture model have been provided.')
            logger.info('Note: Providing folded-architecture model is recommended.')
            logger.info('If there is no BatchNormalization in the model, '
                        'you don\'t need to provide folded-architecture model')
            self.model = folded

        self.compile()
        logger.info('Evaluating folded model...')
        steps = None
        if isinstance(x, tf.data.Dataset):
            steps = steps_per_epoch
        loss, metric = self.model.evaluate(x, y, batch_size, steps=steps)
        logger.info('Evaluation result:')
        logger.info('  Loss: {}'.format(loss))
        logger.info('  Metric: {}'.format(metric))
        logger.info('If the metric or loss result are bad, you must fine-tune the model.')

    def calibrate_relu(self, x, batch_size, steps_per_epoch, relu_calibration_sample_steps, checkpoint_path):
        relus = [layer for layer in self.model.layers if isinstance(layer, ReLU)]
        relu_caps = {layer.name: 0.0 for layer in relus}

        test_cases = []
        for layer in self.model.layers:
            if isinstance(layer, ReLU):
                test_case = K.function(inputs=self.model.layers[0].input,
                                       outputs=layer.output)
                test_cases.append((test_case, layer.name))

        logger.info('Feed-forward model to record ReLU outputs')
        if isinstance(x, np.ndarray) or isinstance(x, tf.Tensor):
            if batch_size is not None:
                length = tf.shape(x)[0]
                steps = length // batch_size
                if relu_calibration_sample_steps is not None:
                    steps = min(steps, relu_calibration_sample_steps)

                for i in tqdm(range(int(steps))):
                    batch = x[i * batch_size:min(length, (i + 1) * batch_size)]
                    relu_caps = self.calibrate_relu_caps(batch, relu_caps, test_cases)
            else:
                relu_caps = self.calibrate_relu_caps(x, relu_caps, test_cases)
        else:
            if steps_per_epoch is None:
                raise ValueError('steps_per_epoch cannot be None when the dataflow is iterator')

            steps = steps_per_epoch
            if relu_calibration_sample_steps is not None:
                steps = min(steps, relu_calibration_sample_steps)
            if inspect.isgeneratorfunction(x):
                for value in tqdm(x, total=int(steps)):
                    inputs, _ = value
                    relu_caps = self.calibrate_relu_caps(inputs, relu_caps, test_cases)
            elif hasattr(x, '__getitem__'):
                for i in tqdm(range(int(steps))):
                    inputs, _ = x[i]
                    relu_caps = self.calibrate_relu_caps(inputs, relu_caps, test_cases)
            elif isinstance(x, tf.data.Dataset):
                for inputs, _ in tqdm(x.take(steps)):
                    relu_caps = self.calibrate_relu_caps(inputs, relu_caps, test_cases)
            else:
                raise ValueError('It is exceptional case to calibrate ReLU caps')

        logger.info('Initialized ReLU caps:')

        for k in sorted(relu_caps.keys()):
            logger.info('\t"{}": {}'.format(k, relu_caps[k]))

        for k, v in relu_caps.items():
            layer = self.model.get_layer(k)
            if isinstance(layer, ReLU):
                layer.cap = v
                layer.set_weights([tf.constant([v], dtype=tf.float32)])

        self.model.save_weights(checkpoint_path.format(epoch=1))

    def calibrate_relu_caps(self, inputs, initial_relu_caps, test_cases):
        # Feed-forward to record outputs
        for test_case, layer_name in test_cases:
            outputs = test_case(inputs)
            cap = np.percentile(outputs, 99)
            if cap <= initial_relu_caps[layer_name]:
                continue

            initial_relu_caps[layer_name] = cap

        return initial_relu_caps

    def get_probe(self, stage: LearningStage):
        return self.probes[stage]

    def get_model(self):
        return self.model

    def get_checkpoint_stage_dir(self, stage):
        return os.path.join(self.checkpoints_dir, 'stage-{}'.format(stage.value))

    def fuse_convolutional_and_batch_norm_v0(self,
                                             conv: Layer,
                                             prev_relu_cap,
                                             relu: ReLU,
                                             batch_normalization=None):
        if batch_normalization is not None and not isinstance(batch_normalization, BatchNormalization):
            raise AttributeError('batch_normalization must be BatchNormalization')

        conv_weights = conv.get_weights()
        if batch_normalization is not None:
            gamma, beta, mean, variance = batch_normalization.get_weights()
            eps = batch_normalization.epsilon
            kernel = conv_weights[0]

            stddev = np.sqrt(variance + eps)
            factor = gamma / stddev

            # for depthwise convolution weights, reshape batch norm factor,
            # so that multiplication of the factor is not broadcast to the last dimension
            if kernel.shape[3] == 1:
                factor = factor.reshape((1, 1, factor.shape[0], 1))

            kernel *= factor
            bias = beta - gamma / stddev * mean
        else:
            kernel = conv_weights[0]
            bias = conv_weights[1]

        # Fuse using ReLU caps
        current_relu_cap = relu.get_weights()[0][0]  # relu_cap
        max_activation = self.specification.max_activation(bits=relu.activation_bits)
        gain_bias = max_activation / current_relu_cap
        gain_kernel = prev_relu_cap / current_relu_cap

        kernel *= gain_kernel
        bias *= gain_bias
        return kernel, bias

    def fuse_convolutional_and_batch_norm(self,
                                          conv: Layer,
                                          prev_relu_cap,
                                          relu: ReLU,
                                          batch_normalization=None):
        if batch_normalization is not None and not isinstance(batch_normalization, BatchNormalization):
            raise AttributeError('batch_normalization must be BatchNormalization')

        conv_weights = conv.get_weights()
        if batch_normalization is not None:
            # Fusing batch normalization and convolution in runtime
            # Source: https://nenadmarkus.com/p/fusing-batchnorm-and-conv/
            gamma, beta, mean, variance = batch_normalization.get_weights()
            eps = batch_normalization.epsilon

            kernel = conv_weights[0]

            shape = kernel.shape
            w_conv = tf.transpose(kernel, perm=(3, 2, 0, 1))
            w_conv = tf.reshape(w_conv, (conv.filters, -1))

            w_bn = tf.linalg.diag(gamma / (tf.sqrt(eps + variance)))

            kernel = tf.matmul(w_bn, w_conv)
            if len(conv_weights) == 2:
                b_conv = conv_weights[1]
            else:
                b_conv = tf.zeros((kernel.shape[-1],))

            b_bn = beta - (gamma * mean) / tf.sqrt(variance + eps)

            kernel = tf.reshape(kernel, (shape[-1], shape[-2], shape[0], shape[1]))
            kernel = tf.transpose(kernel, perm=(2, 3, 1, 0))

            b_conv = tf.expand_dims(b_conv, 1)
            b_bn = tf.expand_dims(b_bn, 1)
            bias = tf.matmul(w_bn, b_conv) + b_bn
            bias = tf.reshape(bias, (-1,))
        elif len(conv_weights) == 2:
            kernel, bias = conv_weights
        else:
            kernel = conv_weights[0]
            bias = tf.zeros((kernel.shape[-1],))

        # Fuse using ReLU caps
        current_relu_cap = relu.get_weights()[0][0]  # relu_cap
        max_activation = self.specification.max_activation(bits=relu.activation_bits)
        gain_bias = max_activation / current_relu_cap
        gain_kernel = prev_relu_cap / current_relu_cap

        kernel *= gain_kernel
        bias *= gain_bias
        return kernel, bias

    def fold_batch_normalization(self, src_model, clip_bias=False):
        fused_conv_layers = {}
        fused_relu_layers = {}
        popped_layers = []
        num_fusing_layers = 0
        length = len(src_model.layers)
        for group in MODEL_FUSION_GROUPS:
            for i in range(length - len(group)):
                eligible = True
                for offset, layer_type in enumerate(group):
                    if not isinstance(src_model.get_layer(index=i + offset), layer_type):
                        eligible = False
                        break

                if not eligible:
                    continue

                conv = src_model.get_layer(index=i)  # first layer
                relu: ReLU = src_model.get_layer(index=i + len(group) - 1)  # last layer

                if len(group) == 3:
                    batch_normalization = src_model.get_layer(index=i + 1)
                    popped_layers.append(batch_normalization)
                else:
                    batch_normalization = None

                prev_relu_layers = [layer
                                    for index, layer in enumerate(src_model.layers)
                                    if isinstance(layer, ReLU) and index < i]
                if len(prev_relu_layers) == 0:
                    prev_relu_cap = self.specification.max_activation()
                else:
                    prev_relu_cap = prev_relu_layers[-1].cap

                kernel, bias = self.fuse_convolutional_and_batch_norm_v0(conv,
                                                                         prev_relu_cap,
                                                                         relu, batch_normalization)
                fused_conv_layers[conv.name] = (kernel, bias)
                fused_relu_layers[relu.name] = self.specification.max_activation(relu.activation_bits)

                num_fusing_layers += 1

        logger.info('{} layer groups will be fused.'.format(num_fusing_layers))

        last_relu_layer = None
        rebuilt_layers = []
        for layer in src_model.layers:
            if isinstance(layer, BatchNormalization):
                continue

            if isinstance(layer, ReLU):
                last_relu_layer = layer

            if layer.name in fused_conv_layers:
                # create a new conv layer
                config = layer.get_config()
                config.update({
                    'use_bias': True,
                    'clip_bias': clip_bias
                })
                conv = layer.__class__.from_config(config)

                inbound_outputs_map = {inbound_layer.name: [inbound_layer.output]
                                       for inbound_layer in get_inbound_layers(layer)}
                organize_layer(conv, inbound_outputs_map, popped_layers, force=True, recreate=False)
                rebuilt_layers.append(conv)
            else:
                rebuilt_layers.append(layer)

        dense_layers = []
        if last_relu_layer is not None:
            dense_layers = self.find_next_first_dense_layers(last_relu_layer)
            logger.info('Next dense layers to be scaled: {}'.format([dense.name for dense in dense_layers]))

        new_model = reorganize_layers(src_model.name, rebuilt_layers, popped_layers)
        for layer_name, (kernel, bias) in fused_conv_layers.items():
            new_model.get_layer(layer_name).set_weights([kernel, bias])

        if last_relu_layer is not None:
            max_cap = self.specification.max_activation(last_relu_layer.activation_bits)
            kernel_scale = max_cap / last_relu_layer.cap
            logger.info('Dense kernel scale: {} ({} / {})'.format(kernel_scale, max_cap, last_relu_layer.cap))
            for dense_layer in dense_layers:
                layer = new_model.get_layer(dense_layer.name)

                old_weights = dense_layer.get_weights()
                kernel = old_weights[0] / kernel_scale
                if len(old_weights) == 2:
                    new_weights = [kernel, old_weights[1]]
                else:
                    new_weights = [kernel]
                layer.set_weights(new_weights)

        for layer_name, relu_cap in fused_relu_layers.items():
            layer = new_model.get_layer(layer_name)
            layer.cap = relu_cap
            layer.set_weights([tf.constant([relu_cap], dtype=tf.float32)])

        return new_model

    def find_next_first_dense_layers(self, search_from):
        dense_layers = []

        outbounds = search_from.outbound_nodes
        for node in outbounds:
            layer = node.outbound_layer
            if isinstance(layer, Dense):
                dense_layers.append(layer)
            else:
                dense_layers += self.find_next_first_dense_layers(layer)

        return dense_layers
