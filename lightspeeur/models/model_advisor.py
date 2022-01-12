import os
import inspect
import logging
import shutil

import numpy as np
import tensorflow as tf

from enum import Enum
from tqdm import tqdm
from tensorflow.keras import Model, backend as K
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import BatchNormalization, Layer
from lightspeeur.layers import Conv2D, Conv2DTranspose, DepthwiseConv2D, ReLU
from lightspeeur.drivers import Specification
from .model_reorganizer import reorganize_layers, get_inbound_layers, organize_layer


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
                 checkpoints_dir=None,
                 cleanup_checkpoints=False):
        if checkpoints_dir is None:
            checkpoints_dir = '/tmp/advisor_checkpoints/{}'.format(model.name)

        self.specification = Specification(chip_id=chip_id)
        self.model = model
        self.current_stage = None
        self.compile_options = compile_options
        self.checkpoints_dir = checkpoints_dir
        self.cleanup_checkpoints = cleanup_checkpoints
        self.probes = {stage: None for stage in LearningStage.__members__.values()}

        if self.cleanup_checkpoints and os.path.exists(self.checkpoints_dir):
            shutil.rmtree(self.checkpoints_dir)

    def compile(self, model=None, options=None):
        model = self.model if model is None else model

        compile_options = self.compile_options
        if options is not None:
            for k, v in options.items():
                compile_options[k] = v

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
            logger.info('All stages have finished')
            return False
        else:
            self.current_stage = LearningStage(self.current_stage.value + 1)

        if previous_stage is not None:
            logger.info('Previous stage was {}'.format(stage_name(previous_stage)))
            stage_dir = self.get_checkpoint_stage_dir(previous_stage)
            checkpoint = tf.train.latest_checkpoint(stage_dir)
            if checkpoint is not None:
                self.model.load_weights(checkpoint)
                logger.info('Loaded best checkpoints from previous stage')
            else:
                logger.info('Checkpoints from previous stage is not available. Skipped.')

        logger.info('Next stage is {}'.format(stage_name(self.current_stage)))
        return True

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
                clip_bias=False):

        # Checkpoints
        stage_dir = self.get_checkpoint_stage_dir(self.current_stage)
        os.makedirs(stage_dir, exist_ok=True)

        checkpoint_path = os.path.join(stage_dir, '%s-{epoch:04d}.ckpt' % self.model.name)
        callback_checkpoint = ModelCheckpoint(filepath=checkpoint_path,
                                              verbose=True,
                                              save_best_only=True,
                                              save_weights_only=True)
        if callbacks is None:
            callbacks = []

        if type(callbacks) is list:
            callbacks.append(callback_checkpoint)

        if self.current_stage == LearningStage.CALIBRATE_RELU_CAPS:
            self.compile()

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
                    for i in tqdm(range(int(steps))):
                        batch = x[i * batch_size:min(length, (i + 1) * batch_size)]
                        relu_caps = self.calibrate_relu_caps(batch, relu_caps, test_cases)
                else:
                    relu_caps = self.calibrate_relu_caps(x, relu_caps, test_cases)
            else:
                if steps_per_epoch is None:
                    raise ValueError('steps_per_epoch cannot be None when the dataflow is iterator')

                if inspect.isgeneratorfunction(x):
                    for value in tqdm(x, total=int(steps_per_epoch)):
                        inputs, _ = value
                        relu_caps = self.calibrate_relu_caps(inputs, relu_caps, test_cases)
                elif hasattr(x, '__getitem__'):
                    for i in tqdm(range(int(steps_per_epoch))):
                        inputs, _ = x[i]
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
        else:
            is_model_fusion = self.current_stage == LearningStage.MODEL_FUSION
            if is_model_fusion:
                fused_conv_layers = {}
                fused_relu_layers = {}
                popped_layers = []
                num_fusing_layers = 0
                length = len(self.model.layers)
                for group in MODEL_FUSION_GROUPS:
                    for i in range(length - len(group)):
                        eligible = True
                        for offset, layer_type in enumerate(group):
                            if not isinstance(self.model.get_layer(index=i + offset), layer_type):
                                eligible = False
                                break

                        if not eligible:
                            continue

                        conv = self.model.get_layer(index=i)  # first layer
                        relu: ReLU = self.model.get_layer(index=i + len(group) - 1)  # last layer

                        if len(group) == 3:
                            batch_normalization = self.model.get_layer(index=i + 1)
                            popped_layers.append(batch_normalization)
                        else:
                            batch_normalization = None

                        prev_relu_layers = [layer
                                            for index, layer in enumerate(self.model.layers)
                                            if isinstance(layer, ReLU) and index < i]
                        if len(prev_relu_layers) == 0:
                            prev_relu_cap = self.specification.max_activation()
                        else:
                            prev_relu_cap = prev_relu_layers[-1].cap

                        kernel, bias = self.fuse_convolutional_and_batch_norm(conv,
                                                                              prev_relu_cap,
                                                                              relu, batch_normalization)
                        fused_conv_layers[conv.name] = (kernel, bias)
                        fused_relu_layers[relu.name] = self.specification.max_activation(relu.activation_bits)

                        num_fusing_layers += 1

                logger.info('{} layer groups will be fused.'.format(num_fusing_layers))
                
                rebuilt_layers = []
                for layer in self.model.layers:
                    if isinstance(layer, BatchNormalization):
                        continue

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
                        organize_layer(conv, inbound_outputs_map, popped_layers, force=True)
                        rebuilt_layers.append(conv)
                    else:
                        rebuilt_layers.append(layer)

                new_model = reorganize_layers(self.model.name, rebuilt_layers, popped_layers)
                for layer_name, (kernel, bias) in fused_conv_layers.items():
                    new_model.get_layer(layer_name).set_weights([kernel, bias])
                
                for layer_name, relu_cap in fused_relu_layers.items():
                    layer = new_model.get_layer(layer_name)
                    layer.cap = relu_cap
                    layer.set_weights([tf.constant([relu_cap], dtype=tf.float32)])

                self.model = new_model
            else:
                is_conv_quantization = self.current_stage == LearningStage.QUANTIZED_CONVOLUTION_TRAINING
                is_activation_quantization = self.current_stage == LearningStage.QUANTIZED_ACTIVATION_TRAINING

                eligible_types = None
                if is_conv_quantization:
                    eligible_types = QUANTIZABLE_CONVOLUTION
                elif is_activation_quantization:
                    eligible_types = QUANTIZABLE_ACTIVATION

                # Convolutional and Activation Layer Quantization
                if eligible_types is not None:
                    layers = [layer for layer in self.model.layers if is_eligible(layer, eligible_types)]
                    for layer in layers:
                        layer.quantize = True

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

    def fuse_convolutional_and_batch_norm(self,
                                          conv: Layer,
                                          prev_relu_cap,
                                          relu: ReLU,
                                          batch_normalization=None):
        if batch_normalization is not None and not isinstance(batch_normalization, BatchNormalization):
            raise AttributeError('batch_normalization must be BatchNormalization')

        conv_weights = conv.get_weights()
        if batch_normalization is not None:
            w_conv = conv_weights[0]
            if len(conv_weights) == 2:
                w_bias = conv_weights[1]
            else:
                w_bias = tf.zeros_like(w_conv)

            gamma, beta, mean, variance = batch_normalization.get_weights()
            eps = batch_normalization.epsilon
            if isinstance(conv, DepthwiseConv2D):
                gamma = gamma.reshape((1, 1, gamma.shape[0], 1))
                variance = variance.reshape((1, 1, variance.shape[0], 1))
            else:
                gamma = gamma.reshape((1, 1, 1, gamma.shape[0]))
                variance = variance.reshape((1, 1, 1, variance.shape[0]))

            std = np.sqrt(variance + eps)
            kernel = w_conv * gamma / std
            bias = beta + (w_bias - mean) * gamma / std
        elif len(conv_weights) == 2:
            kernel, bias = conv_weights
        else:
            kernel = conv_weights[0]
            bias = tf.zeros_like(kernel)

        # if len(conv_weights) == 2 or batch_normalization is None:
        #     # already fused
        #     kernel, bias = conv_weights
        # else:
        #     # Fuse batch normalization
        #     kernel = conv_weights[0]
        #     batch_gamma, batch_beta, batch_moving_mean, batch_moving_variance = batch_normalization.get_weights()
        #     eps = batch_normalization.epsilon
        #     std = np.sqrt(batch_moving_variance + eps)
        #     factor = batch_gamma / std
        #
        #     if isinstance(conv, DepthwiseConv2D):
        #         factor = factor.reshape((1, 1, factor.shape[0], 1))
        #
        #     kernel *= factor
        #     bias = batch_beta - batch_gamma / std * batch_moving_mean

        # Fuse using ReLU caps
        current_relu_cap = relu.cap
        max_activation = self.specification.max_activation(bits=relu.activation_bits)
        gain_bias = max_activation / current_relu_cap
        gain_kernel = prev_relu_cap / current_relu_cap

        kernel *= gain_kernel
        bias *= gain_bias
        return kernel, bias

