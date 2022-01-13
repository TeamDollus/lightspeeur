import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import InputLayer


def get_inbound_layers(layer):
    # keep returns an array to make easy to process them
    inbound_layers = layer.inbound_nodes[0].inbound_layers
    return inbound_layers if type(inbound_layers) is list else [inbound_layers]


def is_input_layer(layer):
    inbound_layers = get_inbound_layers(layer)
    return len(inbound_layers) == 0 and isinstance(layer, InputLayer)


def check_popped_layer_and_get(layer, popped_layers):
    for popped_layer in popped_layers:
        if layer.name == popped_layer.name:
            return popped_layer
    return None


def check_tensor_scope(tensor, name):
    if tensor.name == name:
        return True
    elif '/' in tensor.name:
        return tensor.name.startswith('{}/'.format(name))
    else:
        return False


def find_outputs_from_tensor(outputs_map, tensor, popped_layers):
    for key in outputs_map.keys():
        if check_tensor_scope(tensor, key):
            return outputs_map[key]

    for layer in popped_layers:
        if check_tensor_scope(tensor, layer.name):
            outputs = []
            for inbound_layer in get_inbound_layers(layer):
                outputs += find_outputs_from_tensor(outputs_map, inbound_layer, popped_layers)
            return outputs

    raise ValueError('Failed to find proper tensor from outputs map. Expected \'{}\''.format(tensor.name))


def parse_tensor_call_args(outputs_map, elements, args, popped_layers):
    for v in elements:
        if tf.is_tensor(v):
            args += find_outputs_from_tensor(outputs_map, v, popped_layers)
        elif isinstance(v, (list, tuple)):
            args.append(parse_tensor_call_args(outputs_map, v, [], popped_layers))
        elif isinstance(v, dict):
            args.append(parse_tensor_call_kwargs(outputs_map, v, {}, popped_layers))
        else:
            args.append(v)
    return args


def parse_tensor_call_kwargs(outputs_map, elements, kwargs, popped_layers):
    for k, v in elements.items():
        if tf.is_tensor(v):
            tensors = find_outputs_from_tensor(outputs_map, v, popped_layers)
            if len(tensors) == 1:
                kwargs[k] = tensors[0]
            else:
                kwargs[k] = tensors
        elif isinstance(v, (list, tuple)):
            kwargs[k] = parse_tensor_call_args(outputs_map, v, [], popped_layers)
        elif isinstance(v, dict):
            kwargs[k] = parse_tensor_call_kwargs(outputs_map, v, {}, popped_layers)
        else:
            kwargs[k] = v
    return kwargs


def organize_layer(layer, outputs_map, popped_layers, force=False, recreate=True):
    # unwrap array performed from the reorganizer
    args = []
    kwargs = {}

    # force = True to organize layer with outputs map values forcibly
    if force:
        for value in outputs_map.values():
            args += value
    elif len(layer.inbound_nodes) > 0:
        node = layer.inbound_nodes[0]
        args = parse_tensor_call_args(outputs_map, node.call_args, args, popped_layers)
        kwargs = parse_tensor_call_kwargs(outputs_map, node.call_kwargs, kwargs, popped_layers)
    if recreate:
        config = layer.get_config()
        replicated = layer.__class__.from_config(config)
        replicated(*args, **kwargs)
        replicated.set_weights(layer.get_weights())
        return replicated.output
    else:
        return layer(*args, **kwargs)


def flatten_inbound_popped_layers(layer, popped_layers):
    layers = get_inbound_layers(layer)
    inbound_layers = []
    for layer in layers:
        popped_layer = check_popped_layer_and_get(layer, popped_layers)
        if popped_layer is not None:
            inbound_layers += get_inbound_layers(popped_layer)
        else:
            inbound_layers.append(layer)
    return inbound_layers


def reorganize_layers(name, layers, popped_layers):
    outputs_map = {}

    # Prepare input layers
    inputs = [layer for layer in layers if is_input_layer(layer)]
    for layer in inputs:
        outputs_map[layer.name] = [layer.output]

    # Prepare layers
    x = None
    for layer in layers:
        if is_input_layer(layer):
            continue

        # organize layers from inbound nodes recursively
        x = organize_layer(layer, outputs_map, popped_layers)

        # keep outputs map be arrays to make easy to process
        outputs_map[layer.name] = [x]

    inputs = [layer.input for layer in inputs]
    return Model(inputs=inputs, outputs=x, name=name)
