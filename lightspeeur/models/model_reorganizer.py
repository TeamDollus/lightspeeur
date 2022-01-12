from tensorflow.keras import Model
from tensorflow.keras.layers import InputLayer


def get_inbound_layers(layer):
    # keep returns an array to make easy to process them
    try:
        inbound_layers = layer.inbound_nodes[0].inbound_layers
    except IndexError:
        print(layer)
        print()
        print(type(layer))
        print()
        print(layer.inbound_nodes)
        print(len(layer.inbound_nodes))
        print(type(layer.inbound_nodes))
        raise ValueError('WTF')
    return inbound_layers if type(inbound_layers) is list else [inbound_layers]


def is_input_layer(layer):
    inbound_layers = get_inbound_layers(layer)
    return len(inbound_layers) == 0 and isinstance(layer, InputLayer)


def check_popped_layer_and_get(layer, popped_layers):
    for popped_layer in popped_layers:
        if layer.name == popped_layer.name:
            return popped_layer
    return None


def organize_layer(layer, inputs):
    # unwrap array performed from the reorganizer
    kwargs = {}
    if len(layer.inbound_nodes) > 0:
        kwargs = layer.inbound_nodes[0].call_kwargs

    if len(inputs) == 1:
        return layer(inputs[0], **kwargs)
    else:
        return layer(inputs, **kwargs)


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

        # Get all available layers from inbound layers and popped layers' inbound layers
        inbound_layers = flatten_inbound_popped_layers(layer, popped_layers)

        # keep outputs be arrays to make easy to process
        outputs = []
        for inbound_layer in inbound_layers:
            outputs += outputs_map[inbound_layer.name]

        x = organize_layer(layer, outputs)

        # keep outputs map be arrays to make easy to process
        outputs_map[layer.name] = [x]

    inputs = [layer.input for layer in inputs]
    return Model(inputs=inputs, outputs=x, name=name)
