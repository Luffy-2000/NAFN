def get_padding(kernel, padding='same'):
    """Calculate padding size
    Args:
    kernel: convolution kernel size
    padding: padding type ('same' or 'valid')
    Returns:
    tuple: padding size (left/top padding, right/bottom padding)
    """
    pad = kernel - 1
    if padding == 'same':
        if kernel % 2:
            return pad // 2, pad // 2
        else:
            return pad // 2, pad // 2 + 1
    return 0, 0


def get_output_dim(dimension, kernels, strides, dilatation=1, padding='same', return_paddings=False):
    """Calculate the output dimension after convolution/pooling
    Args:
    dimension: input dimension
    kernels: convolution kernel size list
    strides: stride list
    dilatation: dilation coefficient
    padding: padding type ('same' or 'valid')
    return_paddings: whether to return padding value
    Returns:
    int or tuple: output dimension, if return_paddings is True, it also returns padding value
    """
    out_dim = dimension
    paddings = []
    if padding == 'same':
        for kernel, stride in zip(kernels, strides):
            paddings.append(get_padding(kernel, padding))
            out_dim = (out_dim + stride - 1) // stride
    else:
        for kernel, stride in zip(kernels, strides):
            paddings.append(get_padding(kernel, padding))
            out_dim = (out_dim - kernel + stride) // stride

    if return_paddings:
        return out_dim, paddings
    return out_dim 