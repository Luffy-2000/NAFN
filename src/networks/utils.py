def get_padding(kernel, padding='same'):
    """计算padding大小
    Args:
        kernel: 卷积核大小
        padding: padding类型 ('same' 或 'valid')
    Returns:
        tuple: padding的大小 (左/上padding, 右/下padding)
    """
    pad = kernel - 1
    if padding == 'same':
        if kernel % 2:
            return pad // 2, pad // 2
        else:
            return pad // 2, pad // 2 + 1
    return 0, 0


def get_output_dim(dimension, kernels, strides, dilatation=1, padding='same', return_paddings=False):
    """计算卷积/池化后的输出维度
    Args:
        dimension: 输入维度
        kernels: 卷积核尺寸列表
        strides: 步长列表
        dilatation: 膨胀系数
        padding: padding类型 ('same' 或 'valid')
        return_paddings: 是否返回padding值
    Returns:
        int 或 tuple: 输出维度，如果return_paddings为True则同时返回padding值
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