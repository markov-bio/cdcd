import einops

def bmult(a,b):
    """element-wise mutiplication of two tensors along the batch dimention

    Args:
        a (torch.Tensor): shape=(b,...)
        b (torch.Tensor): shape=(b)

    Returns:
        torch.Tensor: shape=(b,...)
    """
    return einops.einsum(a,b,'b ..., b -> b ...')