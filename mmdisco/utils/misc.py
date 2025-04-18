def set_torch_flags(allow_matmul_tf32=False, allow_cudnn_tf32=False):
    import torch

    torch.backends.cuda.matmul.allow_tf32 = allow_matmul_tf32
    torch.backends.cudnn.allow_tf32 = allow_cudnn_tf32
