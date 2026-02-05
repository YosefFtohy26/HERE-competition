import torch
from numpy.lib.stride_tricks import sliding_window_view

def _sliding_window_snapshots(arr , kernel=None , stride=None):
    """
    Extract sliding-window patches from a 2D image or multi-channel image and
    return them as a flat torch.Tensor of patches.

    Parameters
    ----------
    arr : numpy.ndarray
        Input array. Allowed shapes:
            - (H, W)            : single-channel image
            - (C, H, W)         : (channels, height, width)
            - (N, C, H, W)      : already batched
        The function standardizes these into a batched array of shape (N, C, H, W).

    kernel : tuple[int, int]
        Patch size as (kH, kW). Required.

    stride : int
        Stride step used when sampling sliding windows (applies equally to height and width). Required.

    Returns
    -------
    torch.Tensor
        Tensor of extracted patches with shape (P, C, kH, kW), where:
            - kH, kW = kernel
            - C = number of channels
            - P = total number of patches across the batch (N * H_pos_stride * W_pos_stride)

    Raises
    ------
    AssertionError
        If `kernel` or `stride` is None (they must be provided).
    ValueError
        If `arr` dimensionality is unsupported or kernel is larger than the image axes (raised by sliding_window_view).
    TypeError
        If `arr` is not a NumPy array.

    Notes
    -----
    - The function uses numpy.lib.stride_tricks.sliding_window_view to create windowed views,
      then copies the selected windows to a contiguous array before converting to torch.
    - The final `.squeeze()` may remove singleton dimensions (for example if C==1),
      so the returned tensor shape may lose the channel axis when C==1. If you require a stable
      4D shape, avoid `.squeeze()`.
    - The dtype of the returned torch tensor follows the dtype of the input numpy array
      (e.g., np.float32 -> torch.float32).
    """

    if kernel is None or stride is None:
        raise AssertionError("kernel and stride must be given values")

    # arr shape: (C,H,W) or (H,W)
    #arr shape has to be standaralized to (N,C,H,W)
    if len(arr.shape)==3: #(C , H , W) --> (1 , C , H , W)
        arr = arr[None,...]
        C = arr.shape[1]
    elif len(arr.shape)==2: #(H , W) --> (1 , 1 , H , W)
        arr = arr[None,None,...]
        C = 1
    else:
        C=3
        
    H, W = kernel
    windows = sliding_window_view(arr, window_shape=(H,W), axis=(-2,-1))
    windows = windows[:, :, ::stride, ::stride, :, :].copy()
    windows = torch.from_numpy(windows).permute( 0 , 2 , 3 , 1 , 4 , 5 ).reshape(-1, C, H, W)
    return windows.squeeze()
