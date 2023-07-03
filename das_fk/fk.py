import numpy as np
import warnings
from typing import Tuple

def to_fk_domain(signal, dt=None, dl=None, rotate=True, odd_K=True):
    """
    Computes the real 2D FFT of a real signal with multiple equidistant spatial channels.

    Parameters:
        signal (ndarray): Array with shape (L, T) where L represents the number of spatial channels and T represents the number of time samples.
        dt (float): Time differential.
        dl (float): Spatial differential.
        rotate (bool): Whether to rotate the output. If True, the output will have shape (K, F). If False, the output will have shape (F, K). Defaults to True.
        odd_K (bool): Wether the wavenumber length is forced to be odd. This ensures DC is at the center.

    Returns:
        ndarray: FFT output with shape (K, F) if rotate is True, else (F, K).
        ndarray or None: Frequencies corresponding to the time axis if dt is provided, None otherwise.
        ndarray or None: Wavenumbers corresponding to the spatial axis if dl is provided, None otherwise.
    """

    L, T = signal.shape[-2:]
    
    # Make odd, so that 0 freq is in the middle
    if odd_K:
        L = L - (L+1)%2
    
    # Compute FFT
    output = np.fft.rfft2(signal, s=(L, T), axes=(-2,-1))
    output = np.fft.fftshift(output, axes=-2)
    
    if rotate:
        output = np.rot90(output, axes=(-2,-1))
    
    frequency = None
    wavenumber = None
    
    if dt is not None and dl is not None:
        # Compute frequencies and wavenumbers
        frequency = np.fft.rfftfreq(T, d=dt)
        wavenumber = np.fft.fftshift(np.fft.fftfreq(L, d=dl))
        if rotate:
            frequency = frequency[::-1]
    
    return output, frequency, wavenumber

def _compute_v_range(k: float, v: Tuple[float, ...], f: Tuple[float, ...], dt: float, nF: int) -> Tuple[int, int]:
    """
    Compute the vertical index range (v_min, v_max) corresponding to a given wavenumber k.

    Args:
        k (float): Wavenumber value.
        v (Tuple[float, ...]): Tuple containing velocity range (v_min, v_max).
        f (Tuple[float, ...]): Tuple containing frequency range (f_min, f_max).
        dt (float): Time step.
        nF (int): Number of frequencies.

    Returns:
        Tuple[int, int]: Vertical index range (v_min, v_max).
    """
    v_min, v_max = v
    f_min, f_max = f
    
    # Compute frequency ranges for given velocity constraint and wavenumber
    # then, compare to the frequency constraint
    f_i_min = max(k * v_min, f_min)
    f_i_max = np.nanmin([k * v_max, f_max])

    # Compute vertical index for given frequencies
    v_i_min = int(nF - f_i_min * (dt * nF * 2))
    v_i_max = int(nF - f_i_max * (dt * nF * 2))
    # Handle overflow
    v_i_min = max(v_i_min, -1)
    v_i_max = max(v_i_max, 0)

    return v_i_min, v_i_max

def vfk_crop_sum(data: np.array, v: Tuple[float, ...], f: Tuple[float, ...],
                     k: Tuple[float, ...], dt: float, dl: float, testing: bool = False):
    """
    Crop and sum a frequency-wavenumber representation obtained from a 2DFFT.

    Args:
        data (np.array): 2D array representing frequency-wavenumber data.
        v (Tuple[float, ...]): Tuple containing velocity range (v_min, v_max).
        f (Tuple[float, ...]): Tuple containing frequency range (f_min, f_max).
        k (Tuple[float, ...]): Tuple containing wavenumber range (k_min, k_max).
        dt (float): Time step.
        dl (float): Spatial step.
        testing (bool, optional): If True, return a modified copy of the input data with non-cropped values set to zero. Defaults to False.

    Returns:
        np.array or Tuple[np.array, np.array]: Total value obtained by summing the cropped regions. If testing=True, also returns the modified copy of the input data.

    Raises:
        ValueError: If f or k ranges are invalid or inconsistent with data dimensions.
    """

    nF, nK = data.shape[-2:]  # number of frequencies & wavelengths
    
    # Input checks for f and k
    max_f = ((2*nF-1)/2)/(dt*2*nF)
    max_k = ((nK-1)/2)/(dl*nK)
    if not (0 <= f[0] <= f[1] <= max_f):
        warnings.warn(f"\nInvalid frequency range. Valid range: (0, {max_f})\n")
    if not (0 <= k[0] <= k[1] <= max_k):
        warnings.warn(f"\nInvalid wavenumber range. Valid range: (0, {max_k})\n")
    if data.shape[-1] % 2 == 0:
        raise ValueError("The wavenumber length should be odd.")

    if testing:
        data_testing = np.zeros_like(data, dtype=bool)

    # Filter k indices
    k_min, k_max = k
    k_max_id = int(nK // 2 - k_max * dl * nK)
    k_min_id = int(nK // 2 - k_min * dl * nK)
    h_idx = np.arange(k_max_id, k_min_id + 1)  # Horizontal indices
    
    # Exclude zero from for loop
    h_zero = False
    if k_min==0:
        h_idx = h_idx[:-1]
        h_zero = nK//2
        
    total = np.zeros(data.shape[:-2])
    for h_i in h_idx:
        k_i = (nK // 2 - h_i) / (dl * nK)  # convert horizontal index to wavenumber

        v_i_min, v_i_max = _compute_v_range(k_i, v, f, dt, nF)

        total += np.sum(data[..., v_i_max:v_i_min + 1, h_i], axis=-1)  # sum negative side
        total += np.sum(data[..., v_i_max:v_i_min + 1, -(h_i + 1)], axis=-1)  # sum positive side
        
        if testing:
            data_testing[..., v_i_max:v_i_min + 1, h_i] = 1
            data_testing[..., v_i_max:v_i_min + 1, -(h_i + 1)] = 1
                
    if h_zero:
        k_i = (nK // 2 - h_zero) / (dl * nK)
        v_i_min, v_i_max = _compute_v_range(k_i, v, f, dt, nF)
        total += np.sum(data[..., v_i_max:v_i_min + 1, h_i], axis=-1)
        if testing:
            data_testing[..., v_i_max:v_i_min + 1, h_zero] = 1

    if testing:       
        data_copy = np.where(data_testing, data, 0)
        return total, data_copy

    return total
