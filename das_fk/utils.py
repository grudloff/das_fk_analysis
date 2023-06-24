import numpy as np
import gzip
from scipy.interpolate import interp1d

# Function to load the DAS data
def read_PASSCAL_segy(infile, nTraces=1250, nSample=900000, TraceOff=0):
    """Function to read PASSCAL segy raw data
    For Ridgecrest data, there are 1250 channels in total,
    Sampling rate is 250 Hz so for one hour data: 250 * 3600 samples
    """
    fs = nSample/ 3600 # sampling rate
    data = np.zeros((nTraces, nSample), dtype=np.float32)
    gzFile = False
    if infile.split(".")[-1] == "segy":
        fid = open(infile, 'rb')
    elif infile.split(".")[-1] == "gz":
        gzFile = True
        fid = gzip.open(infile, 'rb')
    fid.seek(3600)
    # Skipping traces if necessary
    fid.seek(TraceOff*(240+nSample*4),1)
    # Looping over traces
    for ii in range(nTraces):
        fid.seek(240, 1)
        if gzFile:
            # np.fromfile does not work on gzip file
            BinDataBuffer = fid.read(nSample*4) # read binary bytes from file
            data[ii, :] = struct.unpack_from(">"+('f')*nSample, BinDataBuffer)
        else:
            data[ii, :] = np.fromfile(fid, dtype=np.float32, count=nSample)
    fid.close()

    # Convert the phase-shift to strain (in nanostrain)
    Ridgecrest_conversion_factor = 1550.12 / (0.78 * 4 * np.pi * 1.46 * 8)
    data = data * Ridgecrest_conversion_factor

    das_time = np.arange(0, data.shape[1]) * 1 / fs

    return data, das_time

# downsample functions: downsample the DAS data to a given frequency fd
def downsample_das(data, das_time, fd=100):
    """
    Down sample das data to fd Hz
    data, das_time_downsampled = donwsample_das(data, das_time, fd=100)
    """
    das_dt_ds = 1/fd
    das_time_downsampled = np.arange(0, das_time[-1], das_dt_ds)

    downsample_f = interp1d(das_time, data, axis=1, bounds_error=False, fill_value=0)
    data = downsample_f(das_time_downsampled)
    return data, das_time_downsampled

import numpy as np

def _non_overlapping_sliding_window_view(arr, window_shape):
    """
    Create a non-overlapping sliding window view of a multidimensional array.

    Args:
        arr (ndarray): The input array.
        window_shape (tuple): The shape of the sliding window.

    Returns:
        ndarray: A view of the input array with non-overlapping sliding windows.

    """
    # exchange axes
    window_shape = window_shape[::-1]

    # Compute the shape of the output view
    shape = tuple(np.floor_divide(arr.shape, window_shape))
    new_shape = shape + window_shape

    # Compute the new strides for the view
    new_strides = tuple(np.array(arr.strides) * window_shape) + arr.strides

    # Create the sliding window view using stride tricks
    strided_arr = np.lib.stride_tricks.as_strided(arr, shape=new_shape, strides=new_strides)
    # exchange axis
    output = strided_arr.swapaxes(-1,-2)
    return output

def split_time_spatial(data, dT, dL, dt, dl):
    # This returns a sliding window view of the data (no realocation).
    # TODO: This may be efficient in memory but slow, should check alternatives
    nT = int(dT/dt)
    nL = int(dL/dl)
    data_view = _non_overlapping_sliding_window_view(data, window_shape=(nT, nL))
    return data_view