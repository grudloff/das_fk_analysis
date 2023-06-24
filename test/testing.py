import numpy as np
from typing import Tuple
import math

def neg(a):
    return -a if a!=0 else None

def filter_velocity_test(data: np.array, v: Tuple[float, ...], dt: float, dl: float, f_min: float=0, k_min: float=0) -> np.array:
    rows, cols = data.shape
    v_idx, h_idx = np.indices((rows, cols))  # Create an array of indices
    
    # Compute f and k indices
    f_idx = f_min + v_idx/(dt*rows*2)
    k_idx = k_min + (h_idx-cols//2)/(dl*cols) # TODO: Check even and odd offset difference

    v_min, v_max = v
    if v_max is np.inf:
        filtered_idx = f_idx >= np.abs(k_idx*v_min)
    else:
        filtered_idx = np.logical_and(f_idx >= np.abs(k_idx*v_min),
                                      f_idx <= np.abs(k_idx*v_max)
                                      )
    return filtered_idx

def crop_velocity_test(data: np.array, v: Tuple[float, ...], dt: float, dl: float) -> np.array:
    data = data.copy()
    data = data[::-1]
    filtered_idx = filter_velocity_test(data, v, dt, dl)
    data[np.logical_not(filtered_idx)]=0
    data = data[::-1]
    return data

def fk_crop(data: np.array, f: Tuple[float, ...],
                k:Tuple[float, ...], dt: float, dl: float):
    f_min, f_max = f
    k_min, k_max = k

    nF, nK = data.shape
    v_ind_min, v_ind_max = [int(nF - f*dt*nF*2) for f in [f_min, f_max]]
    h_ind_min, h_ind_max = [int((nK//2) - k*dl*nK) for k in [k_min, k_max]]
    print("h_ind_min, h_ind_max: ", h_ind_min, h_ind_max)
    
    data = np.zeros_like(data)
    data[v_ind_max:v_ind_min+1, h_ind_max:h_ind_min+1] = 1
    data[v_ind_max:v_ind_min+1, -(h_ind_min+1):neg(h_ind_max)] = 1
    return data
    
def vfk_crop_sum_test(data: np.array, v: Tuple[float, ...], f: Tuple[float, ...],
                k:Tuple[float, ...], dt: float, dl: float, testing: bool=False):
    f_min, f_max = f
    k_min, k_max = k
    
    # --- Slice first for k and f cropping ---
    nF, nK = data.shape
    # Convert to vertical and horizontal indices
    v_ind_min, v_ind_max = [int(nF - f*dt*nF*2) for f in [f_min, f_max]]
    h_ind_min, h_ind_max = [int((nK//2) - k*dl*nK) for k in [k_min, k_max]] # TODO: Check even and odd ofset difference
    
    data = np.concatenate((data[v_ind_max:v_ind_min+1, h_ind_max:min(h_ind_min+1, nK//2)],
                           data[v_ind_max:v_ind_min+1, -(h_ind_min+1):neg(h_ind_max)]), 
                          axis=-1)
    
    filtered_idx = filter_velocity_test(data, v, dt, dl, f_min, k_min)
    
    # Use the filtered indices to sum the corresponding values in the array
    total_sum = np.sum(data[filtered_idx])

    if testing:
        croped_data = data.copy()
        croped_data[np.logical_not(filtered_idx)] = 0
        croped_data=croped_data[::-1]
        return total_sum, croped_data

    return total_sum