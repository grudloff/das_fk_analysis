import numpy as np
import matplotlib.pyplot as plt

# plot the DAS data
def show_data(data, das_time, pclip=99.5):
    fig, ax1 = plt.subplots(figsize=(8,4))
    clipVal = np.percentile(np.absolute(data), pclip)
    ax1.imshow(data.T, 
                extent=[0, data.shape[0], das_time[-1], das_time[0]],
                aspect='auto', vmin=-clipVal, vmax=clipVal, cmap=plt.get_cmap('seismic'))

    ax1.set_xlabel("Channel number")
    ax1.set_ylabel("Time [s]")
    ax1.grid()
    return fig, ax1

def plot_fk(data, frequency, wavenumber):
    if isinstance(float, frequency) and isinstance(float, wavenumber):
        max_frequency = max_frequency
        min_frequency = 0
        max_wavenumber = wavenumber
        min_wavenumber = - wavenumber
    elif isinstance(np.ndarray, frequency) and isinstance(np.ndarray, wavenumber):
        max_frequency = max(frequency)
        min_frequency = min(frequency)
        max_wavenumber = max(wavenumber)
        min_wavenumber = min(wavenumber)
    else:
        raise ValueError("Both frequency and wavenumber should be floats or arrays")
    extent = [min_wavenumber, max_wavenumber, min_frequency, max_frequency]
    plt.figure(figsize=(12,4))
    plt.imshow(data, cmap='hot',aspect = max_wavenumber/max_frequency,
            extent=extent, norm = "log")
    plt.colorbar()
    plt.xlabel("Wavenumber")
    plt.ylabel("Frequency")

# def plot_velocity(v, frequency, wavenumber, ax, color="w"):

#     max_freq = max(frequency)
#     max_wav = max(wavenumber)
    
#     nF = len(frequency)
#     nK = len(wavenumber)

#     # Intercept with sides
#     if(max_freq > max_wav*v):
#         idx = (np.abs(frequency - max_wav*v)).argmin()
#         idx_left = max(idx - 1, 0)
#         idx_right = min(idx + 1, nF-1)
#         interp_value = np.interp(max_wav*v, [frequency[idx_left], frequency[idx_right]],
#                                            [idx_left, idx_right])
#         freq_id_interp = interp_value.item()

#         # points at intercepts and at origin
#         line = np.array([[0, freq_id_interp],
#                          [np.abs(wavenumber - 0).argmin(), len(frequency)],
#                          [len(wavenumber)-1, freq_id_interp]],
#                         dtype=np.float64)
#     # Intercept with ceiling
#     else:     
#         idx = (np.abs(max_freq/v - wavenumber)).argmin()
#         idx_left = max(idx - 1, 0)
#         idx_right = min(idx + 1, nK-1)
#         interp_value = np.interp(max_freq/v, [wavenumber[idx_left], wavenumber[idx_right]],
#                                            [idx_left, idx_right])
#         wav_id_interp = interp_value.item()

#         # points at intercepts and at origin
#         line = np.array([[wav_id_interp, 0],
#                          [np.abs(wavenumber - 0).argmin(), len(frequency)],
#                          [len(wavenumber)-(wav_id_interp+1), 0]],
#                         dtype=np.float64)
    
#     # shift to middle of blocks
#     line+=0.5

#     #plot
#     ax.plot(*line.T, color+'--', alpha=0.5)

def plot_fk(data, frequency, wavenumber, norm='log'):
    if isinstance(frequency, float) and isinstance(wavenumber, float):
        max_frequency = max_frequency
        min_frequency = 0
        max_wavenumber = wavenumber
        min_wavenumber = - wavenumber
    elif isinstance(frequency, np.ndarray) and isinstance(wavenumber, np.ndarray):
        max_frequency = max(frequency)
        min_frequency = min(frequency)
        max_wavenumber = max(wavenumber)
        min_wavenumber = min(wavenumber)
    else:
        raise ValueError("Both frequency and wavenumber should be floats or arrays")
    extent = [min_wavenumber, max_wavenumber, min_frequency, max_frequency]
    fig, ax = plt.subplots(figsize=(12,4))
    plt.imshow(data, cmap='rocket',aspect = max_wavenumber/max_frequency,
            extent=extent, interpolation='none', norm = norm)
    plt.colorbar()
    plt.xlabel("Wavenumber")
    plt.ylabel("Frequency")
    return fig,  ax

def plot_velocity(v, frequency, wavenumber, ax, color="w"):

    max_freq = max(frequency)
    max_wav = max(wavenumber)
    min_wav = min(wavenumber)

    # Intercept with sides
    if(max_freq > max_wav*v):
        # points at intercepts and at origin
        line = np.array([[min_wav, max_wav*v],
                         [0, 0],
                         [max_wav, max_wav*v]])
    # Intercept with ceiling
    else:     
        # points at intercepts and at origin
        line = np.array([[-max_freq/v, max_freq],
                         [0, 0],
                         [max_freq/v, max_freq]])

    #plot
    ax.plot(*line.T, color+'--', alpha=0.7)

def timeseries_ridgeplot(data, dT=1, dL=1, figsize=(12,4)):
    """
    Plots multiple timeseries as walls with the height of the timeseries values.
    Each timeseries is displayed one on top of the other in the same axis, with
    the zero for each timeseries equidistant to its neighbors.
    
    Parameters:
        data (2D array): A 2D array representing multiple timeseries. Each row
            in the array represents a timeseries. Shape: (L, T).
        dT (float): Length in time of each step.
        dL (float): Length in space of each step between timeseries.
    """
    
    # Get the number of timeseries
    num_timeseries = len(data)
    
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Set the colors for the timeseries walls
    colors = plt.cm.viridis(np.linspace(0, 1, num_timeseries))
    
    # Calculate the offset for each timeseries
    offset = np.arange(num_timeseries, 0, -1) * 2 * np.max(np.std(data, axis=1))
    
    # Iterate over the timeseries
    for i, series in enumerate(data):
        # Calculate the x-values for the wall
        x = np.arange(len(series))*dT
        
        # Calculate the y-values for the wall
        y = np.zeros_like(series)
        
        # Plot the wall as a filled polygon
        ax.plot(x, series + offset[i], "w", linewidth=0.5)
        ax.fill_between(x, y + offset[i], series + offset[i], color=colors[i])
        
    # Set the y-ticks and labels
    ax.set_yticks(offset)
    ax.set_yticklabels(dL*np.arange(num_timeseries))
    
    # Set the x-axis label
    ax.set_xlabel('Time')
    ax.set_ylabel('Channel Position')
    return fig, ax