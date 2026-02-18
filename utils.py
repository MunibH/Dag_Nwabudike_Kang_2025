import h5py
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def h5_to_dict(h5_item):
    data_dict = {}

    for key, item in h5_item.items():
        if isinstance(item, h5py.Dataset):
            # Check if it's a scalar (shape is empty tuple)
            if item.shape == ():
                # Use [()] to safely extract a scalar value
                data_dict[key] = item[()]
            else:
                # Use [:] for actual arrays/matrices
                data_dict[key] = item[:]
        elif isinstance(item, h5py.Group):
            data_dict[key] = h5_to_dict(item)
            
    return data_dict

def reload_data(path_h5):
    # don't use this to load the data (use h5_to_dict instead), but this is a function that loads the data in the same way as the original Julia code, for comparison purposes
    # path_h5 = f"/data1/candy/data/processed_h5/{data_uid}-data.h5"

    with h5py.File(path_h5, 'r') as h5f:
        # Loading datasets from the 'behavior' group
        behavior = h5f['behavior']
        velocity = behavior['velocity'][:]
        # .abs() equivalent in NumPy for Julia's abs.(velocity)
        speed = np.abs(velocity)
        
        rev_bin = behavior['reversal_vec'][:]
        rev_start_end = behavior['reversal_events'][:]
        stage_x = behavior['stage_x'][:]
        stage_y = behavior['stage_y'][:]
        pumping = behavior['pumping'][:]
        
        head_curvature = behavior['head_angle'][:]
        
        head_curv_deriv = np.gradient(head_curvature)
        
        # Loading datasets from the 'gcamp' group
        gcamp = h5f['gcamp']
        traces_array = gcamp['trace_array'][:]
        traces_array_F_F20 = gcamp['traces_array_F_F20'][:]
        traces_array_original = gcamp['trace_array_original'][:]
        
        # Loading datasets from the 'timing' group
        time_encounter = h5f['timing']['time_food_encounter'][:]

    return (
        velocity, speed, rev_bin, rev_start_end, stage_x, stage_y, 
        pumping, head_curvature, head_curv_deriv, traces_array, 
        traces_array_F_F20, traces_array_original, time_encounter
    )

def default_plt_params():
    font = {'family' : 'Arial',
            'weight' : 'normal',
            'size'   : 15}
    matplotlib.rc('font', **font)

def pretty_plot(figsize=(6,4), tick_dir='out', tick_length=5, tick_width=1, spine_width=0.75, fontsize=20, top_border=False, right_border=False):
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams.update({'font.size': fontsize})
    plt.rcParams['svg.fonttype'] = 'none'
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    ax.tick_params(direction=tick_dir, length=tick_length, width=tick_width)
    for spine in ax.spines.values():
        spine.set_linewidth(spine_width)
    if not top_border:
        ax.spines['top'].set_visible(False)
    if not right_border:
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    return fig, ax