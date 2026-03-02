import h5py
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import json
from typing import Dict, Any

fs = 15 # default font size for plots

def load_metadata(input_dir: str) -> Dict[str, Any]:
    """Load metadata.json and return a dict (with numpy arrays for lists)."""
    meta_path = os.path.join(input_dir, 'metadata.json')
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    meta['bad_frames'] = np.array(meta['bad_frames'], dtype=int)
    meta['frame_index'] = np.array(meta['frame_index'], dtype=int)
    return meta

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
            'size'   : fs}
    matplotlib.rc('font', **font)

def pretty_plot(figsize=(6,4), tick_dir='out', tick_length=5, tick_width=1, spine_width=0.75, fontsize=fs, top_border=False, right_border=False):
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


def pretty_subplots(nrows=1, ncols=1, figsize=(10, 6), tick_dir='out', tick_length=5,
                    tick_width=1, spine_width=0.75, fontsize=fs, top_border=False,
                    right_border=False, sharex=False, sharey=False, hspace=None, wspace=None):
    """
    Like pretty_plot but creates a figure with a grid of subplots.
    Returns (fig, axes, nexttile) where axes is the 2D array of Axes
    and nexttile is a helper to iterate through them (see nexttile docs).

    Usage:
        fig, axes, nt = pretty_subplots(2, 3, figsize=(12, 8))

        ax = nt()          # gets axes[0,0]
        ax.plot(x, y1)

        ax = nt()          # gets axes[0,1]
        ax.plot(x, y2)

        # ... continues row-by-row, like MATLAB's nexttile

        # You can also jump to a specific tile (1-indexed):
        ax = nt(5)         # gets the 5th tile (row 1, col 2 in a 2x3 grid)

        # Or access axes directly:
        axes[0, 2].plot(x, y3)

    Parameters:
        nrows, ncols : int
            Number of subplot rows and columns.
        figsize : tuple
            Figure size in inches.
        sharex, sharey : bool or str
            Share x/y axes across subplots (passed to plt.subplots).
        hspace, wspace : float or None
            Vertical/horizontal spacing between subplots.
        (other params same as pretty_plot)
    """
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams.update({'font.size': fontsize})
    plt.rcParams['svg.fonttype'] = 'none'

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize,
                             constrained_layout=(hspace is None and wspace is None),
                             sharex=sharex, sharey=sharey)

    if hspace is not None or wspace is not None:
        fig.subplots_adjust(hspace=hspace, wspace=wspace)

    # Ensure axes is always 2D for consistent indexing
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes[np.newaxis, :]
    elif ncols == 1:
        axes = axes[:, np.newaxis]

    # Apply styling to every subplot
    for ax in axes.flat:
        ax.tick_params(direction=tick_dir, length=tick_length, width=tick_width)
        for spine in ax.spines.values():
            spine.set_linewidth(spine_width)
        if not top_border:
            ax.spines['top'].set_visible(False)
        if not right_border:
            ax.spines['right'].set_visible(False)

    # Create a nexttile iterator bound to this figure's axes
    nt = _Nexttile(axes)

    return fig, axes, nt


class _Nexttile:
    """
    MATLAB-style nexttile iterator for subplots.

    Cycles through axes in row-major order (left-to-right, top-to-bottom),
    just like MATLAB's nexttile / tiledlayout.

    Usage (after creating subplots with pretty_subplots):
        fig, axes, nt = pretty_subplots(2, 3)

        ax = nt()        # 1st tile  — axes[0,0]
        ax.plot(...)

        ax = nt()        # 2nd tile  — axes[0,1]
        ax.bar(...)

        ax = nt(5)       # jump to 5th tile (1-indexed) — axes[1,1]
        ax.imshow(...)

        nt.reset()       # reset counter back to the first tile
    """

    def __init__(self, axes):
        self._axes_flat = axes.flat
        self._n = axes.size
        self._idx = 0

    def __call__(self, tile_number=None):
        """
        Return the next axes, or jump to a specific tile.

        Parameters:
            tile_number : int or None
                If None, returns the next tile in order.
                If an int (1-indexed), jumps to that tile and advances
                the counter past it.
        Returns:
            matplotlib Axes for the requested tile.
        """
        if tile_number is not None:
            # 1-indexed like MATLAB
            idx = tile_number - 1
            if idx < 0 or idx >= self._n:
                raise IndexError(f"Tile {tile_number} is out of range (1–{self._n}).")
            self._idx = idx + 1  # advance counter past this tile
            return self._axes_flat[idx]

        if self._idx >= self._n:
            raise StopIteration(
                f"All {self._n} tiles have been used. Call nt.reset() to start over."
            )
        ax = self._axes_flat[self._idx]
        self._idx += 1
        return ax

    def reset(self):
        """Reset the tile counter back to the first subplot."""
        self._idx = 0