from collections import OrderedDict
from itertools import product

import numpy as np
from scipy.interpolate import RegularGridInterpolator


def interpolate_templates(templates, mag_col, var_cols=['EB_V', 'z'], **kwargs):
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RegularGridInterpolator.html#scipy.interpolate.RegularGridInterpolator
    axes, grid = construct_grid(templates, mag_col, var_cols)
    return RegularGridInterpolator(axes, grid, **kwargs)


def construct_grid(df, mag_col, var_cols):
    # operates on 1 model template (filtered to df)

    index_value_mapping = OrderedDict()
    for var_col in var_cols:
        var_values = df[var_col].unique() # TODO exclude nans
        index_value_mapping[var_col] = OrderedDict(zip(var_values, range(len(var_values))))
    axes = [list(mapping.keys()) for _, mapping in index_value_mapping.items()]

    grid = np.zeros([len(values) for values in axes])
    for row in df[var_cols + [mag_col]].itertuples(name='variables'):
        resulting_index = []
        for var_col in var_cols:  # can do this in one line, but very unclear
            var_value = getattr(row, var_col)
            var_index = index_value_mapping[var_col][var_value]
            resulting_index.append(var_index)
        grid[tuple(resulting_index)] = getattr(row, mag_col)

    return axes, grid
