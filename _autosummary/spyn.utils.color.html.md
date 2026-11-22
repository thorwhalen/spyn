# spyn.utils.color

Color and colormap utilities for matplotlib-based visualizations.

### Functions

| `get_colorbar_tick_labels_as_floats`(cbar)                                                       |                                                |
|--------------------------------------------------------------------------------------------------|------------------------------------------------|
| [`shifted_color_map`](#spyn.utils.color.shifted_color_map)(cmap[, start, midpoint, ...]) | Function to offset the "center" of a colormap. |

### spyn.utils.color.shifted_color_map(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap')

Function to offset the “center” of a colormap. Useful for
data with a negative min and positive max and you want the
middle of the colormap’s dynamic range to be at zero

## Input

> cmap : The matplotlib colormap to be altered
> start : Offset from lowest point in the colormap’s range.

> > Defaults to 0.0 (no lower ofset). Should be between
> > 0.0 and `midpoint`.

> midpoint
> : 0.5 (no shift). Should be between 0.0 and 1.0. In
>   general, this should be  1 - vmax/(vmax + abs(vmin))
>   For example if your data range from -15.0 to +5.0 and
>   you want the center of the colormap at 0.0, `midpoint`
>   should be set to  1 - 5/(5 + 15)) or 0.75

> stop
> : Defaults to 1.0 (no upper ofset). Should be between
>   `midpoint` and 1.0.
