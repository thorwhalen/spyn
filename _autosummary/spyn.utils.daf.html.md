# spyn.utils.daf

DataFrame helpers: variable-combination completion and cartesian product.

### Functions

| `cartesian_product`(df1, df2)                                                                     |                                                                                                                                                                                                                                                                                                                             |
|---------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `ch_col_names`(df[, new_names, old_names, inplace])                                               |                                                                                                                                                                                                                                                                                                                             |
| `complete_df_with_all_var_combinations`(df[, ...])                                                |                                                                                                                                                                                                                                                                                                                             |
| [`free_col_name`](#spyn.utils.daf.free_col_name)(df, candidate_cols[, raise_error]) | Will look for the first string in candidate_cols that is not a column name of df.                                                                                                                                                                                                                                           |
| `group_and_count`(df[, count_col, frequency])                                                     |                                                                                                                                                                                                                                                                                                                             |
| `intersect`(A, B)                                                                                 |                                                                                                                                                                                                                                                                                                                             |
| [`map_vals_to_ints_inplace`](#spyn.utils.daf.map_vals_to_ints_inplace)(df, cols_to_map)        | map_vals_to_ints_inplace(df, cols_to_map) will map columns of df inplace If cols_to_map is a string, a single column will be mapped to consecutive integers from 0 onwards. Elif cols_to_map is a dict, the function will attempt to map the columns dict.keys() using the (list-like)     maps specified in dict.values(). |
| [`reorder_as`](#spyn.utils.daf.reorder_as)(A, B)                                 | reorders A so as to respect the order in B.                                                                                                                                                                                                                                                                                 |
| [`reorder_columns_as`](#spyn.utils.daf.reorder_columns_as)(df, col_order[, inplace])     | reorders columns so that they respect the order in col_order.                                                                                                                                                                                                                                                               |
| `setdiff`(A, B)                                                                                   |                                                                                                                                                                                                                                                                                                                             |
| `sort_as`(sort_x, as_y, \*\*sorted_kwargs)                                                        |                                                                                                                                                                                                                                                                                                                             |

### spyn.utils.daf.free_col_name(df, candidate_cols, raise_error=True)

Will look for the first string in candidate_cols that is not a column name of df.
If no free column is found, will raise error (default) or return None.

### spyn.utils.daf.map_vals_to_ints_inplace(df, cols_to_map)

map_vals_to_ints_inplace(df, cols_to_map) will map columns of df inplace
If cols_to_map is a string, a single column will be mapped to consecutive integers from 0 onwards.
Elif cols_to_map is a dict, the function will attempt to map the columns dict.keys() using the (list-like)

> maps specified in dict.values()

Else assumed that cols_to_map is list-like specifying columns to be mapped

#### NOTE
If no mapping is specified (i.e. cols_to_map is not a dict), the function will return the mapping dict,
which is of the form mapping_dict[col][val] := mapped val for col.
This can be used to reverse the operation, i.e.:

mapping_dict = map_vals_to_ints_inplace(df, cols_to_map)
map_vals_to_ints_inplace(df, mapping_dict)
will leave df unchanged

This can also be used to apply the same mapping to multiple columns. To apply the same mapping to A and B:

```default
mapping_dict = map_vals_to_ints_inplace(df, ['A'])
map_vals_to_ints_inplace(df,
    cols_to_map={'B': dict(zip(mapping_dict['A'], range(len(mapping_dict['A']))))})
```

### spyn.utils.daf.reorder_as(A, B)

reorders A so as to respect the order in B.
Only the elements of A that are also in B will be reordered (and placed in front),
those that are not will be put at the end of the returned iterable, in their original order

### spyn.utils.daf.reorder_columns_as(df, col_order, inplace=False)

reorders columns so that they respect the order in col_order.
Only the columns of df that are also in col_order will be reordered (and placed in front),
those that are not will be put at the end of the returned dataframe, in their original order
