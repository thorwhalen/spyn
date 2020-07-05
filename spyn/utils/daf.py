import pandas as pd
from numpy import array, unique
from spyn.utils.ordered_set import OrderedSet


def complete_df_with_all_var_combinations(df, var_cols=None, fill_value=None):
    if var_cols is None:
        var_cols = df.columns
    multi_index = pd.MultiIndex.from_product([df[var].unique() for var in var_cols], names=var_cols)
    return df.set_index(var_cols).reindex(multi_index, fill_value=fill_value).reset_index()

def cartesian_product(df1, df2):
    join_col = 'this is the joining col that wont show up elsewhere'
    df1[join_col] = 1
    df2[join_col] = 1
    df = df1.merge(df2, on=join_col)
    df1.drop(labels=join_col, axis=1, inplace=True)
    df2.drop(labels=join_col, axis=1, inplace=True)
    df.drop(labels=join_col, axis=1, inplace=True)
    return df


def _force_list(list_wannabe):
    if isinstance(list_wannabe, str):
        return [list_wannabe]
    elif not isinstance(list_wannabe, list):
        return list(list_wannabe)
    else:
        return list_wannabe


def ch_col_names(df, new_names=(), old_names=None, inplace=False):
    # changes the names listed in new_names to the names listed in old_names
    new_names = _force_list(new_names)
    if isinstance(df, pd.Series):
        df = df.copy()
        df.name = new_names[0]
        return df
    else:
        if old_names is None:
            old_names = list(df.columns)
        else:
            old_names = _force_list(old_names)
        assert len(new_names) == len(old_names), "old_names and new_names must be the same length"
        # return df.rename(columns={k: v for (k, v) in zip(old_names, new_names)}, inplace=inplace)
        return df.rename(columns=dict(list(zip(old_names, new_names))), inplace=inplace)


def free_col_name(df, candidate_cols, raise_error=True):
    '''
    Will look for the first string in candidate_cols that is not a column name of df.
    If no free column is found, will raise error (default) or return None.
    '''
    for col in candidate_cols:
        if col not in df.columns:
            return col
    if raise_error:
        ValueError("All candidate_cols were already taken")
    else:
        return None


def group_and_count(df, count_col=None, frequency=False):
    if isinstance(df, pd.Series):
        t = pd.DataFrame()
        t[df.name] = df
        df = t
        del t
    count_col = count_col or free_col_name(df, ['count', 'gr_count'])
    d = df.copy()
    d[count_col] = 1
    d = d.groupby(list(df.columns)).count().reset_index()
    if frequency:
        d[count_col] /= float(d[count_col].sum())
    return d


def intersect(A, B):
    C = OrderedSet(B) & OrderedSet(A)
    try:
        return type(A)(C)
    except TypeError:
        return list(C)


def setdiff(A, B):
    # C = OrderedSet(union(B, A)) - OrderedSet(intersect(B, A)) # no, that's the symmetric difference!
    C = OrderedSet(A) - OrderedSet(intersect(B, A))
    try:
        return type(A)(C)
    except TypeError:
        return list(C)


def reorder_as(A, B):
    """
    reorders A so as to respect the order in B.
    Only the elements of A that are also in B will be reordered (and placed in front),
    those that are not will be put at the end of the returned iterable, in their original order
    """
    C = intersect(B, A) + setdiff(A, B)
    try:
        return type(A)(C)
    except TypeError:
        return list(C)


def reorder_columns_as(df, col_order, inplace=False):
    """
    reorders columns so that they respect the order in col_order.
    Only the columns of df that are also in col_order will be reordered (and placed in front),
    those that are not will be put at the end of the returned dataframe, in their original order
    """
    if hasattr(col_order, 'columns'):
        col_order = col_order.columns
    col_order = reorder_as(list(df.columns), list(col_order))
    if not inplace:
        return df[col_order]
    else:
        col_idx_map = dict(list(zip(col_order, list(range(len(col_order))))))
        col_idx = [col_idx_map[c] for c in df.columns]
        df.columns = col_idx
        df.sort_index(axis=1, inplace=True)
        df.columns = [col_order[x] for x in df.columns]


def sort_as(sort_x, as_y, **sorted_kwargs):
    return [x for (y, x) in sorted(zip(as_y, sort_x), **sorted_kwargs)]


def map_vals_to_ints_inplace(df, cols_to_map):
    """
    map_vals_to_ints_inplace(df, cols_to_map) will map columns of df inplace
    If cols_to_map is a string, a single column will be mapped to consecutive integers from 0 onwards.
    Elif cols_to_map is a dict, the function will attempt to map the columns dict.keys() using the (list-like)
        maps specified in dict.values()
    Else assumed that cols_to_map is list-like specifying columns to be mapped

    Note: If no mapping is specified (i.e. cols_to_map is not a dict), the function will return the mapping dict,
    which is of the form mapping_dict[col][val] := mapped val for col.
    This can be used to reverse the operation, i.e.:
        mapping_dict = map_vals_to_ints_inplace(df, cols_to_map)
        map_vals_to_ints_inplace(df, mapping_dict)
        will leave df unchanged
    This can also be used to apply the same mapping to multiple columns. To apply the same mapping to A and B:
        mapping_dict = map_vals_to_ints_inplace(df, ['A'])
        map_vals_to_ints_inplace(df,
            cols_to_map={'B': dict(zip(mapping_dict['A'], range(len(mapping_dict['A']))))})

    """
    mapping_dict = dict()

    if isinstance(cols_to_map, str):  # mapping a single column
        mapping_dict, df[cols_to_map] = unique(df[cols_to_map], return_inverse=True)
        mapping_dict = {cols_to_map: mapping_dict}
    elif isinstance(cols_to_map, dict):  # mapping with a user specified map
        assert set(cols_to_map.keys()).issubset(df.columns), "cols_to_map keys must be a subset of df columns"
        for c in list(cols_to_map.keys()):
            this_map = cols_to_map[c]
            if isinstance(this_map, dict):
                assert all(unique(list(this_map.values())) == array(list(range(len(this_map))))), \
                    "you must map to consecutive integers starting at 0"
                df[c] = df[c].apply(lambda x: this_map[x])
                mapping_dict[c] = sort_as(list(this_map.keys()), list(this_map.values()))
            else:
                df[c] = array(this_map)[list(df[c])]
                mapping_dict[c] = this_map
    else:  # mapping multiple columns
        for c in cols_to_map:
            mapping_dict.update(map_vals_to_ints_inplace(df, c))
    # return the mapping dict
    return mapping_dict
