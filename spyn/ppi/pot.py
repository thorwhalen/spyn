"""Potentials: probability/likelihood tables with algebraic operators.

The Pot class represents discrete probability tables as DataFrames with a 'pval'
column and uses operator overloading for probabilistic inference:

- ``*`` factor product
- ``/`` normalization, conditioning, or factor division
- ``[]`` marginalization (list/str) or slicing (dict)
- ``>>`` projection (marginalization)

>>> p = Pot({'A': [0, 0, 1, 1], 'B': [0, 1, 0, 1], 'pval': [2, 4, 5, 9]})
>>> p['A']  # marginalize to A
     pval
A
0       6
1      14
>>> (p / []).round(2)  # normalize
        pval
A B
0 0     0.10
  1     0.20
1 0     0.25
  1     0.45
"""

from __future__ import annotations

__author__ = 'thor'

from collections import Counter
from collections.abc import Iterable
from decimal import Decimal
from functools import cached_property, reduce
from typing import Any

import numpy as np
import pandas as pd

import spyn.utils.order_conserving as colloc
from spyn.utils.daf import (
    cartesian_product,
    ch_col_names,
    complete_df_with_all_var_combinations,
    group_and_count,
    map_vals_to_ints_inplace,
    reorder_columns_as,
)


class Pot:
    """A discrete potential (probability/likelihood table).

    The core data structure is a pandas DataFrame (``self.tb``) with variable
    columns and a mandatory ``pval`` column holding the values.

    >>> p = Pot({'A': [0, 1], 'B': [0, 1], 'pval': [0.3, 0.7]})
    >>> p.vars
    ['A', 'B']
    >>> p.n
    2
    """

    def __init__(self, data: Pot | float | int | pd.DataFrame | dict | None = None):
        if isinstance(data, Pot):
            self.tb = data.tb
        elif isinstance(data, (float, int)):
            self.tb = pd.DataFrame([{'pval': data}])
        elif data is not None:
            if isinstance(data, pd.DataFrame):
                assert 'pval' in data.columns, "dataframe had no pval column"
                self.tb = data
            elif isinstance(data, dict):
                if 'pval' not in list(data.keys()):
                    data = dict(data, pval=len(data[list(data.keys())[0]]) * [1])
                self.tb = pd.DataFrame(data=data)
            else:
                try:
                    self.tb = data.tb.copy()
                except Exception:
                    raise ValueError("Unknown construction type")
        else:
            self.tb = pd.DataFrame({'pval': 1}, index=[''])  # default "unit" potential
        self.tb.index = range(len(self.tb))

    @cached_property
    def vars(self) -> list[str]:
        """Variable names (all columns except 'pval').

        >>> Pot({'X': [0, 1], 'pval': [1, 1]}).vars
        ['X']
        """
        return [c for c in self.tb.columns if c != 'pval']

    @cached_property
    def vars_set(self) -> set[str]:
        """Set of variable names."""
        return set(self.vars)

    @property
    def n(self) -> int:
        """Number of rows in the table."""
        return len(self.tb)

    def _ipython_key_completions_(self):
        return self.vars

    # -------------------------------------------------------------------
    # OPERATIONS
    # -------------------------------------------------------------------

    def get_slice(self, intercept_dict: dict[str, Any]) -> Pot:
        """Return sub-pot at specific variable values.

        >>> p = Pot({'A': [0, 0, 1, 1], 'B': [0, 1, 0, 1], 'pval': [2, 4, 5, 9]})
        >>> p[{'A': 1}]  # slice where A=1
           pval
        B
        0     5
        1     9
        """
        tb = self.tb.copy()
        for k, v in intercept_dict.items():
            tb = tb[tb[k] == v]
            del tb[k]
        return self.__class__(tb)

    def select(self, selection: callable) -> Pot:
        """Filter rows using a callable predicate on row dicts."""
        assert callable(selection), f"selection needs to be a callable. Was a {type(selection)}"
        tb = pd.DataFrame(list(filter(selection, self.tb.to_dict(orient='records'))))
        return self.__class__(tb[self.vars + ['pval']])

    def project_to(
        self, var_list: list[str] | str, *, assert_subset: bool = False
    ) -> Pot:
        """Marginalize to a subset of variables.

        Non-strict by default: ignores variables not in the pot.

        >>> p = Pot({'A': [0, 0, 1, 1], 'B': [0, 1, 0, 1], 'pval': [2, 4, 5, 9]})
        >>> p.project_to('A')  # sum out B
             pval
        A
        0       6
        1      14
        """
        var_list = _ascertain_list(var_list)

        if assert_subset and not self.vars_set.issuperset(var_list):
            raise KeyError(
                "You requested strict projection, so "
                "you can only project onto variables that are in the pot. "
                f"You wanted {var_list}. The pot has these vars: {self.vars}"
            )
        else:
            var_list = colloc.intersect(var_list, self.vars)

        if var_list:
            return self.__class__(self.tb[var_list + ['pval']].groupby(var_list).sum().reset_index())
        else:
            return self.__class__(pd.DataFrame({'pval': self.tb['pval'].sum()}, index=[0]))

    def __rshift__(self, var_list) -> Pot:
        """Syntactic sugar: ``pot >> ['A']`` is ``pot.project_to(['A'])``."""
        return self.project_to(var_list)

    def normalize(self, var_list: list[str] | str | tuple = ()) -> Pot:
        """Normalize the pot with respect to var_list.

        If var_list is empty, fully normalizes (divides by total sum).
        Otherwise, divides by the projection onto var_list (conditional prob).

        >>> p = Pot({'A': [0, 1], 'pval': [3, 7]})
        >>> p.normalize()  # full normalization
           pval
        A
        0   0.3
        1   0.7
        """
        return self / self.project_to(var_list)

    def __or__(self, item):
        """Deprecated. Use ``/`` instead."""
        import warnings
        warnings.warn(
            "Pot.__or__ (|) is deprecated for normalization. Use / instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if isinstance(item, str):
            return self / self.project_to([item])
        elif isinstance(item, list):
            return self / self.project_to(item)
        elif isinstance(item, dict):
            intercept_dict = item
            var_list = colloc.intersect(self.vars, list(intercept_dict.keys()))
            return (self / self.project_to(var_list)).get_slice(intercept_dict)
        else:
            raise TypeError('Unknown item type')

    def __getitem__(self, item: dict | list | tuple | str | None) -> Pot:
        """Polymorphic access: slice (dict), project (list/str), or sum (None/[]).

        >>> p = Pot({'A': [0, 0, 1, 1], 'B': [0, 1, 0, 1], 'pval': [2, 4, 5, 9]})
        >>> p[['A']]  # project to A
             pval
        A
        0       6
        1      14
        >>> p[{'A': 0}]  # slice where A=0
           pval
        B
        0     2
        1     4
        """
        match item:
            case dict():
                return self.get_slice(item)
            case list() | tuple():
                return self.project_to(list(item))
            case str():
                return self.project_to(item)
            case None:
                return self.__class__(pd.DataFrame({'pval': self.tb['pval'].sum()}, index=[0]))
            case _:
                if not item:  # handles empty list, False, 0, etc.
                    return self.__class__(pd.DataFrame({'pval': self.tb['pval'].sum()}, index=[0]))
                raise TypeError(
                    f"Unknown type for item (must be None, dict, list, or string). Was: {type(item)}"
                )

    def add_count(self, num: float | int) -> Pot:
        """Add a count to all variable combinations (including missing ones)."""
        tb = complete_df_with_all_var_combinations(self.tb, var_cols=self.vars, fill_value=0)
        tb['pval'] += num
        return self.__class__(tb)

    def __add__(self, pot: Pot | float | int) -> Pot:
        if isinstance(pot, self.__class__):
            common_vars = list(set(self.vars).intersection(pot.vars))
            tb = pd.merge(pot.tb, self.tb, how='outer', on=common_vars).fillna(0)
            return self.__class__(_val_add_(tb))
        else:
            return self.add_count(pot)

    def __mul__(self, pot: Pot) -> Pot:
        """Factor product of two potentials.

        >>> a = Pot({'X': [0, 1], 'pval': [0.4, 0.6]})
        >>> b = Pot({'X': [0, 0, 1, 1], 'Y': [0, 1, 0, 1], 'pval': [0.9, 0.1, 0.3, 0.7]})
        >>> joint = a * b
        >>> joint.n
        4
        """
        return self.__class__(_val_prod_(self._merge_(pot)))

    def __div__(self, item):
        """Division: by Pot, or normalization by str/list/dict.

        >>> p = Pot({'A': [0, 0, 1, 1], 'B': [0, 1, 0, 1], 'pval': [2, 4, 5, 9]})
        >>> p / 'A'  # P(B|A) = P(A,B) / P(A)
                  pval
        A B
        0 0  0.333333
          1  0.666667
        1 0  0.357143
          1  0.642857
        """
        if isinstance(item, Pot):
            return self.__class__(_val_div_(self._merge_(item)))
        elif isinstance(item, str):
            return self.normalize([item])
        elif isinstance(item, list):
            return self.normalize(item)
        elif isinstance(item, dict):
            intercept_dict = item
            var_list = colloc.intersect(self.vars, list(intercept_dict.keys()))
            return self.normalize(var_list).get_slice(intercept_dict)
        else:
            raise TypeError('Unknown item type')

    def __truediv__(self, item) -> Pot:
        return self.__div__(item)

    def assimilate(self, pot: Pot) -> Pot:
        """Bayes rule: multiply by pot, normalize, project back to own vars.

        This computes P(X|D) ∝ P(D|X) * P(X), i.e. prior.assimilate(likelihood).

        >>> prior = Pot({'H': [0, 1], 'pval': [0.5, 0.5]})
        >>> likelihood = Pot({'H': [0, 1], 'pval': [0.2, 0.8]})
        >>> posterior = prior.assimilate(likelihood)
        >>> posterior.n
        2
        """
        return self.__mul__(pot).normalize([]).project_to(self.vars)

    def unassimilate(self, pot: Pot) -> Pot:
        """Inverse of assimilate."""
        return self.__div__(pot).normalize([]).project_to(self.vars)

    # -------------------------------------------------------------------
    # UTILITIES
    # -------------------------------------------------------------------

    def order_vars(
        self, var_list: list[str] | str, *, sort_pts: bool = True
    ) -> Pot:
        """Return a new Pot with reordered variables.

        >>> p = Pot({'B': [0, 1], 'A': [0, 1], 'pval': [3, 7]})
        >>> p.order_vars(['A', 'B']).vars
        ['A', 'B']
        """
        new_tb = reorder_columns_as(self.tb.copy(), _ascertain_list(var_list))
        result = self.__class__(new_tb)
        if sort_pts:
            result = result.sort_pts()
        return result

    def sort_pts(self, var_list: list[str] | None = None, **kwargs) -> Pot:
        """Return a new Pot with rows sorted by variables."""
        var_list = var_list or self.vars
        new_tb = self.tb.sort_values(by=var_list, **kwargs).reset_index(drop=True)
        return self.__class__(new_tb)

    @property
    def pval(self) -> pd.Series:
        """The pval column as a Series."""
        return self.tb.pval

    @property
    def values(self) -> np.ndarray:
        """The pval column as a numpy array."""
        return np.array(self.tb.pval)

    def pval_of(self, var_val_dict: dict[str, Any], default_val: float = 0.0) -> float:
        """Look up the pval for a specific variable assignment.

        >>> p = Pot({'A': [0, 1], 'pval': [3, 7]})
        >>> int(p.pval_of({'A': 1}))
        7
        """
        t = self.get_slice(var_val_dict)
        n = len(t.tb)
        if n == 0:
            return default_val
        elif n == 1:
            return t.tb.pval.iloc[0]
        else:
            raise RuntimeError("In pval_of(): get_slice returned more than one value")

    def binarize(self, var_values_to_map_to_1_dict: dict) -> Pot:
        """Map specified variables to {0, 1} and re-aggregate."""
        tb = self.tb.copy()
        for var_name, vals_to_map_to_1 in var_values_to_map_to_1_dict.items():
            if not hasattr(vals_to_map_to_1, '__iter__'):
                vals_to_map_to_1 = [vals_to_map_to_1]
            lidx = tb[var_name].isin(vals_to_map_to_1)
            tb[var_name] = 0
            tb.loc[lidx, var_name] = 1
        tb = tb.groupby(self.vars).sum().reset_index(drop=False)
        return self.__class__(tb)

    def round(self, ndigits: int | None = None, *, inplace: bool = False) -> Pot:
        """Round pval values.

        >>> Pot({'A': [0, 1], 'pval': [0.333, 0.667]}).round(2)
           pval
        A
        0  0.33
        1  0.67
        """
        if ndigits is None:
            import math
            ndigits = abs(int(math.log10(self.tb['pval'].min()))) + 1 + 2
        rounded_pvals = [round(x, ndigits) for x in self.tb['pval']]
        if inplace:
            self.tb['pval'] = rounded_pvals
            return self
        else:
            x = self.__class__(self)
            x.tb['pval'] = rounded_pvals
            return x

    def rect_perspective_df(self) -> pd.DataFrame:
        """For a 2-variable pot, return a matrix-form DataFrame.

        >>> p = Pot({'A': [0, 0, 1, 1], 'B': [0, 1, 0, 1], 'pval': [1, 2, 3, 4]})
        >>> p.rect_perspective_df()
        B  0  1
        A
        0  1  2
        1  3  4
        """
        vars = self.vars
        assert len(self.vars) == 2, "You can only get the rect_perspective_df of a pot with exactly two variables"
        return self.tb.set_index([vars[0], vars[1]]).unstack(vars[1])['pval']

    # -------------------------------------------------------------------
    # Internal
    # -------------------------------------------------------------------

    def _merge_(self, pot: Pot) -> pd.DataFrame:
        """Merge (join) two pots on their common variables."""
        on = colloc.intersect(self.vars, pot.vars)
        if on:
            return pd.merge(self.tb, pot.tb, how='inner', on=on, sort=True, suffixes=('_x', '_y'))
        else:
            return cartesian_product(self.tb, pot.tb)

    def __str__(self) -> str:
        return self.tb.__repr__()

    def __repr__(self) -> str:
        if self.vars:
            return self.tb.set_index(self.vars).__str__()
        else:
            return self.tb.__repr__()

    # -------------------------------------------------------------------
    # FACTORIES
    # -------------------------------------------------------------------

    @classmethod
    def binary_pot(cls, varname: str, prob: float = 1) -> Pot:
        """Create a binary variable potential.

        >>> Pot.binary_pot('coin', 0.6)
              pval
        coin
        0      0.4
        1      0.6
        """
        return cls(pd.DataFrame({varname: [0, 1], 'pval': [1 - prob, prob]}))

    @classmethod
    def zero_potential(cls, varnames_and_scopes) -> Pot:
        """Create a pot with all variable combinations and pval=0.

        >>> pot = Pot.zero_potential({'A': 2, 'B': [10, 20]})
        >>> pot.n
        4
        """
        from itertools import product

        if isinstance(varnames_and_scopes, dict):
            varnames_and_scopes = list(varnames_and_scopes.items())

        def gen():
            for t in varnames_and_scopes:
                if isinstance(t, str) or not isinstance(t, Iterable):
                    t = [t, 2]
                varname, scope = t
                if isinstance(scope, int):
                    scope = list(range(scope))
                yield varname, scope

        varnames_and_scopes = dict(gen())

        names = list(varnames_and_scopes.keys()) + ['pval']
        vals = np.array(list(product(*varnames_and_scopes.values(), [0]))).T

        return cls({k: v for k, v in zip(names, vals)})

    @classmethod
    def from_points_to_count(
        cls,
        pts: pd.DataFrame | list,
        vars: list[str] | None = None,
        scope_for_var=None,
    ) -> Pot:
        """Create a count pot from a collection of data points.

        >>> import pandas as pd
        >>> df = pd.DataFrame({'A': [0, 0, 1, 1, 1], 'B': [0, 1, 0, 0, 1]})
        >>> Pot.from_points_to_count(df).n
        4
        """
        if isinstance(pts, pd.DataFrame):
            pot = cls(group_and_count(pts, count_col='pval'))
        else:
            pot = cls.from_counter(Counter(map(tuple, pts)), vars)

        if scope_for_var is not None:
            if isinstance(scope_for_var, int):
                val_for_all = scope_for_var
                scope_for_var = {var: val_for_all for var in pot.vars}
            else:
                def ensure_iterable_scope(scope):
                    if isinstance(scope, int):
                        scope = list(range(scope))
                    return scope

                scope_for_var = {var: ensure_iterable_scope(scope) for var, scope in scope_for_var.items()}
                assert set(scope_for_var) == set(pot.vars), (
                    f"set(scope_for_var) == set(pot.vars)\n"
                    f"set(scope_for_var)={set(scope_for_var)}\n"
                    f"set(pot.vars)={set(pot.vars)}\n"
                )

            return pot + cls.zero_potential(scope_for_var)
        else:
            return pot

    @classmethod
    def from_counter(cls, counts: Counter, vars: list[str] | None = None) -> Pot:
        """Create a pot from a Counter of tuples."""
        if vars is None:
            example_key = list(counts.keys())[0]
            vars = [str(i) for i in range(len(example_key))]
        return cls(pd.DataFrame(
            [dict(pval=v, **{kk: vv for kk, vv in zip(vars, k)}) for k, v in counts.items()])
        )

    @classmethod
    def from_count_df_to_count(
        cls, count_df: pd.DataFrame, count_col: str = 'pval'
    ) -> Pot:
        """Create a pot from a DataFrame with a count column."""
        pot_vars = list(colloc.setdiff(count_df.columns, [count_col]))
        tb = count_df[pot_vars + [count_col]].groupby(pot_vars).sum().reset_index()
        tb = ch_col_names(tb, 'pval', count_col)
        return cls(tb)

    @classmethod
    def from_points_to_bins(cls, pts: pd.DataFrame, **kwargs) -> Pot:
        """Create a pot from points with binning (alias for count)."""
        if isinstance(pts, pd.DataFrame):
            tb = group_and_count(pts)
            tb = ch_col_names(tb, 'pval', 'count')
            return cls(tb)

    @classmethod
    def from_hard_evidence(cls, **var_val) -> Pot:
        """Create a singleton potential for hard evidence.

        >>> Pot.from_hard_evidence(X=1, Y=0).n
        1
        """
        df = pd.DataFrame([{var: val for var, val in var_val.items()}])
        df['pval'] = 1
        return cls.from_count_df_to_count(df)

    @classmethod
    def rand(
        cls,
        n_var_vals: list[int] | None = None,
        var_names: list[str] | None = None,
        granularity: float | None = None,
        try_to_get_unique_values: bool | int = False,
    ) -> Pot:
        """Generate a random probability pot.

        >>> p = Pot.rand([2, 3])
        >>> p.n
        6
        >>> bool(abs(p.values.sum() - 1.0) < 1e-10)
        True
        """
        if n_var_vals is None:
            n_var_vals = [2, 2]
        assert len(n_var_vals) <= 26, "You can't request more than 26 variables"
        if var_names is None:
            var_names = [str(chr(x)) for x in range(ord('A'), ord('Z'))]
        assert len(n_var_vals) <= len(var_names)
        assert min(np.array(n_var_vals)) >= 2

        df = reduce(cartesian_product,
                    [pd.DataFrame(data=list(range(x)), columns=[y]) for x, y in zip(n_var_vals, var_names)])

        n_vals = len(df)

        def _get_random_pvals():
            if granularity is None:
                if n_vals > 18:
                    x = np.random.rand(n_vals)
                    return x / sum(x)
                elif n_vals == 4:
                    return np.random.permutation([0.1, 0.2, 0.3, 0.4])
                else:
                    if n_vals <= 12:
                        return _rand_numbers_summing_to_one(n_vals, 0.05)
                    else:
                        return _rand_numbers_summing_to_one(n_vals, 0.01)
            else:
                return _rand_numbers_summing_to_one(n_vals, granularity)

        if try_to_get_unique_values:
            if not isinstance(try_to_get_unique_values, int):
                try_to_get_unique_values = 1000
            for i in range(try_to_get_unique_values):
                pvals = _get_random_pvals()
                if len(np.unique(pvals)) == n_vals:
                    break
        else:
            pvals = _get_random_pvals()

        df['pval'] = list(map(float, pvals))

        return cls(df)

    def prob_of(self, var_val_dict: dict[str, Any]) -> float:
        """Get probability for a variable assignment."""
        t = self.get_slice(var_val_dict)
        n = len(t.tb)
        if n == 0:
            return 0.0
        elif n == 1:
            return t.tb.pval.iloc[0]
        else:
            raise RuntimeError("In prob_of(): get_slice returned more than one value")

    def given(self, conditional_vars) -> Pot:
        """Create conditional potential: P(remaining | conditional_vars)."""
        return self.__class__(self.__div__(conditional_vars))

    def relative_risk(
        self,
        event_var: str,
        exposure_var: str,
        event_val: int = 1,
        exposure_val: int = 1,
        smooth_count: float | None = None,
    ) -> float:
        """Compute epidemiological relative risk from a joint pot."""
        prob = self >> [event_var, exposure_var]
        if smooth_count is not None:
            prob = prob.count_pot_to_prob_pot(prior_count=smooth_count)
        prob = prob.binarize({event_var: event_val, exposure_var: exposure_val})
        prob_when_exposed = (prob / {exposure_var: 1})[{event_var: 1}]
        prob_when_not_exposed = (prob / {exposure_var: 0})[{event_var: 1}]
        rel_risk = (prob_when_exposed / prob_when_not_exposed).values
        if rel_risk.size:
            return rel_risk[0]
        else:
            return np.nan

    def count_pot_to_prob_pot(
        self, prior_count: float = 1, possible_vals_for_var=None
    ) -> Pot:
        """Convert count pot to probability pot with Laplace-like smoothing.

        Adds prior_count to all counts, then normalizes.
        """
        if possible_vals_for_var is not None:
            raise NotImplementedError("possible_vals_for_var is not yet implemented")
        c = (self + prior_count).tb
        c['pval'] /= c['pval'].sum()
        return self.__class__(c)


class ProbPot(Pot):
    """A Pot subclass for probability-specific operations."""

    def __init__(self, data=None):
        super().__init__(data=data)

    @staticmethod
    def plot_relrisk_matrix(relrisk):
        import matplotlib.pyplot as plt
        from spyn.utils.color import shifted_color_map, get_colorbar_tick_labels_as_floats

        t = relrisk.copy()
        matrix_shape = (t['exposure'].nunique(), t['event'].nunique())
        m = map_vals_to_ints_inplace(t, cols_to_map=['exposure'])
        m = m['exposure']
        map_vals_to_ints_inplace(t, cols_to_map={'event': dict(list(zip(m, list(range(len(m))))))})
        RR = np.zeros(matrix_shape)
        RR[t['exposure'], t['event']] = t['relative_risk']
        RR[list(range(len(m))), list(range(len(m)))] = np.nan

        RRL = np.log2(RR)

        def normalizor(X):
            min_x = np.nanmin(X)
            range_x = np.nanmax(X) - min_x
            return lambda x: (x - min_x) / range_x

        normalize_this = normalizor(RRL)
        center = normalize_this(0)

        color_map = shifted_color_map(cmap=plt.cm.get_cmap('coolwarm'), start=0, midpoint=center, stop=1)
        plt.imshow(RRL, cmap=color_map, interpolation='none')

        plt.xticks(list(range(np.shape(RRL)[0])), m, rotation=90)
        plt.yticks(list(range(np.shape(RRL)[1])), m)
        cbar = plt.colorbar()
        cbar.ax.set_yticklabels(
            ["%.02f" % x for x in np.exp2(np.array(get_colorbar_tick_labels_as_floats(cbar)))])


# -------------------------------------------------------------------
# Data Prep utils
# -------------------------------------------------------------------

def from_points_to_binary(d, mid_fun=np.median):
    """Convert data to binary based on a midpoint function."""
    dd = d.copy()
    columns = d.columns
    for c in columns:
        dd[c] = list(map(int, d[c] > mid_fun(d[c])))
    return dd


# -------------------------------------------------------------------
# Module-level utility functions
# -------------------------------------------------------------------

def relative_risk(joint_prob_pot, event_var, exposure_var):
    """Compute relative risk from a joint probability pot."""
    prob = joint_prob_pot >> [event_var, exposure_var]
    return ((prob / {exposure_var: 1})[{event_var: 1}]
            / (prob / {exposure_var: 0})[{event_var: 1}]).values[0]


def _val_prod_(tb):
    """Multiply pval_x and pval_y columns into pval."""
    tb['pval'] = tb['pval_x'] * tb['pval_y']
    tb.drop(labels=['pval_x', 'pval_y'], axis=1, inplace=True)
    return tb


def _val_div_(tb):
    """Divide pval_x by pval_y. 0/0 → 0."""
    tb['pval'] = np.true_divide(tb['pval_x'], tb['pval_y']).fillna(0)
    tb.drop(labels=['pval_x', 'pval_y'], axis=1, inplace=True)
    return tb


def _val_add_(tb):
    """Add pval_x and pval_y columns into pval."""
    tb['pval'] = tb['pval_x'] + tb['pval_y']
    tb.drop(labels=['pval_x', 'pval_y'], axis=1, inplace=True)
    return tb


def _ascertain_list(x):
    """Ensure x is a list. Strings become [str], iterables become list(iter)."""
    if not isinstance(x, list):
        if isinstance(x, str):
            x = [x]
        elif hasattr(x, '__iter__') and not isinstance(x, dict):
            x = list(x)
        else:
            x = [x]
    return x


def _rand_numbers_summing_to_one(n_numbers, granularity=0.01):
    """Generate n random numbers summing to 1 with given granularity."""
    n_choices = 1.0 / granularity
    assert round(n_choices) == int(n_choices), "granularity must be an integer divisor of 1.0"
    x = np.linspace(granularity, 1.0 - granularity, int(n_choices) - 1)
    x = sorted(x[np.random.choice(list(range(1, len(x))), size=n_numbers - 1, replace=False)])
    x = np.concatenate([[0.0], x, [1.0]])
    x = np.diff(x)
    x = np.array([Decimal(xi).quantize(Decimal(str(granularity))) for xi in x])
    return x
