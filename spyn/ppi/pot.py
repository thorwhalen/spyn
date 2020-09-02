__author__ = 'thor'

from typing import Iterable

import pandas as pd
from numpy import *
import numpy as np
from collections import Counter
from functools import reduce

import spyn.utils.order_conserving as colloc
# from spyn.utils.color import shifted_color_map, get_colorbar_tick_labels_as_floats
from spyn.utils.daf import cartesian_product, ch_col_names, group_and_count, reorder_columns_as
from spyn.utils.daf import map_vals_to_ints_inplace, complete_df_with_all_var_combinations
from spyn.util import lazyprop


# import ut.pplot.get


class Pot(object):
    def __init__(self, data=None):
        if isinstance(data, Pot):
            self.tb = data.tb
        elif isinstance(data, float) or isinstance(data, int):
            self.tb = pd.DataFrame([{'pval': data}])
        elif data is not None:
            if isinstance(data, pd.DataFrame):
                # inject the dataframe in the tb attribute: It's the potential data
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
        self.tb.index = [''] * len(self.tb)

    @lazyprop
    def vars(self):
        return [c for c in self.tb.columns if c != 'pval']
        # return colloc.setdiff(list(self.tb.columns), ['pval'])

    @lazyprop
    def vars_set(self):
        return set(self.vars)

    def _ipython_key_completions_(self):
        return self.vars

    ###########################################
    # OPERATIONS
    ###########################################
    def get_slice(self, intercept_dict):
        """
        Return sub-pot going through specific "intercept points"
        For example, if X is a pot on ABC, then X.get_slice({'A':0, 'B':1}) is the pot on C taken from ABC where
        A=0 and B=1.
        It's like a subplane of points defined by given axis intercepts.
        """
        tb = self.tb.copy()
        for k, v in intercept_dict.items():
            tb = tb[tb[k] == v]
            del tb[k]
        return self.__class__(tb)

    def select(self, selection):
        """
        Select a sub-pot
        :param selection:
        :return:
        """
        assert callable(selection), f"selection needs to be a callable. Was a {type(selection)}"
        tb = pd.DataFrame(list(filter(selection, self.tb.to_dict(orient='records'))))
        return self.__class__(tb[self.vars + ['pval']])

    def project_to(self, var_list, assert_subset=False):
        """Project to a subset of variables (marginalize out other variables)
        Note that this projection is not strict; It will NOT complain if your list contains extra vars it doesn't have.
        It will simply take the intersection of the it's vars, with the vars you requested.
        This can be confusing, and lead to bugs when it's not what you desire.
        If you want the strict version, request it by doing ``assert_subset=True``
        """
        var_list = _ascertain_list(var_list)

        if assert_subset and not self.vars_set.issuperset(var_list):
            print('adifasdkfjalsdkjflaksjdlfkjasldkf')
            raise KeyError(
                "You requested strict projection, so "
                "you can only project onto variables that are in the pot. "
                f"You wanted {var_list}. The pot has these vars: {self.vars}"
            )
        else:
            var_list = colloc.intersect(var_list, self.vars)

        if var_list:  # if non-empty, marginalize out other variables
            return self.__class__(self.tb[var_list + ['pval']].groupby(var_list).sum().reset_index())
        else:  # if var_list is empty, return a singleton potential containing the sum of the vals of self.tb
            return self.__class__(pd.DataFrame({'pval': self.tb['pval'].sum()}, index=['']))

    def __rshift__(self, var_list):
        return self.project_to(var_list)

    def normalize(self, var_list=()):
        """
        'Normalization' of the pot with respect to _var_list.
        Will define the pot by the projection of the pot on a subset of the variables.

        Note: If this subset is the empty set, this will correspond to "full normalization", i.e. dividing the vals by
        the sum of all vals.

        Use:
            * This can be used to transform a count potential into a probability potential
            (if your sample is large enough!)
            * Conditional Probability: P(A|B) = P(AB) / P(B)
        """
        return self / self.project_to(var_list)

    def __or__(self, item):
        """
        If item is empty/none/false, a string or a list, it normalizes according to item.
        If item is a dict, it normalizes according to the keys, and slices according to the dict.
        --> This resembles P(A|B=1) kind of thing...
        """
        print("I'm trying to discourage using | now (might want to use it for fuzzy logic at some point")
        print("--> Use / instead of |. ")
        if isinstance(item, str):
            return self / self.project_to([item])
        elif isinstance(item, list):
            return self / self.project_to(item)
        elif isinstance(item, dict):
            intercept_dict = item
            var_list = colloc.intersect(self.vars, list(intercept_dict.keys()))
            return (self / self.project_to(var_list)).get_slice(intercept_dict)
        else:
            TypeError('Unknown item type')

    def __getitem__(self, item):
        """
        This function is called when accessing the pot with [] brackets, and will return a slice of projection of the
        pot depending on the type of item.
        """
        if item:
            if isinstance(item, dict):
                return self.get_slice(item)
            elif isinstance(item, (list, tuple)):
                return self.project_to(item)
            elif isinstance(item, str):
                return self.project_to(item)
            else:
                raise TypeError(
                    "Unknown type for item (must be None, dict, list, or string). Was: {}".format(type(item)))
        else:
            return self.__class__(pd.DataFrame({'pval': self.tb['pval'].sum()}, index=['']))

    def add_count(self, num):
        tb = complete_df_with_all_var_combinations(self.tb, var_cols=self.vars, fill_value=0)
        tb['pval'] += num
        return self.__class__(tb)

    def __add__(self, pot):
        # TODO: See if both cases are consistent
        if isinstance(pot, self.__class__):
            common_vars = list(set(self.vars).intersection(pot.vars))
            tb = pd.merge(pot.tb, self.tb, how='outer', on=common_vars).fillna(0)
            return self.__class__(_val_add_(tb))
        else:
            return self.add_count(pot)
        # if isinstance(y, float) | isinstance(y, int):
        #     self.tb['pval'] += y
        # else:
        #     pass

    def __mul__(self, pot):
        """
        Multiply two potentials
        """
        return self.__class__(_val_prod_(self._merge_(pot)))

    def __div__(self, item):
        """
        Operation depends on what item's type is. If item is a:
            Pot: perform potential division (like multiplication but with pvals divided).
            empty/none/false, a string or a list: normalize according to item.
            dict: it normalizes according to the keys, and slices according to the dict.
        --> This resembles P(A|B=1) kind of thing...
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

    def __truediv__(self, item):
        return self.__div__(item)

    def assimilate(self, pot):
        """
        Assimilate information given by input pot (returning the result).
        Assimilation means multiplication followed by a projection to the original variables.
        This is used, for example, when wanting to compute P(X|D=data) as the normalization of P(D=data|X) * P(X)
        (Bayes rule). We can write that as P(X) absorbing P(D=data|X). The result has the dimensions of X.
        """
        return self.__mul__(pot).normalize([]).project_to(self.vars)

    def unassimilate(self, pot):
        """
        Inverse of assimilate.
        """
        return self.__div__(pot).normalize([]).project_to(self.vars)

    ###########################################
    # Usable UTILS
    ###########################################
    def order_vars(self, var_list, sort_pts=True):
        self.tb = reorder_columns_as(self.tb, _ascertain_list(var_list))
        if sort_pts:
            self.sort_pts()
        return self

    def sort_pts(self, var_list=None, **kwargs):
        var_list = var_list or self.vars
        self.tb = self.tb.sort_values(by=var_list, **kwargs)
        return self

    @property
    def pval(self):
        return self.tb.pval

    @property
    def values(self):
        return np.array(self.tb.pval)

    def pval_of(self, var_val_dict, default_val=0.0):
        t = self.get_slice(var_val_dict)
        n = len(t.tb)
        if n == 0:
            return default_val
        elif n == 1:
            return t.tb.pval[0]
        else:
            raise RuntimeError("In pval_of(): get_slice returned more than one value")

    def binarize(self, var_values_to_map_to_1_dict):
        """
        maps specified variables to {0, 1}
            var_values_to_map_to_1_dict is a {variable_name: values to map to 1} specification dict
        """
        tb = self.tb.copy()
        for var_name, vals_to_map_to_1 in var_values_to_map_to_1_dict.items():
            if not hasattr(vals_to_map_to_1, '__iter__'):
                vals_to_map_to_1 = [vals_to_map_to_1]
            lidx = tb[var_name].isin(vals_to_map_to_1)
            tb[var_name] = 0
            tb.loc[lidx, var_name] = 1
        tb = tb.groupby(self.vars).sum().reset_index(drop=False)
        return self.__class__(tb)

    def round(self, ndigits=None, inplace=False):
        if ndigits is None:
            ndigits = abs(int(math.log10(self.tb['pval'].min()))) + 1 + 2
            # print(ndigits)
        rounded_pvals = [round(x, ndigits) for x in self.tb['pval']]
        if inplace:
            self.tb['pval'] = rounded_pvals
        else:
            x = self.__class__(self)
            x.tb['pval'] = rounded_pvals
            return x

    def rect_perspective_df(self):
        vars = self.vars
        assert len(self.vars) == 2, "You can only get the rect_perspective_df of a pot with exactly two variables"
        return self.tb.set_index([vars[0], vars[1]]).unstack(vars[1])['pval']

    ###########################################
    # Hidden UTILS
    ###########################################
    def _merge_(self, pot):
        """
        Util function. Shouldn't really be used directly by the user.
        Merge (join) two pots.
        An inner merge of the two pots, on the intersection of their variables (if non-empty) will be performed,
        producing val_x and val_y columns that will contain the original left and right values, aligned with the join.
        Note: If the vars intersection is empty, the join will correspond to the cartesian product of the variables.
        """
        on = colloc.intersect(self.vars, pot.vars)  # we will merge on the intersection of the variables (not pval)
        if on:
            return pd.merge(self.tb, pot.tb, how='inner', on=on, sort=True, suffixes=('_x', '_y'))
        else:  # if no common variables, take the cartesian product
            return cartesian_product(self.tb, pot.tb)

    def __str__(self):
        """
        This will return a string that represents the underlying dataframe (used when printing the pot)
        """
        return self.tb.__repr__()

    def __repr__(self):
        """
        This is used by iPython to display a variable.
        I chose to do thing differently than __str__.
        Here the dataframe is indexed by the vars and then made into a string.
        This provides a hierarchical progression perspective to the variable combinations.
        """
        if self.vars:
            return self.tb.set_index(self.vars).__str__()
        else:
            return self.tb.__repr__()

    # def assert_pot_validity(self):
    #    assert 'pval' in self.tb.columns, "the potential dataframe has no column named 'pval'"
    #    assert len(self.tb.)

    #################################################################################
    # FACTORIES

    @classmethod
    def binary_pot(cls, varname, prob=1):
        return cls(pd.DataFrame({varname: [0, 1], 'pval': [1 - prob, prob]}))

    @classmethod
    def zero_potential(cls, varnames_and_scopes):
        """Makes a potential by taking the cartesian product of combinations of variable (discrete0 scopes, and pval=0

        This is useful when you need to fill a Pot with all possible combinations.

        :param varnames_and_scopes: list of varnames or (varname, scope)
            or {varname: scope,...} dict
            scope should be an integer (interpreted as range(scope)) or an iterable of the domain of the variable.

        >>> pot = Pot.zero_potential(varnames_and_scopes=['hi', ('there', 3), (10, (7, 77))])
        >>> print(pot)
          hi  there  10  pval
           0      0   7     0
           0      0  77     0
           0      1   7     0
           0      1  77     0
           0      2   7     0
           0      2  77     0
           1      0   7     0
           1      0  77     0
           1      1   7     0
           1      1  77     0
           1      2   7     0
           1      2  77     0

        """
        from itertools import product  # here, because numpy * import pollutes the space with its own

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
        vals = array(list(product(*varnames_and_scopes.values(), [0]))).T

        return cls({k: v for k, v in zip(names, vals)})

    @classmethod
    def from_points_to_count(cls, pts, vars=None, scope_for_var=None):
        """
        By "points" I mean a collection (through some data structure) of multi-dimensional coordinates.
        By default, all unique points will be grouped and the pval will be the cardinality of each group.
        """
        if isinstance(pts, pd.DataFrame):
            # tb = group_and_count(pts)
            # tb = ch_col_names(tb, 'pval', 'count')
            pot = cls(group_and_count(pts, count_col='pval'))
        else:
            pot = cls.from_counter(Counter(map(tuple, pts)), vars)

        # If scope_for_var is given, we want to have a full potential (all the var combos)
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
    def from_counter(cls, counts, vars=None):
        if vars is None:
            example_key = list(counts.keys())[0]
            vars = [str(i) for i in range(len(example_key))]
        return cls(pd.DataFrame(
            [dict(pval=v, **{kk: vv for kk, vv in zip(vars, k)}) for k, v in counts.items()])
        )

    # @classmethod
    # def from_x_and_y_combos(cls, X, y, x_combo_size = 1, y_combo_size = 1):
    #     return cls.from_points_to_count(combo_gen(X, y, x_combo_size, y_combo_size))

    @classmethod
    def from_count_df_to_count(cls, count_df, count_col='pval'):
        """
        Creates a potential from a dataframe specifying point counts (where the count column name is specified by
        count_col
        """
        pot_vars = list(colloc.setdiff(count_df.columns, [count_col]))
        tb = count_df[pot_vars + [count_col]].groupby(pot_vars).sum().reset_index()
        tb = ch_col_names(tb, 'pval', count_col)
        return cls(tb)

    @classmethod
    def from_points_to_bins(cls, pts, **kwargs):
        """
        Creates a potential from a dataframe specifying point counts (where the count column name is specified by
        count_col
        """
        if isinstance(pts, pd.DataFrame):
            tb = group_and_count(pts)
            tb = ch_col_names(tb, 'pval', 'count')
            return cls(tb)

    @classmethod
    def from_hard_evidence(cls, **var_val):
        df = pd.DataFrame([{var: val for var, val in var_val.items()}])
        df['pval'] = 1
        return cls.from_count_df_to_count(df)

    @classmethod
    def rand(cls, n_var_vals=[2, 2], var_names=None, granularity=None, try_to_get_unique_values=False):
        # check inputs
        assert len(n_var_vals) <= 26, "You can't request more than 26 variables: That's just crazy"
        if var_names is None:
            var_names = [str(chr(x)) for x in range(ord('A'), ord('Z'))]
        assert len(n_var_vals) <= len(var_names), "You can't have less var_names than you have n_var_vals"
        assert min(array(n_var_vals)) >= 2, "n_var_vals elements should be >= 2"

        # make the df by taking the cartesian product of the n_var_vals defined ranges
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

        # choose random vals
        if try_to_get_unique_values:
            if not isinstance(try_to_get_unique_values, int):
                try_to_get_unique_values = 1000
            for i in range(try_to_get_unique_values):
                pvals = _get_random_pvals()
                if len(unique(pvals)) == n_vals:
                    break
        else:
            pvals = _get_random_pvals()

        df['pval'] = list(map(float, pvals))

        return cls(df)

    def prob_of(self, var_val_dict):
        t = self.get_slice(var_val_dict)
        n = len(t.tb)
        if n == 0:
            return 0.0
        elif n == 1:
            return t.tb.pval[0]
        else:
            raise RuntimeError("In prob_of(): get_slice returned more than one value")

    def given(self, conditional_vars):
        return self.__class__(self.__div__(conditional_vars))

    def relative_risk(self, event_var, exposure_var, event_val=1, exposure_val=1, smooth_count=None):
        prob = self >> [event_var, exposure_var]
        if smooth_count is not None:
            prob = prob.count_pot_to_prob_pot(prior_count=smooth_count)
        prob = prob.binarize({event_var: event_val, exposure_var: exposure_val})
        prob_when_exposed = (prob / {exposure_var: 1})[{event_var: 1}]
        prob_when_not_exposed = (prob / {exposure_var: 0})[{event_var: 1}]
        rel_risk = (prob_when_exposed / prob_when_not_exposed).values
        if rel_risk:
            return rel_risk[0]
        else:
            return np.nan

    def count_pot_to_prob_pot(self, prior_count=1, possible_vals_for_var=None):
        """

        :param prior_count: The number to add to every (possible) combination of variable values (that is,
        the cartesian product of possible_vals_for_var
        :param possible_vals_for_var: A dict providing the list of values each variable can have
        :return: A probability pot, obtained from the count pot after adding prior_count to all counts (included those
        vars coordinates that didn't show up (i.e. had count of 0)
        """
        if possible_vals_for_var is not None:
            raise NotImplementedError("You wan't it? Implement it!")
        c = (self + prior_count).tb
        c['pval'] /= c['pval'].sum()
        return self.__class__(c)


class ProbPot(Pot):
    def __init__(self, data=None):
        super(ProbPot, self).__init__(data=data)

    @staticmethod
    def plot_relrisk_matrix(relrisk):
        import matplotlib.pyplot as plt
        from spyn.utils.color import shifted_color_map, get_colorbar_tick_labels_as_floats

        t = relrisk.copy()
        matrix_shape = (t['exposure'].nunique(), t['event'].nunique())
        m = map_vals_to_ints_inplace(t, cols_to_map=['exposure'])
        m = m['exposure']
        map_vals_to_ints_inplace(t, cols_to_map={'event': dict(list(zip(m, list(range(len(m))))))})
        RR = zeros(matrix_shape)
        RR[t['exposure'], t['event']] = t['relative_risk']
        RR[list(range(len(m))), list(range(len(m)))] = nan

        RRL = np.log2(RR)

        def normalizor(X):
            min_x = nanmin(X)
            range_x = nanmax(X) - min_x
            return lambda x: (x - min_x) / range_x

        normalize_this = normalizor(RRL)
        center = normalize_this(0)

        color_map = shifted_color_map(cmap=plt.cm.get_cmap('coolwarm'), start=0, midpoint=center, stop=1)
        plt.imshow(RRL, cmap=color_map, interpolation='none');

        plt.xticks(list(range(shape(RRL)[0])), m, rotation=90)
        plt.yticks(list(range(shape(RRL)[1])), m)
        cbar = plt.colorbar()
        cbar.ax.set_yticklabels(
            ["%.02f" % x for x in np.exp2(array(get_colorbar_tick_labels_as_floats(cbar)))])


#
#
# class ValPot(Pot):
#     def __init__(self, **kwargs):
#         super(ValPot, self).__init__(**kwargs)

##### Data Prep utils
def from_points_to_binary(d, mid_fun=median):
    dd = d.copy()
    columns = d.columns
    for c in columns:
        dd[c] = list(map(int, d[c] > mid_fun(d[c])))
    return dd


##### Other utils

def relative_risk(joint_prob_pot, event_var, exposure_var):
    prob = joint_prob_pot >> [event_var, exposure_var]
    return ((prob / {exposure_var: 1})[{event_var: 1}]
            / (prob / {exposure_var: 0})[{event_var: 1}]).values[0]


def _val_prod_(tb):
    """
    multiplies column val_x and val_y creating column pval (and removing val_x and val_y)
    """
    tb['pval'] = tb['pval_x'] * tb['pval_y']
    tb.drop(labels=['pval_x', 'pval_y'], axis=1, inplace=True)
    return tb


def _val_div_(tb):
    """
    divides column val_x and val_y creating column pval (and removing val_x and val_y)
    Note: 0/0 will be equal to 0
    """
    tb['pval'] = np.true_divide(tb['pval_x'], tb['pval_y']).fillna(0)
    tb.drop(labels=['pval_x', 'pval_y'], axis=1, inplace=True)
    return tb


def _val_add_(tb):
    """
    adds column val_x and val_y creating column pval (and removing val_x and val_y)
    """
    tb['pval'] = tb['pval_x'] + tb['pval_y']
    tb.drop(labels=['pval_x', 'pval_y'], axis=1, inplace=True)
    return tb


def _ascertain_list(x):
    """
    _ascertain_list(x) blah blah returns [x] if x is not already a list, and x itself if it's already a list
    Use: This is useful when a function expects a list, but you want to also input a single element without putting this
    this element in a list
    """
    if not isinstance(x, list):
        if isinstance(x, str):
            x = [x]
        elif hasattr(x, '__iter__') and not isinstance(x, dict):
            x = list(x)
        else:
            x = [x]
    return x


from decimal import Decimal


def _rand_numbers_summing_to_one(n_numbers, granularity=0.01):
    n_choices = 1.0 / granularity
    assert round(n_choices) == int(n_choices), "granularity must be an integer divisor of 1.0"
    x = np.linspace(granularity, 1.0 - granularity, n_choices - 1)
    x = sorted(x[np.random.choice(list(range(1, len(x))), size=n_numbers - 1, replace=False)])
    x = np.concatenate([[0.0], x, [1.0]])
    x = np.diff(x)
    x = np.array([Decimal(xi).quantize(Decimal(str(granularity))) for xi in x])
    return x
