from typing import Iterable
from math import prod

from numpy import array, vstack, argmax
from itertools import product
from spyn.util import col_combo_gen
from spyn.ppi.pot import Pot


def _ensure_iterable(x):
    if isinstance(x, str) or not isinstance(x, Iterable):
        return [x]
    return x


def xy_to_pots(
        X,
        y,
        x_names=None,
        y_names=None,
        scope_for_var=None,
        x_combo_size: int = 1,
        y_combo_size: int = 1):
    assert isinstance(x_combo_size, int), f"x_combo_size should be an int. Was {x_combo_size}"
    assert isinstance(y_combo_size, int), f"y_combo_size should be an int. Was {y_combo_size}"
    X = array(X)
    y = array(y)
    if x_names is None:
        x_names = _ensure_iterable([str(i) for i in range(X.shape[-1])])
    if y_names is None:
        y_names = _ensure_iterable([str(i) for i in range(y.shape[-1])])

    pts_gen = col_combo_gen(X, y, x_combo_size, y_combo_size)
    vars_gen = col_combo_gen([x_names], [y_names], x_combo_size, y_combo_size)
    for pts, varnames in zip(pts_gen, vars_gen):
        varnames = [*varnames[0], *varnames[1]]

        yield Pot.from_points_to_count(vstack(pts).T, vars=varnames, scope_for_var=scope_for_var)


def _preprocess_naive_pots_fit_input(X, y, x_names, y_names):
    assert len(X) > 0 and len(X) == len(y), "X and y sizes don't match (or are empty)"

    x0 = X[0]
    y0 = y[0]

    if x_names is None:
        x_names = [f"x{i:2.0f}" for i in range(len(x0))]

    if y_names is None:
        if isinstance(y0, Iterable):
            n = len(y0)
        else:
            n = 1
        y_names = [f"x{i:2.0f}" for i in range(n)]

    x_names = _ensure_iterable(x_names)
    y_names = _ensure_iterable(y_names)

    assert len(X[0]) == len(x_names), "names and dimensions of X don't match"
    if isinstance(y0, Iterable):
        assert len(y0) == len(y_names), "names and dimensions of y don't match"

    return X, y, _ensure_iterable(x_names), _ensure_iterable(y_names)


from sklearn.base import BaseEstimator, ClusterMixin


class NaiveBayesPot(BaseEstimator, ClusterMixin):
    def __init__(self, additive_smoother=1):
        self.additive_smoother = additive_smoother

    def fit(self,
            X,
            y,
            x_names=None,
            y_names='truth',
            scope_for_var=None,
            ):
        X, y, x_names, y_names = _preprocess_naive_pots_fit_input(X, y, x_names, y_names)

        if isinstance(y[0], Iterable):
            raise NotImplementedError("Only single dimension ys have been implemented so far")
        y_name = y_names[0]

        if scope_for_var is None:
            scope_for_var = {name: list(set(col)) for name, col in zip(x_names, X.T)}
            scope_for_var.update({y_name: list(set(y))})  # TODO: Change when allowing multi-dim y

        pots = xy_to_pots(X, y, x_names, y_names, scope_for_var)
        if self.additive_smoother:
            pots = [pot + self.additive_smoother for pot in pots]

        truth_pot = next(iter(pots))[y_name]
        assert all((pot['truth'].tb == truth_pot.tb).all().all() for pot in pots), (
            "Your truth is not consistent. Perhaps you fed me some bad data."
        )
        truth_pot = truth_pot.normalize()

        pots = [pot.normalize(y_name) for pot in pots]
        pots.append(truth_pot)
        pots = {name: pot for name, pot in zip(list(x_names) + [y_name], pots)}

        self.pots_ = pots
        self.y_name_ = y_name
        self.x_names_ = x_names
        self.x_ndims_ = len(x_names)
        self.classes_ = truth_pot.tb[y_name].values

        return self

        # pots = {model: Pot.from_counter(v[model], vars=[model, 'truth']) for model in v}

        # truth_pot = truth_pot.normalize()
        # pots = {model: (pot + 1).normalize('truth') for model, pot in pots.items()}
        # # models = list(pots.keys())
        # pots['truth'] = truth_pot
        # yield pump_type, pots

    def post_pots(self, x):
        if not isinstance(x, dict):
            assert len(x) == self.x_ndims_, "That evidence array does not have the right length"
            x = {name: val for name, val in zip(self.x_names_, x)}

        assert len(x.keys() - self.x_names_) == 0, (
            f"You entered evidence on variables I don't have: {x.keys() - self.x_names_}"
        )
        for name, pot in self.pots_.items():
            if name != self.y_name_:
                if name in x:
                    new_pot = (self.pots_[name] * Pot({name: [x[name]]}))
                    yield new_pot.project_to(self.y_name_).normalize()
                # else:
                #     yield self.pots_[name].project_to(self.y_name_).normalize()

    def single_predict_pot(self, x):
        return prod(self.post_pots(x), start=self.pots_[self.y_name_]).normalize()

    def single_predict_proba(self, x):
        return self.single_predict_pot(x).tb['pval'].values

    def predict_proba(self, X):
        return array([self.single_predict_proba(x) for x in X])

    def predict(self, X):
        return self.classes_[argmax(self.predict_proba(X), axis=1)]
