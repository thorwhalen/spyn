"""Tests for spyn.ppi.pot.Pot — the core potential class."""

import numpy as np
import pandas as pd
import pytest

from spyn.ppi.pot import Pot


# -- Construction ----------------------------------------------------------


class TestConstruction:
    def test_from_dict(self):
        p = Pot({'A': [0, 1], 'B': [0, 1], 'pval': [3, 7]})
        assert p.vars == ['A', 'B']
        assert p.n == 2

    def test_from_dict_without_pval(self):
        p = Pot({'A': [0, 1], 'B': [0, 1]})
        assert 'pval' in p.tb.columns
        assert all(p.tb['pval'] == 1)

    def test_from_dataframe(self):
        df = pd.DataFrame({'X': [0, 1], 'pval': [0.4, 0.6]})
        p = Pot(df)
        assert p.vars == ['X']

    def test_from_pot(self):
        p1 = Pot({'A': [0, 1], 'pval': [3, 7]})
        p2 = Pot(p1)
        assert p2.vars == ['A']

    def test_from_scalar(self):
        p = Pot(5.0)
        assert p.vars == []
        assert p.values[0] == 5.0

    def test_default_unit(self):
        p = Pot()
        assert p.values[0] == 1

    def test_from_dataframe_missing_pval_raises(self):
        with pytest.raises(AssertionError, match="pval"):
            Pot(pd.DataFrame({'A': [1, 2]}))


# -- Properties ------------------------------------------------------------


class TestProperties:
    def test_vars(self):
        p = Pot({'X': [0, 1], 'Y': [0, 1], 'pval': [1, 2]})
        assert p.vars == ['X', 'Y']

    def test_vars_set(self):
        p = Pot({'X': [0, 1], 'Y': [0, 1], 'pval': [1, 2]})
        assert p.vars_set == {'X', 'Y'}

    def test_pval(self):
        p = Pot({'A': [0, 1], 'pval': [3, 7]})
        assert list(p.pval) == [3, 7]

    def test_values(self):
        p = Pot({'A': [0, 1], 'pval': [3, 7]})
        np.testing.assert_array_equal(p.values, [3, 7])

    def test_n(self):
        p = Pot({'A': [0, 1, 2], 'pval': [1, 2, 3]})
        assert p.n == 3


# -- Projection (marginalization) ------------------------------------------


class TestProjection:
    @pytest.fixture()
    def p(self):
        return Pot({'A': [0, 0, 1, 1], 'B': [0, 1, 0, 1], 'pval': [2, 4, 5, 9]})

    def test_project_to_single(self, p):
        result = p.project_to('A')
        assert result.n == 2
        # A=0: 2+4=6, A=1: 5+9=14
        assert result.pval_of({'A': 0}) == 6
        assert result.pval_of({'A': 1}) == 14

    def test_project_to_list(self, p):
        result = p.project_to(['A'])
        assert result.n == 2

    def test_bracket_str(self, p):
        result = p['A']
        assert result.n == 2

    def test_bracket_list(self, p):
        result = p[['A', 'B']]
        assert result.n == 4

    def test_rshift(self, p):
        result = p >> ['A']
        assert result.n == 2

    def test_project_empty(self, p):
        result = p[None]
        assert result.values[0] == 20  # sum of all pvals

    def test_project_to_strict_raises(self, p):
        with pytest.raises(KeyError):
            p.project_to(['A', 'NONEXISTENT'], assert_subset=True)


# -- Slicing ---------------------------------------------------------------


class TestSlicing:
    @pytest.fixture()
    def p(self):
        return Pot({'A': [0, 0, 1, 1], 'B': [0, 1, 0, 1], 'pval': [2, 4, 5, 9]})

    def test_get_slice(self, p):
        result = p.get_slice({'A': 1})
        assert result.n == 2
        assert result.vars == ['B']

    def test_bracket_dict(self, p):
        result = p[{'A': 0}]
        assert result.n == 2
        assert result.pval_of({'B': 0}) == 2


# -- Normalization ---------------------------------------------------------


class TestNormalization:
    @pytest.fixture()
    def p(self):
        return Pot({'A': [0, 0, 1, 1], 'B': [0, 1, 0, 1], 'pval': [2, 4, 5, 9]})

    def test_full_normalize(self, p):
        result = p / []
        assert pytest.approx(result.values.sum()) == 1.0

    def test_conditional(self, p):
        """P(B|A) = P(A,B) / P(A)."""
        result = p / 'A'
        # For A=0: B=0 -> 2/6, B=1 -> 4/6
        assert pytest.approx(result.pval_of({'A': 0, 'B': 0}), rel=1e-6) == 2 / 6
        assert pytest.approx(result.pval_of({'A': 0, 'B': 1}), rel=1e-6) == 4 / 6

    def test_normalize_method(self, p):
        result = p.normalize()
        assert pytest.approx(result.values.sum()) == 1.0


# -- Multiplication (factor product) ---------------------------------------


class TestMultiplication:
    def test_independent_factors(self):
        a = Pot({'X': [0, 1], 'pval': [0.4, 0.6]})
        b = Pot({'Y': [0, 1], 'pval': [0.3, 0.7]})
        joint = a * b
        assert joint.n == 4
        assert pytest.approx(joint.pval_of({'X': 0, 'Y': 0})) == 0.12

    def test_shared_variable(self):
        a = Pot({'X': [0, 1], 'pval': [0.4, 0.6]})
        b = Pot({'X': [0, 0, 1, 1], 'Y': [0, 1, 0, 1],
                 'pval': [0.9, 0.1, 0.3, 0.7]})
        joint = a * b
        assert joint.n == 4
        # X=0, Y=0: 0.4 * 0.9 = 0.36
        assert pytest.approx(joint.pval_of({'X': 0, 'Y': 0})) == 0.36


# -- Assimilate (Bayes rule) -----------------------------------------------


class TestAssimilate:
    def test_basic_bayes(self):
        prior = Pot({'H': [0, 1], 'pval': [0.5, 0.5]})
        likelihood = Pot({'H': [0, 1], 'pval': [0.2, 0.8]})
        posterior = prior.assimilate(likelihood)
        assert posterior.n == 2
        # posterior should be proportional to prior * likelihood
        assert posterior.pval_of({'H': 1}) > posterior.pval_of({'H': 0})
        assert pytest.approx(posterior.values.sum()) == 1.0

    def test_roundtrip(self):
        prior = Pot({'H': [0, 1], 'pval': [0.5, 0.5]})
        likelihood = Pot({'H': [0, 1], 'pval': [0.2, 0.8]})
        posterior = prior.assimilate(likelihood)
        recovered = posterior.unassimilate(likelihood)
        # Should recover something proportional to the prior
        assert pytest.approx(recovered.values.sum()) == 1.0


# -- Other operations ------------------------------------------------------


class TestOther:
    def test_addition(self):
        a = Pot({'X': [0, 1], 'pval': [1, 2]})
        b = Pot({'X': [0, 1], 'pval': [3, 4]})
        result = a + b
        assert result.pval_of({'X': 0}) == 4
        assert result.pval_of({'X': 1}) == 6

    def test_add_scalar(self):
        p = Pot({'A': [0, 1], 'pval': [3, 7]})
        result = p + 1
        # add_count fills all combinations, adds 1 to each
        assert all(v > 0 for v in result.values)

    def test_order_vars_returns_new(self):
        p = Pot({'B': [0, 1], 'A': [0, 1], 'pval': [3, 7]})
        result = p.order_vars(['A', 'B'])
        assert result.vars == ['A', 'B']
        # Original should be unchanged
        assert p.vars == ['B', 'A']

    def test_sort_pts_returns_new(self):
        p = Pot({'A': [1, 0], 'pval': [7, 3]})
        result = p.sort_pts()
        assert list(result.tb['A']) == [0, 1]

    def test_pval_of(self):
        p = Pot({'A': [0, 1], 'pval': [3, 7]})
        assert p.pval_of({'A': 1}) == 7
        assert p.pval_of({'A': 99}) == 0.0  # default

    def test_binarize(self):
        p = Pot({'color': ['r', 'g', 'b'], 'pval': [3, 5, 2]})
        result = p.binarize({'color': ['r', 'g']})
        assert result.n == 2
        assert result.pval_of({'color': 1}) == 8  # r+g

    def test_round(self):
        p = Pot({'A': [0, 1], 'pval': [0.333, 0.667]})
        result = p.round(2)
        assert list(result.tb['pval']) == [0.33, 0.67]

    def test_rect_perspective(self):
        p = Pot({'A': [0, 0, 1, 1], 'B': [0, 1, 0, 1], 'pval': [1, 2, 3, 4]})
        df = p.rect_perspective_df()
        assert df.shape == (2, 2)
        assert df.loc[0, 0] == 1


# -- Factories -------------------------------------------------------------


class TestFactories:
    def test_binary_pot(self):
        p = Pot.binary_pot('coin', 0.6)
        assert p.n == 2
        assert pytest.approx(p.pval_of({'coin': 1})) == 0.6

    def test_zero_potential(self):
        p = Pot.zero_potential({'A': 2, 'B': 3})
        assert p.n == 6
        assert all(v == 0 for v in p.values)

    def test_from_points_to_count(self):
        df = pd.DataFrame({'A': [0, 0, 1, 1, 1], 'B': [0, 1, 0, 0, 1]})
        p = Pot.from_points_to_count(df)
        assert p.pval_of({'A': 1, 'B': 0}) == 2

    def test_from_hard_evidence(self):
        p = Pot.from_hard_evidence(X=1, Y=0)
        assert p.n == 1
        assert p.pval_of({'X': 1, 'Y': 0}) == 1

    def test_rand(self):
        p = Pot.rand([2, 3])
        assert p.n == 6
        assert pytest.approx(p.values.sum()) == 1.0

    def test_count_pot_to_prob_pot(self):
        p = Pot({'A': [0, 1], 'pval': [3, 7]})
        prob = p.count_pot_to_prob_pot(prior_count=1)
        assert pytest.approx(prob.values.sum()) == 1.0


# -- Deprecated operator --------------------------------------------------


class TestDeprecated:
    def test_pipe_warns(self):
        p = Pot({'A': [0, 1], 'pval': [3, 7]})
        with pytest.warns(DeprecationWarning, match="deprecated"):
            p | 'A'
