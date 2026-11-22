"""Tests for spyn.utils: ordered set, order-conserving ops, DictDefaultDict."""

import pytest

from spyn.utils.ordered_set import OrderedSet
from spyn.utils.order_conserving import (
    unique_list,
    unique_from_iter,
    unique_for_non_hashables,
    union,
    intersect,
    setdiff,
    reorder_as,
)
from spyn.utils.special import DictDefaultDict


def test_ordered_set_basics():
    s = OrderedSet("abracadaba")
    assert list(s) == ["a", "b", "r", "c", "d"]
    assert len(s) == 5 and "r" in s and "z" not in s
    assert list(reversed(s)) == ["d", "c", "r", "b", "a"]
    s.discard("r")
    s.discard("nope")  # no error
    assert list(s) == ["a", "b", "c", "d"]


def test_ordered_set_pop_and_repr():
    s = OrderedSet([1, 2, 3])
    assert s.pop() == 3
    assert s.pop(last=False) == 1
    assert repr(s) == "OrderedSet([2])"
    s.pop()
    assert repr(s) == "OrderedSet()"
    with pytest.raises(KeyError):
        s.pop()


def test_ordered_set_eq_and_ops():
    assert OrderedSet([1, 2]) == OrderedSet([1, 2])
    assert OrderedSet([1, 2]) != OrderedSet([2, 1])
    assert OrderedSet([1, 2]) == {2, 1}
    assert list(OrderedSet("abc") | OrderedSet("cd")) == list("abcd")
    assert list(OrderedSet("abc") & OrderedSet("cb")) == ["c", "b"]  # follows other operand (Set ABC)
    assert list(OrderedSet("abc") -OrderedSet("b")) == ["a", "c"]


def test_unique_functions():
    assert unique_list([3, 1, 3, 2]) == [3, 1, 2]
    assert unique_from_iter([[1, 2], [2, 3]]) == [1, 2, 3]
    assert unique_for_non_hashables([1, 2, 1]) == [1, 2]
    assert unique_for_non_hashables((1, 2, 1)) == (1, 2)


def test_set_ops_preserve_type_and_order():
    assert union([1, 2, 3], [3, 4]) == [3, 4, 1, 2]
    assert union((1, 2), (2, 5)) == (2, 5, 1)
    assert intersect([1, 2, 2, 3], [3, 1]) == [1, 3]
    assert setdiff((1, 2, 3), (2,)) == (1, 3)
    assert reorder_as([1, 2, 3], [3, 1]) == [3, 1, 2]


def test_set_ops_fall_back_to_list_for_uncallable_types():
    class Bad(list):
        def __init__(self, *a):
            if a:
                raise TypeError

    assert union(Bad(), [1]) == [1]
    assert intersect(Bad(), []) == []
    assert setdiff(Bad(), []) == []


def test_dict_default_dict():
    d = DictDefaultDict({"a": 1, "b": 2})
    d["a"] = 10
    assert d["a"] == 10 and d["b"] == 2
    with pytest.raises(KeyError):
        d["zzz"]
