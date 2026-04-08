"""Order-preserving set operations using plain dicts (Python 3.7+)."""

__author__ = 'thorwhalen'


def unique_list(x):
    """Return unique items from x, preserving order.

    >>> unique_list([3, 1, 2, 1, 3])
    [3, 1, 2]
    """
    return list(dict.fromkeys(x))


def unique_from_iter(it):
    """Return unique items from an iterable of iterables, preserving order."""
    d = dict()
    for item in it:
        d.update(dict.fromkeys(item))
    return list(d)


def unique_for_non_hashables(X):
    """Return unique items for unhashable types, preserving order."""
    seen = set()
    seen_add = seen.add
    return type(X)([x for x in X if x not in seen and not seen_add(x)])


def union(A, B):
    """Order-preserving union: elements of B then remaining elements of A.

    >>> union([1, 2, 3], [3, 4])
    [3, 4, 1, 2]
    """
    combined = list(dict.fromkeys(list(B) + list(A)))
    try:
        return type(A)(combined)
    except TypeError:
        return combined


def intersect(A, B):
    """Order-preserving intersection: elements of A that are also in B.

    >>> intersect([1, 2, 3], [3, 1])
    [1, 3]
    """
    b_set = set(B)
    result = [x for x in A if x in b_set]
    # Deduplicate while preserving order
    result = list(dict.fromkeys(result))
    try:
        return type(A)(result)
    except TypeError:
        return result


def setdiff(A, B):
    """Order-preserving set difference: elements of A not in B.

    >>> setdiff([1, 2, 3], [2])
    [1, 3]
    """
    b_set = set(B)
    result = list(dict.fromkeys(x for x in A if x not in b_set))
    try:
        return type(A)(result)
    except TypeError:
        return result


def reorder_as(A, B):
    """Reorder A to respect order in B. Elements of A not in B go at the end.

    >>> reorder_as([1, 2, 3], [3, 1])
    [3, 1, 2]
    """
    C = intersect(B, A) + setdiff(A, B)
    try:
        return type(A)(C)
    except TypeError:
        return list(C)
