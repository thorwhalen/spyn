# spyn.utils.order_conserving

Order-preserving set operations using plain dicts (Python 3.7+).

### Functions

| [`intersect`](#spyn.utils.order_conserving.intersect)(A, B)             | Order-preserving intersection: elements of A that are also in B.     |
|------------------------------------------------------------------------------|----------------------------------------------------------------------|
| [`reorder_as`](#spyn.utils.order_conserving.reorder_as)(A, B)            | Reorder A to respect order in B.                                     |
| [`setdiff`](#spyn.utils.order_conserving.setdiff)(A, B)               | Order-preserving set difference: elements of A not in B.             |
| [`union`](#spyn.utils.order_conserving.union)(A, B)                 | Order-preserving union: elements of B then remaining elements of A.  |
| [`unique_for_non_hashables`](#spyn.utils.order_conserving.unique_for_non_hashables)(X) | Return unique items for unhashable types, preserving order.          |
| [`unique_from_iter`](#spyn.utils.order_conserving.unique_from_iter)(it)        | Return unique items from an iterable of iterables, preserving order. |
| [`unique_list`](#spyn.utils.order_conserving.unique_list)(x)              | Return unique items from x, preserving order.                        |

### spyn.utils.order_conserving.intersect(A, B)

Order-preserving intersection: elements of A that are also in B.

```pycon
>>> intersect([1, 2, 3], [3, 1])
[1, 3]
```

### spyn.utils.order_conserving.reorder_as(A, B)

Reorder A to respect order in B. Elements of A not in B go at the end.

```pycon
>>> reorder_as([1, 2, 3], [3, 1])
[3, 1, 2]
```

### spyn.utils.order_conserving.setdiff(A, B)

Order-preserving set difference: elements of A not in B.

```pycon
>>> setdiff([1, 2, 3], [2])
[1, 3]
```

### spyn.utils.order_conserving.union(A, B)

Order-preserving union: elements of B then remaining elements of A.

```pycon
>>> union([1, 2, 3], [3, 4])
[3, 4, 1, 2]
```

### spyn.utils.order_conserving.unique_for_non_hashables(X)

Return unique items for unhashable types, preserving order.

### spyn.utils.order_conserving.unique_from_iter(it)

Return unique items from an iterable of iterables, preserving order.

### spyn.utils.order_conserving.unique_list(x)

Return unique items from x, preserving order.

```pycon
>>> unique_list([3, 1, 2, 1, 3])
[3, 1, 2]
```
