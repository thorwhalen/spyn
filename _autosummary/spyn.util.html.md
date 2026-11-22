# spyn.util

General-purpose utilities used across the spyn package.

### Functions

| [`col_combo_gen`](#spyn.util.col_combo_gen)(X, y[, x_combo_size, y_combo_size])   | Generate all cartesian products of all combinations of the columns of X and y.   |
|------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------|
| [`combo_pt_gen`](#spyn.util.combo_pt_gen)(X, y[, x_combo_size, y_combo_size])    | Generate all cartesian products of all combinations of the rows of X and y.      |

### Classes

| [`lazyprop`](#spyn.util.lazyprop)(func)   | A descriptor implementation of lazyprop (cached property).   |
|-------------------------------------------------------------------|--------------------------------------------------------------|

### spyn.util.col_combo_gen(X, y, x_combo_size=1, y_combo_size=1)

Generate all cartesian products of all combinations of the columns of X and y.
Yeah… have a look at the examples.

* **Parameters:**
  * **X** – List of lists (of the same size), or numpy array, or such
  * **y** – List of numbers, or list of lists. Length should be the same as X
  * **x_combo_size** – The size of the X column combinations
  * **y_combo_size** – The size of the y column combinations
* **Returns:**
  Yield (\*x_combo, \*y_combo) tuples (note to the star-ignorant: That’s a FLAT tuple!)

```pycon
>>> XX = [[1,2,3], [4,5,6]]
>>> yy = [10, 20]
>>>
>>> for combos in col_combo_gen(XX, yy):
...     print(*combos)
[1 4] [10 20]
[2 5] [10 20]
[3 6] [10 20]
>>> for combos in col_combo_gen(XX, yy, x_combo_size=2):
...     print(*combos)
[1 4] [2 5] [10 20]
[1 4] [3 6] [10 20]
[2 5] [3 6] [10 20]
>>>
>>>
>>> yy = [[10, 100, 1000], [20, 200, 2000]]
>>>
>>> for combos in col_combo_gen(XX, yy, x_combo_size=2, y_combo_size=2):
...     print(*combos)
[1 4] [2 5] [10 20] [100 200]
[1 4] [2 5] [10 20] [1000 2000]
[1 4] [2 5] [100 200] [1000 2000]
[1 4] [3 6] [10 20] [100 200]
[1 4] [3 6] [10 20] [1000 2000]
[1 4] [3 6] [100 200] [1000 2000]
[2 5] [3 6] [10 20] [100 200]
[2 5] [3 6] [10 20] [1000 2000]
[2 5] [3 6] [100 200] [1000 2000]
```

### spyn.util.combo_pt_gen(X, y, x_combo_size=1, y_combo_size=1)

Generate all cartesian products of all combinations of the rows of X and y.
Yeah… have a look at the examples.

* **Parameters:**
  * **X** – List of lists (of the same size), or numpy array, or such
  * **y** – List of numbers, or list of lists. Length should be the same as X
  * **x_combo_size** – The size of the combinations to take from X rows values.
  * **y_combo_size** – The size of the combinations to take from y rows values.
* **Returns:**
  Yield (\*x_combo, \*y_combo) tuples

```pycon
>>> XX = [[1,2,3], [4,5,6]]
>>> yy = [10, 20]
>>> XX, yy
([[1, 2, 3], [4, 5, 6]], [10, 20])
>>>
>>> list(combo_pt_gen(XX, yy))
[(1, 10), (2, 10), (3, 10), (4, 20), (5, 20), (6, 20)]
>>>
>>> list(combo_pt_gen(XX, yy, x_combo_size=2))
[(1, 2, 10), (1, 3, 10), (2, 3, 10), (4, 5, 20), (4, 6, 20), (5, 6, 20)]
>>>
>>> yy = [[10, 100, 1000], [20, 200, 2000]]
>>> list(combo_pt_gen(XX, yy, x_combo_size=2, y_combo_size=2))
[(1, 2, 10, 100), (1, 2, 10, 1000), (1, 2, 100, 1000), ..., (5, 6, 20, 200), (5, 6, 20, 2000), (5, 6, 200, 2000)]
```

### *class* spyn.util.lazyprop(func)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

A descriptor implementation of lazyprop (cached property).
Made based on David Beazley’s “Python Cookbook” book and enhanced with boltons.cacheutils ideas.

```pycon
>>> class Test:
...     def __init__(self, a):
...         self.a = a
...     @lazyprop
...     def len(self):
...         print('generating "len"')
...         return len(self.a)
>>> t = Test([0, 1, 2, 3, 4])
>>> t.__dict__
{'a': [0, 1, 2, 3, 4]}
>>> t.len
generating "len"
5
>>> t.__dict__
{'a': [0, 1, 2, 3, 4], 'len': 5}
>>> t.len
5
>>> # But careful when using lazyprop that no one will change the value of a without deleting the property first
>>> t.a = [0, 1, 2]  # if we change a...
>>> t.len  # ... we still get the old cached value of len
5
>>> del t.len  # if we delete the len prop
>>> t.len  # ... then len being recomputed again
generating "len"
3
```
