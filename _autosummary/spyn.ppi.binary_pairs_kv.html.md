# spyn.ppi.binary_pairs_kv

Edge statistics over binary pair sets using key-value count structures.

### Functions

| `get_a_list_from_item_default`(pair_set)   |    |
|--------------------------------------------|----|
| `get_b_list_from_item_default`(pair_set)   |    |

### Classes

| [`BipartiteEdgeCounts`](#spyn.ppi.binary_pairs_kv.BipartiteEdgeCounts)([get_a_list_from_item, ...])   | The class that manages the count data.                                                                      |
|-----------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------|
| [`CountVal`](#spyn.ppi.binary_pairs_kv.CountVal)([v])                                      |                                                                                                             |
| `EdgeStats`()                                                                                       |                                                                                                             |
| [`KeyCount`](#spyn.ppi.binary_pairs_kv.KeyCount)([v])                                      | Extends a map so that one can add and subtract dict pairs by adding or subtracting the (key-aligned) values |
| [`KeyVal`](#spyn.ppi.binary_pairs_kv.KeyVal)([v])                                        | Here the type of the value is a dict (to implement a map).                                                  |
| [`LHVal`](#spyn.ppi.binary_pairs_kv.LHVal)([v])                                         | An LHVal manages a binary likelihood.                                                                       |
| [`Val`](#spyn.ppi.binary_pairs_kv.Val)(v)                                             | The mother class of other Val classes.                                                                      |

### *class* spyn.ppi.binary_pairs_kv.BipartiteEdgeCounts(get_a_list_from_item=None, get_b_list_from_item=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

The class that manages the count data.

### *class* spyn.ppi.binary_pairs_kv.CountVal(v=0.0)

Bases: [`Val`](#spyn.ppi.binary_pairs_kv.Val)

### *class* spyn.ppi.binary_pairs_kv.KeyCount(v=None)

Bases: [`KeyVal`](#spyn.ppi.binary_pairs_kv.KeyVal)

Extends a map so that one can add and subtract dict pairs by adding or subtracting the (key-aligned) values

### *class* spyn.ppi.binary_pairs_kv.KeyVal(v=None)

Bases: [`Val`](#spyn.ppi.binary_pairs_kv.Val)

Here the type of the value is a dict (to implement a map).
The addition of two dicts (therefore the add() method) v and w.

The add(val) method will here be defined to be a sum-update of the (key,value)
pairs of the
Extends a map so that one can add and subtract dict pairs by adding or subtracting

> the (key-aligned) values

### *class* spyn.ppi.binary_pairs_kv.LHVal(v=0.5)

Bases: [`Val`](#spyn.ppi.binary_pairs_kv.Val)

An LHVal manages a binary likelihood.
That is, it holds (as a single float) the binary likelihood distribution and allows one to
merge two such distributions.

### *class* spyn.ppi.binary_pairs_kv.Val(v)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

The mother class of other Val classes.
A Val should hold a value and be able to add and subtract from it.

This mother class implements normal addition of floats, but should be overridden to
implement other types of values such as multiplication, addition of vectors,
merging of likelihoods etc.

Most of the time, you’ll only need to override the add() and the sub() methods.
You may also want to override the default value. This value should act as the
‘unit’ or ‘neutral’ value of the add operation (therefore the sub operation as well).
For example, the unit value of multiplication (which will still be called “add”) is 1.0.
