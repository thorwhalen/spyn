# spyn.utils.ordered_set

OrderedSet collection (Python recipe: [http://code.activestate.com/recipes/576694/](http://code.activestate.com/recipes/576694/)).

### Classes

| [`OrderedSet`](#spyn.utils.ordered_set.OrderedSet)([iterable])   |    |
|---------------------------------------------------------------------------|----|

### *class* spyn.utils.ordered_set.OrderedSet(iterable=None)

Bases: [`MutableSet`](https://docs.python.org/3/library/collections.abc.html#collections.abc.MutableSet)

#### add(key)

Add an element.

#### discard(key)

Remove an element.  Do not raise an exception if absent.

#### pop(last=True)

Return the popped value.  Raise KeyError if empty.
