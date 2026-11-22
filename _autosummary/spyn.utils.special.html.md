# spyn.utils.special

Specialized dict subclasses used as default-value mappings.

### Classes

| [`DictDefaultDict`](#spyn.utils.special.DictDefaultDict)(default_dict)   | Acts similarly to collections.defaultdict, except   |
|----------------------------------------------------------------------------------|-----------------------------------------------------|

### *class* spyn.utils.special.DictDefaultDict(default_dict)

Bases: [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)

Acts similarly to collections.defaultdict, except
: 1. the defaults depend on the key (given by a dict of key–>default_val at construction)
  2. it is not a function that is called to create the default value (so careful with referenced variables)
