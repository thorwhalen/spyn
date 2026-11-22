# spyn: hyper-graphical probabilistic inference

Discrete probabilistic inference whose model is a set of "hyper-edges": tables (potentials) that carry probabilistic information about subsets of random variables. It generalizes Bayesian networks (the join tree of a Bayesian network is a hyper-tree). You don't need to know Bayesian statistics to start: if you can count things and put the counts in a table, you can use it.

```
pip install spyn
```

Requirements are `numpy`, `pandas` and `matplotlib`. Core variables are discrete; for non-discrete ones, wrap them in your own binning.

## Quick example

A potential is a table that maps each combination of values of some discrete variables to a number (a count, a probability, a likelihood):

```python
import pandas as pd
from spyn.ppi.pot import Pot

ab_counts = Pot(
    pd.DataFrame({"A": [0, 0, 1, 1], "B": [0, 1, 0, 1], "pval": [2, 4, 5, 9]})
)
print(ab_counts["A"])  # project to A: sum out B
```

```
   A  pval
0  0     6
1  1    14
```

`ab_counts["A"]` is shorthand for `ab_counts.project_to(var_list=["A"])`, which can also be written `ab_counts >> ["A"]`.

## Learn more

The [tutorial](https://github.com/thorwhalen/spyn/blob/master/docs/tutorial.md) builds up from counting: projecting, slicing, multiplying and dividing potentials, integrating evidence, and building a join tree from your own data.

For `spyn` used in a real project, see [odus](https://github.com/thorwhalen/odus).
