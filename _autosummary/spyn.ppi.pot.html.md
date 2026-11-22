# spyn.ppi.pot

Potentials: probability/likelihood tables with algebraic operators.

The Pot class represents discrete probability tables as DataFrames with a ‘pval’
column and uses operator overloading for probabilistic inference:

- `*` factor product
- `/` normalization, conditioning, or factor division
- `[]` marginalization (list/str) or slicing (dict)
- `>>` projection (marginalization)

```pycon
>>> p = Pot({'A': [0, 0, 1, 1], 'B': [0, 1, 0, 1], 'pval': [2, 4, 5, 9]})
>>> p['A']  # marginalize to A
     pval
A
0       6
1      14
>>> (p / []).round(2)  # normalize
        pval
A B
0 0     0.10
  1     0.20
1 0     0.25
  1     0.45
```

### Functions

| [`from_points_to_binary`](#spyn.ppi.pot.from_points_to_binary)(d[, mid_fun])           | Convert data to binary based on a midpoint function.   |
|------------------------------------------------------------------------------------------------|--------------------------------------------------------|
| [`relative_risk`](#spyn.ppi.pot.relative_risk)(joint_prob_pot, event_var, ...) | Compute relative risk from a joint probability pot.    |

### Classes

| [`Pot`](#spyn.ppi.pot.Pot)([data])     | A discrete potential (probability/likelihood table).   |
|------------------------------------------------------------------|--------------------------------------------------------|
| [`ProbPot`](#spyn.ppi.pot.ProbPot)([data]) | A Pot subclass for probability-specific operations.    |

### *class* spyn.ppi.pot.Pot(data=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

A discrete potential (probability/likelihood table).

The core data structure is a pandas DataFrame (`self.tb`) with variable
columns and a mandatory `pval` column holding the values.

```pycon
>>> p = Pot({'A': [0, 1], 'B': [0, 1], 'pval': [0.3, 0.7]})
>>> p.vars
['A', 'B']
>>> p.n
2
```

#### add_count(num)

Add a count to all variable combinations (including missing ones).

* **Return type:**
  [`Pot`](#spyn.ppi.pot.Pot)

#### assimilate(pot)

Bayes rule: multiply by pot, normalize, project back to own vars.

This computes P(X|D) ∝ P(D|X) \* P(X), i.e. prior.assimilate(likelihood).

```pycon
>>> prior = Pot({'H': [0, 1], 'pval': [0.5, 0.5]})
>>> likelihood = Pot({'H': [0, 1], 'pval': [0.2, 0.8]})
>>> posterior = prior.assimilate(likelihood)
>>> posterior.n
2
```

* **Return type:**
  [`Pot`](#spyn.ppi.pot.Pot)

#### binarize(var_values_to_map_to_1_dict)

Map specified variables to {0, 1} and re-aggregate.

* **Return type:**
  [`Pot`](#spyn.ppi.pot.Pot)

#### *classmethod* binary_pot(varname, prob=1)

Create a binary variable potential.

```pycon
>>> Pot.binary_pot('coin', 0.6)
      pval
coin
0      0.4
1      0.6
```

* **Return type:**
  [`Pot`](#spyn.ppi.pot.Pot)

#### count_pot_to_prob_pot(prior_count=1, possible_vals_for_var=None)

Convert count pot to probability pot with Laplace-like smoothing.

Adds prior_count to all counts, then normalizes.

* **Return type:**
  [`Pot`](#spyn.ppi.pot.Pot)

#### *classmethod* from_count_df_to_count(count_df, count_col='pval')

Create a pot from a DataFrame with a count column.

* **Return type:**
  [`Pot`](#spyn.ppi.pot.Pot)

#### *classmethod* from_counter(counts, vars=None)

Create a pot from a Counter of tuples.

* **Return type:**
  [`Pot`](#spyn.ppi.pot.Pot)

#### *classmethod* from_hard_evidence(\*\*var_val)

Create a singleton potential for hard evidence.

```pycon
>>> Pot.from_hard_evidence(X=1, Y=0).n
1
```

* **Return type:**
  [`Pot`](#spyn.ppi.pot.Pot)

#### *classmethod* from_points_to_bins(pts, \*\*kwargs)

Create a pot from points with binning (alias for count).

* **Return type:**
  [`Pot`](#spyn.ppi.pot.Pot)

#### *classmethod* from_points_to_count(pts, vars=None, scope_for_var=None)

Create a count pot from a collection of data points.

```pycon
>>> import pandas as pd
>>> df = pd.DataFrame({'A': [0, 0, 1, 1, 1], 'B': [0, 1, 0, 0, 1]})
>>> Pot.from_points_to_count(df).n
4
```

* **Return type:**
  [`Pot`](#spyn.ppi.pot.Pot)

#### get_slice(intercept_dict)

Return sub-pot at specific variable values.

```pycon
>>> p = Pot({'A': [0, 0, 1, 1], 'B': [0, 1, 0, 1], 'pval': [2, 4, 5, 9]})
>>> p[{'A': 1}]  # slice where A=1
   pval
B
0     5
1     9
```

* **Return type:**
  [`Pot`](#spyn.ppi.pot.Pot)

#### given(conditional_vars)

Create conditional potential: P(remaining | conditional_vars).

* **Return type:**
  [`Pot`](#spyn.ppi.pot.Pot)

#### *property* n *: [int](https://docs.python.org/3/builtins/functions.html#int)*

Number of rows in the table.

#### normalize(var_list=())

Normalize the pot with respect to var_list.

If var_list is empty, fully normalizes (divides by total sum).
Otherwise, divides by the projection onto var_list (conditional prob).

```pycon
>>> p = Pot({'A': [0, 1], 'pval': [3, 7]})
>>> p.normalize()  # full normalization
   pval
A
0   0.3
1   0.7
```

* **Return type:**
  [`Pot`](#spyn.ppi.pot.Pot)

#### order_vars(var_list, , sort_pts=True)

Return a new Pot with reordered variables.

```pycon
>>> p = Pot({'B': [0, 1], 'A': [0, 1], 'pval': [3, 7]})
>>> p.order_vars(['A', 'B']).vars
['A', 'B']
```

* **Return type:**
  [`Pot`](#spyn.ppi.pot.Pot)

#### prob_of(var_val_dict)

Get probability for a variable assignment.

* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)

#### project_to(var_list, , assert_subset=False)

Marginalize to a subset of variables.

Non-strict by default: ignores variables not in the pot.

```pycon
>>> p = Pot({'A': [0, 0, 1, 1], 'B': [0, 1, 0, 1], 'pval': [2, 4, 5, 9]})
>>> p.project_to('A')  # sum out B
     pval
A
0       6
1      14
```

* **Return type:**
  [`Pot`](#spyn.ppi.pot.Pot)

#### *property* pval *: Series*

The pval column as a Series.

#### pval_of(var_val_dict, default_val=0.0)

Look up the pval for a specific variable assignment.

```pycon
>>> p = Pot({'A': [0, 1], 'pval': [3, 7]})
>>> int(p.pval_of({'A': 1}))
7
```

* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)

#### *classmethod* rand(n_var_vals=None, var_names=None, granularity=None, try_to_get_unique_values=False)

Generate a random probability pot.

```pycon
>>> p = Pot.rand([2, 3])
>>> p.n
6
>>> bool(abs(p.values.sum() - 1.0) < 1e-10)
True
```

* **Return type:**
  [`Pot`](#spyn.ppi.pot.Pot)

#### rect_perspective_df()

For a 2-variable pot, return a matrix-form DataFrame.

```pycon
>>> p = Pot({'A': [0, 0, 1, 1], 'B': [0, 1, 0, 1], 'pval': [1, 2, 3, 4]})
>>> p.rect_perspective_df()
B  0  1
A
0  1  2
1  3  4
```

* **Return type:**
  `DataFrame`

#### relative_risk(event_var, exposure_var, event_val=1, exposure_val=1, smooth_count=None)

Compute epidemiological relative risk from a joint pot.

* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)

#### round(ndigits=None, , inplace=False)

Round pval values.

```pycon
>>> Pot({'A': [0, 1], 'pval': [0.333, 0.667]}).round(2)
   pval
A
0  0.33
1  0.67
```

* **Return type:**
  [`Pot`](#spyn.ppi.pot.Pot)

#### select(selection)

Filter rows using a callable predicate on row dicts.

* **Return type:**
  [`Pot`](#spyn.ppi.pot.Pot)

#### sort_pts(var_list=None, \*\*kwargs)

Return a new Pot with rows sorted by variables.

* **Return type:**
  [`Pot`](#spyn.ppi.pot.Pot)

#### unassimilate(pot)

Inverse of assimilate.

* **Return type:**
  [`Pot`](#spyn.ppi.pot.Pot)

#### *property* values *: ndarray*

The pval column as a numpy array.

#### *property* vars *: [list](https://docs.python.org/3/builtins/stdtypes.html#list)[[str](https://docs.python.org/3/builtins/stdtypes.html#str)]*

Variable names (all columns except ‘pval’).

```pycon
>>> Pot({'X': [0, 1], 'pval': [1, 1]}).vars
['X']
```

#### *property* vars_set *: [set](https://docs.python.org/3/builtins/stdtypes.html#set)[[str](https://docs.python.org/3/builtins/stdtypes.html#str)]*

Set of variable names.

#### *classmethod* zero_potential(varnames_and_scopes)

Create a pot with all variable combinations and pval=0.

```pycon
>>> pot = Pot.zero_potential({'A': 2, 'B': [10, 20]})
>>> pot.n
4
```

* **Return type:**
  [`Pot`](#spyn.ppi.pot.Pot)

### *class* spyn.ppi.pot.ProbPot(data=None)

Bases: [`Pot`](#spyn.ppi.pot.Pot)

A Pot subclass for probability-specific operations.

### spyn.ppi.pot.from_points_to_binary(d, mid_fun=<function median>)

Convert data to binary based on a midpoint function.

### spyn.ppi.pot.relative_risk(joint_prob_pot, event_var, exposure_var)

Compute relative risk from a joint probability pot.
