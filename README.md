# Permutation Test - Lecture & Code 

A short statistics lecture (in hebrew) on the **permutation test**, a non-parametric way to
test whether two groups differ - without assuming a particular distribution.

The talk is framed around a story: a dairy farmer runs a milk-supplement trial on
20 cows (10 control, 10 treatment). The treatment group yields more milk on
average (observed difference = **1.18**), but with such a small sample, could that
gap just be luck? The permutation test answers exactly that question by repeatedly
re-shuffling the group labels and asking how often pure chance produces a gap as
large as the one observed.

**Result:** only ~12 of 500 shuffles beat the observed difference, giving a
one-sided p ≈ **0.024 < 0.05** - evidence against the null that the supplement
does nothing.

## Contents

- `slides/` - the lecture deck (`.pptx`)
- `code/` - the permutation-test implementation and the plot that produces the
- `figures/`
  null distribution

## Running the code

```bash
pip install numpy matplotlib
python code/permutation_test.py
```

## References

- Wasserman, L. (2004). *All of Statistics: A Concise Course in Statistical Inference*. Springer.
