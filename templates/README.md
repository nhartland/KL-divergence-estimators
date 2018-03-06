# KL divergence estimators

Here I test a few implementations of a KL-divergence estimator
based on k-Nearest-Neighbours probability density estimation.

The estimator is that of 

> Qing Wang, Sanjeev R. Kulkarni, and Sergio Verdú. "Divergence estimation for multidimensional densities via k-nearest-neighbor distances." Information Theory, IEEE Transactions on 55.5 (2009): 2392-2405.

This study is far from exhaustive, and timings are sensitive to implementation
details. Please take with a pinch of salt.

# Estimator implementations

{% for estimator in Estimators %}
 - **{{ estimator.__name__ }}**

   {{ estimator.__doc__.splitlines()[0] }}
{% endfor %}

These estimators have been benchmarked against `slaypni/universal-divergence`.

# Tests

{% for test in Tests %}

## {{ test.name }}
{{ test.__doc__ }}

|    Estimator    |  D(P\|Q) | Time (s)|
|-----------------|----------|---------|
{%- for comp in Comparisons[test.filename] %}
{{'|%-17s|% -.3e|%.5f|' % (comp.Estimator, comp.Mean, comp.Time) }}
{%- endfor %}

### Convergence of estimator with *N*
![Convergence Plot]({{ConvergencePlots[test.filename]}})

{% endfor %}


# Generating this document

```Shell
 python run_tests.py
```

Which will then likely take some time to complete.

#### Requirements

- Python >= 3.6
- scipy, scikit-learn, jinja2 

#### Important settings

The number of resamples used to estimate uncertainties is defined by
`n_resamples` in `tests.py`. This is naturally an *extremely* sensitive variable
for how long the tests take to run.

