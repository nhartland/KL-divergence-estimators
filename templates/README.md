# KL divergence estimators

Here I test a few implementations of a KL-divergence estimator
based on k-Nearest-Neighbours probability density estimation.

The estimator is that of 

> Qing Wang, Sanjeev R. Kulkarni, and Sergio Verdú. "Divergence estimation for multidimensional densities via k-nearest-neighbor distances." Information Theory, IEEE Transactions on 55.5 (2009): 2392-2405.

# Estimator implementations

{% for estimator in Estimators %}
 - *{{ estimator.__name__ }}*

   {{ estimator.__doc__.splitlines()[0] }}
{% endfor %}

# Tests

{% for test in Tests %}

## {{ test.name }}
{{ test.__doc__ }}

|    Estimator    |  D(P\|Q) | Time (s)|
|-----------------|----------|---------|
{%- for comp in Comparisons[test.filename] %}
{{'|%-17s|% -.3e|%.7f|' % (comp.Estimator, comp.Mean, comp.Time) }}
{%- endfor %}

{% endfor %}


## Requirements

- Python >= 3.6
- scipy, scikit-learn 
- slaypni/universal-divergence
