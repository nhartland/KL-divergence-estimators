# Estimator implementations

{% for estimator in Estimators %}
 - **{{ estimator.__name__ }}**

   {{ estimator.__doc__.splitlines()[0] }}
{% endfor %}

These estimators have been benchmarked against [slaypni/universal-divergence](https://github.com/slaypni/universal-divergence).

# Tests

{% for test in Tests %}

## {{ test.name }}
{{ test.__doc__ }}
The expected value for the divergence in this test is **D={{test.expectation}}**.

#### Comparison of estimator implementations 

|    Estimator    |  D(P\|Q) | Time (s)|
|-----------------|----------|---------|
{%- for comp in Comparisons[test.filename] %}
{{'|%-17s|% -.3e|%.3f|' % (comp.Estimator, comp.Mean, comp.Time) }}
{%- endfor %}

#### Convergence of estimator with *N*
![Convergence Plot]({{ConvergencePlots[test.filename]}})

{% endfor %}
