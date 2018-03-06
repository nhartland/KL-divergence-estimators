#!/usr/bin/env python

import os
from jinja2 import Environment, FileSystemLoader

from knn_divergence import Estimators
from stats import Tests


def generate_estimator_comparison():
    """ Generates a comparison of the estimator results for all estimators and
    tests.  Returns a dictionary indexed by test 'filename' containing lists of
    EstimatorStats."""
    #TODO change up to N=100
    N, k = 10, 5
    estimator_comparison = {}
    for test in Tests:
        estimator_comparison[test.filename] = []
        for estimator in Estimators:
            result = test.compute(estimator, N, k)
            estimator_comparison[test.filename].append(result)
    return estimator_comparison


PATH = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_ENVIRONMENT = Environment(
    autoescape=False,
    loader=FileSystemLoader(os.path.join(PATH, 'templates')),
    trim_blocks=False)


def render_template(template_filename, context):
    return TEMPLATE_ENVIRONMENT.get_template(template_filename).render(context)


def create_index_html():
    fname = "README.md"

    context = {
        'Estimators': Estimators,
        'Tests': Tests,
        'Comparisons': generate_estimator_comparison()
    }
    #
    with open(fname, 'w') as f:
        html = render_template(fname, context)
        f.write(html)


def main():
    create_index_html()

#############################################################################

if __name__ == "__main__":
    main()
