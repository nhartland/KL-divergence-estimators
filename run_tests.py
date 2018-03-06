#!/usr/bin/env python

import os
import sys
import logging

from jinja2 import Environment, FileSystemLoader

from tests import Tests
from plots import convergence_plot
from knn_divergence import Estimators, naive_estimator

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

def generate_estimator_comparison():
    """ Generates a comparison of the estimator results for all estimators and
    tests.  Returns a dictionary indexed by test 'filename' containing lists of
    EstimatorStats."""
    N, k = 1000, 5
    estimator_comparison = {}
    for test in Tests:
        estimator_comparison[test.filename] = []
        for estimator in Estimators:
            result = test.compute(estimator, N, k)
            estimator_comparison[test.filename].append(result)
    return estimator_comparison


def generate_convergence_plots():
    """ Generates plots of the convergence of the estimator with `N` """
    convergence_plots = {}
    for test in Tests:
        convergence_plots[test.filename] = convergence_plot(naive_estimator, test)
    return convergence_plots


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
        'Comparisons': generate_estimator_comparison(),
        'ConvergencePlots': generate_convergence_plots()
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
