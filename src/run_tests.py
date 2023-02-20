#!/usr/bin/env python
""" Rendering of report.md

    This script loops over all tests defined in test.py and all
    estimators defined in knn_divergence.py and generates the
    quanitites needed for the report.
"""
import os
import sys
import logging

from jinja2 import Environment, FileSystemLoader

from tests import Tests
from plots import convergence_plot
from knn_divergence import Estimators, scipy_estimator

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

# Jinja rendering
TEMPLATE_ENVIRONMENT = Environment(
    autoescape=False,
    loader=FileSystemLoader(os.path.join(os.getcwd(), "templates")),
    trim_blocks=False,
)


def render_template(template_filename, context):
    return TEMPLATE_ENVIRONMENT.get_template(template_filename).render(context)


def generate_estimator_comparison():
    """Generates a comparison of the estimator results for all estimators and
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
    """Generates plots of the convergence of the estimator with `N`"""
    convergence_plots = {}
    for test in Tests:
        convergence_plots[test.filename] = convergence_plot(scipy_estimator, test)
    return convergence_plots


def main():
    data = {
        "Estimators": Estimators,
        "Tests": Tests,
        "Comparisons": generate_estimator_comparison(),
        "ConvergencePlots": generate_convergence_plots(),
    }
    with open("report.md", "w") as f:
        f.write(render_template("report.md", data))


if __name__ == "__main__":
    main()
