from setuptools import setup, find_packages

setup(
    name="kl_divergence_estimators",
    version="0.1",
    author="",
    author_email="",
    install_requires=["numpy", "scipy", "scikit-learn"],
    packages=find_packages("src"),
    package_dir={"": "src"},
    zip_safe=True,
)
