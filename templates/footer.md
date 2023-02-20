# Generating this document

Start in a clean python 3.10 environment and run the following

```Shell
 # Setup dependencies
 pip install -r requirements.txt
 # Run the tests and generate the figures
 ./src/run_tests.py
 # Add the header and footer to the report
 cat templates/header.md report.md templates/footer.md > README.md
```

Which will then likely take some time to complete.

#### Important settings

The number of resamples used to estimate uncertainties is defined by
`n_resamples` in `tests.py`. This is naturally an *extremely* sensitive variable
for how long the tests take to run.

