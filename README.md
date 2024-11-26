# TW Experimentation: A library for automated A/B testing


**TW Experimentation** is a library to design experiments, check data, run statistical tests and make decisions

<summary><strong><em>Table of Contents</em></strong></summary>

- [TW Experimentation: A library for automated A/B testing and causal inference](#tw-experimentation-a-library-for-automated-ab-testing-and-causal-inference)
  - [What can this do for you?](#what-can-this-do-for-you)
    - [1. Designing experiments](#1-designing-experiments)
    - [2. Evaluating results](#2-evaluating-results)
  - [Installation](#installation)
  - [Quick Start](#quick-start)
    - [Notebooks](#notebooks)
    - [Streamlit web app](#streamlit-web-app)
  - [For Developers](#for-developers)
    - [Testing](#testing)

 ## What can this do for you?

The experimentation library can help you with:
- Sample Size Calculator  [link](https://github.com/transferwise/tw-experimentation/blob/main/notebooks/1_pre_experiment.ipynb) - here you can estimate how many observations you need to perform an experiment on the given parameters.

![plot](https://github.com/transferwise/tw-experimentation/blob/main/docs/images/sample_size_notebook_1.png?raw=True)

![plot](https://github.com/transferwise/tw-experimentation/blob/main/docs/images/sample_size_notebook_2.png?raw=True)

- Integrity checks + Evaluation [link](https://github.com/transferwise/tw-experimentation/blob/main/notebooks/2_integrity_checks%20%2B%20evaluation.ipynb) - here you can check important plots to understand your data better and then run statistical evaluation. Our library will automatically detect if it is a binary metric or continuous and then apply specific method as well as multiple hypothesis correction if needed.

![plot](https://github.com/transferwise/tw-experimentation/blob/main/docs/images/integrity_checks_notebook_1.png?raw=True)

![plot](https://github.com/transferwise/tw-experimentation/blob/main/docs/images/integrity_checks_notebook_2.png?raw=True)

![plot](https://github.com/transferwise/tw-experimentation/blob/main/docs/images/evaluation_notebook_1.png?raw=True)

![plot](https://github.com/transferwise/tw-experimentation/blob/main/docs/images/evaluation_notebook_2.png?raw=True)

- Evaluation (Bayesian A/B testing) [link](https://github.com/transferwise/tw-experimentation/blob/main/notebooks/2a_evaluation_bayesian.ipynb) - here you can apply bayesian evaluation on your dataset

### 1. Designing experiments
By using **TW Experimentation** you can design your experiments, choose sample size, evaluate the experiment.


### 2. Evaluating results
You can use various statistical tests for the metrics provided. (Frequentist / Bayesian)
For this goal you can use jupyter notebooks or streamlit app with user-friendly interface.
Using this repo you can:
- Run frequentist evaluation
- Run bayesian evaluation
- Apply multiple hypothesis correction
- Remove outliers
- Check different plots
- Run segmentation to check evaluation on specific segments as well as find unusual segments using our another tool: wise-pizza


## Installation

1. You can easily install this repo using "pip"
```
pip install tw-experimentation
```

Then you can just use all functionality.

For running streamlit app please just open the terminal and run:
```
run_tw_experimentation_streamlit
```

2. You can install the package  via the dependency manager poetry after cloning/git pull/download it as a zip from this repository.

To do so, clone the repository by running
```
git clone git@github.com:transferwise/tw-experimentation.git
```
from terminal.
To set up poetry, run
```
make set-up-poetry-mac
```
for mac (or linux) and
```
make set-up-poetry-windows
```
for windows.
Then, run
```
make run-streamlit-poetry
```
from the root of the package folder.

3. **Alernative:** TW Experimentation requires the following libraries to work which you can find in the .yml file. To install requirements please make sure you have installed the package manager Anaconda and then run the following commands in the terminal:

```
conda env create -n <my_env> -f envs/environment.yml
conda activate <my_env>
```

If you are using Windows, please do these additional steps:

1. pick a jaxlib-0.3.7 wheel from here https://whls.blob.core.windows.net/unstable/index.html and install it manually (pip install <wheel_url>)
2. Install jax==0.3.7

If you have any problems with jax on Mac, please do the following:
```
pip uninstall jax jaxlib
conda install -c conda-forge jaxlib
conda install -c conda-forge jax
```

## Quick Start

Make sure you have followed the installation instructions.

### Notebooks
You can use the jupyter notebooks `1_pre_experiment.ipynb` or `2_integrity_checks + evaluation.ipynb` for experiments design and evaluation.
The tw experimentation package can be used for different things, for example for analyzing results:

```Python
df = pd.read_csv('experiment.csv')

ed = ExperimentDataset(
    data=df,
    variant="T",
    targets=['conversion', 'revenue'],
    date='trigger_dates',
    pre_experiment_cols=None,
    n_variants=2,
)
ed.preprocess_dataset(remove_outliers=True)
```

This code will generate the data model for experiment analysis

And then you can run evaluation

```Python
evaluation = FrequentistEvaluation(experiment_dataset)
evaluation.start()
```

### Streamlit web app

For running streamlit app please just open the terminal and run after "pip install" section:
```
run_tw_experimentation_streamlit
```

Or:

Open terminal and navigate to the repository.
Then navigate to the folder `./tw_experimentation/streamlit`.

Now run the command `streamlit run Main.py` and the app should open in your browser.

For testing, you can load the test dataset from `data/test_data.csv`. You should either set the fields as shown in the 
screenshot below; or just click on the "Load configuration from json" section and load the whole 
configuration from the file `data/test_config.json`.

![plot](https://github.com/transferwise/tw-experimentation/blob/main/docs/images/data_loading_streamlit.png?raw=True)
![plot](https://github.com/transferwise/tw-experimentation/blob/main/docs/images/evaluation_streamlit.png?raw=True)

Tip on navigation:
`ls` - show files in current directory
`pwd` - print current directory address
`cd` - change directory, e.g. `cd ./tw_experimentation/streamlit`

## For Developers
### Testing
We use [PyTest](https://docs.pytest.org/) for testing. If you want to contribute code, make sure that the tests in tests/ run without errors.
