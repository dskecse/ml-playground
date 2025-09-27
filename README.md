# ML Playground

Machine Learning Playground with Python

https://youtu.be/_uQrJ0TkZlc?feature=shared&t=15057

## Libraries and Tools for ML

* [numpy](https://numpy.org/) - provides multi-dimensional arrays
* [pandas](https://pandas.pydata.org/) - a data-analysis library that provides a concept called *data frame*
* [matplotlib](https://matplotlib.org/) - a 2D plotting library for creating graphs and plots
* [scikit-learn](https://scikit-learn.org/stable/) - provides common algorithms like decision trees, neural networks and so on
* [JupyterLab](https://jupyterlab.readthedocs.io/en/stable/index.html) - environment for interactive computing with computational notebooks

## Prerequisites

* git
* Docker Desktop / OrbStack

## Setup

```sh
git clone https://github.com/dskecse/ml-playground
cd $_
docker compose up
```

This will:

* clone the repo
* `cd` into the repo dir
* pull up the official [`scipy-notebook` Docker image](https://quay.io/repository/jupyter/scipy-notebook?tab=tags) - includes JupyterLab and a `scikit-learn` package, [based off of the `minimal-notebook`](https://jupyter-docker-stacks.readthedocs.io/en/latest/using/selecting.html#image-relationships)
* map the repo's root dir to the container's work dir
* spin up the Jupyter Server on port `8888`
* serve notebooks from the repo dir.

To access the server (Jupyter Dashboard), open up http://localhost:8888/lab?token=TOKEN.

By default it points to the user's `HOME` dir. Switch to the `work` dir in the Jupyter Dashboard and create a Jupyter notebook:

```
Launcher -> Notebook -> Python 3 (ipykernel)
```

This will create a notebook file with an `.ipynb` extension.

## Importing a Dataset

How to load a dataset from the CSV file in Jupyter.

Dataset: [Video Game Sales](https://www.kaggle.com/datasets/gregorut/videogamesales)

```python
import pandas as pd
df = pd.read_csv("vgsales.csv")
df # to inspect the resulting data frame
```

This returns a data frame object like an Excel spreadsheet.

Below are some of the most useful methods and attributes:

* `df.shape` - returns a tuple of the number of records and columns in the dataset
* `df.describe()` - basic statistics of the data
* `df.values` - a 2D array of a dataset representation

## ML in Action

High-level steps to follow in a ML project:

1. Import the data - often comes in the form of a CSV file
2. Clean the data - involves removing irrelevant, duplicate or incomplete data
3. Split the data into training/test sets - usually 80% for training the model & 20% for testing
4. Create a model - involves selecting an algorithm to analyze the data, trade-off: accuracy vs performance
5. Train the model
6. Make predictions - not always accurate
7. Evaluate predictions and improve - involves evaluating predictions and measuring their accuracy

Depending on the accuracy of predictions, we could get back to the model and:

* select a different algorithm to produce a more accurate result for our problem
* or fine-tune the parameters of a model

Each algorithm has parameters that can be modified to optimize the accuracy.
