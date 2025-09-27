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

## Creating Jupyter Notebooks

By default Jupyter Dashboard points to the user's `HOME` dir.

Switch to the `work` dir there and create a Jupyter notebook:

```
Launcher -> Notebook -> Python 3 (ipykernel)
```

This will create a notebook file with an `.ipynb` extension.

## Importing a Dataset

How to load a dataset from the CSV file in Jupyter.

Given the [Video Game Sales](https://www.kaggle.com/datasets/gregorut/videogamesales) dataset:

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

## Jupyter Shortcuts

Jupyter notebooks are structured around units called **cells**. They serve as the fundamental building blocks for organizing and executing code and text within a notebook.

An activated cell can either be in:

* *Edit mode* or
* *Command mode*

Depending on the mode, there are different shortcuts. When a cell is in the *Command mode* (press <kbd>Esc</kbd> to enable):

* <kbd>Cmd</kbd>+<kbd>Shift</kbd>+<kbd>H</kbd> - to list all the keyboard shortcuts
* <kbd>Cmd</kbd>+<kbd>Shift</kbd>+<kbd>C</kbd> - to open the Command Palette

Most used shortcuts:

* <kbd>B</kbd> (below) - to Insert a Cell below the current cell (in Command mode)
* <kbd>A</kbd> (above) - to Insert a Cell above the current cell (in Command mode)
* <kbd>DD</kbd> - to Delete the Cell
* <kbd>J</kbd> and <kbd>K</kbd> - to move down and up the cells (VIM-like navigation)
* <kbd>Enter</kbd> - to Enter the *Edit mode* in the activated Cell
* <kbd>Cmd</kbd>+<kbd>/</kbd> - to Comment out the line
* <kbd>Cmd</kbd>+<kbd>Enter</kbd> - to Run only the selected Cell and do not advance

NOTE: Jupyter saves the output of each cell, so we don't have to rerun the code if it hasn't changed!

To run all cells: activate the Command Palette and search for `Run All Cells`.

The `.ipynb` notebook files include the source code organized in cells as well as the output for each cell. That's why it's different from a regular `.py` file where we only have the source code.

There's also autocompletion and IntelliSense in Jupyter notebooks:

1. Type in `df.` and press <kbd>Tab</kbd> to see all the attributes and methods on this data frame object.
2. With the cursor on the name of the method press <kbd>Shift</kbd>+<kbd>Tab</kbd> to see the tooltip on what this method does and what parameters it takes.

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

## ML Project: Online Music Player

Users sign up, we ask their age and gender.
Based on their profile, we recommend the music they're likely to listen to.

GOAL: use ML to improve recommendations.

We want to build a model:

* we feed this model some sample data based on existing users
* our model learns patterns in the data, so we can ask it to make predictions.

When a new user signs up, we tell our model we have a new user with this profile, and ask what kind of music this user is interested in. Our model will say "jazz" or "hip-hop" or whatever, and based on that we could make suggestions to the user.

Given: [Music](http://bit.ly/music-csv) dataset.
