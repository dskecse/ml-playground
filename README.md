# ML Playground

Machine Learning Playground with Python

https://youtu.be/_uQrJ0TkZlc?feature=shared&t=15057

## Libraries and Tools for ML

* [numpy](https://numpy.org/) - provides multi-dimensional arrays
* [pandas](https://pandas.pydata.org/) - a data-analysis library that provides a concept called *data frame*
* [matplotlib](https://matplotlib.org/) - a 2D plotting library for creating graphs and plots
* [scikit-learn](https://scikit-learn.org/stable/) - provides common algorithms like decision trees, neural networks and so on
* [joblib](https://joblib.readthedocs.io/en/stable/) - provides fast compressed persistence capabilities
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

### Learning and Predicting

Building a model requires using an ML algorithm.

There's many algorithms out there, and each algorithm has its pros and cons in terms of

* performance
* accuracy.

We'll use a very simple algorithm called **decision tree**.

The good news is we don't have to explicitly program these algorithms.
They're already implemented for us in a library called `skicit-learn`!

In the `sklearn` package there's a module called `tree` with a `DecisionTreeClassifier` class.
This class implements the **decision tree** algorithm.

We need to create a new instance of this class, and let's call this object a `model`.

Now that we have a model, we need to train it, so it learns patterns in the data!

Finally, we need to ask our model to **make predictions**, so we can ask it:
what's the kind of music that a 24-year-old male likes?

There's no sample for a 24-year-old male. We *expect* the model to say "HipHop".
Similarly, we *expect* the model to say a 23-year-old female likes dance music.

```python
import os
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

music_data = pd.read_csv(os.path.join("..", "datasets", "music.csv")) # import the dataset
X = music_data.drop(columns=["genre"]) # create an input set
y = music_data["genre"] # create an output set

model = DecisionTreeClassifier() # create a model
model.fit(X, y) # train the model, takes 2 datasets: input & output set

# make predictions for a 21-year-old male & a 22-year-old female
predictions = model.predict(pd.DataFrame([ [21, 1], [22, 0] ], columns=["age", "gender"]))
predictions
```

So our model can successfully make predictions now!

But building a model that makes predictions accurately is not always that easy.

After we build a model, we need to **measure** its **accuracy**.
And if it's not accurate enough, we should either:

* fine-tune it
* or build a model using a different algorithm.

### Calculating the Accuracy

To measure the **accuracy** of the model, we need to split our dataset into 2 sets:

* one for *training*
* the other for *testing*.

Above we're passing the entire dataset for training the model and using 2 samples for making predictions. That's not enough to calculate the accuracy.

A general rule of thumb is to allocate:

* 70-80% of our data for training
* the other 20–30% for testing.

Then, instead of passing only 2 samples for making predictions, we can pass the *testing* dataset,
we'll get predictions and then compare them with the expected values in the output set for testing.
Based on that we can calculate the **accuracy**.

```python
import os
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

TEST_SET_RATIO = 0.2

music_data = pd.read_csv(os.path.join("..", "datasets", "music.csv")) # import the dataset
X = music_data.drop(columns=["genre"]) # create an input set
y = music_data["genre"] # create an output set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SET_RATIO)

model = DecisionTreeClassifier() # create a model
model.fit(X_train, y_train) # train the model, takes 2 sets: input & output set for training
predictions = model.predict(X_test) # make predictions

score = accuracy_score(y_test, predictions) # compare expected values & predictions, score: 0..1
score
```

With the `train_test_split` function we can easily split out dataset into 2 sets:

* training
* and testing.

That way, we're allocating 20% our data for testing.
This function returns a tuple (immutable list), so we unpack it into 4 variables:

* input sets for training (`X_train`) and testing (`x_test`)
* output sets for training (`y_train`) and testing (`y_test`).

When training our model, instead of passing the entire dataset, we pass only the *training* set (`X_train`, `y_train`).

When making predictions, instead of passing 2 samples, we pass the input set for testing (`X_test`).
Now we get predictions.

To calculate the **accuracy** we simply have to compare these predictions with the expected values we have in our output set for testing (`y_test`).

The `accuracy_score` function returns an **accuracy score** between 0 and 1.

In this case it's 1, or 100%. But if we run this one more time, we'll see a different result, e.g. `0.75`. Because every time we split our dataset into *training* and *testing sets*, we'll have different datasets, because the function randomly picks data for training and testing.

If we change the `test_size` from `0.2` to `0.8`, essentially using only 20% of our data for training this model and using the other 80% for testing, and we run this cell multiple times, the **accuracy** immediately drops to `0.4`, or 40%, or even lower. That's really bad.

The reason this is happening is because we're using very little data for training this model.
This is one of the **key concepts** in ML!
The more data we give to our model, and the cleaner the data is, the better result we get.
So if we have duplicates, irrelevant data or incomplete values, our model will learn *bad patterns* in our data. That's why it's really important to *clean* our data *before* training our model!

Even if we change the `test_size` back to `0.2`, the model's accuracy score could drop to `0.5`.
The reason is that we don't have enough data.
Some ML problems require 1000s or even millions of samples to train a model.
The more complex a problem is, the more data we need.

Here we're only dealing with a table of 3 columns.
But if we want to build a model to tell if a picture is a cat or a dog or a horse or a lion, we'll need millions of pictures. The more animals we want to support, the more pictures we need.

### Persisting Models

Basically, we:

1. Import our dataset
2. Create a model
3. Train it
4. Ask the model to make predictions

Steps 1-3 is NOT what we want to run every time we have a new user or every time we want to make recommendations to an existing user, because training a model can sometimes be really time- and resource-consuming.

In this example we're dealing with a very small dataset that has only 20 records.
But in real applications we might have a dataset with 1000s or millions of samples.
Training a model for that might take seconds or minutes or even hours.
So that's why a **model persistence** is important.

Once in a while, we build and train our model and then we save it to a file.
Next time we want to make predictions, we simply load the model from the file, and ask it to make predictions. That model is already trained. We don't need to retrain it.

We'll use the `joblib` library for saving and loading models.

> [!NOTE]
> `joblib` was removed from `sklearn.externals` starting with `scikit-learn` version `0.21`.
> The correct way now is to install `joblib` separately and import it directly.

Use `joblib.dump(model, file)` and `joblib.load(file)` as before.

After we train our model, we simply call `joblib.dump(model, file)`.

```python
import os
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
# from sklearn.externals import joblib # older way
import joblib

music_data = pd.read_csv(os.path.join("..", "datasets", "music.csv"))
X = music_data.drop(columns=["genre"])
y = music_data["genre"]

model = DecisionTreeClassifier()
model.fit(X, y)

model_file = os.path.join("..", "models", "music-recommender.joblib")
joblib.dump(model, model_file) # store the model
```

The file produced is simply a binary file.

Comment out steps 1-3, *load* the model instead, and make predictions:

```python
import os
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import joblib

model_file = os.path.join("..", "models", "music-recommender.joblib")
model = joblib.load(model_file) # load the model

predictions = model.predict(pd.DataFrame([[21, 1]], columns=X.columns))
predictions
```

### Visualizing a Decision Tree

Earlier it was mentioned that decision trees are the easiest to understand.
That's why we started ML with decision trees.

We're going to *export* our model in a *visual* format,
so we'll see how this model makes predictions!

Once again, the code was simplified, so we just:

1. Import our dataset
2. Create input and output sets
3. Create a model
4. And train it.

```python
import os
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

music_data = pd.read_csv(os.path.join("..", "datasets", "music.csv"))
X = music_data.drop(columns=["genre"])
y = music_data["genre"]

model = DecisionTreeClassifier()
model.fit(X, y)

out_file = os.path.join("..", "music-recommender.dot")
tree.export_graphviz(model, out_file=out_file,
                            feature_names=X.columns, # ["age", "gender"]
                            class_names=sorted(y.unique()),
                            label="all",
                            rounded=True,
                            filled=True)
```

The `tree` object has an `export_graphviz` method for exporting the decision tree in a graphical format. It's called after we've trained our model.

The method takes a model and an output file. We want to selectively pass keyword arguments without worrying about their order.

The [DOT format](https://graphviz.org/doc/info/lang.html) is a graph description language.

Other parameters we want to set are:

* `feature_names` – these are the **features** or the **columns** of our **dataset**, so they are the properties or features of our data, set to an array of 2 strings: age and gender
* `class_names` - the list of **classes** or **labels** we have in our output dataset (`y`), which includes all the genres or all the classes of our data, set to the sorted list of unique values

This produces a new `music-recommender.dot` file in a DOT format. It could be opened in an editor.

To visualize this graph, look for the tools in the list:

* https://www.graphviz.org/about/#viewers
* https://www.graphviz.org/resources/#editor-plugins

Online graphviz visual editors:

* https://magjac.com/graphviz-visual-editor/
* https://dreampuf.github.io/GraphvizOnline/

![Decision Tree Graph Visualization](/music-recommender.png)

This is exactly how our model makes predictions.
We have this **binary tree**, which means every node can have a maximum of 2 children.
On top of each node we have a condition.
If it's `True` then we go to the child node on the left side.
Otherwise, we go to the child on the right side.

The meaning of all the parameters:

* `feature_names` are set, so we can see the rules in our nodes
* `class_names` are set to the unique list of genres for displaying the `class` for each node
* `label="all"` means every node has labels that we can read
* `rounded=True` means nodes have rounded corners
* `filled=True` means that each box or each node is filled with a color
