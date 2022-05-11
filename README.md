# Forest

This package allows to predict cover type of a forest. Data was taken from [Forest Dataset](https://www.kaggle.com/competitions/forest-cover-type-prediction).

## Usage
### Train model
1. Clone this repository to your machine.
2. Download [Forest](https://www.kaggle.com/competitions/forest-cover-type-prediction) dataset, save csv locally (default path is *data/forest_data.csv* in repository's root).
3. Make sure Python 3.9 and [Poetry](https://python-poetry.org/docs/) are installed on your machine (I use Poetry 1.1.13).
4. Install the project dependencies (*run this and following commands in a terminal, from the root of a cloned repository(from Forest folder)*):
```sh
poetry install --no-dev
```
5. Run train with the following command:
```sh
poetry run train -d <path to csv with data> -s <path to save trained model>
```
You can configure additional options (such as hyperparameters/making feature engineering) in the CLI. To get a full list of them, use help:
```sh
poetry run train --help
```
6. Run MLflow UI to see the information about experiments you conducted:
```sh
poetry run mlflow ui
```
7. You can also use Nested CV for choosing best hyperparameters and evaluating the model:
```sh
poetry run train --use-nested-cv=True
```
### EDA report
Eda package lets generate you an eda report in html format.
1. Run creating a report with following command:
```sh
poetry run eda -d <path to csv with data> -s <path to save report>
```
Default for loading and saving is *data/forest_data.csv* and *data/eda_report.html* 

## Screenshots
### Mlflow
mlflow experiments:
![изображение](https://user-images.githubusercontent.com/44612254/167252439-52d8d943-0951-46c8-bf32-1c7908e55515.png)
(As main metric was Accuracy selected, beacause distribution of target class CoverType is normal and not skewed)

### Black & Flake8
![изображение](https://user-images.githubusercontent.com/44612254/167374487-c499b540-1f00-442f-ad55-7bef545d91b9.png)
![изображение](https://user-images.githubusercontent.com/44612254/167375179-1d3a0bc6-932c-440e-8995-b1bc491813ed.png)

### Pytest
In test_train were:
* logreg-c and classifier inputs tested, with right and not correct inputs, since logreg-c can be only btw 0 and 1, and classifier KNeighbors or LogReg:
* correct creating and saving of model tested 
In test_data was:
* correct features and target extracting tested
![изображение](https://user-images.githubusercontent.com/44612254/167865680-a4893b08-d01d-472f-9e5a-e685207af401.png)
