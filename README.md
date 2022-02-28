# MAAML : Make Applications Apply Machine Learning


A package of under 2000 line of code for researchers in data analysis, preprocessing, machine learning and deep learning. This package can help you transform your 100 to 1000 lines of code to 10 with simple to use piplines, currently specialized in time series data analysis and uses pandas Dataframes in backend.
The package includes:

- Datasets section that includes a time_series module.
- Utils module with useful functions and classes.
- Data cleaning  with a cleaning module.
- Data preprocessing with a preprocessing module.
- Machine learning evaluation with machine_learning module.
- Deep learning evaluation with deep_learning module.

## Installation

```bash
$ pip install maaml
```
## Usage

`maaml` can be used to clean and preprocess data and also do machine learning and deep learning training and evaluation.
exemple:

```python
from maam.utils import read_csv,save_parquet
from maaml.cleaning import DataCleaner
from maaml.preprocessing import DataPreprocessor,columns_mean
from maaml.machine_learning import Evaluator as MLEvaluator
from maaml.deep_learning import Evaluator as DLEvaluator
from maaml.deep_learning import DeepRCNModel
file_path = "dataset/test.csv"  # path to your file
# read csv data
df = read_csv(file_path, delimiter=" ", header=None, verbose=0, prompt=None)
# save data in an optimized format 
save_parquet(df,path="dataset/",name="test")
# clean the dataset : interpolate, remove raws with missing values,merge dataframes, add columns ..
cleaner_dataset = DataCleaner(
    df,
    drop_duplicates=True,
    add_columns_dictionnary: dict = {"driver": "driver1","road": "HIGHWAY","target": "Normal"}, # add columns with a spesific values
    save_to="csv", # save data after cleaning process to csv file, can also be saved to parquet with the argument "parquet" instead of "csv".
    save_tag="dataset",
    timestamp_column="Timestamp (seconds)",
    verbose=1,
    )
# preprocess the dataset: filter the data, scale the data , window stepping, encode categorical data, one hot encode the classification target
preprocessed_dataset = DataPreprocessor(
    dataset=cleaner_dataset(),# the data can also be passed from the cleaner_dataset class instance using the data attribute such as : cleaner_dataset.data 
    target_name="target",
    scaler="minmax",
    droped_columns=["Timestamp (seconds)"],
    window_size=10,
    step=10,
    window_transformation=True,
    window_transformation_function=columns_mean, # columns_mean is a pre-build function available in the utils module and the preprocessing module that takes a dataframe as input and applies mean to all induvidual columns.
    save_to="csv", # The same as the DataCleaner class, you can use "csv" or "parquet".
    save_tag="preprocessed_dataset",
    verbose=0,
    )
# evaluate the dataset with different machine learning models and different metrics(accuracy,precision,recall,f1..) using cross validation, feature importance ranking
ml_evaluation = MLEvaluator(
    dataset=preprocessed.dataset.ml_dataset, # ML ready to evaluate data from the preprocessor
    target_name="target",
    nb_splits=5,
    test_size=0.3,
    full_eval=True, # cross validation evaluation using 9 diffrent ML models
    save_eval=True,
    save_tag="mlevaluation",
    preprocessing_alias="minmax_scaling",
    verbose=0)
# use our DeepRCNModel or build your own model and evaluate it easily with our evaluator with diffrent metrics and customization options for training, and automatically save your model and training results.
dl_model=DeepRCNModel
dl_model.show()
dl_evaluation = DLEvaluator(
    dl_model(),
    dataset=preprocessed.dataset.dl_dataset, # another ready to evaluate data from the preprocessor for DL
    target_name="target",
    model_name:"MyDeepRCN model", # model name to include in the evaluation result table
    input_shape= dl_model.input_shape,
    preprocessing_alias="minmax_scaling",
    cross_eval=True,
    save_tag="deep_learning",
    nb_splits=5,
    test_size=0.3,
    opt="adam",
    loss="categorical_crossentropy",
    epochs=600,
    batch_size=60,
    verbose=1,
    )

```
## Contributing
The project is still in it's Beta phase and looking for expanding the project and improving it, we are open to contributers and collaborations.
For more information, discussions and possible collaborations please contact :
- email: najemabdennour@gmail.com  
- linkedin: https://www.linkedin.com/in/najemeddine-abdennour/

## License

`maaml` was created by Najemeddine Abdennour. It is licensed under the terms
of the GNU General Public License v3 (GPLv3).
