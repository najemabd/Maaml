import pkg_resources
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    MaxAbsScaler,
    RobustScaler,
    QuantileTransformer,
    PowerTransformer,
    Normalizer,
)
from maaml.utils import save_csv
import time


class DataPreprocessor:
    """A class for Data preprocessing specialized in time series data analysis from dataframes. includes these attributes: `raw_dataset`, `filtered_dataset` ,`numeric_dataset` ,`scaled_dataset` ,`scaler_name` ,`ml_dataset` ,`features` ,`target` ,`target_ohe` ,`preprocessed_dataset` ,
    `dl_dataset` and in the case of window stepping `windowed_dataset` ,`ml_dataset_w` ,`features_w` ,`target_w` ,`target_ohe_w` ,`preprocessed_dataset_w` ,`dl_dataset_w`.
    It also includes useful static methods a `uahdataset_loading` for loading the UAHdataset,`label_encoding` for encoding categorical data,`data_scaling` for scaling the data,`one_hot_encoding` for one hot encoding and `window_stepping` for window stepping.

    Args:
    * data_path (str, optional): The data file name in the working directory or the data file path with the file name used in case the dataset prarameter is not set. Defaults to `""`.
    * specific_data (str or int, optional): A parameter to define a specific grouping from the UAHdataset used in case the dataset prarameter is not set. Defaults to `None`.
    * target_name (str, optional): The name of the dataset target as a string. Defaults to `"target"`.
    * dataset (pandas.DataFrame or array or numpy.array, optional): A dataset that includes features in columns and a target in one column with a name that match the target provided in the `target_name`. Defaults to `None`.
    * scaler (str, optional): selects the scaling technique used as integers from `"0"` to `"8"` passed as strings, or the name of the scaling technique such as `"minmax"` or `"normalizer"`. Defaults to no scaling with the value `"0"`.
    * droped_columns (list, optional): list of strings with the name of the columns to be removed or droped from the dataset after preprocessing. Defaults to `["Timestamp (seconds)"]`.
    * no_encoding_columns (list, optional): list of strings with the name of columns that will not be included in the label encoding process of cataegorical data. Defaults to `[]`.
    * no_scaling_columns (list, optional): list of strings with the name of columns not to be included in the data scaling. Defaults to `["target"]`.
    * window_size (int, optional): the size of the window in the case of window stepping the data, in case of `0` will not perform the window stepping. Defaults to `0`.
    * step (int, optional): The length of the step for window stepping, if smaller than `window_size` will result in overlapping windows, if equal to `window_size` performs standard window stepping, if bigger will skip some rows (not recommended). Defaults to `0`.
    * window_transformation (bool, optional): in case of True applies the function in `window_transformation_function` parameter to the window. Defaults to `False`.
    * window_transformation_function (function, optional): A function to be applied to the window preferably a lambda function. Defaults to the mean value with: `lambda x:sum(x)/len(x)`.
    * from_csv (bool, optional): Specifies if the data loaded from data_path is a csv file or not. Defaults to `True`.
    * save_dataset (bool, optional): saves in a newly created directory under the working directory in the case of `True`, the preprocessed dataset with an ML specified and DL specified datasets, and windowed data for each case if window stepping is applied. Defaults to `False`.
    * save_tag (str, optional): add a custom tag to the name of the files to be saved in the case of save_dataset is `True`. Defaults to `"dataset"`.
    * verbose (int, optional): An integer of the verbosity of the process can be ``0`` or ``1``. Defaults to ``0``.
    """

    def __init__(
        self,
        data_path="",
        specific_data=None,
        target_name="target",
        dataset=None,
        scaler="0",
        droped_columns=["Timestamp (seconds)"],
        no_encoding_columns=[],
        no_scaling_columns=["target"],
        window_size=0,
        step=0,
        window_transformation=False,
        window_transformation_function=lambda x: sum(x) / len(x),
        from_csv=True,
        save_dataset=False,
        save_tag="dataset",
        verbose=0,
    ):
        """A constructor for the DataProcessor class

        Args:
            * data_path (str, optional): The data file name in the working directory or the data file path with the file name used in case the dataset prarameter is not set. Defaults to `""`.
            * specific_data (str or int, optional): A parameter to define a specific grouping from the UAHdataset used in case the dataset prarameter is not set. Defaults to `None`.
            * target_name (str, optional): The name of the dataset target as a string. Defaults to `"target"`.
            * dataset (pandas.DataFrame or array or numpy.array, optional): A dataset that includes features in columns and a target in one column with a name that match the target provided in the `target_name`. Defaults to `None`.
            * scaler (str, optional): selects the scaling technique used as integers from `"0"` to `"8"` passed as strings, or the name of the scaling technique such as `"minmax"` or `"normalizer"`. Defaults to no scaling with the value `"0"`.
            * droped_columns (list, optional): list of strings with the name of the columns to be removed or droped from the dataset after preprocessing. Defaults to `["Timestamp (seconds)"]`.
            * no_encoding_columns (list, optional): list of strings with the name of columns that will not be included in the label encoding process of cataegorical data. Defaults to `[]`.
            * no_scaling_columns (list, optional): list of strings with the name of columns not to be included in the data scaling. Defaults to `["target"]`.
            * window_size (int, optional): the size of the window in the case of window stepping the data, in case of `0` will not perform the window stepping. Defaults to `0`.
            * step (int, optional): The length of the step for window stepping, if smaller than `window_size` will result in overlapping windows, if equal to `window_size` performs standard window stepping, if bigger will skip some rows (not recommended). Defaults to `0`.
            * window_transformation (bool, optional): in case of True applies the function in `window_transformation_function` parameter to the window. Defaults to `False`.
            * window_transformation_function (function, optional): A function to be applied to the window preferably a lambda function. Defaults to the mean value with: `lambda x:sum(x)/len(x)`.
            * from_csv (bool, optional): Specifies if the data loaded from data_path is a csv file or not. Defaults to `True`.
            * save_dataset (bool, optional): saves in a newly created directory under the working directory in the case of `True`, the preprocessed dataset with an ML specified and DL specified datasets, and windowed data for each case if window stepping is applied. Defaults to `False`.
            * save_tag (str, optional): add a custom tag to the name of the files to be saved in the case of save_dataset is `True`. Defaults to `"dataset"`.
            * verbose (int, optional): An integer of the verbosity of the process can be ``0`` or ``1``. Defaults to ``0``.
        """
        start_time = time.perf_counter()
        if dataset is None or isinstance(dataset, str):
            if from_csv is True:
                if dataset in [
                    "UAHdataset",
                    "uahdataset",
                    "UAHDataset",
                    "UAHDATASET",
                    "uah",
                    "UAH",
                ]:
                    self.raw_dataset = self.uahdataset_loading(
                        data_path, specific=specific_data, verbose=verbose
                    )
                else:
                    try:
                        self.raw_dataset = pd.read_csv(data_path)
                        if verbose == 1:
                            print(
                                "Reading data from provided path to the data csv file"
                            )
                    except Exception:
                        print("\nError reading data, verify the provided path")
        elif dataset is not None:
            self.raw_dataset = dataset
            if verbose == 1:
                print("Reading from the dataset argement the provided dataframe")
        self.filtered_dataset = self.raw_dataset.drop(labels=droped_columns, axis=1)
        self.numeric_dataset = self.filtered_dataset.copy(deep=True)
        for column in self.numeric_dataset.columns:
            if (
                self.numeric_dataset.dtypes[column] != float
                and self.numeric_dataset.dtypes[column] != int
            ):
                if column in no_encoding_columns:
                    if verbose == 1:
                        print(
                            f"skipping \033[1m{column}\033[0m label encoding for being in the no_encoding_columns"
                        )
                else:
                    self.numeric_dataset = self.label_encoding(
                        self.numeric_dataset, target=column, verbose=verbose
                    )
        self.scaled_dataset, self.scaler_name = self.data_scaling(
            self.numeric_dataset,
            excluded_axis=no_scaling_columns,
            scaler=scaler,
            verbose=verbose,
        )
        self.ml_dataset = self.scaled_dataset
        self.features = self.ml_dataset.drop(target_name, axis=1)
        self.target = self.ml_dataset[target_name]
        self.target_ohe = self.one_hot_encoding(
            self.ml_dataset, target=target_name, verbose=verbose
        )
        self.preprocessed_dataset = self.ml_dataset.copy(deep=True)
        for i in self.target_ohe.columns:
            column_name = f"target {i}"
            self.preprocessed_dataset[column_name] = self.target_ohe[i]
        self.dl_dataset = self.preprocessed_dataset
        if window_size > 0:
            if verbose == 1:
                print(
                    "\n\033[1mThe window stepping can take some time depending on the dataset \033[0m"
                )
            self.windowed_dataset = self.ml_dataset.copy(deep=True)
            self.windowed_dataset = self.window_stepping(
                self.windowed_dataset,
                window_size=window_size,
                step=step,
                window_transformation=window_transformation,
                transformation_fn=window_transformation_function,
                verbose=verbose,
            )
            self.ml_dataset_w = self.windowed_dataset
            self.features_w = self.ml_dataset_w.drop(target_name, axis=1)
            if window_transformation == True:
                self.target_w = self.ml_dataset_w[target_name].round()
            else:
                self.target_w = self.ml_dataset_w[target_name]
            self.target_ohe_w = self.one_hot_encoding(
                self.target_w, target=target_name, verbose=verbose
            )
            self.preprocessed_dataset_w = self.ml_dataset_w.copy(deep=True)
            for i in self.target_ohe_w.columns:
                column_name = f"target {i}"
                self.preprocessed_dataset_w[column_name] = self.target_ohe_w[i]
            self.dl_dataset_w = self.preprocessed_dataset_w
        if save_dataset == True:
            PATH = "preprocessed_dataset"
            save_csv(self.ml_dataset, PATH, f"ml_{save_tag}", verbose=verbose)
            save_csv(self.dl_dataset, PATH, f"dl_{save_tag}", verbose=verbose)
            if window_size > 0:
                save_csv(
                    self.ml_dataset_w,
                    PATH,
                    f"ml_{save_tag}_w({window_size})_s({step})",
                    verbose=verbose,
                )
                save_csv(
                    self.dl_dataset_w,
                    PATH,
                    f"dl_{save_tag}_w({window_size})_s({step})",
                    verbose=verbose,
                )
        self.preprocessing_time = f"{(time.perf_counter() - start_time):.2f} (s)"
        exec_time = self.preprocessing_time.replace("(", "").replace(")", "")
        self.preprocessing_info = self.scaler_name + f"({exec_time})"

    @staticmethod
    def uahdataset_loading(path="", specific=None, verbose=1):
        """A static method for loading the uahdatset with various configirations.

        Args:
            * path (str, optional): The path to the dataset, if not provided, the internal dataset will be loaded. Defaults to "".
            * specific_data (str or int, optional): A parameter to define a specific grouping from the UAHdataset, if an integer used, the function will return the dataset of the driver matching that integer, if `""` or `"secondary road"` will return the data in the secondary road only, the same goes for `"0"` and `"motorway road"` returning the motorway road data only, if `None` the whole dataset is returned.the Defaults to `None`.
            * verbose (int, optional): An integer of the verbosity of the process can be ``0`` or ``1``. Defaults to ``1``.

        Returns:
            * pandas.DataFrame: A UAHdataset grouping for time series data
        """
        if path == "":
            DATA_PATH = pkg_resources.resource_filename(
                __name__, "Datasets/UAH_dataset/dataset/UAHDataset.csv"
            )
            print(f"\nloading the internal \033[1mUAHDataset\033[0m from maaml\n")
            data = pd.read_csv(DATA_PATH)
        else:
            try:
                data = pd.read_csv(path)
                if verbose == 1:
                    print("\nUAHDataset read successfully\n")
            except Exception:
                print("\nERROR: bad path entry\nEmpty data variable returned")
                data = []
                return data
        if specific is None:
            data_info = "full data loaded successfully\n"
        elif str(specific) == "secondary road" or str(specific) == "":
            data = data.loc[data["road"] == "secondary"]
            data = data.drop("road", axis=1)
            data_info = "data of secondary road loaded successfully"
        elif str(specific) == "motorway road" or str(specific) == "0":
            data = data.loc[data["road"] == "motorway"]
            data = data.drop("road", axis=1)
            data_info = "data of motorway road loaded successfully"
        elif int(specific) < 7:
            data = data.loc[data["driver"] == int(specific)]
            data = data.drop("driver", axis=1)
            data_info = f"data of driver number {int(specific)} loaded successfully \n"
        else:
            print(
                "ERROR: wrong specific entry or specific entry does not exist\nEmpty data returned "
            )
            data = []
        if verbose == 1:
            print(data_info)
        return data

    @staticmethod
    def label_encoding(data, target, verbose=1):
        """A static method to to convert categorical data column to numeric data via label encoding.

        Args:
            * data (pandas.DataFrame or array or numpy.array): An array of data with a column of categorical data.
            * target ([str]): The name of the column to be converted.
            * verbose (int, optional): An integer of the verbosity of the operation can be ``0`` or ``1``. Defaults to ``1``.

        Returns:
            * pandas.DataFrame: the data with the cateorical data column converted to numeric data.
        """
        encoder = LabelEncoder()
        df = pd.DataFrame(data)
        try:
            if verbose == 1:
                print(
                    f"encoding the \033[1m{target}\033[0m column. The target labels are: {data[target].unique()} "
                )
            df[target] = encoder.fit_transform(data[target])
            if verbose == 1:
                print(f"The target labels after encoding : {df[target].unique()}")
        except Exception:
            print(
                f"ERROR: the column name '{target}' is not available in data\nno label encoding realized for this target\n"
            )
        return data

    @staticmethod
    def data_scaling(data, excluded_axis=[], scaler="minmax", verbose=1):
        """A static method to scale the data using 8 diffrent scaling techniques or returning the raw data with all column values conveted to floats.

        Args:
            * data (pandas.DataFrame): A numeric dataset in a pandas.DataFrame format.
            * excluded_axis (list, optional): A list of column names for the columns to be excluded from the scaling process. Defaults to `[]`.
            * scaler (str, optional): selects the scaling technique used as integers from `"0"` to `"8"` passed as strings, or the name of the scaling technique such as `"RawData (no scaling)"` or `"normalizer"`. Defaults to `"minmax"`.
            * verbose (int, optional): An integer of the verbosity of the operation can be ``0`` or ``1``. Defaults to ``1``.

        Returns:
            * tuple: (pandas.DataFrame,str) A scaled pandas.DataFrame Data and the name of the scaling technique used as string.
        """
        scaled_df = data
        scaled_df = scaled_df.drop(excluded_axis, axis=1)
        columns_names_list = scaled_df.columns
        scaler = str(scaler)
        if scaler == "0" or scaler == "raw_data":
            scaler_name = "RawData (no scaling)"
            scaled_df = pd.DataFrame()
            for column in data.columns:
                scaled_df[column] = data[column].astype("float")
                scaled_df = scaled_df.reset_index(drop=True)
            scaled_df = scaled_df.fillna(0)
            if verbose == 1:
                print(f"data was not scaled, returned: {scaler_name}")
            return scaled_df, scaler_name
        elif scaler == "1" or scaler == "minmax":
            scalerfunction = MinMaxScaler()
            scaler_name = "MinMaxscaler"
        elif scaler == "2" or scaler == "standard":
            scalerfunction = StandardScaler()
            scaler_name = "Standardscaler"
        elif scaler == "3" or scaler == "maxabs":
            scalerfunction = MaxAbsScaler()
            scaler_name = "MaxAbsScaler"
        elif scaler == "4" or scaler == "robust":
            scalerfunction = RobustScaler()
            scaler_name = "RobustScaler"
        elif scaler == "5" or scaler == "quantile_normal":
            scalerfunction = QuantileTransformer(output_distribution="normal")
            scaler_name = "QuantileTransformer using normal distribution"
        elif scaler == "6" or scaler == "quantile_uniform":
            scalerfunction = QuantileTransformer(output_distribution="uniform")
            scaler_name = "QuantileTransformer using uniform distribution"
        elif scaler == "7" or scaler == "power_transform":
            scalerfunction = PowerTransformer(method="yeo-johnson")
            scaler_name = "PowerTransformer using the yeo-johnson method"
        elif scaler == "8" or scaler == "normalizer":
            scalerfunction = Normalizer()
            scaler_name = "Normalizer"
        else:
            print("\nERROR: wrong data entry or wrong scaler type\ninput data returned")
            scaler_name = "Worning : No scaling (something went wrong)"
            return data, scaler_name
        scaled_df = scalerfunction.fit_transform(scaled_df)
        scaled_df = pd.DataFrame(scaled_df, columns=columns_names_list)
        for i in excluded_axis:
            scaled_df[i] = data[i]
        scaled_df = scaled_df.fillna(0)
        if verbose == 1:
            print(f"data scaled with the {scaler_name}")
        return scaled_df, scaler_name

    @staticmethod
    def one_hot_encoding(data, target="target", verbose=1):
        """A static method to convert a single column to a number of columns corresponding to the number of unique values in that columns by One Hot encoding.
        Example:
            >>> df
            index              Timestamp              Speed              driver               road              target
            0                  7                  65.2                  1                  secondary                  normal
            1                  8                  68.5                  1                  secondary                  normal
            2                  9                  73.6                  1                  secondary                  normal
            3                 10                  80.2                  1                  secondary                  agressif
            4                 11                  90.9                  1                  secondary                  agressif
            >>> print(df[target].unique())
            ['normal' 'agressif']
            >>> df_ohe = one_hot_encoding(df,target="target",verbose=0)
            >>> df_ohe
                    0    1
            0      0.0  1.0
            1      0.0  1.0
            2      0.0  1.0
            3      1.0  0.0
            4      1.0  0.0

        Args:
            * data (pandas.DataFrame): A data array in pandas.DataFrame format.
            * target (str, optional): The name of the target column that is going to be one hot encoded. Defaults to `"target"`.
            * verbose (int, optional): An integer of the verbosity of the operation can be ``0`` or ``1``. Defaults to ``1``.

        Returns:
            * pandas.DataFrame: the target column converted to binary format in a pandas.DataFrame with the number of columns corresponds to the unique values of the target column.
        """
        encoder = OneHotEncoder()
        try:
            if verbose == 1:
                print(f"One Hot Encoder target: {data[target].unique()}")
            encoded = encoder.fit_transform(
                data[target].values.reshape(-1, 1)
            ).toarray()
        except Exception:
            try:
                if verbose == 1:
                    print(f"One Hot Encoder target: {data.unique()}")
                encoded = encoder.fit_transform(data.values.reshape(-1, 1)).toarray()
            except Exception:
                if verbose == 1:
                    print(
                        f"ERROR: target name '{target}' is not available in data\nNo One hot encoding realized"
                    )
                return data
        if verbose == 1:
            print(f"example of the target after One Hot encoding : {encoded[0]}")
        df = pd.DataFrame(encoded)
        return df

    @staticmethod
    def window_stepping(
        data=None,
        window_size=0,
        step=0,
        window_transformation=False,
        transformation_fn=lambda x: sum(x) / len(x),
        verbose=1,
    ):
        """A static method for window stepping a time series data.

        Args:
            * data (pandas.DataFrame, optional): A data array in pandas.DataFrame format. Defaults to `None`.
            * window_size (int, optional): the size of the window, in case of `0` will not perform the window stepping. Defaults to `0`.
            * step (int, optional): The length of the step, if smaller than `window_size` will result in overlapping windows, if equal to `window_size` performs standard window stepping, if bigger will skip some rows (not recommended). Defaults to `0`.
            * window_transformation (bool, optional): in case of True applies the function in `window_transformation_function` parameter to the window. Defaults to `False`.
            * window_transformation_function (function, optional): A function to be applied to the window preferably a lambda function. Defaults to the mean value with: `lambda x:sum(x)/len(x)`.
            * verbose (int, optional): An integer of the verbosity of the operation can be ``0`` or ``1``. Defaults to ``1``.

        Returns:
            * pandas.DataFrame: A window stepped data in case the window was bigger than 0 or the entry dataframe in case window_size is equal to 0.
        """
        final_data = pd.DataFrame()
        if len(data) != 0:
            if window_size == 0:
                final_data = data
                if verbose == 1:
                    print("\nATTENTION: Entry data returned without window stepping")
                return final_data
            else:
                if verbose == 1:
                    if window_transformation is True:
                        print("\n\033[1mWindow transformation applied\033[0m")
                    else:
                        print(
                            f"\nwindow stepping applied with window size: {window_size} and step : {step}"
                        )
                for i in range(0, len(data) - 1, step):
                    window_segment = data[i : i + window_size]
                    if window_transformation is True:
                        window_segment = window_segment.apply(transformation_fn, axis=0)
                    final_data = final_data.append(window_segment, ignore_index=True)
        else:
            print("ERROR: Empty data entry")
        return final_data


if __name__ == "__main__":
    preprocessor = DataPreprocessor(
        dataset="UAHdataset",
        no_encoding_columns=[],
        scaler=2,
        window_size=60,
        step=10,
        window_transformation=True,
        window_transformation_function=lambda x: sum(x) / len(x),
        save_dataset=False,
        verbose=1,
    )
    print(f"\nthe raw dataset is: \n{preprocessor.raw_dataset}")
    print(f"\nthe dataset(after dropping columns) is\n{preprocessor.filtered_dataset}")
    print(f"the label encoded dataset: \n{preprocessor.numeric_dataset}")
    print(f"The used scaler is: {preprocessor.scaler_name}")
    print(f"\nthe scaled dataset is: \n{preprocessor.scaled_dataset}")
    print(f"\nthe dataset features are: \n{preprocessor.features}")
    print(f"\nthe dataset target column is: \n{preprocessor.target}")
    print(f"\nthe dataset one hot encoded target is: \n{preprocessor.target_ohe}")
    print(f"\nthe full preprocessed dataset is: \n{preprocessor.preprocessed_dataset}")
    print("\n ******* windowed data ******* \n")
    print(f"\nthe dataset windowed features are: \n{preprocessor.features_w}")
    print(f"\nthe dataset windowed target column is: \n{preprocessor.target_w}")
    print(
        f"\nthe dataset windowed one hot encoded target is: \n{preprocessor.target_ohe_w}"
    )
    print(
        f"\nthe full windowed preprocessed dataset is: \n{preprocessor.preprocessed_dataset_w}"
    )
    print(f"\nthe preprocessing time is : {preprocessor.preprocessing_time}")
    print(f"\npreprocessing info : {preprocessor.preprocessing_info}")
