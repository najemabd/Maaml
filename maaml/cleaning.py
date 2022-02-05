import pandas as pd
from maaml.utils import save_csv


class DataReader:
    """A class for reading data from a csv file. includes a `path` attribute and `data` attribute.

    Args:
        * path (str): The data file name in the working directory or the data file path with the file name.
        * header (int, optional):  The specification of the technique used to define the columns names: `None` in case of no columns names in the file, `0` in case that first row is the header. Defaults to `None`.
        * delimiter (str, optional): A string for the type of separation used in the csv file. Defaults to `" "`.
    """

    def __init__(self, path, header=None, delimiter=" "):
        """A constuctor for DataReader class

        Args:
        * path (str): The data file name in the working directory or the data file path with the file name.
        * header (int, optional):  The specification of the technique used to define the columns names: `None` in case of no columns names in the file, `0` in case that first row is the header. Defaults to `None`.
        * delimiter (str, optional): A string for the type of separation used in the csv file. Defaults to `" "`.
        """
        self.path = str(path)
        self.data = pd.read_table(path, header=header, delimiter=delimiter)

    def __call__(self):
        """A method for the class instance call

        Returns:
            * pandas.DataFrame: The read dataset from the file.
        """
        return self.data


class DataCleaner:
    """A class for data cleaning.

    Args:
    * data (pandas.DataFrame or array or numpy.array): A time series dataset.
    * merge_data (pandas.DataFrame or array or numpy.array, optional): A time series dataset to be merged with the data in the `data` parameter. Defaults to `None`.
    * window_size (int, optional): the size of the window in the case of window stepping the data, in case of `0` will not perform the window stepping. Defaults to `0`.
    * step (int, optional): The length of the step for window stepping, if smaller than `window_size` will result in overlapping windows, if equal to `window_size` performs standard window stepping, if bigger will skip some rows (not recommended). Defaults to `0`.
    * drop_duplicates (bool, optional): if `True` removes the duplicate values using the Timestamp column as refrence. Defaults to `True`.
    * window_transformation (bool, optional): in case of True applies the function in `window_transformation_function` parameter to the window. Defaults to `False`.
    * window_transformation_function (function, optional): A function to be applied to the window preferably a lambda function. Defaults to the mean value with: `lambda x:sum(x)/len(x)`.
    * add_columns_dictionnary (dict, optional): A dictionnary of keys (column names) and values to be added to the array or the pandas.DataFrame, if set to `None` will skip adding data. Defaults to `None`.
    * save_dataset (bool, optional): in the case of `True` saves the dataset in a newly created directory under the working directory. Defaults to `False`.
    * save_tag (str, optional): add the name tag of the file to be saved in the case of save_dataset is `True`. Defaults to `"dataset"`.
    * timestamp_column (str, optional): the name of the column that has the timpestamps in seconds of the time series data. Defaults to `"Timestamp (seconds)"`.
    * verbose (int, optional): An integer of the verbosity of the process can be ``0`` or ``1``. Defaults to ``0``.
    """

    def __init__(
        self,
        data,
        merge_data=None,
        window_size=0,
        step=0,
        drop_duplicates=True,
        window_transformation=False,
        window_transformation_function=lambda x: sum(x) / len(x),
        add_columns_dictionnary: dict = None,
        save_dataset=False,
        save_tag="dataset",
        timestamp_column="Timestamp (seconds)",
        verbose=0,
    ):
        """A constructor for DataCleaner class.

        Args:
        * data (pandas.DataFrame or array or numpy.array): A time series dataset.
        * merge_data (pandas.DataFrame or array or numpy.array, optional): A time series dataset to be merged with the data in the `data` parameter. Defaults to `None`.
        * window_size (int, optional): the size of the window in the case of window stepping the data, in case of `0` will not perform the window stepping. Defaults to `0`.
        * step (int, optional): The length of the step for window stepping, if smaller than `window_size` will result in overlapping windows, if equal to `window_size` performs standard window stepping, if bigger will skip some rows (not recommended). Defaults to `0`.
        * drop_duplicates (bool, optional): if `True` removes the duplicate values using the Timestamp column as reference. Defaults to `True`.
        * window_transformation (bool, optional): in case of True applies the function in `window_transformation_function` parameter to the window. Defaults to `False`.
        * window_transformation_function (function, optional): A function to be applied to the window preferably a lambda function. Defaults to the mean value with: `lambda x:sum(x)/len(x)`.
        * add_columns_dictionnary (dict, optional): A dictionnary of keys (column names) and values to be added to the array or the pandas.DataFrame, if set to `None` will skip adding data. Defaults to `None`.
        * save_dataset (bool, optional): in the case of `True` saves the dataset in a newly created directory under the working directory. Defaults to `False`.
        * save_tag (str, optional): add the name tag of the file to be saved in the case of save_dataset is `True`. Defaults to `"dataset"`.
        * timestamp_column (str, optional): the name of the column that has the timpestamps in seconds of the time series data. Defaults to `"Timestamp (seconds)"`.
        * verbose (int, optional): An integer of the verbosity of the process can be ``0`` or ``1``. Defaults to ``0``.
        """
        self.raw_data = data
        if drop_duplicates is True:
            self.filtered_data = data.drop_duplicates(subset=[timestamp_column])
        else:
            self.filtered_data = self.raw_data
            print(f"no duplicates droped, filtered_data is raw_data")
        self.windowed_data = self.window_stepping(
            self.filtered_data,
            window_size=window_size,
            step=step,
            window_transformation=window_transformation,
            transformation_fn=window_transformation_function,
            verbose=verbose,
        )
        if merge_data is not None:
            self.merged_data = self.dataframes_merging(
                self.windowed_data,
                merge_data,
                timestamp_column=timestamp_column,
                drop_duplicates=drop_duplicates,
                verbose=verbose,
            )
        else:
            self.merged_data = self.windowed_data
        self.interpolated_data = self.data_interpolating(
            self.merged_data, timestamp_column=timestamp_column, verbose=verbose
        )
        self.dataset = self.removing_incomplete_raws(
            self.interpolated_data, verbose=verbose
        )
        if add_columns_dictionnary is not None:
            self.dataset = self.column_adding(
                self.dataset,
                add_columns_dictionnary=add_columns_dictionnary,
                verbose=verbose,
            )
        if save_dataset == True:
            PATH = "cleaned_dataset"
            save_csv(self.dataset, PATH, save_tag, verbose=verbose)

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

    @staticmethod
    def dataframes_merging(
        data=None,
        new_data=None,
        timestamp_column="Timestamp (seconds)",
        drop_duplicates=True,
        verbose=1,
    ):
        """A static method for merging two dataframes.

        Args:
            * data (pandas.DataFrame, optional): A time series dataframe. Defaults to `None`.
            * new_data (pandas.DataFrame or array or numpy.array, optional): A time series dataset to be merged with the data in the `data` parameter. Defaults to `None`.
            * timestamp_column (str, optional): The name of the timestamp column to be used as reference for the merge operation. Defaults to `"Timestamp (seconds)"`.
            * drop_duplicates (bool, optional): if `True` removes the duplicate values  in  both dataframes using the Timestamp column as reference before the merge operation. Defaults to `True`.
            * verbose (int, optional): An integer of the verbosity of the operation can be ``0`` or ``1``. Defaults to ``1``.

        Returns:
            * pandas.DataFrame: a pandas.Dataframe with the two orginal dataframes merged using `timestamp_column` as reference.
        """
        try:
            while data.dtypes[timestamp_column] != "int64":
                if verbose == 1:
                    print(
                        "\nWarning: data Timestamp type is: ",
                        data.dtypes[timestamp_column],
                        "\n",
                    )
                data = data.astype({timestamp_column: "int"})
                if verbose == 1:
                    print(
                        "data timestamp type changed to : ",
                        data.dtypes[timestamp_column],
                        "\n",
                    )
            while new_data.dtypes[timestamp_column] != "int64":
                if verbose == 1:
                    print(
                        "Warning: new_data Timestamp type is: ",
                        data.dtypes[timestamp_column],
                        "\n",
                    )
                new_data = new_data.astype({timestamp_column: "int"})
                if verbose == 1:
                    print(
                        "new_data timestamp type changed to : ",
                        new_data.dtypes[timestamp_column],
                        "\n",
                    )
            if drop_duplicates is True:
                data = data.drop_duplicates([timestamp_column])
                new_data = new_data.drop_duplicates([timestamp_column])
            data_merged = data.set_index(timestamp_column).join(
                new_data.set_index(timestamp_column)
            )
            data_merged = data_merged.reset_index()
            if verbose == 1:
                print(f"Shape of the megred data: {data_merged.shape}\n")
                print("\033[1m", "******* DATA SUCCESSFULLY MERGED *******", "\033[0m")
        except Exception:
            print(
                "ERROR: empty data entries or one data entry or both do not have Timestamp column, \nplease renter your two dataframes and check their columns before entry "
            )
            print("\nEmpty data returned")
            data_merged = None
        return data_merged

    @staticmethod
    def data_interpolating(
        data=None, timestamp_column="Timestamp (seconds)", verbose=1
    ):
        """A static method for data interpolation.

        Args:
            * data (pandas.DataFrame, optional): A time series dataframe with missing values. Defaults to `None`.
            * timestamp_columns (str, optional): The name of the timestamp column for the time series data (to ignore this column in the interpolation). Defaults to `"Timestamp (seconds)"`.
            * verbose (int, optional): An integer of the verbosity of the operation can be ``0`` or ``1``. Defaults to ``1``.

        Returns:
            * pandas.DataFrame: A pandas.DataFrame with filled missing values.
        """
        try:
            if verbose == 1:
                print(
                    f"\n    State before interpolation    \nCOLUMNS                   NUMBER OF RAWS WITH MISSING DATA\n{data.isnull().sum()}\n"
                )
            if data.isnull().values.any() == True:
                if verbose == 1:
                    print("\n       Executing interpolation     \n")
                missing_values = data.drop(timestamp_column, axis=1)
                missing_values = missing_values.interpolate(method="cubic", limit=3)
                data[missing_values.columns] = missing_values
                data_interpolated = data
                if verbose == 1:
                    print(
                        f"\n    State after interpolation    \nCOLUMNS                   NUMBER OF RAWS WITH MISSING DATA\n{data_interpolated.isnull().sum()}\n"
                    )
            else:
                data_interpolated = data
                if verbose == 1:
                    print("\n   Interpolation not needed    \n")
        except Exception:
            data_interpolated = None
            print(
                f"\nERROR: empty data entry or non dataframe type\nEmpty data returned"
            )
        return data_interpolated

    @staticmethod
    def removing_incomplete_raws(data=None, verbose=1):
        """A static method for removing all Nan or Na values.

        Args:
            * data (pandas.DataFrame, optional): A dataframe. Defaults to `None`.
            * verbose (int, optional): An integer of the verbosity of the operation can be ``0`` or ``1``. Defaults to ``1``.

        Returns:
            * pandas.DataFrame: a dataframe with no missing or Nan/ Na/ None values.
        """
        try:
            if verbose == 1:
                print(
                    f"\n    Data count before removing any rows :     \n{data.count()}"
                )
                print(
                    "\nis there any missing data values? :",
                    "\033[1m",
                    data.isnull().values.any(),
                    "\033[0m",
                )
            data = data.dropna()
            data = data.reset_index(drop=True)
            if verbose == 1:
                print(f"\n  Final Data count :     \n{data.count()}")
                print(
                    "\nis there any missing data values? :",
                    "\033[1m",
                    data.isnull().values.any(),
                    "\033[0m",
                )
        except Exception:
            print(
                "ERROR: empty data entry or non dataframe type, please enter your data dataframe\nEmpty data returned"
            )
            data = []
        return data

    @staticmethod
    def column_adding(
        data,
        add_columns_dictionnary: dict = None,
        column_name: str = None,
        value: str = None,
        verbose=0,
    ):
        """A static method for adding columns into a pandas.DataFrame.

        Args:
            * data (pandas.DataFrame): A dataframe. Defaults to `None`.
            * add_columns_dictionnary (dict, optional): A dictionnary of keys (column names) and corresponding values to be added, if set to `None` will check `column_name` and `value` if one of them is `None` will skip adding data. Defaults to `None`.
            * column_name (str, optional): the name of the new column. Defaults to `None`.
            * value (Any, optional): The value of the new column. Defaults to `None`.
            * verbose (int, optional): An integer of the verbosity of the operation can be ``0`` or ``1``. Defaults to ``0``.

        Returns:
            [type]: [description]
        """
        if add_columns_dictionnary is None:
            if column_name is not None and value is not None:
                data[column_name] = value
            else:
                data = data
                if verbose == 1:
                    print("\n    No columns added    \n")
        else:
            for key in add_columns_dictionnary:
                try:
                    data[key] = add_columns_dictionnary[key]
                    if verbose == 1:
                        print(f"\nThe '{key}' column was added successfully   \n")
                except ValueError:
                    print(
                        f"\nAbort column adding operation : The '{key}' column length is {len(add_columns_dictionnary[key])}, while the data length {len(data)}\n"
                    )
                    break
        return data


if __name__ == "__main__":
    DATA_DIR_PATH = "/run/media/najem/34b207a8-0f0c-4398-bba2-f31339727706/home/stock/The_stock/dev & datasets/PhD/datasets/UAH-DRIVESET-v1/"
    my_dict = {
        "Timestamp (seconds)": [4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
        "speed": [40, 45, 60, 62, 70, 75, 80, None, 72, 70],
        "loc": [3, 4, 7, 10, None, 15, 17, 20, 24, 27],
        "driver": ["D1", "D1", "D1", "D1", "D2", "D4", None, "D2", "D1", "D5"],
        "target": [
            "normal",
            "normal",
            "normal",
            "agressif",
            "agressif",
            "drowsy",
            "normal",
            "normal",
            "normal",
            "drowsy",
        ],
    }
    data = pd.DataFrame(my_dict)
    cleaning = DataCleaner(
        data,
        drop_duplicates=False,
        add_columns_dictionnary={"axis": [12, 4, 5, 7, 5, 8, 2, 5, 4]},
        save_dataset=True,
        verbose=1,
    )
    print(cleaning.raw_data)
    print(cleaning.dataset)
