import pandas as pd
from maaml.utils import save_csv, FileScraper, pattern_search, read_csv


class UahDatasetPathFinder:
    """A class for generating a path for a specific file in the UahDataset. includes an attribute `file_path` and a method `numeric_to_string_from_list` for converting a nemeric value to the a corresponding indexed string from a list and a `__call__ ` method for calling an instance of the class to return the `file_path` attribute.

        Args:
        * dataset_dir (str): The path to the parent directory of the UahDataset.
        * driver (str or int): The number of the driver, can be an integer from `1` to `6` or a string that ends with integer in the same range such as `"driver1"` or `"D6"`.
        * state (str or int): The state of the driving session, can be an integer from `1` to `3` or a string such as `"normal"`,`"aggressive"` or `"drowsy"` not case sensitive.
        * road (str or int): The type of road in the driving session, can be an integer from `1` to `2` or a string such as `"motorway"` or `"secondary"` not case sensitive.
        * data_type (str or int): The type of data or the file of data chosen, can be an integer from 1 to 9 or a string such as `"gps"` or `"accelerometer"` not case sensitive.
        * standard (bool, optional): if `True` gives you the first session and if `False` gives the second session, of course for both cases it depends on the availability of the data. Defaults to `True`.
        * verbose (int, optional): An integer of the verbosity of the operation can be ``0`` or ``1``. Defaults to ``0``.

    Raises:
        * ValueError: In the case of wrong argement type or entry, with a description for the reason.
    """

    SENSOR_FILES = [
        "RAW_GPS.txt",
        "RAW_ACCELEROMETERS.txt",
        "PROC_LANE_DETECTION.txt",
        "PROC_VEHICLE_DETECTION.txt",
        "PROC_OPENSTREETMAP_DATA.txt",
        "EVENTS_LIST_LANE_CHANGES.txt",
        "EVENTS_INERTIAL.txt",
        "SEMANTIC_FINAL.txt",
        "SEMANTIC_ONLINE.txt",
    ]
    STATES = ["NORMAL", "AGGRESSIVE", "DROWSY"]
    ROADS = ["MOTORWAY", "SECONDARY"]

    def __init__(
        self, dataset_dir, driver, state, road, data_type, standard=True, verbose=0
    ):
        """A constructor for the UahDatasetPathFinder class.

            Args:
            * dataset_dir (str): The path to the parent directory of the UahDataset.
            * driver (str or int): The number of the driver, can be an integer from `1` to `6` or a string that ends with integer in the same range such as `"driver1"` or `"D6"`.
            * state (str or int): The state of the driving session, can be an integer from `1` to `3` or a string such as `"normal"`,`"aggressive"` or `"drowsy"` not case sensitive.
            * road (str or int): The type of road in the driving session, can be an integer from `1` to `2` or a string such as `"motorway"` or `"secondary"` not case sensitive.
            * data_type (str or int): The type of data or the file of data chosen, can be an integer from 1 to 9 or a string such as `"gps"` or `"accelerometer"` not case sensitive.
            * standard (bool, optional): if `True` gives you the first session and if `False` gives the second session, of course for both cases it depends on the availability of the data. Defaults to `True`.
            * verbose (int, optional): An integer of the verbosity of the operation can be ``0`` or ``1``. Defaults to ``0``.

        Raises:
            * ValueError: In the case of wrong argement type or entry, with a description for the reason.
        """
        file_error_msg = (
            "please give a correct data_type from the files available for the dataset"
        )
        numeric_file_error_msg = (
            "please give the correct number of the file, only 9 file available"
        )
        data_type = self.numeric_to_string_from_list(
            data_type, self.SENSOR_FILES, numeric_file_error_msg
        ).replace(".txt", "")
        file_name = pattern_search(data_type.upper(), self.SENSOR_FILES, file_error_msg)
        if len(file_name) > 1:
            raise ValueError(
                f"There is more than one data_type with that entry:\n{file_name}\nplease be more specific."
            )
        else:
            for file in file_name:
                self.file = file
                print(
                    f"Searching '{file}' file in all directories and sub_directories.. "
                )
            path_list = set(FileScraper(dataset_dir, file_name).path_list)
            driver = (
                driver
                if isinstance(driver, int) or driver.isnumeric()
                else int(driver[-1])
            )
            driver_name = f"D{driver}"
            driver_error_msg = (
                "please give a correct driver number, only 6 drivers available."
            )
            driver_paths = pattern_search(driver_name, path_list, driver_error_msg)
            if verbose == 1:
                print(
                    f"\033[1mdriver {driver_name} has {len(driver_paths)} diffrent paths\033[0m"
                )
            numeric_state_error_msg = (
                "please give the correct driving state number, only 3 states available."
            )
            state = self.numeric_to_string_from_list(
                state, self.STATES, error_msg=numeric_state_error_msg
            )
            state_error_msg = f"\nplease give driving state number or give the correct driving state from:\n{self.STATES}"
            state_paths = pattern_search(
                state.upper(), self.STATES, state_error_msg, global_set=driver_paths
            )
            numeric_road_error_msg = (
                "please give the correct road number, only 2 roads available."
            )
            road = self.numeric_to_string_from_list(
                road, self.ROADS, numeric_road_error_msg
            )
            road_error_msg = f"\nplease give road number or give the correct road name from:\n{self.ROADS}"
            road_paths = pattern_search(
                road.upper(), self.ROADS, road_error_msg, global_set=state_paths
            )
            road_paths = list(road_paths)
            if len(road_paths) > 1 and verbose == 1:
                print(
                    f"\nThere is more than one path for this case,the selection will depend on the standard parameter"
                )
                for path in road_paths:
                    print("* path:", path)
            if standard:
                try:
                    self.file_path = road_paths[0]
                except IndexError:
                    raise ValueError(
                        "driver 6 does not have data for state: AGGRESSIVE and road: SECONDARY"
                    )
            elif not standard:
                try:
                    self.file_path = road_paths[1]
                except IndexError:
                    raise ValueError("Only standard file available for this case.")
            if verbose == 1:
                print(f"The file path is:\n\033[1m'{self.file_path}'\033[0m")

    def __call__(self):
        """A method for the class instance call

        Returns:
            * str: The file path.
        """
        return self.file_path

    def numeric_to_string_from_list(self, element, element_list, error_msg):
        """A method for checking if an element is an integer or numeric to return a string from a list matching it's index.

        Args:
            * element (int or str): An element to be checked if it is numeric.
            * element_list (set or list): A list to have matching strings for the numeric element.
            * error_msg (str): An error message to be displayed when a ValueError is raised.

        Raises:
            * ValueError: if the numeric element is not in the range of the element_list.

        Returns:
            * str: The string matching the index of the numeric elemnt or the entry element if the element is not a numeric value.
        """
        if isinstance(element, int) or element.isnumeric():
            if 0 < int(element) < len(element_list) + 1:
                element = element_list[int(element) - 1]
                return element
            else:
                raise ValueError(error_msg)
        return element


class UahDatasetloader(UahDatasetPathFinder):
    files_column_names = {
        "RAW_GPS.txt": [
            "Timestamp (seconds)",
            "Speed (km/h)",
            "Latitude coordinate (degrees)",
            "Longitude coordinate (degrees)",
            "Altitude (meters)",
            "Vertical accuracy (degrees)",
            "Horizontal accuracy (degrees)",
            "Course (degrees)",
            "Difcourse: course variation (degrees)",
        ],
        "RAW_ACCELEROMETERS.txt": [
            "Timestamp (seconds)",
            "Boolean of system activated (1 if >50km/h)",
            "Acceleration in X (Gs)",
            "Acceleration in Y (Gs)",
            "Acceleration in Z (Gs)",
            "Acceleration in X filtered by KF (Gs)",
            "Acceleration in Y filtered by KF (Gs)",
            "Acceleration in Z filtered by KF (Gs)",
            "Roll (degrees)",
            "Pitch (degrees)",
            "Yaw (degrees)",
        ],
        "PROC_LANE_DETECTION.txt": [
            "Timestamp (seconds)",
            "X: car position relative to lane center (meters)",
            "Phi: car angle relative to lane curvature (degrees)",
            "W: road width (meters)",
            "State of the lane det. algorithm [-1=calibrating,0=initializing, 1=undetected, 2=detected/running]",
        ],
        "PROC_VEHICLE_DETECTION.txt": [
            "Timestamp (seconds)",
            "Distance to ahead vehicle in current lane (meters) [value -1 means no car is detected in front]",
            "Time of impact to ahead vehicle (seconds) [distance related to own speed]",
            "Number of detected vehicles in this frame (traffic)",
            "GPS speed (km/h) [same as in RAW GPS]",
        ],
        "PROC_OPENSTREETMAP_DATA.txt": [
            "Timestamp (seconds)",
            "Maximum allowed speed of current road (km/h)",
            "Reliability of obtained maxspeed (0=unknown,1=reliable, 2=used previously obtained maxspeed,3=estimated by type of road)",
            "Type of road (motorway, trunk, secondary...)",
            "Number of lanes in current road",
            "Estimated current lane (1=right lane, 2=first left lane, 3=second left lane, etc) [experimental]",
            "GPS Latitude used to query OSM (degrees)",
            "GPS Longitude used to query OSM (degrees)",
            "OSM delay to answer query (seconds)",
            "GPS speed (km/h) [same as in RAW GPS]",
        ],
        "EVENTS_LIST_LANE_CHANGES.txt": [
            "Timestamp (seconds)",
            "Type [+ indicates right and - left, 1 indicates normal lane change and 2 slow lane change]",
            "GPS Latitude of the event (degrees)",
            "GPS Longitude of the event (degrees)",
            "Duration of the lane change (seconds) [measured since the car position is near the lane marks]",
            "Time threshold to consider irregular change (secs.) [slow if change duration is over this threshold and fast if duration is lower than threshold/3]",
        ],
        "EVENTS_INERTIAL.txt": [
            "Timestamp (seconds)",
            "Type (1=braking, 2=turning, 3=acceleration)",
            "Level (1=low, 2=medium, 3=high)",
            "GPS Latitude of the event",
            "GPS Longitude of the event ",
            "Date of the event in YYYYMMDDhhmmss format",
        ],
        "SEMANTIC_FINAL.txt": [
            "Hour of route start",
            "Minute of route start",
            "Second of route start",
            "Average speed during trip (km/h)",
            "Maximum achieved speed during route (km/h)",
            "Lanex score (internal value, related to lane drifting)",
            "Driving time (in minutes)",
            "Hour of route end",
            "Minute of route end",
            "Second of route end",
            "Trip distance (km)",
            "ScoreLongDist	(internal value, Score accelerations)",
            "ScoreTranDist	(internal value, Score turnings)",
            "ScoreSpeedDist (internal value, Score brakings)",
            "ScoreGlobal (internal value, old score that is not used anymore)",
            "Alerts Long (internal value)",
            "Alerts Late (internal value)",
            "Alerts Lanex (internal value)",
            "Number of vehicle stops during route (experimental, related to fuel efficiency estimation)",
            "Speed variability (experimental, related to fuel efficiency estimation)",
            "Acceleration noise (experimental, related to fuel efficiency estimation)",
            "Kinetic energy (experimental, related to fuel efficiency estimation)",
            "Driving time (in seconds)",
            "Number of curves in the route",
            "Power exherted (experimental, related to fuel efficiency estimation)",
            "Acceleration events (internal value)",
            "Braking events (internal value)",
            "Turning events (internal value)",
            "Longitudinal-distraction Global Score (internal value, combines mean[31] and std[32])",
            "Transversal-distraction Global Score (internal value, combines mean[33] and std[34])",
            "Mean Long.-dist. score (internal value)",
            "STD Long.-dist. score (internal value)",
            "Average Trans.-dist. score (internal value)",
            "STD Trans.-dist. score (internal value)",
            "Lacc (number of low accelerations)",
            "Macc (number of medium accelerations)",
            "Hacc (number of high accelerations)",
            "Lbra (number of low brakings)",
            "Mbra (number of medium brakings)",
            "Hbra (number of high brakings)",
            "Ltur (number of low turnings)",
            "Mtur (number of medium turnings)",
            "Htur (number of high turnings)",
            "Score total (base 100, direct mean of the other 7 scores [45-51])",
            "Score accelerations (base 100)",
            "Score brakings (base 100)",
            "Score turnings (base 100)",
            "Score lane-weaving (base 100)",
            "Score lane-drifting (base 100)",
            "Score overspeeding (base 100)",
            "Score car-following (base 100)",
            "Ratio normal (base 1)",
            "Ratio drowsy (base 1)",
            "Ratio aggressive (base 1)",
        ],
        "SEMANTIC_ONLINE.txt": [
            "TimeStamp since route start (seconds)",
            "GPS Latitude (degrees)",
            "GPS Longitude (degrees)",
            "Score total WINDOW (base 100, direct mean of the other 7 scores)",
            "Score accelerations WINDOW (base 100)",
            "Score brakings WINDOW (base 100)",
            "Score turnings WINDOW (base 100)",
            "Score weaving WINDOW (base 100)",
            "Score drifting WINDOW (base 100)",
            "Score overspeeding WINDOW (base 100)",
            "Score car-following WINDOW (base 100)",
            "Ratio normal WINDOW (base 1)",
            "Ratio drowsy WINDOW (base 1)",
            "Ratio aggressive WINDOW (base 1)",
            "Ratio distracted WINDOW (1=distraction detected in last 2 seconds, 0=otherwise)",
            "Score total (base 100, direct mean of the other 7 scores)",
            "Score accelerations (base 100)",
            "Score brakings (base 100)",
            "Score turnings (base 100)",
            "Score weaving (base 100)",
            "Score drifting (base 100)",
            "Score overspeeding (base 100)",
            "Score car-following (base 100)",
            "Ratio normal (base 1)",
            "Ratio drowsy (base 1)",
            "Ratio aggressive (base 1)",
            "Ratio distracted (1=distraction detected in last 2 seconds, 0=otherwise)",
        ],
    }

    def __init__(
        self, parent_path, driver_id, state_id, road_id, data_type_id, standard=True
    ):
        super().__init__(
            parent_path, driver_id, state_id, road_id, data_type_id, standard, verbose=0
        )
        if self.file == "SEMANTIC_FINAL.txt":
            delimiter = None
        else:
            delimiter = " "
        self.data = read_csv(self.file_path, delimiter=delimiter, header=None)
        if self.file == self.SENSOR_FILES[0]:
            self.data = self.data.drop([9, 10, 11, 12], axis=1)
        elif self.file == self.SENSOR_FILES[1]:
            try:
                self.data = self.data.drop(11, axis=1)
            except Exception:
                self.data = self.data
        elif self.file == self.SENSOR_FILES[3]:
            self.data = self.data.drop(5, axis=1)
        elif self.file == self.SENSOR_FILES[4]:
            self.data = self.data.drop(10, axis=1)
        elif self.file == self.SENSOR_FILES[6]:
            self.data = self.data.drop(6, axis=1)
        elif self.file == self.SENSOR_FILES[7]:
            self.data = self.data.transpose()
        elif self.file == self.SENSOR_FILES[8]:
            self.data = self.data.drop(27, axis=1)
        self.data.columns = self.files_column_names[self.file]

    def __call__(self):
        return self.data


class DataCleaner:
    def __init__(
        self,
        data,
        new_data=None,
        average_window=True,
        window_size=0,
        step=0,
        uah_dataset_vector: list = None,
        save_dataset=False,
        name_dataset="dataset",
        timestamp_column="Timestamp (seconds)",
        verbose=0,
    ):
        self.data_raw = data
        self.data_filtered = data.drop_duplicates(subset=[timestamp_column])
        self.average_window = average_window
        self.window_size = window_size
        self.step = step
        self.data_windowed = self.window_stepping(
            self.data_filtered,
            average_window=average_window,
            window_size=window_size,
            step=step,
            verbose=verbose,
        )
        if new_data is not None:
            self.data_merged = self.dataframes_merging(
                self.data_windowed,
                new_data,
                timestamp_column=timestamp_column,
                drop_duplicates=average_window,
                verbose=verbose,
            )
        else:
            self.data_merged = self.data_windowed
        self.data_interpolated = self.data_interpolating(
            self.data_merged, timestamp_columns=timestamp_column, verbose=verbose
        )
        self.dataset = self.removing_incomplete_raws(
            self.data_interpolated, verbose=verbose
        )
        if uah_dataset_vector is not None:
            self.dataset = self.column_adding(
                self.dataset,
                uah_dataset_vector=uah_dataset_vector,
                verbose=verbose,
            )
        if save_dataset == True:
            PATH = "dataset"
            save_csv(self.dataset, PATH, name_dataset, verbose=verbose)

    @staticmethod
    def window_stepping(data=[], window_size=0, step=0, average_window=True, verbose=1):
        segment = []
        final_data = pd.DataFrame()
        if len(data) != 0:
            if window_size == 0:
                final_data = data
                if verbose == 1:
                    print("\nATTENTION: Entry data returned without window stepping")
                return final_data
            else:
                if average_window is True:
                    if verbose == 1:
                        print("\nAverage window applied")
                    for i in range(0, len(data) - 1, step):
                        segment = data[i : i + window_size]
                        row = segment.mean()
                        final_data = final_data.append(row, ignore_index=True)
                else:
                    for i in range(0, len(data) - 1, step):
                        window = data[i : i + window_size]
                        final_data = final_data.append(window, ignore_index=True)
                    if verbose == 1:
                        print(
                            f"\nwindow stepping applied with window size: {window_size} and step : {step}"
                        )
        else:
            final_data = []
            print("ERROR: Empty data entry")
        return final_data

    @staticmethod
    def dataframes_merging(
        data=[],
        new_data=[],
        timestamp_column="Timestamp (seconds)",
        drop_duplicates=True,
        verbose=1,
    ):
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
            data_merged = []
        return data_merged

    @staticmethod
    def data_interpolating(
        data=[], timestamp_columns=["Timestamp (seconds)"], verbose=1
    ):
        try:
            if verbose == 1:
                print(
                    f"\n    State before interpolation    \nCOLUMNS                   NUMBER OF RAWS WITH MISSING DATA\n{data.isnull().sum()}\n"
                )
            if data.isnull().values.any() == True:
                if verbose == 1:
                    print("\n       Executing interpolation     \n")
                missing_values = data.drop(timestamp_columns, axis=1)
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
            data_interpolated = []
            print(
                f"{data_interpolated}\nERROR: empty data entry or non dataframe type\nEmpty data returned"
            )
        return data_interpolated

    @staticmethod
    def removing_incomplete_raws(data=[], verbose=1):
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
        column_name: str = None,
        value: str = None,
        uah_dataset_vector: list = None,
        verbose=0,
    ):
        if uah_dataset_vector is None:
            if column_name is not None and value is not None:
                data[column_name] = value
            else:
                data = data
                if verbose == 1:
                    print("\n    No label columns added    \n")
        else:
            if uah_dataset_vector[0] == 1:
                data["driver"] = "1"
            elif uah_dataset_vector[0] == 2:
                data["driver"] = "2"
            elif uah_dataset_vector[0] == 3:
                data["driver"] = "3"
            elif uah_dataset_vector[0] == 4:
                data["driver"] = "4"
            elif uah_dataset_vector[0] == 5:
                data["driver"] = "5"
            elif uah_dataset_vector[0] == 6:
                data["driver"] = "6"
            if uah_dataset_vector[2] == 1:
                data["road"] = "secondary"
            elif uah_dataset_vector[2] == 2:
                data["road"] = "motorway"
            if uah_dataset_vector[1] == 1:
                data["target"] = "normal"
            elif uah_dataset_vector[1] == 2:
                data["target"] = "agressif"
            elif uah_dataset_vector[1] == 3:
                data["target"] = "drowsy"
            if verbose == 1:
                print("\n   labels columns added successfully   \n")
        return data


class UAHDatasetBuilder:
    def __init__(
        self,
        path,
        datatype1,
        datatype2,
        window_size_dt1=0,
        step_dt1=0,
        window_size_dt2=0,
        step_dt2=0,
        save_dataset=False,
        name_dataset="UAHDataset",
        verbose=0,
        verbose1=0,
        verbose2=0,
    ):
        self.path_list = []
        self.path_list2 = []
        self.data = pd.DataFrame()
        count = 0
        merge_count = 0
        for i in (1, 2, 3, 4, 5, 6):
            for j in (1, 2, 3):
                for k in (1, 2):
                    for l in (1, 2):
                        # file1 = PathBuilder(path, conditions_vector=[i, j, k, l], datatype=datatype1)
                        if l == 1:
                            standard = True
                        elif l == 2:
                            standard = False
                        try:
                            raw_data1 = UahDatasetloader(
                                path, i, j, k, datatype1, standard=standard
                            )
                            raw_data2 = UahDatasetloader(
                                path, i, j, k, datatype2, standard=standard
                            )
                            count += 1
                            print("raw data scaning files number ", count)
                        except ValueError:
                            print("no file for this case..continue")
                        self.data_chunk1 = DataCleaner(
                            raw_data1.data,
                            window_size=window_size_dt1,
                            step=step_dt1,
                            verbose=verbose1,
                        ).dataset
                        self.data_chunk2 = DataCleaner(
                            raw_data2.data,
                            window_size=window_size_dt2,
                            step=step_dt2,
                            verbose=verbose2,
                        ).dataset
                        print("****merge Data****\n")
                        self.data_chunk_merged = DataCleaner(
                            data=self.data_chunk1,
                            new_data=self.data_chunk2,
                            uah_dataset_vector=[i, j, k, l],
                            verbose=verbose,
                        ).dataset
                        merge_count += 1
                        print(
                            "merging data successfully in the attempt number ",
                            merge_count,
                        )
                        self.data = self.data.append(self.data_chunk_merged)
                        self.data = self.data.reset_index()
                        self.data = self.data.drop("index", axis=1)
                        try:
                            self.path_list.append(raw_data1.file_path)
                            self.path_list2.append(raw_data1.file_path)
                        except ValueError:
                            print("no file path for this case..continue")
        print("final scaning count", count)
        print("final merging attempts count", merge_count)

        if save_dataset == True:
            PATH = "dataset"
            save_csv(self.data, PATH, name_dataset, verbose=verbose)


if __name__ == "__main__":
    DATA_DIR_PATH = "/run/media/najem/34b207a8-0f0c-4398-bba2-f31339727706/home/stock/The_stock/dev & datasets/PhD/datasets/UAH-DRIVESET-v1/"
    dataset = UAHDatasetBuilder(
        DATA_DIR_PATH,
        "gps",
        "acc",
        window_size_dt2=10,
        step_dt2=10,
        verbose=1,
        save_dataset=False,
    )
    print(f"The dataset shape: {dataset.data.shape}")

# %%
