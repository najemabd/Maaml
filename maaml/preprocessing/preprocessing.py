import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


class DataPreprocessor:
    def __init__(
        self, path="", specific_data=None, target_name="target", dataset=None, verbose=0
    ):
        if dataset is None:
            self.dataset = self.uahdataset_loading(
                path, specific=specific_data, verbose=verbose
            )
        else:
            self.dataset = dataset
        for column in dataset.columns:
            if isinstance(self.dataset[column], str):
                self.numeric_dataset = self.label_encoding(self.dataset, target=column)

    def uahdataset_loading(path="", specific=None, verbose=1):
        if path == "":
            print("\nset path to default:", "\ndataset/dataset.csv\n")
            data = pd.read_csv("dataset/dataset.csv")
        else:
            try:
                data = pd.read_csv(path)
            except:
                print("\nERROR: bad path entry\nEmpty data variable returned")
                data = []
                return data
        if specific is None:
            data_info = "full data loaded"
            return data
        elif str(specific) == "secondary road" or str(specific) == "":
            data = data.loc[data["road"] == "secondary"]
            data = data.drop("road", axis=1)
            data_info = "data of secondary road loaded"
        elif str(specific) == "motorway road" or str(specific) == "0":
            data = data.loc[data["road"] == "motorway"]
            data = data.drop("road", axis=1)
            data_info = "data of motorway road loaded"
        elif int(specific) < 7:
            data = data.loc[data["driver"] == int(specific)]
            data = data.drop("driver", axis=1)
            data_info = f"data from driver number {int(specific)} data loaded\n"
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
        encoder = LabelEncoder()
        df = pd.DataFrame(data)
        try:
            if verbose == 1:
                print("label encoder target: ", data[target].unique())
            df[target] = encoder.fit_transform(data[target])
            if verbose == 1:
                print("target after label encoding : ", df[target].unique())
        except:
            print(
                f"ERROR: target name '{target}' is not available in data\n",
                f"no label encoding realized for '{target}'\n",
            )
            return data


if __name__ == "__main__":
    preprocessor = DataPreprocessor()
