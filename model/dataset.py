""" 
Method to create the random dataset
"""
from typing import TypedDict
# External imports
from numpy import ndarray
import pandas as pd
# SK Learn imports
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# ================================= #
#               DATA                #
# ================================= #

class Data(TypedDict):
    """Train data expected"""
    x: pd.DataFrame
    y: pd.Series

class TrainAndValidateData(TypedDict):
    """Splitted data to train and validate the model"""
    train: Data
    validate: Data

class DataVar(TypedDict):
    """Params needed for the SK Learn classification"""
    n_samples: int
    n_features: int
    n_informative: int
    n_redundant: int
    n_repeated: int
    n_classes: int
    flip_y: float
    random_state: int
    shuffle: bool

# ================================= #
#            FUNCTIONS              #
# ================================= #

def __generate_parameters(dni: str) -> DataVar:
    """Generate the parameters based on the DNI"""
    # Get the first 4 elements of the DNI
    dni_1, dni_2 = int(dni[0]), int(dni[1])
    dni_3, dni_4 = int(dni[2]), int(dni[3])
    return {
        "n_samples": 200 + 10 * dni_1,
        "n_features":  10 + dni_2 + dni_3,
        "n_informative": 10 + dni_2,
        "n_redundant": 0,
        "n_repeated": 0,
        "n_classes": 2,
        "flip_y": (10 * dni_4) / 100,
        "random_state": int(dni),
        "shuffle": False
    }

def __generate_dataframe(x: ndarray, y: ndarray, dni_params: DataVar) -> pd.DataFrame:
    """From the dataset created from SK Learn, convert it to a dataframe"""
    # Create the DataFrame
    data_frame = pd.DataFrame(
        x,
        columns=[f"feature_{i+1}" for i in range(dni_params["n_features"])]
    )
    # Add the target
    data_frame["target"] = y
    return data_frame

def generate_dataset(dni: str) -> pd.DataFrame:
    """From the given DNI, create a random Dataset
    using the make classification of SkLearn

    Args:
        - dni (str): The DNI to generate unique parameters

    Returns:
        - DataFrame with the data
    """
    # DNI parameters
    dni_params = __generate_parameters(dni)
    # Get the X and Y data from the DNI parameters
    x, y = make_classification(**dni_params)
    return __generate_dataframe(x, y, dni_params)

def split_data(data: pd.DataFrame, dni: str) -> TrainAndValidateData:
    """From a given dataframe, split the values and get:

    - X data to train
    - X to validate
    - Y data to train
    - Y data to validate

    Args:
        - Dataframe with the data

    Returns:
        - Group of data
    """
    # Drop the target
    data_to_use = data.drop(columns=["target"])
    # Split the data
    x_train, x_val, y_train, y_val = train_test_split(
        data_to_use, # X
        data["target"], # y,
        test_size=30,
        random_state=int(dni)
    )
    # REturn the dictionary with the data splitted
    return {
        "train":{
            "x": x_train,
            "y": y_train
        },
        "validate":{
            "x": x_val,
            "y": y_val
        }
    }
