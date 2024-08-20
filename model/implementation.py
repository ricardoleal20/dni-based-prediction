"""
Implement the methods for prediction of data
"""
import time
from typing import TypeVar, Callable, Any
# SK Learn imports
from sklearn.ensemble import (
    BaggingClassifier, RandomForestClassifier,
    GradientBoostingClassifier
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression, ElasticNetCV
from sklearn.metrics import mean_squared_error
# Extern imports
import statsmodels.api as sm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Local import
from model.dataset import split_data, generate_dataset, TrainAndValidateData
from model.utils import df_describe_to_latex, write_to_file

MODEL = TypeVar("MODEL", LinearRegression, ElasticNetCV)

# ================================= #
#          Public function          #
# ================================= #

def generate_model(
    dni: str,
    model: str = "all",
    verbose: bool = False,
    to_latex: bool = False
) -> None:
    """Generate one or more models. To generate all the models, write model = all

    Args:
        - dni (str): The DNI to generate the random dataset
    """
    data_frame = generate_dataset(dni)
    if verbose is True:
        # Describe the DataFrame
        print(data_frame.info())
        print(32*"-")
        print(data_frame.describe())
        print(32*"-")
        data_frame.hist(figsize=(7, 7), color="black")
        plt.tight_layout()
        plt.savefig("data.pdf", format='pdf')
    if to_latex is True:
        latex_text = df_describe_to_latex(data_frame)
        # Write the file
        write_to_file(latex_text)
    # Split the data
    data = split_data(data_frame, dni)
    if model in ["all", "regression", "lineal", "step_wise"]:
        # Generate the start time
        __calculate_time(
            __generate_lineal_model,
            [data],
            "lineal"
        )
    if model in ["all", "regression", "elastic"]:
        __calculate_time(
            __generate_elastic_model,
            [data, dni],
            "elastic"
        )
    if model in ["all", "regression", "step_wise"]:
        __calculate_time(
            __generate_lineal_model_with_step_wise,
            [data],
            "step_wise"
        )

    if model in ["all", "classification", "decision_tree"]:
        __calculate_time(
            __generate_decision_tree_model,
            [data, dni],
            "decision_tree"
        )
    if model in ["all", "classification", "bagging"]:
        __calculate_time(
            __generate_bagging_model,
            [data, dni],
            "bagging"
        )
    if model in ["all", "classification", "pasting"]:
        __calculate_time(
            __generate_pasting_model,
            [data, dni],
            "pasting"
        )
    if model in ["all", "classification", "random_forest"]:
        __calculate_time(
            __generate_random_forest_model,
            [data, dni],
            "random_forest"
        )
    if model in ["all", "classification", "gbm"]:
        __calculate_time(
            __generate_gbm_model,
            [data, dni],
            "gbm"
        )

# ================================= #
#           Time function           #
# ================================= #
def __calculate_time(
        function: Callable[..., MODEL],
        args: list[Any],
        model: str
    ) -> MODEL:
    """Calculate the execution time of the execution of the models"""
    start_time = time.time()
    # Execute the model
    generated_model = function(*args)
    print(f"The model {model} takes {time.time() - start_time}s\n")
    # Return the model
    return generated_model

# ================================= #
#       Methods implementation      #
# ================================= #

def __generate_lineal_model(data: TrainAndValidateData) -> LinearRegression:
    """Generate the Multiple Regression model using the data
    fitted

    Args:
        - data (TrainAndValidateData): The data to train the model
    """
    # Then, generate the lineal model
    model = LinearRegression()
    model.fit(data["train"]["x"], data["train"]["y"])
    # Validate the model
    __validate_model(model, data)
    # Return the model
    return model

def __generate_elastic_model(data: TrainAndValidateData, dni: str) -> ElasticNetCV:
    """Generate the Elastic model using the data
    fitted

    Args:
        - data (TrainAndValidateData): The data to train the model
    """
    # Generate the elastic model
    model = ElasticNetCV(cv=5, random_state=int(dni))
    # Then, fit the model
    model.fit(data["train"]["x"], data["train"]["y"])
    # Validate the model
    __validate_model(model, data)
    # Return the model
    return model

def __generate_lineal_model_with_step_wise(data: TrainAndValidateData) -> LinearRegression:
    """Generate the Multiple Regression model using the data
    fitted

    Args:
        - data (TrainAndValidateData): The data to train the model
    """
    # Get the significative vars
    vars_data = __step_wise_selection(data)
    new_data: TrainAndValidateData = {
        "train": {
            "x": data["train"]["x"][vars_data], # type: ignore
            "y": data["train"]["y"]
        },
        "validate": {
            "x": data["validate"]["x"][vars_data], # type: ignore
            "y": data["validate"]["y"]
        }
    }
    # With the vars, select the x val and x train
    # Get the model
    model = __generate_lineal_model(new_data)
    # Validate the model
    __validate_model(model, new_data)
    # Return the model
    return model

def __step_wise_selection(
    data: TrainAndValidateData,
    threshold_in: float = 0.01,
    threshold_out: float = 0.05,
) -> list[pd.Series]:
    """Implement a selection due to Step Wise, to generate those
    variables that are needed

    Args:
        data_x (pd.DataFrame): Data for the X value (the variables)
        data_y (pd.DataFrame): Data for the Y value (the objectives)
        initial_list (list): List of initial elements in the Step Wise list
        threshold_in (float, optional): Parameter for the Threshold input. Defaults to 0.01.
        threshold_out (float, optional): Parameter for the Threshold output. Defaults to 0.05.
    """
    # Instance the included vars
    included_vars = []
    # Instance the data x and data y
    data_x, data_y = data["train"]["x"], data["train"]["y"]
    # Then, keep the execution until we find all the vars
    while True:
        changed: bool = False
        # Take a step forward
        excluded_vars = list(set(data_x.columns) - set(included_vars))
        # Generate the new P Value for this
        p_val = pd.Series(index=excluded_vars)
        # Iterate over this excluded vars
        for column in excluded_vars:
            model = sm.OLS(
                data_y,
                sm.add_constant(pd.DataFrame(data_x[included_vars + [column]]))
            ).fit()
            # With the model, obtain the P Values
            p_val[column] = model.pvalues[column]
        # From here, obtain the best P Value
        best_p_val = p_val.min()
        # And evaluate if this is lower than the threshold in
        if best_p_val > threshold_in:
            # Get the new best feature
            included_vars.append(p_val.idxmin())
            # Move the changed flag to true
            changed = True
        # Now, do the backward step
        model = sm.OLS(
            data_y,
            sm.add_constant(pd.DataFrame(data_x[included_vars]))
        ).fit()
        # Get the p values
        p_val = model.pvalues.iloc[1:]
        worst_p_val = p_val.max()
        if worst_p_val > threshold_out:
            included_vars.remove(p_val.idxmax())
            changed = True

        if changed is True:
            break
    # At the very end, return the list of included vars
    return list(included_vars)


def __generate_decision_tree_model(data: TrainAndValidateData, dni: str) -> DecisionTreeClassifier:
    """Generate the Decision Tree model using the data provided.

    Args:
        - data (TrainAndValidateData): The data to train the model
        - dni (str): Used as random state to ensure reproducibility
    """
    # Generate the decision tree model
    model = DecisionTreeClassifier(random_state=int(dni))
    # Fit the model
    model.fit(data["train"]["x"], data["train"]["y"])
    # Get the variable importances
    __get_variables_importances(model, data)
    # Validate the model
    __validate_model(model, data)
    # Return the model
    return model


def __generate_bagging_model(data: TrainAndValidateData, dni: str) -> BaggingClassifier:
    """Generate the Bagging model with replacement using the data provided.

    Args:
        - data (TrainAndValidateData): The data to train the model
        - dni (str): Used as random state to ensure reproducibility
    """
    # Generate the Bagging model with DecisionTreeClassifier as the base estimator
    model = BaggingClassifier(
        estimator=DecisionTreeClassifier(random_state=int(dni)),
        n_estimators=50,
        random_state=int(dni),
        bootstrap=True  # With replacement
    )
    # Fit the model
    model.fit(data["train"]["x"], data["train"]["y"])
    # Get the variable importances
    model.feature_importances_ = np.mean([
        tree.feature_importances_ for tree in model.estimators_
    ], axis=0)
    __get_variables_importances(model, data)
    # Validate the model
    __validate_model(model, data)
    # Return the model
    return model


def __generate_pasting_model(data: TrainAndValidateData, dni: str) -> BaggingClassifier:
    """Generate the Pasting model without replacement using the data provided.

    Args:
        - data (TrainAndValidateData): The data to train the model
        - dni (str): Used as random state to ensure reproducibility
    """
    # Generate the Pasting model with DecisionTreeClassifier as the base estimator
    model = BaggingClassifier(
        estimator=DecisionTreeClassifier(random_state=int(dni)),
        n_estimators=50,
        random_state=int(dni),
        bootstrap=False  # Without replacement
    )
    # Fit the model
    model.fit(data["train"]["x"], data["train"]["y"])
    # Get the variable importances
    model.feature_importances_ = np.mean([
        tree.feature_importances_ for tree in model.estimators_
    ], axis=0)
    __get_variables_importances(model, data)
    # Validate the model
    __validate_model(model, data)
    # Return the model
    return model


def __generate_random_forest_model(data: TrainAndValidateData, dni: str) -> RandomForestClassifier:
    """Generate the Random Forest model using the data provided.

    Args:
        - data (TrainAndValidateData): The data to train the model
        - dni (str): Used as random state to ensure reproducibility
    """
    # Generate the Random Forest model
    model = RandomForestClassifier(
        random_state=int(dni),
        max_leaf_nodes=4  # Set the maximum number of leaf nodes to 4
    )
    # Fit the model
    model.fit(data["train"]["x"], data["train"]["y"])
    # Get the variable importances
    __get_variables_importances(model, data)
    # Validate the model
    __validate_model(model, data)
    # Return the model
    return model


def __generate_gbm_model(data: TrainAndValidateData, dni: str) -> GradientBoostingClassifier:
    """Generate the Gradient Boosting Machine model using the data provided.

    Args:
        - data (TrainAndValidateData): The data to train the model
        - dni (str): Used as random state to ensure reproducibility
    """
    # Generate the Gradient Boosting model
    model = GradientBoostingClassifier(
        random_state=int(dni),
        n_estimators=100,  # Number of boosting stages
        learning_rate=0.1,  # Learning rate
        max_depth=3  # Maximum depth of the individual trees
    )
    # Fit the model
    model.fit(data["train"]["x"], data["train"]["y"])
    # Get the variable importances
    __get_variables_importances(model, data)
    # Validate the model
    __validate_model(model, data)
    # Return the model
    return model

# ================================= #
#           Validation              #
# ================================= #

def __validate_model(
    model: MODEL,
    data: TrainAndValidateData
) -> None:
    """Validate the model"""
    y_predicted = model.predict(data["validate"]["x"])
    # From here, get the MSE for this model
    mse = mean_squared_error(data["validate"]["y"], y_predicted)
    print(f"The MSE is {mse}")


def __get_variables_importances(
    model: MODEL,
    data: TrainAndValidateData
) -> None:
    """Get the variables importances"""
    # Get the imprtances
    importances = model.feature_importances_
    # Print each feature and their importance
    for feature, importance in zip(data["train"]["x"].columns, importances):
        print(f"{feature}: {importance}")
    print(36*"=")
