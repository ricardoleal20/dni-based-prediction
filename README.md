# Multiple Linear Regression, Step-wise and Elastic Net Project

## Project Description

This project implements and compares three different regression methods: multiple linear regression, multiple linear regression with step-wise selection, and elastic net regression. The objective is to generate a dataset based on a provided National Identity Document (DNI), train regression models, and evaluate their performance.

## Motivation

The motivation behind this project is to explore and understand how different regression techniques can be applied to a randomly generated dataset, and how variable selection techniques can improve model performance. The addition of noise and bias to the dataset also allows evaluating the robustness of the models against imperfect data.

## Mathematical Logic

### 1. Multiple Linear Regression

Multiple linear regression models the relationship between a dependent variable \( y \) and multiple independent variables \( X_1, X_2, \ldots, X_p \) using the following formula:

\[
y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \ldots + \beta_p X_p + \epsilon
\]

where:
- \( y \) is the dependent variable.
- \( \beta_0 \) is the intercept.
- \( \beta_1, \beta_2, \ldots, \beta_p \) are the coefficients of the independent variables.
- \( X_1, X_2, \ldots, X_p \) are the independent variables.
- \( \epsilon \) is the error term.

The coefficients are estimated by minimizing the sum of squared residuals (RSS):

\[
RSS = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 = \sum_{i=1}^{n} \left(y_i - \left(\beta_0 + \sum_{j=1}^{p} \beta_j X_{ij}\right)\right)^2
\]

### 2. Multiple Linear Regression with Step-wise Selection

Step-wise selection is an iterative method that combines forward and backward regression to select the subset of independent variables that best explains the variability in the dependent variable.

1. **Forward Selection**: Starts with an empty model and adds variables one by one, selecting the variable that most reduces the RSS or has the lowest p-value at each step.
2. **Backward Elimination**: Starts with a full model and removes variables one by one, removing the variable with the highest p-value at each step.
3. **Step-wise**: Starts with an empty model and in each step, a variable is added or removed based on a predefined criterion (e.g., p-value, AIC, BIC).

### 3. Elastic Net Regression

Elastic net regression is a regularization technique that linearly combines L1 (Lasso) and L2 (Ridge) penalties. It is useful when there are multiple correlated features.

\[
\hat{\beta} = \arg \min_{\beta} \left\{ \sum_{i=1}^{n} (y_i - \beta_0 - \sum_{j=1}^{p} \beta_j X_{ij})^2 + \lambda_1 \sum_{j=1}^{p} |\beta_j| + \lambda_2 \sum_{j=1}^{p} \beta_j^2 \right\}
\]

where:
- \( \lambda_1 \) controls the L1 penalty.
- \( \lambda_2 \) controls the L2 penalty.

## Project Structure

- **main.py**: Main module that handles command-line arguments and executes the corresponding functions.
- **model/implementation.py**: Implementation of the regression methods and auxiliary functions.
- **model/create_dni_number.py**: Generates a random DNI to create a unique dataset.
- **model/dataset.py**: Functions to generate and split the dataset.
- **model/utils.py**: Auxiliary functions to convert DataFrame information to LaTeX and write to files.

## Project Usage

### Command-Line Arguments

- `-a`, `--algorithm`: Algorithm to use (`all`, `lineal`, `elastic`, `step_wise`). Default is `all`.
- `-v`, `--verbose`: Verbose mode (True or False). Default is False.
- `-l`, `--latex`: Export results to LaTeX (True or False). Default is False.
- `-f`, `--function`: Function to execute (`dni` or `models`).
- `-d`, `--dni`: DNI to generate the dataset.

### Usage Examples

To generate a random DNI:

```bash
python main.py -f dni
```

To run the models with a specific DNI:

```bash
python main.py -f models -d 32425632
```

To run a specific model with verbosity and export to LaTeX:

```bash
python main.py -f models -d 32425632 -a lineal -v True -l True
```
