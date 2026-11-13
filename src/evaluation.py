import numpy as np


# checks that the actual and predicted values are valid for evaluation
def prepare_values(actual, predicted):
    # converts both inputs into NumPy arrays
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    # makes sure both inputs contain one-dimensional values (1d array)
    if actual.ndim != 1 or predicted.ndim != 1:
        raise ValueError("Actual and predicted values must be one-dimensional.")

    # checks that both arrays have the same number of values
    if len(actual) == 0 or len(actual) != len(predicted):
        raise ValueError("Actual and predicted values must have matching lengths.")

    # prevents invalid numbers from affecting the evaluation results
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Actual and predicted values must be finite.")

    return actual, predicted


# calculates the average absolute difference between actual and predicted budgets
def mean_absolute_error(actual, predicted):
    actual, predicted = prepare_values(actual, predicted)

    # gets the absolute error for each prediction
    errors = np.abs(actual - predicted)

    # returns the average error
    return np.mean(errors)


# calculates the average squared difference between actual and predicted budgets
def mean_squared_error(actual, predicted):
    actual, predicted = prepare_values(actual, predicted)

    # gets the difference between each actual and predicted budget
    errors = actual - predicted

    # squares the errors so larger mistakes have greater penalties
    squared_errors = errors ** 2

    # returns the average squared error
    return np.mean(squared_errors)


# calculates the square root of the average squared prediction error
def root_mean_squared_error(actual, predicted):
    # reuses the MSE function to calculate the squared error
    mse = mean_squared_error(actual, predicted)

    # converts the squared error back into the original budget unit
    return np.sqrt(mse)


# compares model predictions against the average actual budget
def r_squared(actual, predicted):
    actual, predicted = prepare_values(actual, predicted)

    # calculates the total squared prediction error
    prediction_errors = actual - predicted
    residual_sum = np.sum(prediction_errors ** 2)

    # calculates how far actual budgets are from their average
    mean_actual = np.mean(actual)
    total_sum = np.sum((actual - mean_actual) ** 2)

    # R-squared is undefined when all actual values are identical
    if total_sum == 0:
        return float("nan")

    # compares the model's squared error against the mean baseline
    return 1 - (residual_sum / total_sum)
