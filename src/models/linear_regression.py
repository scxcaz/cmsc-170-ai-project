import numpy as np


# calculates the predicted budget using the features, weights, and bias
def predict(features, weights, bias):
    prediction = bias

    for i in range(len(features)):
        prediction += features[i] * weights[i]

    return prediction


# calculates the average squared difference between actual and predicted values
def mean_squared_error(actual, predicted):
    errors = predicted - actual
    squared_errors = errors ** 2

    return np.mean(squared_errors)


# calculates predictions for multiple projects at once
def predict_batch(X, weights, bias):
    return np.dot(X, weights) + bias


# initializes the model's weights and bias before training
def initialize_parameters(feature_count):
    weights = np.zeros(feature_count)
    bias = 0.0

    return weights, bias


# converts budget values from PHP to millions for training
def budgets_to_millions(budgets):
    return budgets / 1_000_000


# converts predicted budgets from millions back to PHP
def millions_to_pesos(predictions):
    return predictions * 1_000_000


# ==================== GRADIENT DESCENT ==================== #


# calculates how the bias affects the prediction error
def calculate_bias_gradient(actual, predicted):
    errors = predicted - actual

    return 2 * np.mean(errors)


# calculates how each feature weight affects the prediction error
def calculate_weight_gradients(X, actual, predicted):
    errors = predicted - actual
    gradients = []

    # goes through each feature column to calculate its weight gradient
    for i in range(X.shape[1]):
        # gets the values of the current feature from all training projects
        feature_values = X[:, i]

        # multiplies each error by its feature value, then calculates the gradient
        gradient = 2 * np.mean(errors * feature_values)

        gradients.append(gradient)

    return np.array(gradients)


# updates the weights and bias to reduce prediction error
def update_parameters(weights, bias, weight_gradients, bias_gradient, learning_rate):
    # moves each weight in the opposite direction of its gradient
    weights = weights - learning_rate * weight_gradients

    # adjusts the bias using its own gradient
    bias = bias - learning_rate * bias_gradient

    return weights, bias


# trains linear regression using gradient descent
def train(X, y, learning_rate=0.05, iterations=200):
    # starts with zero weights for each feature and a zero bias
    weights, bias = initialize_parameters(X.shape[1])

    # repeats the learning process for the specified number of iterations
    for i in range(iterations):
        # predicts budgets using the current weights and bias
        predictions = predict_batch(X, weights, bias)

        # calculates how each parameter affects the prediction error
        weight_gradients = calculate_weight_gradients(X, y, predictions)
        bias_gradient = calculate_bias_gradient(y, predictions)

        # adjusts the parameters to reduce the prediction error
        weights, bias = update_parameters(
            weights, bias, weight_gradients, bias_gradient, learning_rate
        )

    return weights, bias



if __name__ == "__main__":
    from src.preprocessing import prepare_data

    # loads the shared training and testing data
    X_train, y_train, X_test, y_test = prepare_data()

    # converts the prepared data into NumPy arrays
    X_train = np.array(X_train, dtype=float)
    X_test = np.array(X_test, dtype=float)
    y_train = np.array(y_train, dtype=float)
    y_test = np.array(y_test, dtype=float)

    # converts training budgets to millions of pesos
    y_train_millions = budgets_to_millions(y_train)

    # trains the model using gradient descent
    learning_rate = 0.05
    iterations = 200

    weights, bias = train(
        X_train, y_train_millions, learning_rate, iterations
    )

    # predicts testing budgets and converts them back to PHP
    predictions_millions = predict_batch(X_test, weights, bias)
    predictions_php = millions_to_pesos(predictions_millions)

    # calculates the average absolute prediction error in PHP
    test_mae = np.mean(np.abs(y_test - predictions_php))

    # compares the model against a constant training-median prediction
    median_budget = np.median(y_train)
    baseline_mae = np.mean(np.abs(y_test - median_budget))

    # displays the training settings and evaluation results
    print("Training projects:", len(X_train))
    print("Testing projects:", len(X_test))
    print("Learning rate:", learning_rate)
    print("Iterations:", iterations)
    print("All predictions finite:", np.isfinite(predictions_php).all())
    print("Linear Regression MAE (PHP):", test_mae)
    print("Median baseline MAE (PHP):", baseline_mae)
