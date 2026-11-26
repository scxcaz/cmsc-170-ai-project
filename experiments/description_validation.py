import numpy as np

from src.preprocessing import prepare_data
from src.models.linear_regression import (
    train,
    predict_batch,
    budgets_to_millions,
    millions_to_pesos,
)
from src.evaluation import (
    mean_absolute_error,
    mean_squared_error,
    root_mean_squared_error,
    r_squared,
)
from src.models.decision_tree import DecisionTreeRegressor
from src.models.random_forest import RandomForestRegressor


# trains and evaluates linear regression using validation data
def evaluate_linear_regression(data):
    X_train, y_train, X_val, y_val = data

    X_train = np.array(X_train, dtype=float)
    X_val = np.array(X_val, dtype=float)
    y_train = np.array(y_train, dtype=float)
    y_val = np.array(y_val, dtype=float)

    weights, bias = train(
        X_train,
        budgets_to_millions(y_train),
        learning_rate=0.05,
        iterations=200
    )

    predictions = millions_to_pesos(
        predict_batch(X_val, weights, bias)
    )

    print(f"MAE:  PHP {mean_absolute_error(y_val, predictions):,.2f}")
    print(f"MSE:  {mean_squared_error(y_val, predictions):.3e} PHP^2")
    print(f"RMSE: PHP {root_mean_squared_error(y_val, predictions):,.2f}")
    print(f"R2:   {r_squared(y_val, predictions):.4f}")


# trains and evaluates decision tree using validation data
def evaluate_decision_tree(data):
    X_train, y_train, X_val, y_val = data

    model = DecisionTreeRegressor(
        max_depth=16,
        min_samples_split=20,
        min_samples_leaf=10
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_val)

    print(f"MAE:  PHP {mean_absolute_error(y_val, predictions):,.2f}")
    print(f"MSE:  {mean_squared_error(y_val, predictions):.3e} PHP^2")
    print(f"RMSE: PHP {root_mean_squared_error(y_val, predictions):,.2f}")
    print(f"R2:   {r_squared(y_val, predictions):.4f}")


# trains and evaluates random forest using validation data
def evaluate_random_forest(data, max_features=5):
    X_train, y_train, X_val, y_val = data

    model = RandomForestRegressor(
        n_trees=5,
        max_depth=8,
        min_samples_split=20,
        min_samples_leaf=10,
        max_features=max_features,
        random_state=42
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_val)

    print(f"MAE:  PHP {mean_absolute_error(y_val, predictions):,.2f}")
    print(f"MSE:  {mean_squared_error(y_val, predictions):.3e} PHP^2")
    print(f"RMSE: PHP {root_mean_squared_error(y_val, predictions):,.2f}")
    print(f"R2:   {r_squared(y_val, predictions):.4f}")


def main():
    # prepare both feature sets using the same validation split
    original = prepare_data(use_validation=True)
    expanded = prepare_data(
        include_description=True,
        use_validation=True
    )

    # verify that the actual budgets and original features match
    assert original[1] == expanded[1]
    assert original[3] == expanded[3]

    for i in range(len(original[0])):
        assert original[0][i] == expanded[0][i][:25]

    for i in range(len(original[2])):
        assert original[2][i] == expanded[2][i][:25]

    print("Validation datasets match: PASSED")

    print("\nLINEAR REGRESSION - ORIGINAL 25 FEATURES")
    evaluate_linear_regression(original)

    print("\nLINEAR REGRESSION - WITH 10 DESCRIPTION FEATURES")
    evaluate_linear_regression(expanded)
    
    print("\nDECISION TREE - ORIGINAL 25 FEATURES")
    evaluate_decision_tree(original)

    print("\nDECISION TREE - WITH 10 DESCRIPTION FEATURES")
    evaluate_decision_tree(expanded)
    
    print("\nRANDOM FOREST - ORIGINAL 25 FEATURES")
    evaluate_random_forest(original)

    print("\nRANDOM FOREST - WITH 10 DESCRIPTION FEATURES")
    evaluate_random_forest(expanded)


if __name__ == "__main__":
    main()
