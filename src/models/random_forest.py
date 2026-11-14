
import numpy as np
from src.models.decision_tree import DecisionTreeRegressor


# decision tree that randomly selects features at each split
class RandomFeatureTree(DecisionTreeRegressor):

    def __init__(self, max_depth=8, min_samples_split=20,
                 min_samples_leaf=10, max_features=None,
                 random_state=42):

        super().__init__(max_depth, min_samples_split, min_samples_leaf)

        self.max_features = max_features
        self.rng = np.random.default_rng(random_state)

    def search_split(self, X, y):

        # select random features to check
        if self.max_features is None:
            feature_count = max(1, int(np.sqrt(X.shape[1])))
        else:
            feature_count = min(self.max_features, X.shape[1])

        columns = self.rng.choice(
            X.shape[1],
            size=feature_count,
            replace=False
        )

        column, cut, gain = super().search_split(X[:, columns], y)

        if column is None:
            return None, None, 0.0

        return int(columns[column]), cut, gain


# random forest regression
class RandomForestRegressor:

    def __init__(self, n_trees=5, max_depth=8,
                 min_samples_split=20, min_samples_leaf=10,
                 max_features=None, random_state=42):

        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state

        self.trees = []
        self.rng = np.random.default_rng(random_state)

    def fit(self, X, y):

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()

        if X.ndim != 2 or len(X) == 0:
            raise ValueError("Invalid training data.")

        if len(X) != len(y):
            raise ValueError("X and y must have the same number of rows.")

        if X.shape[1] == 0:
            raise ValueError("X must have at least one feature.")

        if self.n_trees < 1:
            raise ValueError("The forest must have at least one tree.")

        # choose how many feats each split can check
        if self.max_features is None:
            feature_count = max(1, int(np.sqrt(X.shape[1])))
        else:
            feature_count = int(self.max_features)

        if feature_count < 1:
            raise ValueError("max_features must be at least 1.")

        feature_count = min(feature_count, X.shape[1])

        self.trees = []
        self.rng = np.random.default_rng(self.random_state)

        sample_count = len(X)

        for i in range(self.n_trees):

            # create bootstrap sample
            sample_indices = self.rng.integers(
                0, sample_count, size=sample_count
            )

            X_sample = X[sample_indices]
            y_sample = y[sample_indices]

            # give each tree different random seed
            tree_seed = int(self.rng.integers(0, 1000000))

            tree = RandomFeatureTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=feature_count,
                random_state=tree_seed
            )

            tree.fit(X_sample, y_sample)

            self.trees.append(tree)

        return self

    def predict(self, X):

        if len(self.trees) == 0:
            raise ValueError("Train the model before predicting.")

        X = np.asarray(X, dtype=np.float64)

        if X.ndim != 2:
            raise ValueError("X must be a two-dimensional array.")

        predictions = []

        # get predictions
        for tree in self.trees:
            tree_predictions = tree.predict(X)
            predictions.append(tree_predictions)

        # average predictions
        return np.mean(predictions, axis=0)


# run forest
if __name__ == "__main__":

    from src.preprocessing import prepare_data
    from src.evaluation import (
        mean_absolute_error,
        mean_squared_error,
        root_mean_squared_error,
        r_squared
    )

    X_train, y_train, X_test, y_test = prepare_data()

    print(
        f"Training projects: {len(y_train)} | "
        f"Testing projects: {len(y_test)}"
    )

    model = RandomForestRegressor(
        n_trees=5,
        max_depth=8,
        min_samples_split=20,
        min_samples_leaf=10,
        max_features=5,
        random_state=42
    )

    print("\nTraining Random Forest...")
    model.fit(X_train, y_train)

    print("Training complete!")

    predicted = model.predict(X_test)
    actual = np.asarray(y_test, dtype=np.float64).ravel()

    # evaluation metrics using shared functions
    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    rmse = root_mean_squared_error(actual, predicted)
    r2 = r_squared(actual, predicted)

    # display hyperparameters
    print("\nRandom Forest Hyperparameters:")
    print("Number of trees:", model.n_trees)
    print("Maximum depth:", model.max_depth)
    print("Minimum samples to split:", model.min_samples_split)
    print("Minimum samples per leaf:", model.min_samples_leaf)
    print("Features per split:", model.max_features)
    print("Random state:", model.random_state)

    print("\nRandom Forest Results:")
    print(f"MAE:  PHP {mae:,.2f}")
    print(f"MSE:  {mse:,.2f}")
    print(f"RMSE: PHP {rmse:,.2f}")
    print(f"R2:   {r2:.4f}")

    print("\nFirst 10 testing projects:")
    print(f"{'Actual':>18} {'Predicted':>18}")

    for i in range(min(10, len(actual))):
        print(f"{actual[i]:>18,.0f} {predicted[i]:>18,.0f}")
