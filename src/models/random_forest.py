
import numpy as np

# temporarily copied decision tree
# will import decision_tree.py after reviews

# node for each tree
class Node:

    def __init__(self, prediction, size):
        self.prediction = prediction
        self.size = size
        self.feature = None
        self.cut = None
        self.left_child = None
        self.right_child = None
        self.gain = 0.0

    def is_leaf(self):
        return self.feature is None


# decision tree used by the forest
class _DecisionTreeRegressor:

    def __init__(self, max_depth=8, min_samples_split=20,
                 min_samples_leaf=10, max_features=None,
                 random_state=42):

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.rng = np.random.default_rng(random_state)
        self.root = None

    def fit(self, X, y):

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()

        if len(X) == 0 or len(X) != len(y):
            raise ValueError("Invalid training data.")

        self.root = self.grow(X, y, 0)
        return self

    def grow(self, X, y, depth):

        # use the average budget as the prediction
        node = Node(float(y.mean()), len(y))

        too_deep = depth >= self.max_depth
        too_small = len(y) < max(
            self.min_samples_split,
            2 * self.min_samples_leaf
        )
        all_equal = y.min() == y.max()

        if too_deep or too_small or all_equal:
            return node

        column, cut, gain = self.search_split(X, y)

        if column is None or gain <= 0:
            return node

        goes_left = X[:, column] <= cut

        node.feature = column
        node.cut = cut
        node.gain = gain

        node.left_child = self.grow(
            X[goes_left], y[goes_left], depth + 1
        )

        node.right_child = self.grow(
            X[~goes_left], y[~goes_left], depth + 1
        )

        return node

    def search_split(self, X, y):

        n = len(y)
        centred = y - y.mean()
        smallest = self.min_samples_leaf

        best_column = None
        best_cut = None
        best_gain = -np.inf

        # select random features to check
        if self.max_features is None:
            feature_count = X.shape[1]
        else:
            feature_count = min(self.max_features, X.shape[1])

        columns = self.rng.choice(
            X.shape[1],
            size=feature_count,
            replace=False
        )

        for column in columns:

            values = X[:, column]
            low = values.min()
            high = values.max()

            if low == high:
                continue

            # check binary features
            if np.all((values == low) | (values == high)):

                left_rows = values == low
                n_left = int(left_rows.sum())
                n_right = n - n_left

                if n_left < smallest or n_right < smallest:
                    continue

                left_total = centred[left_rows].sum()

                gain = self.variance_reduction(
                    n, n_left, n_right, left_total
                )

                cut = (low + high) / 2

            else:

                # check possible split points
                order = np.argsort(values, kind="stable")
                ordered_values = values[order]

                left_totals = np.cumsum(centred[order])[:-1]

                n_left = np.arange(1, n)
                n_right = n - n_left

                usable = (
                    (ordered_values[:-1] != ordered_values[1:])
                    & (n_left >= smallest)
                    & (n_right >= smallest)
                )

                if not usable.any():
                    continue

                gains = self.variance_reduction(
                    n, n_left, n_right, left_totals
                )

                gains[~usable] = -np.inf

                pos = int(gains.argmax())

                gain = gains[pos]
                cut = (
                    ordered_values[pos] + ordered_values[pos + 1]
                ) / 2

            if gain > best_gain:
                best_column = column
                best_cut = cut
                best_gain = gain

        if best_column is None:
            return None, None, 0.0

        return best_column, best_cut, float(best_gain)

    # calculate variance reduction
    def variance_reduction(self, n, n_left, n_right, left_total):

        return (
            left_total ** 2 / n_left
            + left_total ** 2 / n_right
        ) / n

    def predict(self, X):

        X = np.asarray(X, dtype=np.float64)

        if self.root is None:
            raise ValueError("Train the tree before predicting.")

        results = np.empty(len(X))

        self.route(self.root, X, np.arange(len(X)), results)

        return results

    # send rows to their matching leaves
    def route(self, node, X, rows, results):

        if len(rows) == 0:
            return

        if node.is_leaf():
            results[rows] = node.prediction
            return

        goes_left = X[rows, node.feature] <= node.cut

        self.route(node.left_child, X, rows[goes_left], results)
        self.route(node.right_child, X, rows[~goes_left], results)


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

            tree = _DecisionTreeRegressor(
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

    X_train, y_train, X_test, y_test = prepare_data()

    print(
        f"Training projects: {len(y_train)} | "
        f"Testing projects: {len(y_test)}"
    )

    model = RandomForestRegressor(
        n_trees=5,
        max_depth=16,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42
    )

    print("\nTraining Random Forest...")
    model.fit(X_train, y_train)

    print("Training complete!")

    predicted = model.predict(X_test)
    actual = np.asarray(y_test, dtype=np.float64).ravel()

    # evaluation metrics
    mae = np.mean(np.abs(actual - predicted))
    mse = np.mean((actual - predicted) ** 2)
    rmse = np.sqrt(mse)

    total_variance = np.sum((actual - actual.mean()) ** 2)

    if total_variance == 0:
        r2 = float("nan")
    else:
        r2 = 1 - np.sum((actual - predicted) ** 2) / total_variance

    print("\nRandom Forest Results:")
    print(f"MAE:  PHP {mae:,.2f}")
    print(f"MSE:  {mse:,.2f}")
    print(f"RMSE: PHP {rmse:,.2f}")
    print(f"R2:   {r2:.4f}")

    print("\nFirst 10 testing projects:")
    print(f"{'Actual':>18} {'Predicted':>18}")

    for i in range(min(10, len(actual))):
        print(f"{actual[i]:>18,.0f} {predicted[i]:>18,.0f}")