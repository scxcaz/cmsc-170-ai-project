import numpy as np

class Node:
    __slots__ = ("feature", "cut", "left_child", "right_child", "gain", "prediction", "size")

    def __init__(self, prediction, size):
        self.prediction = prediction #avg target value at this node
        self.size = size #num of data points that reach this node
        self.feature = None
        self.cut = None
        self.left_child = None
        self.right_child = None
        self.gain = 0.0 #how useful the split is

    def is_leaf(self):
        return self.feature is None

class DecisionTreeRegressor: #predicts the average target value
    def __init__(self, max_depth=8, min_samples_split=20, min_samples_leaf=10):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.root = None

    def fit(self, X, y):
        #convert x and y into numpy arrays
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel() #ravel make sure it is 1d

        #start building the tree from the root
        self.root = self.grow(X, y, depth=0)
        return self

    def grow(self, X, y, depth): #recursively build tree
        #assume node will be leaf
        node = Node(prediction=float(y.mean()), size=len(y))

        #allowed to split?
        too_deep = depth >= self.max_depth
        too_small = len(y) < max(self.min_samples_split, 2 * self.min_samples_leaf)
        all_equal = y.min() == y.max()
        if too_deep or too_small or all_equal:
            return node

        #search for best possible split
        column, cut, gain = self.search_split(X, y)
        if column is None or gain <= 0:
            return node

        #create the two groups
        goes_left = X[:, column] <= cut
        node.feature, node.cut, node.gain = column, cut, gain
        node.left_child = self.grow(X[goes_left], y[goes_left], depth + 1)
        node.right_child = self.grow(X[~goes_left], y[~goes_left], depth + 1)
        return node

    def search_split(self, X, y): #search for the best split
        n = len(y) #num of projs
        centred = y - y.mean()
        smallest = self.min_samples_leaf

        best_column, best_cut, best_gain = None, None, -np.inf

        #Check every cols
        for column in range(X.shape[1]):
            values = X[:, column]
            low, high = values.min(), values.max()

            #if every value is identical, this feature cannot divide the data
            if low == high:
                continue

            #if binary
            if np.all((values == low) | (values == high)):

                left_rows = values == low
                n_left = int(left_rows.sum())
                n_right = n - n_left
                if n_left < smallest or n_right < smallest:
                    continue
                left_total = centred[left_rows].sum()
                gain = self.variance_reduction(n, n_left, n_right, left_total)
                cut = (low + high) / 2
            else:
                # sort the feature values and test possible splits
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

                #calculate gain for each split
                gains = self.variance_reduction(n, n_left, n_right, left_totals)
                gains[~usable] = -np.inf
                pos = int(gains.argmax())
                gain = gains[pos]
                cut = (ordered_values[pos] + ordered_values[pos + 1]) / 2

            #keep split with highest gain
            if gain > best_gain:
                best_column, best_cut, best_gain = column, cut, gain

        if best_column is None:
            return None, None, 0.0
        return best_column, best_cut, float(best_gain)

    #how much the split lowers the variance of the targets
    def variance_reduction(self, n, n_left, n_right, left_total):
        return (left_total ** 2 / n_left + left_total ** 2 / n_right) / n

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        results = np.empty(len(X)) #create empty array to store predictions
        self.route(self.root, X, np.arange(len(X)), results)
        return results

    # send data thru tree
    def route(self, node, X, rows, results):
        if len(rows) == 0:
            return
        if node.is_leaf():
            results[rows] = node.prediction
            return

        goes_left = X[rows, node.feature] <= node.cut
        self.route(node.left_child, X, rows[goes_left], results)
        self.route(node.right_child, X, rows[~goes_left], results)

    #print decision rules
    def print_tree(self, feature_names=None, max_print_depth=3):
        def label(column):
            return feature_names[column] if feature_names is not None else f"X_{column}"

        def walk(node, level):
            pad = "|   " * level
            if node.is_leaf():
                print(f"{pad}-> predict {node.prediction:,.0f} PHP  (n={node.size})")
            elif level >= max_print_depth:
                print(f"{pad}...")
            else:
                print(f"{pad}if {label(node.feature)} <= {node.cut:.4g}   (n={node.size}, gain={node.gain:,.0f})")
                walk(node.left_child, level + 1)
                print(f"{pad}else:")
                walk(node.right_child, level + 1)

        walk(self.root, 0)


if __name__ == "__main__":
    from src.preprocessing import prepare_data
    from src.evaluation import (
        mean_absolute_error,
        mean_squared_error,
        root_mean_squared_error,
        r_squared,
    )

    X_train, y_train, X_test, y_test = prepare_data()
    print(f"Training projects: {len(y_train)} | Testing projects: {len(y_test)}")
    model = DecisionTreeRegressor(max_depth=16, min_samples_split=20, min_samples_leaf=10)
   
    # train the decision tree using the training data
    model.fit(X_train, y_train)

    print("\nTop of the tree:")
    model.print_tree(max_print_depth=3)

    # make predictions for both the training and testing data
    train_actual = np.asarray(y_train, dtype=np.float64)
    train_predicted = model.predict(X_train)
    actual = np.asarray(y_test, dtype=np.float64)
    predicted = model.predict(X_test)

    # evaluate model on the testing data using the shared metrics
    print("\nDecision Tree Results (testing set):")
    print(f"MAE:  PHP {mean_absolute_error(actual, predicted):,.2f}")
    print(f"MSE:  {mean_squared_error(actual, predicted):,.2f}")
    print(f"RMSE: PHP {root_mean_squared_error(actual, predicted):,.2f}")
    print(f"R2:   {r_squared(actual, predicted):.4f}")

    # evaluate model on the training data
    #see if the tree is overfitting
    print("\nDecision Tree Results (training set):")
    print(f"MAE:  PHP {mean_absolute_error(train_actual, train_predicted):,.2f}")
    print(f"RMSE: PHP {root_mean_squared_error(train_actual, train_predicted):,.2f}")
    print(f"R2:   {r_squared(train_actual, train_predicted):.4f}")

    print("\nFirst 10 testing projects:")
    print(f"{'Actual':>18} {'Predicted':>18}")
    for i in range(10):
        print(f"{actual[i]:>18,.0f} {predicted[i]:>18,.0f}")