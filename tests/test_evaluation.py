import numpy as np

from src.evaluation import (
    mean_absolute_error,
    mean_squared_error,
    root_mean_squared_error,
    r_squared,
)


# checks that all four metrics return the expected results
def test_metrics():
    actual = [10, 20, 30]
    predicted = [12, 18, 32]

    # compares the results against manually calculated answers
    assert np.isclose(mean_absolute_error(actual, predicted), 2.0)
    assert np.isclose(mean_squared_error(actual, predicted), 4.0)
    assert np.isclose(root_mean_squared_error(actual, predicted), 2.0)

    # checks perfect predictions and the mean-based baseline
    assert np.isclose(r_squared(actual, actual), 1.0)
    assert np.isclose(r_squared(actual, [20, 20, 20]), 0.0)

    # checks that predictions worse than the mean baseline get negative R-squared
    assert np.isclose(r_squared(actual, [0, 0, 0]), -6.0)

    print("Metric calculations: PASSED")


# checks that invalid inputs are rejected
def test_invalid_inputs():
    invalid_inputs = [
        ([], []),
        ([10, 20], [10]),
        ([10, float("nan")], [10, 20]),
        ([[10, 20]], [[10, 20]]),
    ]

    # each invalid pair should produce a ValueError
    for actual, predicted in invalid_inputs:
        try:
            mean_absolute_error(actual, predicted)
        except ValueError:
            continue

        raise AssertionError("Invalid input was accepted.")

    print("Input validation: PASSED")


# checks how R-squared handles identical actual values
def test_constant_targets():
    result = r_squared([10, 10, 10], [10, 10, 10])

    # R-squared should be undefined when actual values have no variation
    assert np.isnan(result)

    print("Constant target handling: PASSED")


if __name__ == "__main__":
    test_metrics()
    test_invalid_inputs()
    test_constant_targets()

    print("All evaluation tests passed!")
