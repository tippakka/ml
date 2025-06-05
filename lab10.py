import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Bandwidth for local weighting
tau = 0.5

# Load dataset
data = pd.read_csv("10.csv")
x_train = np.array(data.total_bill)[:, np.newaxis]  # Reshape to 2D
y_train = np.array(data.tip)

# Generate test data
x_test = np.array([[i] for i in range(300)])
y_test = []

for i in range(len(x_test)):
    # Gaussian weights
    diff = x_train - x_test[i]
    w = np.exp(-np.sum(diff**2, axis=1) / (2 * tau**2))
    W = np.diag(w)

    # Locally weighted linear regression using pseudo-inverse
    XTWX = x_train.T.dot(W).dot(x_train)
    XTWy = x_train.T.dot(W).dot(y_train)
    theta = np.linalg.pinv(XTWX).dot(XTWy)

    # Predict
    prediction = x_test[i].dot(theta)
    y_test.append(prediction)

# Convert to array
y_test = np.array(y_test)

# Plot results
plt.plot(x_train.squeeze(), y_train, 'ro', label="Training Data")
plt.plot(x_test.squeeze(), y_test, 'b-', label="Predicted Curve")
plt.xlabel("Total Bill")
plt.ylabel("Tip")
plt.title("Locally Weighted Regression")
plt.legend()
plt.show()