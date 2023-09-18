import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from sklearn.model_selection import KFold

if len(sys.argv)<2:
    print("Error : need dataset!")
    exit()
dataset_file = sys.argv[1]
data = np.loadtxt(dataset_file)
x = data[:,0]
y = data[:,1]

X = np.ones((len(x), 15))
X[:, 0] = x**14
X[:, 1] = x**13
X[:, 2] = x**12
X[:, 3] = x**11
X[:, 4] = x**10
X[:, 5] = x**9
X[:, 6] = x**8
X[:, 7] = x**7
X[:, 8] = x**6
X[:, 9] = x**5
X[:, 10] = x**4
X[:, 11] = x**3
X[:, 12] = x**2
X[:, 13] = x
X_T = X.transpose()
Y = np.array([y])
w = np.linalg.inv(X_T.dot(X)).dot(X_T).dot(Y.T)
print("\nWeight vector w:")
print(w.T)

predic_y = X.dot(w)
train_error = 0 
for i in range(len(x)):
    tmp = predic_y[i]-y[i]
    train_error += tmp**2
train_error /= len(x)

kf = KFold(n_splits=5, shuffle=True, random_state=14)
cv_errors = []
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    w_cv = np.linalg.inv(X_train.T.dot(X_train)).dot(X_train.T).dot(y_train)
    y_pred_cv = X_test.dot(w_cv)
    cv_error = np.mean((y_pred_cv - y_test)**2)
    cv_errors.append(cv_error)
cross_validation_error = np.mean(cv_errors)

print("\n\nTraining error:", train_error)
print("\n\nFive-fold cross-validation errors:", cross_validation_error)

plt.scatter(x, y)
plt.plot(x, X.dot(w), color='red')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Linear Regression Fit')
plt.show()