import numpy as np
import pandas as pd
from sys import argv
from csv import reader, writer
import matplotlib.pyplot as plt

def read_csv(key):
    fname = f"linear_reg_{key}_data.csv"
    with open(fname, 'r') as csvfile:
        csv_reader = reader(csvfile, delimiter=',')
        csv_data = [row for row in csv_reader][1:]
        csv_data = [list(map(float, row)) for row in csv_data]
    df = pd.DataFrame(csv_data).sort_values([0])
    return df

def feature_extraction(df, degree=1):
    N = len(df)
    data_df = pd.DataFrame([])
    for i in range(degree+1):
        data_df[i] = np.power(df[0], i)
    Y = np.resize(df[1], (N, 1))
    X = np.array(data_df)
    return (X, Y)

def opt_solution(X, Y, lambd=0.0):
    D = X.shape[1]
    x = np.dot(X.T, X)
    if lambd != 0.0:
        x += lambd * np.identity(D)
    y = np.dot(X.T, Y)
    W = np.dot(np.linalg.inv(x), y)
    return W

def squared_loss(Y, Y_pred):
    return np.sum((Y - Y_pred) ** 2)

def plot_data(X, Y, Y_pred, keys):
    plt.figure(figsize=(9,6))
    plt.scatter(X, Y, color='g', marker='*', label="actual value")
    plt.plot(X, Y_pred, color='m', label="predicted value")
    plt.xlabel(r"x$\rightarrow$", size=15)
    plt.ylabel(r"y$\rightarrow$", size=15)
    plt.title(f"{keys[0]} data; degree {keys[1]}; lambda: {keys[2]}")
    plt.legend(loc="best")
    plt.ylim(-8, 8)
    plt.show()

def scatter_plot_data(Y, Y_pred, key):
    x_ = np.linspace(-5, 7)
    plt.figure(figsize=(9,6))
    plt.scatter(Y, Y_pred, color='r', marker='*')
    plt.plot(x_, x_, color='b', ls="--")
    plt.title(f"Scatter plot of model output vs expected output for {key} data")
    plt.xlabel("expected output", size=15)
    plt.ylabel("model output", size=15)
    plt.show()

def write_csv(X, Y, fname):
    with open(fname, 'w') as f:
        csv_writer = writer(f)
        csv_writer.writerow(["X", "Y_pred"])
        csv_writer.writerows([[x, y] for (x, y) in zip(X, Y)])

################################################### without regularization ###################################################

degree = int(argv[1])
train_df = read_csv("train")
X_train, Y_train = feature_extraction(train_df, degree=degree)
W = opt_solution(X_train, Y_train)
Y_train_pred = np.dot(X_train, W)
train_error = squared_loss(Y_train, Y_train_pred)
print(f"squared loss for train (degree = {degree}): {train_error}")
plot_data(train_df[0], Y_train, Y_train_pred, keys=["train", degree, 0.0])
scatter_plot_data(Y_train_pred, Y_train, key="train")
print(f"coefficents: {W}\n")

test_df = read_csv("test")
X_test, Y_test = feature_extraction(test_df, degree=degree)
Y_test_pred = np.dot(X_test, W)
test_error = squared_loss(Y_test, Y_test_pred)
print(f"squared loss for test (degree = {degree}): {test_error}")
scatter_plot_data(Y_test_pred, Y_test, key="test")
write_csv(test_df[0], Y_test_pred, fname=f"regress_test_{degree}.csv")

###################################################### with regularization #####################################################

if len(argv) > 2:
    print(f"\n+{'-'*100}+\n")
    lambd = float(argv[2])
    W = opt_solution(X_train, Y_train, lambd=lambd)
    Y_train_pred = np.dot(X_train, W)
    train_error = squared_loss(Y_train, Y_train_pred)
    print(f"squared loss for train with regularization (degree = {degree}): {train_error}")
    plot_data(train_df[0], Y_train, Y_train_pred, keys=["train", degree, lambd])
    scatter_plot_data(Y_train_pred, Y_train, key="train")
    print(f"coefficents: {W}\n")

    Y_test_pred = np.dot(X_test, W)
    test_error = squared_loss(Y_test, Y_test_pred)
    print(f"squared loss for test with regularization (degree = {degree}): {test_error}")
    scatter_plot_data(Y_test_pred, Y_test, key="test")
    write_csv(test_df[0], Y_test_pred, fname=f"ridge_regress_test_{degree}.csv")