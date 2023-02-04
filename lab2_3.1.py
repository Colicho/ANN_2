import numpy as np
import matplotlib.pyplot as plt
import math
import random as rnd


def main():
    step, rbf_nodes, sigmaW, eta, epochs, noise = 0.1, 10, 1, 0.01, 100, 0.2

    # limit to [0, 2pi]
    limit_x = int(2 * math.pi / step)
    patterns = [i * step for i in range(limit_x)]
    targets_sin_train = sin(patterns)
    targets_square_train = square(patterns)
    rbf_placements_x = [(limit_x + 1) / rbf_nodes * i * step for i in range(rbf_nodes)]

    # placing rbfs uniformly with noise
    for i in range(len(rbf_placements_x)):
        if rnd.randint(0, 1) == 1:
            rbf_placements_x[i] += noise
        else:
            rbf_placements_x[i] -= noise

    rbf_placements_sin = sin(rbf_placements_x)
    rbf_placements_square = square(rbf_placements_x)

    # could be wrong variance
    rbf_variance = [3 for i in range(rbf_nodes)]

    # test set
    test_set = [0.05 + i * step for i in range(limit_x)]
    targets_sin_test = sin(test_set)
    targets_square_test = square(test_set)

    # least squares train
    phi_train = rbf(patterns, rbf_placements_x, rbf_variance)
    A_train = np.dot(np.transpose(phi_train), phi_train)

    # least squares for sin test
    B_sin = np.dot(np.dot(np.linalg.inv(A_train), np.transpose(phi_train)), targets_sin_train)
    weights_sin = B_sin
    err_sin = np.sum(np.square(np.dot(phi_train, weights_sin) - targets_sin_train))
    print(err_sin, "error on sin(2x) train")

    # least squares for square test
    B_square = np.dot(np.dot(np.linalg.inv(A_train), np.transpose(phi_train)), targets_square_train)
    weights_square = B_square
    err_square = np.sum(np.square(np.dot(phi_train, weights_square) - targets_square_train))
    print(err_square, "error on square(2x) train")

    # least square test
    phi_test = rbf(test_set, rbf_placements_x, rbf_variance)

    # least squares for sin test
    err_sin_test = np.sum(np.square(np.dot(phi_test, weights_sin) - targets_sin_test))
    print(err_sin_test, "error on sin(2x) test")

    # #least squares for square test
    err_square_test = np.sum(np.square(np.dot(phi_test, weights_square) - targets_square_test))
    print(err_square_test, "error on square(2x) test")

    plt.figure()
    plt.plot(patterns, targets_sin_train, label="sin(2x)")
    plt.plot(patterns, np.dot(phi_test, weights_sin), label="sin(2x) approx")
    plt.plot(patterns, targets_square_train, label="square(2x)")
    plt.plot(patterns, np.dot(phi_test, weights_square), label="square(2x) approx")

    plt.legend()

    plt.show()


def sin(x):
    targets = []
    for i in x:
        targets.append(math.sin(2 * i))
    return targets


def square(x):
    targets = []
    for i in range(len(x)):
        if math.sin(2 * x[i]) >= 0:
            targets.append(1)
        else:
            targets.append(-1)
    return targets


def rbf(x, rbf, variance):
    phi = []
    for j in range(len(x)):
        phi.append([])
        for i in range(len(rbf)):
            phi[j].append(np.exp(-((x[j] - rbf[i]) ** 2) / (2 * variance[i])))
    return phi


if __name__ == '__main__':
    main()