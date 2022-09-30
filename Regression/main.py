import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
train_data = pd.read_csv('trainRegression.csv')
test_data = pd.read_csv('testRegression.csv')
#print(train_data)
#print(test_data)
figure, axis = plt.subplots(1,2)
axis[0].plot(train_data['X'],train_data['R'],'*')
axis[1].plot(test_data['X'],test_data['R'],'*')
numpy_train_data = train_data.values
numpy_test_data = test_data.values
N = len(numpy_train_data)
N2 = len(numpy_test_data)
plt.show()

def mean_square_err(YPrime, Y, N):
    sumE = np.sum(np.square(YPrime - Y))
    J = sumE / (2 * N)
    return J

def linear_model():
    X = numpy_train_data[:, 0]
    Y = numpy_train_data[:, 1]

    # computing cells for matrix A
    sumX = np.sum(X)
    sumX2 = np.sum(np.square(X))

    # computing cells for matrix B
    sumY = np.sum(Y)
    sumXY = np.sum(np.multiply(X, Y))

    # matrix A, B
    A = np.array([[N, sumX], [sumX, sumX2]])
    B = np.array([[sumY], [sumXY]])

    # matrix T
    T = np.dot(np.linalg.inv(A), B)

    # theta values
    theta0 = T[0][0]
    theta1 = T[1][0]

    # prediction on trained data
    predict_train_Y = theta0 + theta1 * X

    # mse for trained data
    JOrignal = mean_square_err(predict_train_Y, Y, N)
    print("Mean Square Error For Linear Model Training Data = ", JOrignal)
    plt.title("Linear Training Data")
    plt.plot(train_data['X'],train_data['R'],label="Original Training")
    plt.plot(train_data['X'], predict_train_Y, label="Predict Training")
    plt.legend()
    plt.show()

    # prediction on testing data
    test_X = numpy_test_data[:, 0]
    test_Y = numpy_test_data[:, 1]
    predict_test_Y = theta0 + theta1 * test_X

    # mse for testing data
    JTest = mean_square_err(predict_test_Y, test_Y, N2)
    print("Mean Square Error For Linear Model Testing  Data = ", JTest)
    plt.title("Linear Testing Data")
    plt.plot(test_data['X'],test_data['R'],label="Original Testing")
    plt.plot(test_data['X'],predict_test_Y,label="Predict Testing")
    plt.legend()
    plt.show()

linear_model()

def quadratic_model():
    X = numpy_train_data[:, 0]
    Y = numpy_train_data[:, 1]

    # computing cells for matrix A
    sumX = np.sum(X)
    sumX2 = np.sum(np.square(X))
    sumX3 = np.sum(np.power(X, 3))
    sumX4 = np.sum(np.power(X, 4))

    # computing cells for matrix B
    sumY = np.sum(Y)
    sumXY = np.sum(np.multiply(X, Y))
    sumX2Y = np.sum(np.multiply(np.square(X), Y))

    # matrix A, B
    A = np.array([[N, sumX, sumX2], [sumX, sumX2, sumX3], [sumX2, sumX3, sumX4]])
    B = np.array([[sumY], [sumXY], [sumX2Y]])

    # matrix T
    T = np.dot(np.linalg.inv(A), B)

    # theta values
    theta0 = T[0][0]
    theta1 = T[1][0]
    theta2 = T[2][0]

    # prediction on trained data
    predict_train_Y = theta0 + (theta1 * X) + (theta2 * np.square(X))

    # mse for trained data
    JOrignal = mean_square_err(predict_train_Y, Y, N)
    print("\nMean Square Error For Quadratic Model Training Data = ", JOrignal)
    plt.title("Quadratic Training Data")
    plt.plot(train_data['X'],train_data['R'],label="Original Training")
    plt.plot(train_data['X'],predict_train_Y,label="Predict Training")
    plt.legend()
    plt.show()

    # prediction on testing data
    test_X = numpy_test_data[:, 0]
    test_Y = numpy_test_data[:, 1]
    predict_test_Y = theta0 + (theta1 * test_X) + (theta2 * np.square(test_X))

    # mse for testing data
    JTest = mean_square_err(predict_test_Y, test_Y, N2)
    print("Mean Square Error For Quadratic Model Testing  Data = ", JTest)
    plt.title("Quadratic Testing Data")
    plt.plot(test_data['X'],test_data['R'],label="Original Testing")
    plt.plot(test_data['X'], predict_test_Y, label="Original Testing")
    plt.legend()
    plt.show()

quadratic_model()

def cubic_model():
    X = numpy_train_data[:, 0]
    Y = numpy_train_data[:, 1]

    # computing cells for matrix A
    sumX = np.sum(X)
    sumX2 = np.sum(np.square(X))
    sumX3 = np.sum(np.power(X, 3))
    sumX4 = np.sum(np.power(X, 4))
    sumX5 = np.sum(np.power(X, 5))
    sumX6 = np.sum(np.power(X, 6))

    # computing cells for matrix B
    sumY = np.sum(Y)
    sumXY = np.sum(np.multiply(X, Y))
    sumX2Y = np.sum(np.multiply(np.square(X), Y))
    sumX3Y = np.sum(np.multiply(np.power(X, 3), Y))

    # matrix A, B
    A = np.array([[N, sumX, sumX2, sumX3], [sumX, sumX2, sumX3, sumX4], [sumX2, sumX3, sumX4, sumX5],[sumX3, sumX4, sumX5, sumX6]])
    B = np.array([[sumY], [sumXY], [sumX2Y], [sumX3Y]])

    # matrix T
    T = np.dot(np.linalg.inv(A), B)

    # theta values
    theta0 = T[0][0]
    theta1 = T[1][0]
    theta2 = T[2][0]
    theta3 = T[3][0]

    # prediction on trained data
    predict_train_Y = theta0 + (theta1 * X) + (theta2 * np.square(X)) + (theta3 * np.power(X, 3))

    # mse for trained data
    JOrignal = mean_square_err(predict_train_Y, Y, N)
    print("\nMean Square Error For Cubic Model Training Data = ", JOrignal)
    plt.title("Cubic Training Data")
    plt.plot(train_data['X'],train_data['R'],label="Original Training")
    plt.plot(train_data['X'],predict_train_Y,label="Predicted Training")
    plt.legend()
    plt.show()

    # prediction on testing data
    test_X = numpy_test_data[:, 0]
    test_Y = numpy_test_data[:, 1]
    predict_test_Y = theta0 + (theta1 * test_X) + (theta2 * np.square(test_X)) + (theta3 * np.power(test_X, 3))

    # mse for testing data
    JTest = mean_square_err(predict_test_Y, test_Y, N2)
    print("Mean Square Error For Cubic Model Testing  Data = ", JTest)
    plt.title("Cubic Testing Data")
    plt.plot(test_data['X'],test_data['R'],label="Original Testing")
    plt.plot(test_data['X'], predict_test_Y, label="Original Testing")
    plt.legend()
    plt.show()
cubic_model()

print("\nCubic Model is best as it has least mean square error!!!!!!!")