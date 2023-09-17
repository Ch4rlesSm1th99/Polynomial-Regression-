"""
Linear Regression - Nonlinear Fits (Polynomial Regression)

linear regression can be used to fit more complex functions to our data than just linear fits. To do this feature
engineering is introduced to the model where new features are created using the original data and this could be as
simple as raising your original features to a power or taking the product of different combinations of the original
features.

Typically, in linear regression feature engineering is referred to as the number of basis the model uses to fit the
function. More generally we use polynomials for our basis but as demonstrated by later scripts on linear regression
we will see that we can use other basis for specific problems and tasks.

Another point to note that will be shown in later scripts is that if some features of our data are less important
than others we can introduce feature selection where we only use feature engineering on certain features of
our data and this should aim to fit an accurate model while keeping the number of parameters the algorithm is solving
for lower. I should also add this that if your having into introduce feature selection to a linear regression model
normally there will be a better alternative method for dealing with data that has high dimensionality.

The reason why this technique that is being exploited still remains linear regression is because the algorithm
solves for the best linear combination of weights (theta in the normal equation) that fits the data best.

"""

import numpy as np
import matplotlib.pyplot as plt
import math

pi = math.pi

# data generation
num_samples = 30
x0 = np.linspace(-1, 1, num_samples)

# generate noisy sine wave data for training and testing
y0 = np.sin(2 * pi * 1 * x0) + 0.3 * np.random.randn(x0.size)
yt = np.sin(2 * pi * 1 * x0) + 0.3 * np.random.randn(x0.size)

# plot the generated noisy data
plt.plot(x0, y0, "bo")
plt.title("training data")
plt.show()

plt.figure(figsize=(12, 7))

# initialise lists store errors for different polynomial degrees
tr_errors = []
te_errors = []
n_basis = []

# loop over varying polynomial degrees (from 1 to 10)
for num_basis in range(1, 11):

    # design matrix construction: [1, x, x^2, x^3, ...]
    X = []
    for i in range(num_basis + 1):
        X.append(x0 ** i)
    X = np.array(X).T

    # compute the weights (theta)
    theta = np.linalg.solve(X.T @ X, X.T @ y0)

    # predict y values using the polynomial model
    yhat = X @ theta

    # plot original data, model's predictions, and true sine curve
    plt.subplot(2, 5, num_basis)
    plt.plot(x0, y0, "bo", x0, yhat, 'r', x0, yt, 'g*')

    # calculate and store the training and test errors
    test_error = np.mean((yhat - yt) ** 2)
    train_error = np.mean((yhat - y0) ** 2)
    plt.title(f"num basis = {num_basis}")
    tr_errors.append(np.log(train_error))
    te_errors.append(np.log(test_error))
    n_basis.append(num_basis)

# show the plots for all polynomial fits
plt.show()

# plotting errors
# plot the logarithm of training and test errors against polynomial order
plt.figure()
plt.plot(n_basis, tr_errors, n_basis, te_errors)
plt.xlabel('Number of Basis')
plt.ylabel('Log Error')
plt.legend(['Training Error', 'Test Error'])
plt.title('Error vs. Polynomial Degree')
plt.grid()
plt.show()

'''
When the script is run hopefully you can see from the error graph that the test and training errors begin to diverge 
after a certain polynomial degree order. This happens because the model begins to suffer from over fitting at a certain
point where it can't distinguish between the trends in the data and the random noise and so when tested on the test set
it under performs.

In order to stop the model from over fitting we have to introduce constraints where it is penalised for over fitting. In
this script we will right another version of the code above where the only constraint we are going to introduce is we 
are going to restrict the polynomial order degree in our basis so that the test and training errors remain closest. 

In later scripts lasso and ridge regularisation will be implemented in order to try to avoid over fitting.
'''
tr_errors = []
te_errors = []

num_basis = 1  # start with polynomial degree 1
max_basis = 11  # some upper limit to ensure the loop doesn't run indefinitely

while num_basis <= max_basis:

    # design matrix construction: [1, x, x^2, x^3, ...]
    X = []
    for i in range(num_basis + 1):
        X.append(x0 ** i)
    X = np.array(X).T

    # compute the weights (theta)
    theta = np.linalg.solve(X.T @ X, X.T @ y0)

    # predict y values using the polynomial model
    yhat = X @ theta

    # calculate and store the training and test errors
    test_error = np.mean((yhat - yt) ** 2)
    train_error = np.mean((yhat - y0) ** 2)

    tr_errors.append(np.log(train_error))
    te_errors.append(np.log(test_error))

    # check if difference between training and test errors is less than a specified error range
    # this method will only work for incredibly simple models, although it stops over fitting, the condition which
    # breaks the training loop can be satisfied too early sometimes resulting in high amounts of under fitting
    if abs(train_error - test_error) < 0.01:
        break

    # increase polynomial degree for the next iteration
    num_basis += 1

# plot original data, model's predictions, and true sine curve
plt.plot(x0, y0, "bo", label="training")
plt.plot(x0, yhat, 'r', label="model")
plt.plot(x0, yt, 'g*', label="testing")
plt.title("constrained model")
plt.legend()
plt.grid()
plt.show()
