"""
@authors: Liscio Alessandro, Cozzi Riccardo
last updated: 1/04/2020 - 18:55
"""

# main libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification  # used to create random datasets
from sklearn.linear_model import LogisticRegression  # used to compare scores
import time

print("OPTIMIZATION @ AI\n  Case study project\n  Cozzi R. - Liscio A.\n-----------------------\n")


###############################################################################
#                         FUNCTIONS DEFINITIONS                               #
###############################################################################


# sigmoid function, used for logistic regression
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# actual predictive function
def predict(X, params):
    return np.round(sigmoid(X @ params))


# loss function, used for logistic regression (different from linear regression)
def loss(X, y, theta):
    m = len(y)
    h = sigmoid(X @ theta)
    eps = 1e-5  # used to avoid the log 0 
    cost = (1 / m) * (((-y).T @ np.log(h + eps)) - ((1 - y).T @ np.log(1 - h + eps)))
    return cost


# gradient descent for boundary model optimization
def gradient_descent(X, y, params, lr, iterations):
    print("\nGradient descent")
    m = len(y)
    costs = np.zeros((iterations, 1))
    t0 = time.time()
    for i in range(iterations):
        gradient = X.T @ (sigmoid(X @ params) - y)      # compute the gradient
        params -= (1 / m) * lr * gradient               # update the params
        costs[i] = loss(X, y, params)                   # compute the loss

    print("Completed")
    return costs, params, time.time() - t0


# mini-batch stochstic gradient descent for boundary model optimization
def stochastic_gradient_descent(X, y, params, lr, iterations, use_batch=True, batch_size=10):
    m = len(y)
    if not use_batch:
        batch_size = 3 # <----------------------- M, not 3! 
    print("\nStochastic gradient descent", end="")
    print("( with batch_size ", batch_size if use_batch else print("."), ")")

    costs = np.zeros((iterations, 1))
    t0 = time.time()

    for i in range(0, iterations):
        batch_cost = 0
        batch_indices = np.random.permutation(m)[:batch_size]   # choose batch_size random indices from the data set
        X_batch = X[batch_indices]                              # extract a subset of X as batch
        y_batch = y[batch_indices]                              # the same for Y

        for b in range(batch_size):                                             # for each batch
            gradient = (1 / batch_size) * X_batch.T @ (sigmoid(X_batch @ params) - y_batch)    # compute the gradient
            params -= lr * gradient                                             # update the params
            batch_cost = loss(X, y, params)                                     # compute the loss locally to the batch

        costs[i] = batch_cost                                                   # compute the loss on the whole dataset

    print("Completed")
    return costs, params, time.time() - t0


# gradient descent for boundary model optimization
def momentum_gradient_descent(X, y, params, lr, iterations, gamma=0.9):
    print("\nMomentum gradient descent")
    m = len(y)
    costs = np.zeros((iterations, 1))
    t0 = time.time()
    momentum = 0
    for i in range(iterations):
        gradient = (1 / m) * X.T @ (sigmoid(X @ params) - y)                # compute the gradient
        momentum = (lr * gradient) + momentum * gamma  # the new movement is incremented proportionally to the previous
        params -= momentum                                                  # update the params
        costs[i] = loss(X, y, params)                                       # compute the loss

    print("Completed")
    return costs, params, time.time() - t0


###############################################################################
#                            DATA INITIALIZATION                              #
###############################################################################

random_state = 42
np.random.seed(random_state)

X, y = make_classification(n_samples=1000,
                           n_features=2,
                           n_redundant=0,
                           n_informative=1,
                           n_clusters_per_class=1,
                           random_state=random_state,
                           class_sep=1.5)

"""
# plot data
sns.set_style('white')
sns.scatterplot(X[:, 0], X[:, 1], hue=y.reshape(-1))
plt.title("Data")
plt.show()
"""

# add intercept column
X = np.hstack((np.ones((len(y), 1)), X))
y = y[:, np.newaxis]

# parameters initialization
epochs = 1000
learning_rate = 0.1

###############################################################################
#                            MODEL COMPUTATION                                #
###############################################################################


""" GRADIENT DESCENT """

# compute costs and best parameters
initial_params = np.zeros((np.size(X, 1), 1))
costs_GD, params_best_GD, time_GD = gradient_descent(X, y, initial_params, learning_rate, epochs)
print("\nBest Parameters GD:\n", params_best_GD)

# compute slope and intercept of the boundary line
slope_GD = -(params_best_GD[1] / params_best_GD[2])
intercept_GD = -(params_best_GD[0] / params_best_GD[2])

# predict data and compute score for the GD
y_pred_GD = predict(X, params_best_GD)
score_GD = float(sum(y_pred_GD == y)) / float(len(y)) * 100

""" STOCHASTIC GRADIENT DESCENT """

# compute costs and best parameters
initial_params = np.zeros((np.size(X, 1), 1))
costs_SGD, params_best_SGD, time_SGD = stochastic_gradient_descent(X, y, initial_params, learning_rate, epochs)
print("\nBest Parameters SGD:\n", params_best_SGD)

# compute slope and intercept of the boundary line
slope_SGD = -(params_best_SGD[1] / params_best_SGD[2])
intercept_SGD = -(params_best_SGD[0] / params_best_SGD[2])
# predict data and compute score for the SGD
y_pred_SGD = predict(X, params_best_SGD)
score_SGD = float(sum(y_pred_SGD == y)) / float(len(y)) * 100

""" GRADIENT DESCENT WITH MOMENTUM """

# compute costs and best parameters
initial_params = np.zeros((np.size(X, 1), 1))
costs_MGD, params_best_MGD, time_MGD = momentum_gradient_descent(X, y, initial_params, learning_rate, epochs)
print("\nBest Parameters MGD:\n", params_best_MGD)

# compute slope and intercept of the boundary line
slope_MGD = -(params_best_MGD[1] / params_best_MGD[2])
intercept_MGD = -(params_best_MGD[0] / params_best_MGD[2])

# predict data and compute score for the MGD
y_pred_MGD = predict(X, params_best_MGD)
score_MGD = float(sum(y_pred_MGD == y)) / float(len(y)) * 100

""" STANDARD METHOD """

clf = LogisticRegression(fit_intercept=True,
                         random_state=random_state,
                         solver='lbfgs')  # default, warning if missing
t0 = time.time()
clf.fit(X, y.reshape(-1))
time_STD = time.time() - t0
clf.predict(X)
score_STD = clf.score(X, y) * 100


# plot convergence of cost function
plt.figure(figsize=(15, 5))
# normal graphic
plt.subplot(121)
plt.title("Convergence of Cost Function")
plt.plot(range(len(costs_GD)), costs_GD, 'r', label="Gradient Descent")
plt.plot(range(len(costs_SGD)), costs_SGD, 'b', label="Stochastic Gradient Descent")
plt.plot(range(len(costs_MGD)), costs_MGD, 'g', label="Gradient Descent + momentum")
plt.xlabel("Epochs")
plt.ylabel("Loss")

# zoomed graphic
plt.subplot(122)
lim = int(epochs / 10)
plt.title("First " + str(lim) + " epochs")
plt.plot(range(lim), costs_GD[:lim], 'r', label="Gradient Descent")
plt.plot(range(lim), costs_SGD[:lim], 'b', label="Stochastic Gradient Descent")
plt.plot(range(lim), costs_MGD[:lim], 'g', label="Gradient Descent + momentum")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

# plot data and regressor line
plt.figure()
sns.scatterplot(X[:, 1], X[:, 2], hue=y.reshape(-1))
ax = plt.gca()
ax.autoscale(False)
x_vals = np.array(ax.get_xlim())
y_vals_GD = intercept_GD + (slope_GD * x_vals)
y_vals_SGD = intercept_SGD + (slope_SGD * x_vals)
y_vals_MGD = intercept_MGD + (slope_MGD * x_vals)
plt.plot(x_vals, y_vals_GD, c="r", label="GD intercept")
plt.plot(x_vals, y_vals_SGD, c="b", label="SGD intercept")
plt.plot(x_vals, y_vals_MGD, c="g", label="MGD intercept")
plt.title("Data and boundary line")
plt.legend()

# print results
print("\n    MODEL                      TIME (sec)    SCORE (%)"
      "\n------------------------------------------------------"
      "\nGradient Descent               {:.3f}         {:.2f}%"
      "\nStochastic Gradient Descent    {:.3f}         {:.2f}%"
      "\nGradient Descent + momentum    {:.3f}         {:.2f}%"
      "\nSklearn.LogisticRegression     {:.3f}         {:.2f}%".format(
        time_GD, score_GD,
        time_SGD, score_SGD,
        time_MGD, score_MGD,
        time_STD, score_STD))

plt.show()
