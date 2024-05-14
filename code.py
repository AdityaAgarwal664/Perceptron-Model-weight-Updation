import numpy as np
import matplotlib.pyplot as plt


np.random.seed(0)
X = 2 * np.random.rand(100, 1)
y = 3 + 4 * X + np.random.randn(100, 1)


X_b = np.c_[np.ones((100, 1)), X]


def gradient_descent(X, y, learning_rate, n_iterations=500):
    m = len(X)
    theta = np.random.randn(2, 1)  
    cost_history = []  
    theta_history = [theta.flatten()]  
    for iteration in range(n_iterations):
        gradients = 2/m * X.T.dot(X.dot(theta) - y)
        theta = theta - learning_rate * gradients
        
        cost = np.mean((X.dot(theta) - y) ** 2)
        cost_history.append(cost)
        theta_history.append(theta.flatten())
    return theta, cost_history, np.array(theta_history)


learning_rate = float(input("Enter the learning rate: "))


optimal_weights, cost_history, theta_history = gradient_descent(X_b, y, learning_rate)


plt.figure(figsize=(10, 6))

plt.plot(X, y, "b.")
plt.xlabel("X")
plt.ylabel("y")
plt.title("Original Data and Linear Regression")
plt.plot(X, X_b.dot(optimal_weights), 'r-')

plt.tight_layout()
plt.show()


plt.figure(figsize=(8, 6))
plt.plot(range(len(cost_history)), cost_history)
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Cost Function Convergence")
plt.show()


print("\nHistory of weights:")
for i, weights in enumerate(theta_history):
    if(i%50==0):
         print(f"Iteration {i}: theta =", weights)

print("\nUpdated weights are: ",(theta_history[len(theta_history)-1]))