import math

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from cvxopt import matrix, solvers
from mpl_toolkits.mplot3d import Axes3D

# Load data
data = pd.read_csv('SDdiabetesS30X3.csv')
x_indexes = [0, 1, 2]  # Indices of feature columns
y_index = len(data.columns)-1        # Index of label column
print(data)


# Prepare matrices
matrix_size = data.shape[0]  # Number of rows in the dataset
varCount = len(x_indexes)

x = np.zeros((matrix_size, varCount))
for row in range(len(x)):
    for var in range(len(x[row])):
        x[row][var] = data.iloc[row, x_indexes[var]]

scaler = StandardScaler()
x = scaler.fit_transform(x)

K = np.zeros((matrix_size, matrix_size))  # Kernel matrix (dot products)
for row in range(len(K)):
    for col in range(len(K[row])):
        sumVar = 0
        for VARx in range(varCount):
            sumVar += data.iloc[row, VARx] * data.iloc[col, VARx]
        K[row][col] = sumVar

y = np.array([0] * matrix_size)
for i in range(len(y)):
    y[i] = data.iloc[i, y_index]

Q = np.zeros((matrix_size, matrix_size))  # Scaled kernel matrix
for row in range(len(Q)):
    for col in range(len(Q[row])):
        Q[row][col] = y[row] * y[col] * K[row][col]

p = [-1 for _ in range(matrix_size)]  # Linear coefficient vector
G = np.eye(matrix_size)               # Identity matrix for inequality constraints
h = np.zeros(matrix_size)             # Vector of zeros for inequality constraints
A = y.astype(float).reshape(1, -1)    # Row vector of labels for equality constraint
b = np.array([0.0])                   # Scalar for equality constraint

# Convert inputs to CVXOPT matrix format
Qm = matrix(Q, tc='d')                # Q must be a symmetric matrix of type 'd'
pm = matrix(np.array(p, dtype='d').reshape(-1, 1))  # p must be a column vector of type 'd'
Gm = matrix(-G, tc='d')               # Negative identity matrix for α ≥ 0
hm = matrix(np.zeros(matrix_size), tc='d')          # h must be a column vector of type 'd'
Am = matrix(A, tc='d')                # A must be a row vector of type 'd'
bm = matrix(b, tc='d')                # b must be a scalar or column vector of type 'd'

# Solve the QP problem
#solvers.options['show_progress'] = False
solution = solvers.qp(Qm, pm, Gm, hm, Am, bm)

# Extract the Lagrange multipliers (α)
alphas = np.array(solution['x']).flatten()
print("Lagrange Multipliers (α):", alphas)

W = [0] * varCount

for i in range(matrix_size):
    for j in range(varCount):
        a = float(alphas[i])
        y = data.iloc[i, y_index]
        x = data.iloc[i, x_indexes[j]]
        W[j] += a*y*x

print(f'w = {W}')

sumW2 = 0
for i in range(len(W)):
    sumW2 += W[i]**2

margin_W = math.sqrt(sumW2)
print(f'||w|| = {margin_W}')

#calculate b
support_vectors = [i for i, alpha in enumerate(alphas) if alpha > 1e-5]
row = support_vectors[0]
print(row)
sumWX = 0
for i in range(len(W)):
    sumWX += W[i]* data.iloc[row, x_indexes[i]]

b = data.iloc[row, y_index] - sumWX




# Plot data points
def plot_2dGraphs(yVarName):
    gradient = -W[0] / W[1]
    c = -b / W[1]
    plt.figure(figsize=(8, 6))
    plt.scatter(data[data[yVarName] == 1]['x1'], data[data[yVarName] == 1]['x2'],
                color='red', label="Class +1", edgecolors='k', s=100)
    plt.scatter(data[data[yVarName] == -1]['x1'], data[data[yVarName] == -1]['x2'],
                color='blue', label="Class -1", edgecolors='k', s=100)

    x1_values = np.linspace(-1, 10, 100)
    x2_values = gradient * x1_values + c

    plt.plot(x1_values, x2_values, label="decision boundary", color='red')

    plt.xlabel(data.columns[0])
    plt.ylabel(data.columns[1])
    plt.title("SVM Data Visualization")
    plt.axhline(0, color='black', linewidth=0.5, linestyle='--')  # x-axis
    plt.axvline(0, color='black', linewidth=0.5, linestyle='--')  # y-axis
    plt.grid(alpha=0.3)
    plt.legend()
    plt.show()

def plot_3dGraphs(yVarName):
    # Generate a grid of x1 and x2 values
    x1_values = np.linspace(data.iloc[:, 0].min(), data.iloc[:, 0].max(), 50)
    x2_values = np.linspace(data.iloc[:, 1].min(), data.iloc[:, 1].max(), 50)
    x1_grid, x2_grid = np.meshgrid(x1_values, x2_values)

    # Compute corresponding x3 values using the plane equation
    x3_grid = -(W[0] * x1_grid + W[1] * x2_grid + b) / W[2]

    # Create a 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the decision boundary
    ax.plot_surface(x1_grid, x2_grid, x3_grid, alpha=0.7, cmap='viridis', edgecolor='none')

    # Plot the data points
    class_1 = data[data[yVarName] == 1]
    class_2 = data[data[yVarName] == -1]
    ax.scatter(class_1.iloc[:, 0], class_1.iloc[:, 1], class_1.iloc[:, 2], label="Class +1", color="green", marker="o")
    ax.scatter(class_2.iloc[:, 0], class_2.iloc[:, 1], class_2.iloc[:, 2], label="Class -1", color="red", marker="x")

    # Add labels and legend
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("x3")
    ax.set_title("3D Decision Boundary")
    ax.legend()

    # Show the plot
    plt.show()

plot_3dGraphs('Outcome')

#printing the equation
print(f'b = {b}')
print(f'Final Equation: ', end='')
for i in range(varCount):
    if W[i] > 0 :
        if i != 0:
            print(" + ", end='')
        print(f'{abs(W[i])}', end='')
    else:
        if i != 0:
            print(f' - {abs(W[i])}', end='')
        else:
            print(f'-{abs(W[i])}', end='')
    print(f'x{i+1}', end='')
if b > 0:
    print(' + ', end='')
print(f'{b} = 0')

margin = 2/margin_W
print(f"Margin: {margin}")
