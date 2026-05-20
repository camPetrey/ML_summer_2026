import matplotlib.pyplot as plt

xData = [1, 2, 3, 4, 5]
yData = [3.1, 4.9, 7.2, 8.8, 11.1]

'''
Mean Squared Error (MSE). What is being reduced using the derivatives with respect to m and b.
'''
def loss_function(m, b, xData, yData):
    totLoss = 0  # accumulate squared error

    for i in range(len(xData)):
        x = xData[i]
        y = yData[i]

        # MSE term for each point
        totLoss += (1/len(xData)) * (y - (m * x + b)) ** 2

    return totLoss


def gradient_decent(m, b, xData, yData, l):
    m_gradient = 0  # dL/dm accumulator
    b_gradient = 0  # dL/db accumulator
    n = len(xData)  # number of samples

    for i in range(n):
        x = xData[i]
        y = yData[i]

        # gradient for m (slope)
        m_gradient += -(2/n) * x * (y - (m * x + b))

        # gradient for b (intercept)
        b_gradient += -(2/n) * (y - (m * x + b))

    # parameter update step (gradient descent)
    m = m - m_gradient * l
    b = b - b_gradient * l

    return m, b


m = 0
b = 0
l = 0.001   # learning rate
epochs = 300  # number of passes over dataset

# training loop
for i in range(epochs):
    m, b = gradient_decent(m, b, xData, yData, l)

# compute predictions using trained parameters
yPredictions = []

for x in xData:
    yPredictions.append(m * x + b)

# plot actual data points
plt.scatter(xData, yData, label="Actual Data")

# plot learned regression line
plt.plot(xData, yPredictions, label="Regression Line")

plt.xlabel("x")
plt.ylabel("y")
plt.title("Linear Regression From Scratch")
plt.legend()

plt.show()