import matplotlib.pyplot as plt


xData = [1, 2, 3, 4, 5]
yData = [3.1, 4.9, 7.2, 8.8, 11.1]

def loss_function(m, b, xData, yData):
    totLoss = 0
    for i in range(len(xData)):
        x = xData[i]
        y = yData[i]

        totLoss += (1/len(xData)) * (y - (m * x + b)) ** 2

    return totLoss

def gradient_decent(m, b, xData, yData, l):
    m_gradient = 0
    b_gradient = 0
    n = len(xData)

    for i in range(n):
        x = xData[i]
        y = yData[i]

        m_gradient += -(2/n) * x * (y - (m * x + b))
        b_gradient += -(2/n) * (y - (m * x + b))

    m = m - m_gradient * l
    b = b - b_gradient * l
    return m, b

m = 0
b = 0
l = 0.001
epochs = 300

for i in range(epochs):
    m, b = gradient_decent(m, b, xData, yData, l)

# Predicted y values
yPredictions = []

for x in xData:
    yPredictions.append(m * x + b)

# Plot
plt.scatter(xData, yData, label="Actual Data")
plt.plot(xData, yPredictions, label="Regression Line")

plt.xlabel("x")
plt.ylabel("y")
plt.title("Linear Regression From Scratch")
plt.legend()

plt.show()