import numpy as np
import matplotlib.pyplot as plt

input2 = np.linspace(-10, 10, 100)


def sigmoid(X):
    val = 1 / (1 + np.exp(-X))
    return val


output = sigmoid(input2)

print(input2)
print(output)

plt.plot(input2, output, c = 'r')
plt.xlabel("Input")
plt.ylabel("Output")
plt.title("sigmoid function")
plt.show()
