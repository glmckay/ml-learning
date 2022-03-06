import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


class Network:
    def __init__(self, inputs, outputs, hidden_layers):
        self.layers = [inputs] + hidden_layers + [outputs]
        self.weights = [
            np.random.normal(size=(m, n))
            for m, n in zip(hidden_layers + [outputs], [inputs] + hidden_layers)
        ]
        self.biases = [np.random.normal(size=(n, 1)) for n in hidden_layers + [outputs]]

    def compute(self, input):
        if len(input) != self.layers[0]:
            raise ValueError("Wrong sized inputs")

        v = input
        for w, b in zip(self.weights, self.biases):
            v = sigmoid(w.dot(v) + b)
        return v

    def evaluate(self, tests):
        return sum(
            1 for inputs, label in tests if np.argmax(self.compute(inputs)) == label
        )

    def backprop(self, input, expected):
        def sigmoid_deriv(sigmoid_z):
            return sigmoid_z * (1 - sigmoid_z)

        def cost_deriv(actual, expected):
            return actual - expected  # scaled by 1/2

        # vs are the activation values of the nodes
        vs = [input]
        for w, b in zip(self.weights, self.biases):
            vs.append(sigmoid(w.dot(vs[-1]) + b))

        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]

        delta = cost_deriv(vs[-1], expected)# * sigmoid_deriv(vs[-1])

        for i in range(len(self.layers) - 2, -1, -1):
            nabla_w[i] = delta.dot(vs[i].transpose())
            nabla_b[i] = delta
            if i == 0:
                break
            delta = self.weights[i].transpose().dot(delta)# * sigmoid_deriv(vs[i])

        return nabla_w, nabla_b

    def train(self, examples, eta):

        nabla_w_sum = [np.zeros(w.shape) for w in self.weights]
        nabla_b_sum = [np.zeros(b.shape) for b in self.biases]

        num_examples = 0
        for inputs, expected in examples:
            nabla_w, nabla_b = self.backprop(inputs, expected)
            for l in range(len(self.layers) - 1):
                nabla_w_sum[l] += nabla_w[l]
                nabla_b_sum[l] += nabla_b[l]
            num_examples += 1

        for l in range(len(self.layers) - 1):
            self.weights[l] -= (eta / num_examples) * nabla_w_sum[l]
            self.biases[l] -= (eta / num_examples) * nabla_b_sum[l]
