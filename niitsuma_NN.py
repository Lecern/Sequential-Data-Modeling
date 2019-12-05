import numpy as np
from numpy import tanh


class SimpleNeuralNetwork:
    def __init__(self, hidden_size, output_size, W_1, W_2, lr):
        self.b_1 = np.matrix(np.zeros(hidden_size))
        self.b_2 = np.matrix(np.zeros(output_size))
        self.W_1 = W_1
        self.W_2 = W_2
        self.lr = lr

    def train(self, X, y):
        Z_2 = X @ self.W_1 + self.b_1
        L_2 = tanh(Z_2)
        Z_3 = L_2 @ self.W_2.T + self.b_2
        output = tanh(Z_3)

        delta_3 = (y - output)
        delta_2 = delta_3 @ self._tanh_grad(Z_3) @ L_2
        delta_2_b = delta_3 @ self._tanh_grad(Z_3)
        delta_1 = delta_2 @ self._tanh_grad(Z_2).T @ X.reshape(1, -1)
        delta_1_b = delta_2 @ self._tanh_grad(Z_2).T

        self.W_1 -= self.lr * delta_1
        self.b_1 -= self.lr * delta_1_b

        self.W_2 -= self.lr * delta_2
        self.b_2 -= self.lr * delta_2_b

        print("y' = {}, W_1 = {}, b_1 = {}, W_2 = {}, b_2 = {}".format(output, self.W_1, self.b_1, self.W_2, self.b_2))

    def _tanh_grad(self, Z):
        return 1 - np.multiply(tanh(Z), tanh(Z))


def build_X(data):
    features = []
    for sent in data:
        words = sent.split()
        features.append([WORD_TO_FEATURE[word] for word in words])
    return np.array(features)


WORD_TO_FEATURE = {
    "very": 1,
    "not": 0,
    "good": 1,
    "bad": 0
}

STUDENT_ID = input("please, input a your id:")
print("Your ID is {}".format(STUDENT_ID))

b_3, b_2, b_1 = [int(sid) for sid in STUDENT_ID[-3:]]
print("b_1 = {}, b_2 = {}, b_3 = {}".format(b_1, b_2, b_3))

w_1_1 = -(b_2 + 1) / 10
w_1_2 = (b_1 + 1) / 10
w_1_3 = -(b_3 + 1) / 10
w_1_4 = (b_3 + 1) / 10

w_2_1 = (b_1 + 1) / 10
w_2_2 = -(b_2 + 1) / 10


INIT_W_1 = np.matrix([[w_1_1, w_1_2],
                      [w_1_3, w_1_4]])
INIT_W_2 = np.matrix([w_2_1, w_2_2])

DATA = ["very good", "not good", "not bad", "very bad"]
LABELS = [1, -1, 1, -1]
INPUT = build_X(DATA)
INIT_LR = 1
HIDDEN_SIZE = 2
OUTPUT_SIZE = 1

print("INPUT = {}".format(INPUT))
print("W_1 = {}, W_2 = {}, X = {}".format(INIT_W_1, INIT_W_2, INPUT))

nn_model = SimpleNeuralNetwork(HIDDEN_SIZE, OUTPUT_SIZE, INIT_W_1, INIT_W_2, INIT_LR)
for n, (input_X, input_y) in enumerate(zip(INPUT, LABELS), start=1):
    print("Iteration {}:".format(n))
    nn_model.train(input_X, input_y)
