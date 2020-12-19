from numpy import exp, log, sum

def swish(self, x):
    return x / (1 + np.exp(-x))


def relu(self, x):
    return max(0.0, x)


# sigmoid function returns a probability between 0 and 1
def sigmoid(self, x):
    return 1 / (1 + exp(-x))


def softplus(self, x, limit=30):
    if x >= limit:
        return x
    return log(1 + exp(x), exp(1))


def softmax(self, x, theta=1.0, axis=None):
    e_x = exp(x)
    sum_e_x = sum(exp(x), axis=0)

    return e_x / sum_e_x