import numpy as np

class RegularizedRegressor:
    def __init__(self, l = 0.01):
        self.l = l

    def combine(self, inputs):
        return sum([i*w for (i,w) in zip([1] + inputs, self.weights)])

    def predict(self, X):
        return [self.combine(x) for x in X]

    def classify(self, inputs):
        return sign(self.predict(inputs))

    def fit(self, X, y, **kwargs):
        self.l = kwargs['l']
        X = np.matrix(X)
        y = np.matrix(y)
        W = (X.transpose() * X).getI() * X.transpose() * y

        self.weights = [w[0] for w in W.tolist()]

    def get_params(self, deep = False):
        return {'l':self.l}
