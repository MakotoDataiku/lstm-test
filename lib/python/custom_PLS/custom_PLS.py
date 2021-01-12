from sklearn.cross_decomposition import PLSRegression

class PLS():
    def __init__(self, n_components=2, scale=True, max_iter=500, tol=1e-06, copy=True):
        self.pls = PLSRegression(n_components, scale, max_iter, tol, copy)
        
    
    def fit(self, X, Y):
        self.pls.fit(X, Y)
        return self.pls
    
    def predict(self, X, copy=True):
        return self.pls.predict(X, copy).flatten()
    
    def score(self, X, Y, sample_weight=None):
        return self.pls.score(X, Y, sample_weight)
    
    def get_params(self, **parameters):
        self.pls.set_params(**parameters)
        return self
    
    @property
    def intercept_(self):
        return 0
    
    @property
    def coeff_(self):
        return self.pls.coef_