import pickle


class InterpolatedModel():
    
    def __init__(self, name, interp, bounds):
        self.name = name
        self.grid_model = interp
        self.bounds = bounds

    def parameters(self):
        return list(self.bounds.keys())
        
    def save(self, loc):
        pickle.dump(self, open(loc, 'wb'))
        
    def __call__(self, x):
        return self.grid_model(x)
        
    def __repr__(self):
        return 'InterpolatedModel "{}", with bounds {}'.format(self.name, self.bounds)
