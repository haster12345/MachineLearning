import numpy as np

class BaseModel:
    def __init__(self, alpha, features_train, target_variables_train) -> None:
        
        self._alpha: float = alpha
        self._features_train_no_intercept: np.array[np.array] = np.array(features_train)
        self.target_variables_train = np.array(target_variables_train)
        self.initial_vector_of_parameters: np.array = np.zeros(self.number_of_parameters)
        self.hypothesis_vector = np.ones(self.number_of_parameters)

    @property
    def alpha(self):
        return self._alpha

    @property
    def features_train_no_intercept(self):
        return self._features_train_no_intercept

    @property
    def features_shape(self):
        return np.shape(self.features_train_no_intercept)

    @property
    def features_train(self):
        return np.c_[np.ones(self.features_shape[0]), self.features_train_no_intercept]
    
    @property
    def number_of_outputs(self):
        return len(self.target_variables_train)
    
    @property
    def number_of_inputs(self):
        """
        number of examples
        """
        return len(self.features_train)
    
    @property
    def number_of_parameters(self):
        return len(self.features_train[0])
    
    @property
    def hypothesis_vector(self):
        return np.ones(self.number_of_parameters)