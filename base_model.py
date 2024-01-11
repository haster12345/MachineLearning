import numpy as np

class BaseModel:
    def __init__(self, alpha, features_train, target_variables_train) -> None:
        
        self.alpha: float = alpha
        self.features_train_no_intercept: np.array[np.array] = np.array(features_train)
        self.features_shape = np.shape(self.features_train_no_intercept)
        self.features_train = np.c_[np.ones(self.features_shape[0]), self.features_train_no_intercept]
        self.target_variables_train = np.array(target_variables_train)
        self.number_of_outputs: int = len(self.target_variables_train)
        self.number_of_inputs: int = len(self.features_train)
        self.number_of_parameters = len(self.features_train[0])
        self.initial_vector_of_parameters: np.array = np.zeros(self.number_of_parameters)
        self.hypothesis_vector = np.ones(self.number_of_parameters)