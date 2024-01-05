import numpy as np

class Optimsiers:

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
        self.hypothesis_vector[0] = 0

    def batch_gradient_descent(self, abs_diff=0.0001):

        vector_of_parameters = self.initial_vector_of_parameters

        for j in range(self.number_of_parameters):
            theta = vector_of_parameters[j]
            sum_LMS = 0
            for i in range(self.number_of_outputs):
                sum_LMS += ((self.target_variables_train[i] - self.hypothesis_equation(vector_of_parameters, i))
                            * self.features_train[i][j])

            theta = theta + self.alpha * sum_LMS

            if abs(vector_of_parameters[j] - theta) < abs_diff:
                vector_of_parameters[j] = theta

                print('convergence error rate reached')
                return vector_of_parameters

            vector_of_parameters[j] = theta
        
        return vector_of_parameters

    def stochastic_gradient_descent(self, abs_diff = 0.0001):

        vector_of_parameters = self.initial_vector_of_parameters

        for j in range(self.number_of_outputs):
            theta_j = vector_of_parameters[j]
            for i in range(self.number_of_outputs):
                theta_j = theta_j +  self.alpha * (self.target_variables_train[i] - self.hypothesis_equation(vector_of_parameters, i )) * self.features_train[i][j]

                if abs(vector_of_parameters[j] - theta_j) < abs_diff:
                    vector_of_parameters[j] = theta_j
                    print('convergence error rate reached')
                    
                    return vector_of_parameters

        return vector_of_parameters

