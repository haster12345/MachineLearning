import numpy as np
from linear_regression import LinearRegression
from classification import LogisticRegression


class TestModel:

    def __init__(self, target_variables_test, features_test) -> None:
        self.target_variables_test = np.array(target_variables_test)
        self.features_test_no_intercept = np.array(features_test)
        self.features_shape = np.shape(self.features_test_no_intercept)
        self.features_test = np.c_[np.ones(self.features_shape[0]), self.features_test_no_intercept]

        # print(self.features_test)

    def model(self):
        pass    

    def mean_square_error(self, features_train, target_variables_train):

        vector_of_params_output = LinearRegression(0.0001, features_train=features_train, target_variables_train=target_variables_train).stochastic_gradient_descent()
        hypothesis_value = 0
        sum_se = 0

        for i in range(self.features_shape[0]):
            for j in range(len(vector_of_params_output)):
                hypothesis_value += self.features_test[i][j] + vector_of_params_output[j]
            
            sum_se += (hypothesis_value - self.target_variables_test[i])**2
        
        return sum_se/self.features_shape[0]