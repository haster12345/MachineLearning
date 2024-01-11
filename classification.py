from base_model import BaseModel
import numpy as np



class LogisticRegression(BaseModel):

    def sigmoid_function(self, vector_of_parameters, row):

        dot_product = 0

        for i in range(self.number_of_parameters):
            dot_product += vector_of_parameters[i] * self.features_train[row][i]

        return 1 / (1 + np.exp(dot_product))

    def threshold_function(self, vector_of_parameters, row):

        dot_product = 0

        for i in range(self.number_of_parameters):
            dot_product += vector_of_parameters[i] * self.features_train[row][i]

        if dot_product >= 0:
            return 1

        else:
            return 0

    def stochastic_gradient_descent(self, abs_diff=0.0001):

        vector_of_parameters = self.initial_vector_of_parameters
        for i in range(self.number_of_outputs):
            for j in range(self.number_of_parameters):

                theta_j = vector_of_parameters[j]
                theta_j = theta_j + self.alpha * (
                            self.target_variables_train[i] - self.threshold_function(vector_of_parameters, i)) * \
                          self.features_train[i][j]

                if abs(vector_of_parameters[j] - theta_j) < abs_diff:
                    vector_of_parameters[j] = theta_j

        return vector_of_parameters


class SoftmaxRegression:

    def __init__(self, alpha, features_train, target_variables_train) -> None:

        self.alpha: float = alpha
        self.features_train_no_intercept: np.array[np.array] = np.array(features_train)
        self.features_shape = np.shape(self.features_train_no_intercept)
        self.features_train = np.c_[np.ones(self.features_shape[0]), self.features_train_no_intercept]
        self.target_variables_train = np.array(target_variables_train)
        self.number_of_outputs: int = len(self.target_variables_train)
        self.number_of_inputs: int = len(self.features_train)
        self.number_of_parameters = len(self.features_train[0])
        self.number_of_examples = self.features_shape[1]
        self.initial_matrix_of_parameters: np.array[np.array] = np.zeros(
            (self.number_of_parameters, self.number_of_outputs))
        self.hypothesis_vector = np.ones(self.number_of_parameters)

    def stochastic_gradient_descent(self, abs_diff=0.00001):

        matrix_of_parameters = np.zeros((self.number_of_parameters, self.number_of_outputs))
        for l in range(self.number_of_outputs):
            vector_of_parameters = self.initial_matrix_of_parameters[l]
            for i in range(self.number_of_examples):
                for j in range(self.number_of_parameters):
                    theta_j = vector_of_parameters[j]
                    theta_j = theta_j + self.alpha * self.deriv_cost_function(vector_of_parameters[j], l)

                    if abs(vector_of_parameters[j] - theta_j) < abs_diff:
                        vector_of_parameters[j] = theta_j
                        continue

            matrix_of_parameters[l] = matrix_of_parameters

        return matrix_of_parameters

    @staticmethod
    def indicator_function(x1, x2):
        if x1 == x2:
            return 1
        else:
            return 0

    def deriv_cost_function(self, vector_of_parameters, k):
        sum = 0
        for i in range(self.number_of_examples):
            sum1 = 1

            for j in range(self.number_of_outputs):
                sum1 += np.exp(np.dot(vector_of_parameters, self.features_train[i]))

            sum += self.features_train[i] * (self.indicator_function(self.target_variables_train[i], k)
                                             - np.exp(np.dot(vector_of_parameters, self.features_train[i])) / (
                                                 sum1
                                             ))
        return -sum