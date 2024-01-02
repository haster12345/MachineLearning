import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from dummy_data import dummy_data_fn

"""
linear regression:
    - gradient descent
        - stochastic
        - batch
    - solved solution
    - locally weighted linear regression

logistic linear regression
"""

# df = pd.read_csv('rent_data/House_Rent_dataset.csv')

df = dummy_data_fn()
features = df.drop('sinxy_values', axis=1)
target_variables = df['sinxy_values']

features_train, features_test, target_variables_train, target_variables_test = (
    train_test_split(features, target_variables, test_size=0.2, random_state=42))

# print(len(features_train))

class Data:
    def __init__(self, path) -> None:
        # self.df = pd.read_csv(path)
        self.df = dummy_data_fn()
        self.feature = self.df.drop('sinxy_values', axis=1)
        self.target_variables = self.df['sinxy_values']

class LinearRegression:

    def __init__(self, alpha, features_train, target_variables_train):
        """
        :param alpha: learning_rate

        :param features_train: list of x_i, where x_i is a vector. E.g :For instance, x_i[j] refers to the i-th row and
        j feature (column). In the notes x_i[j] is x_j^(i)

        :param target_variables_train: these are the y-values

        :param number_of_parameters:
        """

        self.alpha: float = alpha

        self.features_train_no_intercept: np.array[np.array] = np.array(features_train)
        self.features_shape = np.shape(self.features_train_no_intercept)

        self.features_train = np.c_[np.ones(self.features_shape[0]), self.features_train_no_intercept]

        
        self.target_variables_train = np.array(target_variables_train)

        self.number_of_outputs: int = len(self.target_variables_train)
        self.number_of_inputs: int = len(self.features_train)

        self.number_of_parameters = len(self.features_train[0])

        self.initial_vector_of_parameters: np.array = np.zeros(self.number_of_parameters)
        # self.initial_vector_of_parameters[0] = 1

        self.hypothesis_vector = np.ones(self.number_of_parameters) 


    def hypothesis_equation(self, vector_of_parameters, row):
        """
        h(x) = theta^T X
        :return: h(X)
        """

        # print(self.features_train[row])

        hypothesis_value = 0
        for i in range(self.number_of_parameters):
            hypothesis_value += vector_of_parameters[i] * self.features_train[row][i]

        # print(hypothesis_value)
        return hypothesis_value

    def ordinary_least_squares(self, vector_of_parameters):
        """
        J(theta) = 0.5 * sum{h(x^(i) - y^(i))^2}
        :return:  J(theta)
        """

        OLS = 0  # J(theta)
        for row_num in range(len(self.features_train)):
            OLS += 0.5 * (
                    self.hypothesis_equation(vector_of_parameters, row_num) - self.target_variables_train[row_num]) ** 2
        return OLS

    def batch_gradient_descent(self, abs_diff=0.0001):

        vector_of_parameters = self.initial_vector_of_parameters

        for j in range(self.number_of_parameters):
            print(vector_of_parameters)
            theta = vector_of_parameters[j]
            sum_LMS = 0
            for i in range(self.number_of_outputs):
                sum_LMS += ((self.target_variables_train[i] - self.hypothesis_equation(vector_of_parameters, i))
                            * self.features_train[i][j])

            theta = theta + self.alpha * sum_LMS

            # print(abs(vector_of_parameters[j] - theta))

            if abs(vector_of_parameters[j] - theta) < abs_diff:
                vector_of_parameters[j] = theta

                print('convergence error rate reached')
                return vector_of_parameters

            vector_of_parameters[j] = theta
        
        return vector_of_parameters

    def normal_equations(self):
        pass

    def mean_square_error(self):
        pass

x = LinearRegression(0.01, features_train=features_train, target_variables_train=target_variables_train).batch_gradient_descent()
print(x)
