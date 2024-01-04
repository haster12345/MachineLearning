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
        self.hypothesis_vector = np.ones(self.number_of_parameters) 
        self.hypothesis_vector[0] = 0



    def hypothesis_equation(self, vector_of_parameters, row):
        """
        h(x) = theta^T X
        :return: h(X)
        """

        hypothesis_value = 0
        for i in range(self.number_of_parameters):
            hypothesis_value += vector_of_parameters[i] * self.features_train[row][i]

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

    def normal_equations(self):
        pass


class TestModel:

    def __init__(self, target_variables_test, features_test) -> None:
        self.target_variables_test = np.array(target_variables_test)
        self.features_test_no_intercept = np.array(features_test)
        self.features_shape = np.shape(self.features_test_no_intercept)
        self.features_test = np.c_[np.ones(self.features_shape[0]), self.features_test_no_intercept]

        # print(self.features_test)

    def mean_square_error(self):

        vector_of_params_output = LinearRegression(0.0001, features_train=features_train, target_variables_train=target_variables_train).stochastic_gradient_descent()
        hypothesis_value = 0
        sum_se = 0

        for i in range(self.features_shape[0]):
            for j in range(len(vector_of_params_output)):
                hypothesis_value += self.features_test[i][j] + vector_of_params_output[j]
            
            sum_se += (hypothesis_value - self.target_variables_test[i])**2
        
        return sum_se/self.features_shape[0]


x = LinearRegression(0.0001, features_train=features_train, target_variables_train=target_variables_train).batch_gradient_descent()
print(x)

x1 = LinearRegression(0.0001, features_train=features_train, target_variables_train=target_variables_train).stochastic_gradient_descent()
print(x1)

y = TestModel(target_variables_test=target_variables_test, features_test=features_test).mean_square_error()
print(y)
