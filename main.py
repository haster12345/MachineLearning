import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

"""
linear regression:
    - gradient descent
        - stochastic
        - batch
    - solved solution
    - locally weighted linear regression

logistic linear regression
"""

df = pd.read_csv('rent_data/House_Rent_Dataset.csv')

features = df.drop('Rent', axis=1)
target_variables = df['Rent']

features_train, features_test, target_variables_train, target_variables_test = (
    train_test_split(features, target_variables, test_size=0.2, random_state=42))

print(np.array(features_train))


class LinearRegression:

    def __init__(self, alpha, features_train, target_variables_train, number_of_parameters):

        """
        :param alpha: learning_rate

        :param features_train: list of x_i, where x_i is a vector. E.g :For instance, x_i[j] refers to the i-th row and
        j feature (column). In the notes x_i[j] is x_j^(i)

        :param target_variables_train: these are the y-values

        :param number_of_parameters:
        """

        self.alpha: float = alpha

        self.features_train = np.array(features_train)

        self.target_variables_train = np.array(target_variables_train)

        self.number_of_outputs: int = len(self.target_variables_train)
        self.number_of_features: int = len(self.features_train[0])

        self.initial_vector_of_parameters: np.array = np.zeros(number_of_parameters)
        self.hypothesis_vector = np.ones(number_of_parameters)

    def hypothesis_equation(self, vector_of_parameters, row):
        """
        h(x) = theta^T X
        :return: h(X)
        """
        hypothesis_value = 0
        for i in range(self.number_of_features):
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

        for j in range(self.number_of_features):
            theta = vector_of_parameters[j]
            sum_LMS = 0
            for i in range(self.number_of_outputs):
                sum_LMS += ((self.target_variables_train[i] - self.hypothesis_equation(vector_of_parameters, i))
                            * self.features_train[i][j])

            theta = theta * self.alpha * sum_LMS

            if abs(vector_of_parameters[j] - theta) < abs_diff:
                vector_of_parameters[j] = theta
                return vector_of_parameters

            vector_of_parameters[j] = theta
        return vector_of_parameters

    def normal_equations(self):
        pass
# LinearRegression(0.01, )
