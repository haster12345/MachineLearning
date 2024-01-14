from base_model import BaseModel
import numpy as np

class LinearRegression(BaseModel):

    def hypothesis_equation(self, vector_of_parameters, row):
        """
        h(x) = theta^T X
        """
        
        hypothesis_value = np.dot(vector_of_parameters, self.features_train[row])
        return hypothesis_value

    def ordinary_least_squares(self, vector_of_parameters):
        """
        J(theta) = 0.5 * sum{h(x^(i) - y^(i))^2}
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
        for i in range(self.number_of_outputs):
            for j in range(self.number_of_parameters):
                theta_j = vector_of_parameters[j]
                theta_j = theta_j +  self.alpha * (self.target_variables_train[i] - self.hypothesis_equation(vector_of_parameters, i )) * self.features_train[i][j]

                if abs(vector_of_parameters[j] - theta_j) < abs_diff:
                    vector_of_parameters[j] = theta_j
                    print('convergence error rate reached')
                    
                    return vector_of_parameters
        return vector_of_parameters

    def newton_method(self, function, function_deriv, vector_of_parameters):
        for j in range(len(vector_of_parameters)):
            # theta_j = vector_of_parameters[j]
            vector_of_parameters[j] = vector_of_parameters[j] - (function(vector_of_parameters)/function_deriv(vector_of_parameters))
        return vector_of_parameters 


    def weights(self):
        pass

    def locally_weighted(self):
        pass

    def normal_equations(self):

        x_T_y = np.mat_mul(self.features_train.T, self.target_variables_train)
        inverse = np.linalg.inv(np.mat_mul(self.features_train.T, self.features_train))
        self.initial_vector_of_parameters = np.mat_mul(inverse,x_T_y)
        
        return self.initial_vector_of_parameters

