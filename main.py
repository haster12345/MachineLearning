import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from dummy_data import dummy_data_fn
from dummy_data import dummy_data_classication_fn
from linear_regression import LinearRegression
from testmodel import TestModel
from base_model import BaseModel


if __name__ == '__main__':

    # df = pd.read_csv('rent_data/House_Rent_dataset.csv')

    df = dummy_data_fn()
    features = df.drop('sinxy_values', axis=1)
    target_variables = df['sinxy_values']

    features_train, features_test, target_variables_train, target_variables_test = (
        train_test_split(features, target_variables, test_size=0.2, random_state=42))

    df_class = dummy_data_classication_fn()
    features_class = df_class.drop('polxy_values', axis=1)
    target_variables_class = df_class['polxy_values']

    features_train_class, features_test_class, target_variables_train_class, target_variables_test_class = (
        train_test_split(features_class, target_variables_class, test_size=0.2,random_state=42 )
    )

    
    LinearRegression_instance = LinearRegression(0.0001, features_train=features_train, target_variables_train=target_variables_train)
    
    TestModel_instance = TestModel(target_variables_test=target_variables_test, features_test=features_test)

    TestModel_instance_classification = TestModel(target_variables_class, features_test_class)

    BaseModel_instance = BaseModel(0.0001, features_train, target_variables_train)

    x = LinearRegression_instance.batch_gradient_descent()
    
    x1 = LinearRegression_instance.stochastic_gradient_descent()
    
    y = TestModel_instance.mean_square_error(features_train, target_variables_train)

    y1 = TestModel_instance_classification.mean_square_error(features_train_class, target_variables_class)    

    x2 = BaseModel_instance.alpha
    
    x3 = LinearRegression_instance.normal_equations()

    print('batch gradient descent: ',x)
    print('stochastic gradient descent: ',x1)
    print('alpha: ',x2)
    print('normal equations solution: ',x3)
    print('mse : ',y)
    print('mse classification data: ',y1)