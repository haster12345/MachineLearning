import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from dummy_data import dummy_data_fn
from dummy_data import dummy_data_classication_fn
from linear_regression import LinearRegression
from testmodel import TestModel


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


    x = LinearRegression(0.0001, features_train=features_train, target_variables_train=target_variables_train
                         ).batch_gradient_descent()
    print(x)

    x1 = LinearRegression(0.0001, features_train=features_train, target_variables_train=target_variables_train
                          ).stochastic_gradient_descent()
    print(x1)

    y = TestModel(target_variables_test=target_variables_test, features_test=features_test
                  ).mean_square_error(features_train, target_variables)
    print(y)

    y1 = TestModel(target_variables_test=target_variables_test_class, features_test=features_test_class
                   ).mean_square_error(features_train_class, target_variables_class)    

    print(y1)
