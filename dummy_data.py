import numpy as np
import random
import pandas as pd


def dummy_data_fn():
    sinxy = lambda x,y : np.sin((x*y))

    x_values = [random.uniform(0, np.pi/2) for _ in range(1000)]

    y_values = [random.uniform(0, np.pi/2) for _ in range(1000)]

    xy_values = []

    for i in range(1000):
        xy_values.append((x_values[i], y_values[i]))

    sinxy_values = [sinxy(x,y) for (x,y) in xy_values ]

    data_dict = {'x_values' : x_values, 'y_values': y_values, 'sinxy_values': sinxy_values}

    df = pd.DataFrame(data = data_dict) 

    return df

def polarised_fn(x, y):
    if x*y >= 0:
        return 1
    else:
        return 0


def dummy_data_classication_fn():

    x_values = [random.uniform(-1, 1) for _ in range(1000)]
    y_values = [random.uniform(-1,-1) for _ in range(1000)]
    
    xy_values = []

    for i in range(1000):
        xy_values.append((x_values[i], y_values[i]))

    pol_values = [polarised_fn(x,y) for (x,y) in xy_values]

    data_dict = {'x_values' : x_values, 'y_values': y_values, 'polxy_values': pol_values}

    df = pd.DataFrame(data = data_dict) 

    return df 