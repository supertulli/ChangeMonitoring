import numpy as np
import pandas as pd

from scipy.stats import entropy

def JSD_distance(ref_hist: pd.Series, new_hist: pd.Series):
    P_i, P_j = ref_hist.to_numpy(), new_hist.to_numpy()
    M = (P_i + P_j)/2
    return np.sqrt((entropy(P_i,M, base=2)+entropy(P_j,M, base=2))/2)

def geo_mean(iterable):
    return np.exp(np.log(iterable).mean())

def iter_geo_mean_estimator(new_value: float, order: int, previous_value:float|None = None):
    if order == 1 and previous_value is None:
        return new_value
    else:
        # print("regular calc: ", ((previous_value**(order-1))*new_value)**(1/order))
        # print("log calc: ", np.exp((1/order)*np.log(new_value)+((order-1)/order)*np.log(previous_value)))
        # return ((previous_value**(order-1))*new_value)**(1/order)
        return np.exp((1/order)*np.log(new_value)+((order-1)/order)*np.log(previous_value))