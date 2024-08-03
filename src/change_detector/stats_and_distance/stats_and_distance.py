import numpy as np
import pandas as pd

from scipy.stats import entropy

def JSD_distance(ref_hist: pd.Series, new_hist: pd.Series):
    P_i, P_j = ref_hist.to_numpy(), new_hist.to_numpy()
    M = (P_i + P_j)/2
    return np.sqrt((entropy(P_i,M, base=2)+entropy(P_j,M, base=2))/2)

def geo_mean(iterable):
    return np.exp(np.log(iterable).mean())

def iter_geo_mean_estimator(new_value: float, order: int, previous_value:float|None = None, exp_log:bool = True, order_limit:None|int = None):
    if order_limit is not None:
        order = min(order_limit, order)
    if order == 1 and previous_value is None:
        return new_value
    # print("regular calc: ", ((previous_value**(order-1))*new_value)**(1/order))
    # print("log calc: ", np.exp((1/order)*np.log(new_value)+((order-1)/order)*np.log(previous_value)))
    # return ((previous_value**(order-1))*new_value)**(1/order)
    if exp_log and new_value != 0: # avoid Log(0) by using multiplicative formula
        # print("order:", order)
        # print("new value:",new_value)
        # print("log calc: ", np.exp((1/order)*np.log(new_value)+((order-1)/order)*np.log(previous_value)))
        return np.exp((1/order)*np.log(new_value)+((order-1)/order)*np.log(previous_value))
    else:
        return ((previous_value**(order-1))*new_value)**(1/order)
        