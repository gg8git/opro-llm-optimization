import numpy as np
import pandas as pd

def oned_data_gen(lbound,ubound,nx):
    w = np.random.randint(lbound, (ubound+1))
    b = np.random.randint(lbound, (ubound+1))

    xs = np.array([np.random.uniform(1, (nx+1)) for _ in range(nx)])
    ys = [(x*w + b + np.random.normal(0, 1)) for x in xs]
    coords = pd.DataFrame({'xs': xs, 'ys': ys})

    return {'weights': w, 'intercept': b, 'coords': coords}


def multid_data_gen(wdim,lbound,ubound,nx):
    w = np.random.randint(lbound, (ubound+1), wdim)
    b = np.random.randint(lbound, (ubound+1))

    xs = np.array([np.random.uniform(1, (nx+1), wdim) for _ in range(nx)])
    ys = [(np.sum([(xi*wi) for xi in x for wi in w]) + b + np.random.normal(0, 1)) for x in xs]
    coords = pd.DataFrame({'xs': [x for x in xs], 'ys': ys})

    return {'weights': w, 'intercept': b, 'coords': coords}