import numpy as np
import pandas as pd
import scipy as scipy

def get_ffme_returns():
    """
    Load Fama French Data Set
    """
    me_m = pd.read_csv('/Users/tiuchienyi/Desktop/Courses/EDHEC\Data\Analytics/Personal\Coding\Projects/data/Portfolios_Formed_on_ME_monthly_EW.csv',
               header=0, index_col=0, na_values=-99.99)
    rets = me_m[['Lo 10', 'Hi 10']]
    rets.columns = ['SmallCap','LargeCap']
    rets = rets/100
    rets.index = pd.to_datetime(rets.index, format='%Y%M').to_period('M')
    return rets

def get_hfi_returns():
    """
    Load and Format EDHEC Hedge Fund Return Index
    """
    hfi = pd.read_csv('/Users/tiuchienyi/Desktop/data/edhec-hedgefundindices.csv',
                     header=0, index_col=0, parse_dates=True, infer_datetime_format=True)
    hfi = hfi/100
    hfi.index = hfi.index.to_period('M')
    return hfi

def semideviation(r):
    """
    Semideviation of r that is negative
    """
    is_negative = r<0
    return r[is_negative].std(ddof=0)

def var_historic(r, level=5):
    """
    VaR Historic
    """
    if isinstance(r, pd.DataFrame):
        return r.aggregate(var_historic, level=level)
    elif isinstance(r, pd.Series):
        return -np.percentile(r, level)
    else:
        raise TypeError('Expected r to be Series or DataFrame')

from scipy.stats import norm
def var_gaussian(r, level=5, modified=False):
    """
    Returns the Parametric Gaussian VaR of a Series or DataFrame
    Computing Z Score assuming it is Gaussian
    If "modified" is True, then the modified VaR is returned using the Cornish-Fisher modification
    """
    z = norm.ppf(level/100)
    if modified:
        s = skewness(r)
        k = kurtosis(r)
        z = (z+
             (z**2 - 1)*s/6 +
             (z**3 -3*z) * (k-3) /24 -
             (2**z**3 - 5*z)*(s**2)/36
            )
    return -(r.mean() + z*r.std(ddof = 0))

def cvar_historic(r, level=5):
    """
    Computes the Conditional VaR of Series or DataFrame
    """
    if isinstance(r, pd.Series):
        is_beyond = r<= -var_historic(r, level=level)
        return -r[is_beyond].mean()
    elif isinstance(r, pd.DataFrame):
        return r.aggregate(cvar_historic, level=level)
    else:
        raise TypeError("Expected r to be a Series or DataFrame")
    
def skewness(r):
    """
    Define Skewness
    """
    demeaned_r = r - r.mean()
    sigma_r = r.std(ddof=0)
    exp = (demeaned_r**3).mean()
    return exp/sigma_r**3

def kurtosis(r):
    """
    Define Kurtosis
    """
    demeaned_r = r - r.mean()
    sigma_r = r.std(ddof=0)
    exp = (demeaned_r**4).mean()
    return exp/sigma_r**4

def is_normal(r, level=0.01):
    """
    Apply Jarque Bera test to see if data is normally distributed
    """
    statistics, p_value = scipy.stats.jarque_bera(r)
    return p_value > level