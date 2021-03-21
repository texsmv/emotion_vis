import numpy as np
import matplotlib.pyplot as plt
import json
from dateutil import parser


def plot_discords(pattern , discords, L, ax):
    
    colori = 0
    colors = 'rgbcm'
    ms = discords
    l = "Discords"
    # for ms,l in zip(discords,labels):
    c =colors[colori % len(colors)]
    starts = list(ms)
    ends = [min(s + L,len(pattern)-1) for s in starts]
    ax.plot(starts, pattern[starts],  c +'o',  label=l)
    ax.plot(ends, pattern[ends],  c +'o', markerfacecolor='none')
    for nn in ms:
        ax.plot(range(nn,nn+L),pattern[nn:nn+L], c , linewidth=2)
    colori += 1

    ax.plot(pattern, 'k', linewidth=1, label="data")
    ax.legend()

def plot_motifs(pattern , mtfs, L, labels, ax):

    colori = 0
    colors = 'rgbcm'
    for ms,l in zip(mtfs,labels):
        c =colors[colori % len(colors)]
        starts = list(ms)
        ends = [min(s + L,len(pattern)-1) for s in starts]
        ax.plot(starts, pattern[starts],  c +'o',  label=l)
        ax.plot(ends, pattern[ends],  c +'o', markerfacecolor='none')
        for nn in ms:
            ax.plot(range(nn,nn+L),pattern[nn:nn+L], c , linewidth=2)
        colori += 1

    ax.plot(pattern, 'k', linewidth=1, label="data")
    ax.legend()

def plotMotifs(pattern, mp, mtfs, motif_d, L):
    mp_adj = np.append(mp[0],np.zeros(L-1)+np.nan)
    #Plot the signal data
    fig, (ax1, ax2, ax3) = plt.subplots(3,1,sharex=True,figsize=(20,10))
    ax1.plot(np.arange(len(pattern)),pattern, label="Synthetic Data")
    ax1.set_ylabel('Signal', size=22)

    #Plot the Matrix Profile
    ax2.plot(np.arange(len(mp_adj)),mp_adj, label="Matrix Profile", color='red')
    ax2.set_ylabel('Matrix Profile', size=22)

    #Plot the Motifs
    plot_motifs(pattern, mtfs, L, [f"{md:.3f}" for md in motif_d], ax3)
    ax3.set_ylabel('Motifs', size=22)


def plotDiscords(pattern, mp, discords, L):
    mp_adj = np.append(mp[0],np.zeros(L-1)+np.nan)
    #Plot the signal data
    fig, (ax1, ax2, ax3) = plt.subplots(3,1,sharex=True,figsize=(20,10))
    ax1.plot(np.arange(len(pattern)),pattern, label="Synthetic Data")
    ax1.set_ylabel('Signal', size=22)

    #Plot the Matrix Profile
    ax2.plot(np.arange(len(mp_adj)),mp_adj, label="Matrix Profile", color='red')
    ax2.set_ylabel('Matrix Profile', size=22)

    #Plot the Motifs
    plot_discords(pattern, discords, L, ax3)
    ax3.set_ylabel('Discords', size=22)

def is_array_like(a):
    """
    Helper function to determine if a value is array like.
    Parameters
    ----------
    a : obj
        Object to test.
    Returns
    -------
    True or false respectively.
    
    from https://github.com/matrix-profile-foundation/matrixprofile repository
    """
    return isinstance(a, (list, tuple, np.ndarray))


def to_np_array(a):
    """
    Helper function to convert tuple or list to np.ndarray.
    Parameters
    ----------
    a : Tuple, list or np.ndarray
        The object to transform.
    Returns
    -------
    The np.ndarray.
    Raises
    ------
    ValueError
        If a is not a valid type.
        
    from https://github.com/matrix-profile-foundation/matrixprofile repository
    """
    if not is_array_like(a):
        raise ValueError('Unable to convert to np.ndarray!')

    return np.array(a)


def clean_nan_inf(ts):
    """
    Converts tuples & lists to Numpy arrays and replaces nan and inf values with zeros

    Parameters
    ----------
    ts: Time series to clean
    
    from https://github.com/matrix-profile-foundation/matrixprofile repository
    """

    #Convert time series to a Numpy array
    ts = to_np_array(ts)

    search = (np.isinf(ts) | np.isnan(ts))
    ts[search] = 0

    return ts


def zNormalize(ts):
    """
    Returns a z-normalized version of a time series.

    Parameters
    ----------
    ts: Time series to be normalized
    
    from https://github.com/matrix-profile-foundation/matrixprofile repository
    """

    ts -= np.mean(ts)
    std = np.std(ts)

    if std == 0:
        raise ValueError("The Standard Deviation cannot be zero")
    else:
        ts /= std

    return ts

# todo: document this
def mtserieQueryToJsonStr(query):
    assert isinstance(query, dict)
    if isinstance(next(iter(query.values())), np.ndarray):
        newQuery = {}
        for id, series in query.items():
            newQuery[id] = series.tolist()
        return json.dumps(newQuery)
    return json.dumps(query)

""" 
    upon request ranks the time series according to how well each time series separates those subsets.
    
    D_list: list of distance matrix D^2_k
"""
def subsetSeparationRanking(D_list, u_ind, v_ind):
    n = len(u_ind)
    m = len(v_ind)
    js = []
    for D_k in D_list:
        firstTerm = 0
        for i in u_ind:
            for j in v_ind:
                firstTerm = firstTerm + D_k[i][j]
        firstTerm =  firstTerm / (n * m)
        
        s_u = 0
        secondTerm = 0
        for i in u_ind:
            for j in u_ind:
                secondTerm = secondTerm + D_k[i][j]
        s_u = secondTerm / (2 * n)
        secondTerm =  secondTerm / (2 * n * n)
        
        s_v = 0
        thirdTerm = 0
        for i in v_ind:
            for j in v_ind:
                thirdTerm = thirdTerm + D_k[i][j]
        s_v = thirdTerm / (2 * m)
        thirdTerm =  thirdTerm / (2 * m * m)
        
        
        num = firstTerm - secondTerm - thirdTerm
        
        den = s_u + s_v
        
        
        j_k = num / (den + 1e-17)
        
        js = js + [j_k]
    return js

def fishersDiscriminantRanking(D_ks, u_ind, v_ind):
    assert isinstance(D_ks, dict)
    
    n = len(u_ind)
    m = len(v_ind)
    js = {}
    for varName in D_ks.keys():
        D_k = D_ks[varName]
        firstTerm = 0
        for i in u_ind:
            for j in v_ind:
                firstTerm = firstTerm + D_k[i][j]
        firstTerm =  firstTerm / (n * m)
        
        s_u = 0
        secondTerm = 0
        for i in u_ind:
            for j in u_ind:
                secondTerm = secondTerm + D_k[i][j]
        s_u = secondTerm / (2 * n)
        secondTerm =  secondTerm / (2 * n * n)
        
        s_v = 0
        thirdTerm = 0
        for i in v_ind:
            for j in v_ind:
                thirdTerm = thirdTerm + D_k[i][j]
        s_v = thirdTerm / (2 * m)
        thirdTerm =  thirdTerm / (2 * m * m)
        
        
        num = firstTerm - secondTerm - thirdTerm
        
        den = s_u + s_v
        
        
        j_k = num / (den + 1e-17)
        
        js[varName] = j_k
        # js = js + [j_k]
    return js

def _timedelta_unit_to_resample_rule(unit):
    if unit == 'Y':
        return 'A'
    elif unit == 'M':
        return 'M'
    elif unit == 'D':
        return 'D'
    elif unit == 'h':
        return 'H'
    elif unit == 'm':
        return 'T'
    elif unit == 's':
        return 'S'
    

def _timedeltaUnits(timedelta):
    years = timedelta.astype('timedelta64[Y]') / np.timedelta64(1, 'Y')
    months = timedelta.astype('timedelta64[M]') / np.timedelta64(1, 'M')
    days = timedelta.astype('timedelta64[D]') / np.timedelta64(1, 'D')
    hours = timedelta.astype('timedelta64[h]') / np.timedelta64(1, 'h')
    minutes = timedelta.astype('timedelta64[m]') / np.timedelta64(1, 'm')
    seconds = timedelta.astype('timedelta64[s]') / np.timedelta64(1, 's')
    nanoSeconds = timedelta.astype('timedelta64[ns]') / np.timedelta64(1, 'ns')
    return {'years': years, 'months': months, 'days':days, 'hours':hours, 'minutes':minutes, 'seconds':seconds, 'nanoseconds':nanoSeconds}

def allowed_downsample_rule(df):
    timeMedia = df.index.to_series().diff().median().to_numpy()
    timeMediaUnits = _timedeltaUnits(timeMedia)
    
    begin = df.index[0]
    end = df.index[-1]
    timeLength = (end - begin).to_numpy()
    timeLengthUnits = _timedeltaUnits(timeLength)
    
    minUnitSize = 3
    
    units = []
    
    
    if timeMediaUnits['years'] != 0:
        units = []
    
    if timeMediaUnits['months'] != 0 :
        units = ['Y']

    if timeMediaUnits['days'] != 0 and timeLengthUnits['months'] >= minUnitSize:
        units = ['Y', 'M']

    if timeMediaUnits['hours'] != 0 and timeLengthUnits['days'] >= minUnitSize:
        units = ['Y', 'M', 'D']
    
    if timeMediaUnits['minutes'] != 0 and timeLengthUnits['hours'] >= minUnitSize:
        units = ['Y', 'M', 'D', 'h']
    
    if timeMediaUnits['seconds'] != 0 and timeLengthUnits['minutes'] >= minUnitSize:
        units = ['Y', 'M', 'D', 'h', 'm']

    if timeMediaUnits['nanoseconds'] != 0 and timeLengthUnits['seconds'] >= minUnitSize:
        units = ['Y', 'M', 'D', 'h', 'm', 's']

    rules = []
    for r in units:
        if r == 'Y' and timeLengthUnits['years'] >= minUnitSize:
            rules = rules  + [_timedelta_unit_to_resample_rule(r)]
        elif r == 'M' and timeLengthUnits['months'] >= minUnitSize:
            rules = rules  + [_timedelta_unit_to_resample_rule(r)]
        elif r == 'D' and timeLengthUnits['days'] >= minUnitSize:
            rules = rules  + [_timedelta_unit_to_resample_rule(r)]
        elif r == 'h' and timeLengthUnits['hours'] >= minUnitSize:
            rules = rules  + [_timedelta_unit_to_resample_rule(r)]
        elif r == 'm' and timeLengthUnits['minutes'] >= minUnitSize:
            rules = rules  + [_timedelta_unit_to_resample_rule(r)]
        elif r == 's' and timeLengthUnits['seconds'] >= minUnitSize:
            rules = rules  + [_timedelta_unit_to_resample_rule(r)]
            
    return rules


def strToDateTime64(dateStr):
    return np.datetime64(parser.parse(dateStr))

def strToDateTime(dateStr):
    return parser.parse(dateStr)
