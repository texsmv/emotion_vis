import math
import numpy as np
from .utils import is_array_like, zNormalize, clean_nan_inf, to_np_array
from matrixprofile.algorithms.pairwise_dist import pairwise_dist
from scipy.spatial.distance import squareform
import mass_ts as mass_ts



def zNormalize_euclidian(tsA,tsB):
    """
    Returns the z-normalized Euclidian distance between two time series.

    Parameters
    ----------
    tsA: Time series #1
    tsB: Time series #2
    
    from https://github.com/matrix-profile-foundation/matrixprofile repository
    """

    if len(tsA) != len(tsB):
        raise ValueError("tsA and tsB must be the same length")

    return np.linalg.norm(zNormalize(tsA.astype("float64")) - zNormalize(tsB.astype("float64")))

def _sequences_indexes(ts, L, begin = 0):
    n = len(ts)
    if n < L:
        return []
    return list(range(begin, begin + n - L + 1))

def subsequences_indexes(ts, L):
    """
    Get the indexes of subsequences not containing Nan values in the 
    time serie ts with a window of length L.

    Args:
        ts (Tuple, list or np.ndarray): temporal serie
        L (int): length of window

    Returns:
        List: indexes of the start of each subsequence of length L
    """
    if not is_array_like(ts):
        return ValueError("Time serie structure is not an array")
    isMissing = (np.isinf(ts) | np.isnan(ts))
    n = len(ts)
    
    isLastValueNan = True
    firstIndex = None
    lastIndex = None
    currIndex = 0
    sequencesIndexes = []
    
    while currIndex != n:
        if isMissing[currIndex]:
            lastIndex = currIndex
        if isLastValueNan and not isMissing[currIndex]:
            firstIndex = currIndex
        if firstIndex != None and lastIndex != None:
            if(lastIndex > firstIndex):
                indexes = _sequences_indexes(ts[firstIndex: lastIndex], L, begin=firstIndex)
                sequencesIndexes = sequencesIndexes + indexes
                firstIndex = None
                lastIndex = None
        
        isLastValueNan = isMissing[currIndex]
        currIndex = currIndex + 1
    if firstIndex != None:
        indexes = _sequences_indexes(ts[firstIndex: ], L, begin=firstIndex)
        sequencesIndexes = sequencesIndexes + indexes
    return sequencesIndexes


def naive_distance_profile(tsA,idx,m, searchIndexes = None, tsB = None):
    """
    Returns the distance profile of a query within tsA against the time 
    series tsB using the naive all-pairs comparison.

    Parameters
    ----------
    tsA: Time series containing the query for which to calculate the distance profile.
    idx: Starting location of the query within tsA
    m: Length of query.
    tsB: Time series to compare the query against. Note that, if no value is provided, tsB = tsA by default.
    searchIndexes: sequences indexes of tsB that don't have Nan or Inf values
    
    modified from https://github.com/matrix-profile-foundation/matrixprofile repository
    """

    selfJoin = False
    if tsB is None:
        selfJoin = True
        tsB = tsA

    query = tsA[idx: (idx+m)]
    distanceProfile = []
    n = len(tsB)
    
    if is_array_like(searchIndexes):
        indexes = searchIndexes
    else:
        indexes = list(range(n-m+1))
    for i in searchIndexes:
        distanceProfile.append(zNormalize_euclidian(query,tsB[i:i+m]))

    dp = np.array(distanceProfile)
    
    if selfJoin:
        for i in range(len(indexes)):
            if abs(indexes[i] - idx) <= np.round(m/2):
                dp[i] = np.inf
    return dp, indexes

def mass_distance_profile(tsA,idx,m, searchIndexes = None, tsB = None):
    """
    Returns the distance profile of a query within tsA against the time 
    series tsB using the naive all-pairs comparison.

    Parameters
    ----------
    tsA: Time series containing the query for which to calculate the distance profile.
    idx: Starting location of the query within tsA
    m: Length of query.
    tsB: Time series to compare the query against. Note that, if no value is provided, tsB = tsA by default.
    searchIndexes: sequences indexes of tsB that don't have Nan or Inf values
    
    modified from https://github.com/matrix-profile-foundation/matrixprofile repository
    """

    selfJoin = False
    if tsB is None:
        selfJoin = True
        tsB = tsA

    query = tsA[idx: (idx+m)]
    distanceProfile = np.array([])
    distanceProfileIds = []
    
    if is_array_like(searchIndexes):
        indexes = searchIndexes
    else:
        n = len(tsB)
        indexes = list(range(n-m+1))
    if len(indexes) == 0:
        # todo check this
        return [np.Inf], [0]
    segmentStart = indexes[0]
    for i in range(1, len(searchIndexes)):
        if indexes[i]!= indexes[i - 1] + 1: 
            segment = tsB[segmentStart: indexes[i - 1] + m]
            segmentDistances = mass_ts.mass(segment, query)
            segmentStart = indexes[i]
            distanceProfile = np.concatenate([distanceProfile, segmentDistances])
    segmentDistances = mass_ts.mass(tsB[segmentStart: indexes[-1] + m], query)
    distanceProfile = np.concatenate([distanceProfile, segmentDistances])

    dp = distanceProfile
    if selfJoin:
        for i in range(len(indexes)):
            if abs(indexes[i] - idx) <= np.round(m/2):
                dp[i] = np.inf
    return dp, indexes

def _self_join_or_not_preprocess(tsA, tsB, m):
    """
    Core method for determining if a self join is occuring and returns appropriate
    profile and index numpy arrays with correct dimensions as all np.nan values.

    Parameters
    ----------
    tsA: Time series containing the queries for which to calculate the Matrix Profile.
    tsB: Time series to compare the query against. Note that, if no value is provided, ts_b = ts_a by default.
    m: Length of subsequence to compare.
    
    from https://github.com/matrix-profile-foundation/matrixprofile repository
    """
    n = len(tsA)
    if tsB is not None:
        n = len(tsB)

    shape = n - m + 1

    return (np.full(shape, np.inf), np.full(shape, np.inf))



def matrix_profile(ts_A, indexes_A, ts_B, indexes_B, L):
    """
    Function for calculating the Matrix Profile

    Args:
        ts_A (List or np.ndarray): Time series containing the queries for which to calculate the Matrix Profile.
        indexes_A (List): Indexes of subsequences of ts_A
        ts_B (List or np.ndarray): Time series to compare the query against. Note that, if no value is provided, ts_b = ts_a by default.
        indexes_B ([type]): Indexes of subsequences of ts_B
        L (int): window length

    Returns:
        np.ndarray : matrix profile distances
    """
    matrix_profile_AB = np.zeros(len(ts_A) - L + 1)
    matrix_profile_AB[:] = np.NaN
    for index_A in indexes_A:
        # dist, _ = naive_distance_profile(ts_A, index_A, L, searchIndexes=indexes_B, tsB = ts_B)
        dist, _ = mass_distance_profile(ts_A, index_A, L, searchIndexes=indexes_B, tsB = ts_B)
        if len(dist) == 0:
            matrix_profile_AB[index_A] = np.Inf
        else:    
            min_pos = np.argmin(dist)
            matrix_profile_AB[index_A] = dist[min_pos]      
    return matrix_profile_AB

def join_matrix_profile(ts_A, indexes_A, ts_B, indexes_B, L):
    """
    Function to obtain an array containing the euclidean distances for each subsequence in ts_A and ts_B to its nearest neighbour in ts_B and ts_A respectively

    Args:
        ts_A (List or np.ndarray): Temporal series #1
        indexes_A (List or np.ndarray): Indexes of subsequences of ts_A
        ts_B (List or np.ndarray): Temporal series #2
        indexes_B (List or np.ndarray): Indexes of subsequences of ts_B
        L (int): window length

    Returns:
        np.ndarray: join matrix profile distances
    """
    matrixProfile_AB = matrix_profile(ts_A, indexes_A, ts_B, indexes_B, L)
    matrixProfile_BA = matrix_profile(ts_B, indexes_B, ts_A, indexes_A, L)

    matrixProfile_ABBA = np.concatenate([matrixProfile_AB , matrixProfile_BA])
    return matrixProfile_ABBA

def calc_MPdist(joinMatrixProfile, dataLength):
    """
    Gets the Matrix Profile Distance of two time series given its joinMatrixProfile and data length

    Args:
        joinMatrixProfile (np.ndarray): euclidean distances
        dataLength (int): quantity of data in ts_A and ts_B

    Returns:
        float: distance
    """
    if not is_array_like(joinMatrixProfile):
        return ValueError("Join matrix profile is not an array")
    
    matrixProfile = to_np_array(joinMatrixProfile)
    
    threshold = 0.05;
    k = math.ceil(threshold * dataLength)
    matrixProfile.sort()
    
    if len(matrixProfile) == 0:
        return np.Inf
    elif len(matrixProfile) >= k:
        return matrixProfile[k]
    else:
        return matrixProfile[len(matrixProfile) - 1]

def mp_distance_matrix(mtseries, variables, alphas, L, n_jobs):
    assert len(variables) == len(alphas)
    
    N = len(mtseries)
    
    # * assumes all mtseries are even and aligned
    d = len(variables)
    T = mtseries[0].timeLen
    
    D_k = np.zeros([d, N, N])
    
    values = []
    for k in range(d):
        for serie in mtseries:
            values = values + [serie.get_serie(variables[k])]
        values =  np.array(values)
        dm = pairwise_dist(values, L, n_jobs=n_jobs)
        print(L)
        print("dm")
        print(dm)
        D_k[k] = squareform(dm)
        
    D_ks =  np.copy(D_k)
    
    for k in range(d):
        D_k[k] = np.power(D_k[k], 2) * (alphas[k] ** 2)
    D = np.sum(D_k, axis=0)
    D = np.power(D, 1/2)
    
    return D, D_ks

    
    
    
    