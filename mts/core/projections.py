import numpy as np
from numpy.core.fromnumeric import var
from sklearn import manifold
from .mtserie import MTSerie
from .distances import ts_euclidean_distance, ts_dtw_distance, ts_mp_distance, DistanceType
from .matrix_profile import mp_distance_matrix


def compute_k_distance_matrixes(mtseries, variables = [], distanceType = DistanceType.EUCLIDEAN, L = 10):
    # todo restore mpdist, maybe
    # if distanceType == DistanceType.PDIST:
    #     return mp_distance_matrix(mtseries, variables, alphas, L, 8)
    N = len(mtseries)
    # * assumes all mtseries are even and aligned
    D_k = {}
    for varName in variables:
        D_k[varName] = np.zeros([N, N])
        for i in range(N):
            for j in range(N):
                assert isinstance(mtseries[i], MTSerie)
                if distanceType == DistanceType.EUCLIDEAN:
                    D_k[varName][i][j] = ts_euclidean_distance(mtseries[i].get_serie(varName), mtseries[j].get_serie(varName))
                elif distanceType == DistanceType.DTW:
                    D_k[varName][i][j] = ts_dtw_distance(mtseries[i].get_serie(varName), mtseries[j].get_serie(varName))
        D_k[varName] = np.power(D_k[varName], 2)
    return D_k
    

def compute_distance_matrix(D_k,alphas,N):
    D = np.zeros([N,N])
    for varName in list(D_k.keys()):
        D = D + (D_k[varName] * (alphas[varName] ** 2))
    D = np.power(D, 1/2)
    return D


def distance_matrix(mtseries, variables = [], alphas = [], distanceType = DistanceType.EUCLIDEAN, L = 10):
    """
    Gets Distance Matrix of multivariate time series using euclidean distance on the selected variables and using the provided alphas

    Args:
        mtseries (List of MTSerie): Multivariate time series list
        variables (List of str): Time dependent variables to use
        alphas (List of float): weigth for each variable

    Returns:
        [type]: [description]
    """
    assert len(variables) == len(alphas)
    
    # todo add jobs arguments
    if distanceType == DistanceType.PDIST:
        return mp_distance_matrix(mtseries, variables, alphas, L, 8)
    
    N = len(mtseries)
    
    # * assumes all mtseries are even and aligned
    D = len(variables)
    T = mtseries[0].timeLen
    
    D_k = np.zeros([D, N, N])
    
    for k in range(D):
        varName = variables[k]
        for i in range(N):
            for j in range(N):
                assert isinstance(mtseries[i], MTSerie)
                if distanceType == DistanceType.EUCLIDEAN:
                    D_k[k][i][j] = ts_euclidean_distance(mtseries[i].get_serie(varName), mtseries[j].get_serie(varName))
                elif distanceType == DistanceType.DTW:
                    D_k[k][i][j] = ts_dtw_distance(mtseries[i].get_serie(varName), mtseries[j].get_serie(varName))
                
                #todo remove
                # elif distanceType == DistanceType.PDIST:
                #     D_k[k][i][j] = ts_mp_distance(mtseries[i].get_serie(varName), mtseries[j].get_serie(varName), L)
    D_ks =  np.copy(D_k)
    
    for k in range(D):
        D_k[k] = np.power(D_k[k], 2) * (alphas[k] ** 2)
    D = np.sum(D_k, axis=0)
    D = np.power(D, 1/2)
    
    return D, D_ks

def euclidean_distance_matrix(mtseries, variables, alphas):
    """
    Gets Distance Matrix of multivariate time series using euclidean distance on the selected variables and using the provided alphas

    Args:
        mtseries (List of MTSerie): Multivariate time series list
        variables (List of str): Time dependent variables to use
        alphas (List of float): weigth for each variable

    Returns:
        [type]: [description]
    """
    assert len(variables) == len(alphas)
    
    N = len(mtseries)
    
    # * assumes all mtseries are even and aligned
    D = len(variables)
    T = mtseries[0].timeLen
    
    D_k = np.zeros([D, N, N])
    
    for k in range(D):
        varName = variables[k]
        for i in range(N):
            for j in range(N):
                assert isinstance(mtseries[i], MTSerie)
                # TODO: maybe normalize
                D_k[k][i][j] = ts_euclidean_distance(mtseries[i].get_serie(varName), mtseries[j].get_serie(varName))
    
    D_ks =  np.copy(D_k)
    
    for k in range(D):
        D_k[k] = np.power(D_k[k], 2) * (alphas[k] ** 2)
    D = np.sum(D_k, axis=0)
    D = np.power(D, 1/2)
    
    return D, D_ks

def dtw_distance_matrix(mtseries, variables, alphas):
    """
    Gets Distance Matrix of multivariate time series using dtw distance on the selected variables and using the provided alphas

    Args:
        mtseries (List of MTSerie): Multivariate time series list
        variables (List of str): Time dependent variables to use
        alphas (List of float): weigth for each variable

    Returns:
        [type]: [description]
    """
    assert len(variables) == len(alphas)
    
    N = len(mtseries)
    
    # * assumes all mtseries are even and aligned
    D = len(variables)
    T = mtseries[0].timeLen
    
    D_k = np.zeros([D, N, N])
    
    for k in range(D):
        varName = variables[k]
        for i in range(N):
            for j in range(N):
                assert isinstance(mtseries[i], MTSerie)
                D_k[k][i][j] = ts_dtw_distance(mtseries[i].get_serie(varName), mtseries[j].get_serie(varName))
    
    D_ks =  np.copy(D_k)
    
    for k in range(D):
        D_k[k] = np.power(D_k[k], 2) * (alphas[k] ** 2)
    D = np.sum(D_k, axis=0)
    D = np.power(D, 1/2)
    
    return D, D_ks

# def mp_distance_matrix(mtseries, variables, alphas, L):
#     """
#     Gets Distance Matrix of multivariate time series using MPdist distance on the selected variables and using the provided alphas

#     Args:
#         mtseries (List of MTSerie): Multivariate time series list
#         variables (List of str): Time dependent variables to use
#         alphas (List of float): weigth for each variable
#         L (int): window size

#     Returns:
#         [type]: [description]
#     """
#     assert len(variables) == len(alphas)
    
#     N = len(mtseries)
    
#     # * assumes all mtseries are even and aligned
#     D = len(variables)
#     T = mtseries[0].timeLen
    
#     D_k = np.zeros([D, N, N])
#     for k in range(D):
#         varName = variables[k]
#         for i in range(N):
#             # print(i)
#             for j in range(N):
#                 # print(j)
#                 assert isinstance(mtseries[i], MTSerie)
#                 # TODO: maybe normalize
#                 D_k[k][i][j] = ts_mp_distance(mtseries[i].get_serie(varName), mtseries[j].get_serie(varName), L)
    
#     D_ks =  np.copy(D_k)
    
#     for k in range(D):
#         D_k[k] = np.power(D_k[k], 2) * (alphas[k] ** 2)
#     D = np.sum(D_k, axis=0)
#     D = np.power(D, 1/2)
    
#     return D, D_ks

def mds_projection(D):
    mds = manifold.MDS(n_components=2, dissimilarity="precomputed", random_state=6)
    results = mds.fit(D)
    return results.embedding_ 

