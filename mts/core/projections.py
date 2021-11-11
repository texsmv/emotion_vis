# import numpy as np
# import umap
# from sklearn.manifold import Isomap
# from numpy.core.fromnumeric import var
# from sklearn import manifold
# from .mtserie import MTSerie
# from .distances import ts_euclidean_distance, ts_dtw_distance, ts_mp_distance, ts_lcs_distance, DistanceType
# # from .matrix_profile import mp_distance_matrix
# from enum import Enum


# class ProjectionAlg(Enum):
#     MDS = 0
#     ISOMAP = 1
#     UMAP = 2
#     TSNE = 3


# def compute_k_distance_matrixes(mtseries, variables=[], distanceType=DistanceType.EUCLIDEAN, L=10):
#     # todo restore mpdist, maybe
#     # if distanceType == DistanceType.PDIST:
#     #     return mp_distance_matrix(mtseries, variables, alphas, L, 8)
#     N = len(mtseries)
#     # * assumes all mtseries are even and aligned
#     D_k = {}
#     for varName in variables:
#         D_k[varName] = np.zeros([N, N])
#         print(varName)
#         for i in range(N):
#             for j in range(i, N):
#                 assert isinstance(mtseries[i], MTSerie)
#                 if distanceType == DistanceType.EUCLIDEAN:
#                     distance = ts_euclidean_distance(
#                         mtseries[i].get_serie(varName), mtseries[j].get_serie(varName))
#                 elif distanceType == DistanceType.DTW:
#                     distance = ts_dtw_distance(
#                         mtseries[i].get_serie(varName), mtseries[j].get_serie(varName))
#                 elif distanceType == DistanceType.LCS:
#                     distance = ts_lcs_distance(
#                         mtseries[i].get_serie(varName), mtseries[j].get_serie(varName))
#                 D_k[varName][i][j] = distance
#                 D_k[varName][j][i] = distance
#         D_k[varName] = np.power(D_k[varName], 2)
#     return D_k


# def compute_distance_matrix(D_k, alphas, N):
#     D = np.zeros([N, N])
#     for varName in list(D_k.keys()):
#         D = D + (D_k[varName] * (alphas[varName] ** 2))
#     D = np.power(D, 1/2)
#     return D


# def distance_matrix(mtseries, variables=[], alphas=[], distanceType=DistanceType.EUCLIDEAN, L=10):
#     """
#     Gets Distance Matrix of multivariate time series using euclidean distance on the selected variables and using the provided alphas

#     Args:
#         mtseries (List of MTSerie): Multivariate time series list
#         variables (List of str): Time dependent variables to use
#         alphas (List of float): weigth for each variable

#     Returns:
#         [type]: [description]
#     """
#     assert len(variables) == len(alphas)

#     # todo add jobs arguments
#     if distanceType == DistanceType.PDIST:
#         return mp_distance_matrix(mtseries, variables, alphas, L, 8)

#     N = len(mtseries)

#     # * assumes all mtseries are even and aligned
#     D = len(variables)
#     T = mtseries[0].timeLen

#     D_k = np.zeros([D, N, N])

#     for k in range(D):
#         varName = variables[k]
#         for i in range(N):
#             for j in range(N):
#                 assert isinstance(mtseries[i], MTSerie)
#                 if distanceType == DistanceType.EUCLIDEAN:
#                     D_k[k][i][j] = ts_euclidean_distance(
#                         mtseries[i].get_serie(varName), mtseries[j].get_serie(varName))
#                 elif distanceType == DistanceType.DTW:
#                     D_k[k][i][j] = ts_dtw_distance(mtseries[i].get_serie(
#                         varName), mtseries[j].get_serie(varName))

#                 # todo remove
#                 # elif distanceType == DistanceType.PDIST:
#                 #     D_k[k][i][j] = ts_mp_distance(mtseries[i].get_serie(varName), mtseries[j].get_serie(varName), L)
#     D_ks = np.copy(D_k)

#     for k in range(D):
#         D_k[k] = np.power(D_k[k], 2) * (alphas[k] ** 2)
#     D = np.sum(D_k, axis=0)
#     D = np.power(D, 1/2)

#     return D, D_ks


# def euclidean_distance_matrix(mtseries, variables, alphas):
#     """
#     Gets Distance Matrix of multivariate time series using euclidean distance on the selected variables and using the provided alphas

#     Args:
#         mtseries (List of MTSerie): Multivariate time series list
#         variables (List of str): Time dependent variables to use
#         alphas (List of float): weigth for each variable

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
#             for j in range(N):
#                 assert isinstance(mtseries[i], MTSerie)
#                 # TODO: maybe normalize
#                 D_k[k][i][j] = ts_euclidean_distance(
#                     mtseries[i].get_serie(varName), mtseries[j].get_serie(varName))

#     D_ks = np.copy(D_k)

#     for k in range(D):
#         D_k[k] = np.power(D_k[k], 2) * (alphas[k] ** 2)
#     D = np.sum(D_k, axis=0)
#     D = np.power(D, 1/2)

#     return D, D_ks


# def dtw_distance_matrix(mtseries, variables, alphas):
#     """
#     Gets Distance Matrix of multivariate time series using dtw distance on the selected variables and using the provided alphas

#     Args:
#         mtseries (List of MTSerie): Multivariate time series list
#         variables (List of str): Time dependent variables to use
#         alphas (List of float): weigth for each variable

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
#             for j in range(N):
#                 assert isinstance(mtseries[i], MTSerie)
#                 D_k[k][i][j] = ts_dtw_distance(mtseries[i].get_serie(
#                     varName), mtseries[j].get_serie(varName))

#     D_ks = np.copy(D_k)

#     for k in range(D):
#         D_k[k] = np.power(D_k[k], 2) * (alphas[k] ** 2)
#     D = np.sum(D_k, axis=0)
#     D = np.power(D, 1/2)

#     return D, D_ks

# # def mp_distance_matrix(mtseries, variables, alphas, L):
# #     """
# #     Gets Distance Matrix of multivariate time series using MPdist distance on the selected variables and using the provided alphas

# #     Args:
# #         mtseries (List of MTSerie): Multivariate time series list
# #         variables (List of str): Time dependent variables to use
# #         alphas (List of float): weigth for each variable
# #         L (int): window size

# #     Returns:
# #         [type]: [description]
# #     """
# #     assert len(variables) == len(alphas)

# #     N = len(mtseries)

# #     # * assumes all mtseries are even and aligned
# #     D = len(variables)
# #     T = mtseries[0].timeLen

# #     D_k = np.zeros([D, N, N])
# #     for k in range(D):
# #         varName = variables[k]
# #         for i in range(N):
# #             # print(i)
# #             for j in range(N):
# #                 # print(j)
# #                 assert isinstance(mtseries[i], MTSerie)
# #                 # TODO: maybe normalize
# #                 D_k[k][i][j] = ts_mp_distance(mtseries[i].get_serie(varName), mtseries[j].get_serie(varName), L)

# #     D_ks =  np.copy(D_k)

# #     for k in range(D):
# #         D_k[k] = np.power(D_k[k], 2) * (alphas[k] ** 2)
# #     D = np.sum(D_k, axis=0)
# #     D = np.power(D, 1/2)

# #     return D, D_ks


# randomState = 2021


# def mds_projection(D: np.ndarray):
#     mds = manifold.MDS(
#         n_components=2,
#         dissimilarity="precomputed",
#         random_state=randomState
#     )
#     print(D)
#     print(D.shape)
#     print(D.dtype)
#     results = mds.fit(D.astype(np.float64))
#     return results.embedding_


# def tsne_projection(D, perplexity: int = 5):
#     model = manifold.TSNE(
#         n_components=2,
#         random_state=randomState,
#         metric='precomputed',
#         perplexity=perplexity
#     )
#     coords = model.fit_transform(D)
#     return coords


# def umap_projection(D, n_neighbors: int = 5):
#     reducer = umap.UMAP(
#         metric='precomputed',
#         n_neighbors=n_neighbors,
#         random_state=randomState
#     )
#     coords = reducer.fit_transform(D)
#     return coords


# def isomap_projection(D, n_neighbors: int = 5):
#     embedding = Isomap(
#         n_components=2,
#         n_neighbors=n_neighbors,
#         metric='precomputed',
#     )
#     coords = embedding.fit_transform(D)
#     return coords
