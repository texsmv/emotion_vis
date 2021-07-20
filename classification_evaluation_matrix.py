import sys
import numpy as np
import argparse

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from sktime.classification.compose import ColumnEnsembleClassifier
from sktime.classification.dictionary_based import BOSSEnsemble
from sktime.classification.interval_based import TimeSeriesForestClassifier
from sktime.classification.shapelet_based import MrSEQLClassifier
from sktime.datasets import load_basic_motions
from sktime.transformations.panel.compose import ColumnConcatenator
from typing import Dict, Any, List
from scipy.spatial.distance import pdist, cdist, squareform
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from umap import UMAP
from sktime.utils.data_io import load_from_tsfile_to_dataframe
from math import *

# import tensorflow as tf
# import numpy as np
# from keras.models import Sequential
# from keras.layers import LSTM
# from keras.layers import Dense
# from keras.layers import RepeatVector
# from keras.layers import TimeDistributed
# from keras import Model
# from keras import backend as K


# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.9
# config.gpu_options.allow_growth = True
# tf.keras.backend.set_session(tf.Session(config=config));



np.random.seed(0)
# sys.path.insert(1, '/home/texs/Documents/Kusisqa/emotion_vis_server')


# eps = np.float(K.epsilon())

def perSerieDistance(x_1, x_2):
    D = x_1.shape[1]
    distance = 0
    for i in range(D):
        distance += sqrt(1 * pow(np.linalg.norm(x_1[:, i] - x_2[:, i]), 2))
    return distance

def scaleSerie(data, scaler = None):
    if(scaler == None):
        scaler = MinMaxScaler()
        scaler.fit(data)
    scaler_data = scaler.transform(data)
    return np.clip(scaler_data, 0, 1), scaler

def labelInt(labels):
    for i in range(len(labels)):
        if labels[i] == 1:
            return i

def compute_test_scores(pred_class, Yte):
    """
    Wrapper to compute classification accuracy and F1 score
    """
    true_class = np.argmax(Yte, axis=1)
    
    accuracy = accuracy_score(true_class, pred_class)
    if Yte.shape[1] > 2:
        f1 = f1_score(true_class, pred_class, average='weighted')
        recall = recall_score(true_class, pred_class, average='weighted')
    else:
        f1 = f1_score(true_class, pred_class, average='weighted')
        recall = recall_score(true_class, pred_class, average='weighted')
    # F1 = 2 * (precision * recall) / (precision + recall)

    return accuracy, recall, f1




def getValues(dataFrame):
    N, D = dataFrame.shape
    T = dataFrame.to_numpy()[0][0].to_numpy().shape[0]
    data = np.zeros((N, D, T), dtype=np.float32)
    # data = np.zeros((N, D, T))
    for i in range(N):
        for j in range(D):
            data[i][j] = dataFrame.to_numpy()[i][j].to_numpy()
    return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "-o", "--alg",
    #     type=str,
    #     required=True,
    #     help="Algorithm"
    # )
    parser.add_argument(
        "-d", "--dataset",
        type=str,
        required=True,
        help="Algorithm"
    )

    args: Dict[str, Any] = vars(parser.parse_args())


    # X, y = load_basic_motions(return_X_y=True)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    # N, D = X_train.shape
    # T = X_train.to_numpy()[0][0].to_numpy().shape[0]
    if args["dataset"] == "motions":
        X, y = load_basic_motions(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

        x_train = getValues(X_train)
        x_train = x_train.transpose([0,2,1])
        x_test = getValues(X_test)
        x_test = x_test.transpose([0,2,1])
        N, D = X_train.shape
        T = X_train.to_numpy()[0][0].to_numpy().shape[0]
        y_train = y_train.to_numpy()
        y_test = y_test.to_numpy()
        N_te = X_test.shape[0]
    else:
        if args["dataset"] == "wafer":
            X_train, y_train = load_from_tsfile_to_dataframe('datasets/Wafer/Wafer_TRAIN.ts')
            X_test, y_test = load_from_tsfile_to_dataframe('datasets/Wafer/Wafer_TRAIN.ts')
        elif args["dataset"] == "libras":
            X_train, y_train = load_from_tsfile_to_dataframe('datasets/Libras/Libras_TRAIN.ts')
            X_test, y_test = load_from_tsfile_to_dataframe('datasets/Libras/Libras_TEST.ts')
        elif args["dataset"] == "uwave":
            X_train, y_train = load_from_tsfile_to_dataframe('datasets/UWaveGestureLibraryAll/UWaveGestureLibraryAll_TRAIN.ts')
            X_test, y_test = load_from_tsfile_to_dataframe('datasets/UWaveGestureLibraryAll/UWaveGestureLibraryAll_TEST.ts')
        elif args["dataset"] == "stand":
            X_train, y_train = load_from_tsfile_to_dataframe('datasets/StandWalkJump/StandWalkJump_TRAIN.ts')
            X_test, y_test = load_from_tsfile_to_dataframe('datasets/StandWalkJump/StandWalkJump_TEST.ts')
        elif args["dataset"] == "handwritting":
            X_train, y_train = load_from_tsfile_to_dataframe('datasets/Handwriting/Handwriting_TRAIN.ts')
            X_test, y_test = load_from_tsfile_to_dataframe('datasets/Handwriting/Handwriting_TEST.ts')
        N, D = X_train.shape
        N_te = X_test.shape[0]
        T = np.array(X_train.to_numpy()[0][0]).shape[0]
        x_train = np.zeros([N, D, T])
        x_test = np.zeros([N_te, D, T])
        for i in range(N):
            for j in range(D):
                x_train[i][j] = np.array(X_train.to_numpy()[i][j])
        
        for i in range(N_te):
            for j in range(D):
                x_test[i][j] = np.array(X_test.to_numpy()[i][j])
        # print(x_train.shape)
        x_train = x_train.transpose([0, 2, 1])
        x_test = x_test.transpose([0, 2, 1])
        
        # print(y_train)
        # x_test, y_test = load_from_tsfile_to_dataframe('datasets/Wafer/Wafer_TEST.ts')


    print(f"N: {N} - T: {T} - D: {D}")
    print(f"N_te: {N_te}")

    # x_train = getValues(X_train)
    # x_train = x_train.transpose([0,2,1])
    # x_test = getValues(X_test)
    # x_test = x_test.transpose([0,2,1])


    # * scale data

    x_train = x_train.transpose([2,0,1])
    x_test = x_test.transpose([2,0,1])
    for i in range(D):
        x_train[i], scaler = scaleSerie(x_train[i])
        x_test[i], _ = scaleSerie(x_test[i], scaler)
    x_train = x_train.transpose([1,2,0])
    x_test = x_test.transpose([1,2,0])

    # * get data labels
    labels = np.unique(y_train)
    lb = preprocessing.LabelBinarizer()
    lb.fit(labels);
    y_train = lb.transform(y_train)
    y_test = lb.transform(y_test)
    if(len(labels) == 2):
        y_train = np.hstack((y_train, 1 - y_train))
        y_test = np.hstack((y_test, 1 - y_test))
    y_train_int = [labelInt(label) for label in y_train]
    y_test_int = [labelInt(label) for label in y_test]

    if(x_train.min() < 0):
        print(x_train.min())
        print("ERROR MIN")
        raise AssertionError()
    if(x_train.max() > 1):
        print("ERROR MAX")
        raise AssertionError()

    N_te = x_test.shape[0]

    D = np.zeros([N, N])
    for i in range(N):
        for j in range(N):
            D[i][j] = perSerieDistance(x_train[i], x_train[j])

    D_te = np.zeros([N_te, N])
    for i in range(N_te):
        for j in range(N):
            D_te[i][j] = perSerieDistance(x_test[i], x_train[j])

    # print(D)

    readout = SVC(C=1, kernel='precomputed')
    svm_gamma = 1
    # Ktr = squareform(pdist(repr_tr, metric='sqeuclidean')) 
    Ktr = D
    Ktr = np.exp(-svm_gamma*Ktr)
    readout.fit(Ktr, np.argmax(y_train,axis=1))


    # Kte = cdist(repr_te, repr_tr, metric='sqeuclidean')
    Kte = D_te
    Kte = np.exp(-svm_gamma*Kte)
    pred_class = readout.predict(Kte)

    accuracy, recall, f1 = compute_test_scores(pred_class, y_test)
    print('Accuracy = %.3f, F1 = %.3f'%(accuracy, f1))
    print(f'Labels: {len(labels)}')

    reducer = UMAP(metric='precomputed')

    X = np.concatenate([x_train, x_test])
    # X = x_train
    N_t = X.shape[0]
    D_t = np.zeros([N_t, N_t])
    for i in range(N_t):
        for j in range(N_t):
            D_t[i][j] = perSerieDistance(X[i], X[j])
    
    Kte_t = np.exp(-svm_gamma*D_t)

    coords = reducer.fit_transform(Kte_t)

    cmap = plt.cm.jet
    # extract all colors from the .jet map
    cmaplist = [cmap(i) for i in range(cmap.N)]
    # create the new map
    cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)

    clusters = np.concatenate([y_train_int, y_test_int])
    # clusters = y_train_int
    
    plt.scatter(
        coords[:, 0], coords[:, 1], marker = 'o', cmap=cmap, c =clusters, s=15
    )
    plt.savefig('images/' + args['dataset'] + '_' + 'mds' + '.png')
    plt.show()









