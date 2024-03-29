import sys
import numpy as np
import argparse

from timeit import default_timer as timer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import KernelPCA
from sktime.classification.compose import ColumnEnsembleClassifier
from sktime.classification.dictionary_based import BOSSEnsemble
from sktime.classification.interval_based import TimeSeriesForestClassifier
from sktime.classification.shapelet_based import MrSEQLClassifier
from sktime.datasets import load_basic_motions
from sktime.transformations.panel.compose import ColumnConcatenator
from typing import Dict, Any, List
from scipy.spatial.distance import pdist, cdist, squareform
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from sktime.utils.data_io import load_from_tsfile_to_dataframe
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier

import numpy as np
import matplotlib.pyplot as plt

# from rc_next_gen import NextGenAE
from esn_ae import EsnAe
from esn_ae_v2 import EsnAe2

# import tensorflow as tf
# from keras.models import Sequential
# from keras.layers import LSTM
# from keras.layers import Dense
# from keras.layers import RepeatVector
# from keras.layers import TimeDistributed
# from keras.optimizers import SGD
# from keras import Model
# from keras import backend as K

# from multidr.tdr import TDR
# from multidr.cl import CL

# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.95
# config.gpu_options.allow_growth = True
# tf.keras.backend.set_session(tf.Session(config=config));
# eps = np.float(K.epsilon())

from sklearn.decomposition import PCA
from umap import UMAP





np.random.seed(2021)

# sys.path.insert(1, '/home/texs/Documents/Kusisqa/emotion_vis_server')
sys.path.insert(1, '/home/texs/Documents/AirQuality/repositories/peax/experiments')
sys.path.insert(1, '/home/texs/Documents/Kusisqa/repositories/Time-series-classification-and-clustering-with-Reservoir-Computing/code')

from modules import RC_model

# from air_quality import trainAutoencoder

esn_ca_config = {}
esn_ca_config['n_internal_units'] = 100
esn_ca_config['W_out_mode'] = False
esn_ca_config['pca_dim'] = None
esn_ca_config['bidir'] = True
esn_ca_config['n_epoch'] = 500
esn_ca_config['embedding'] = 50

esn_ca_config2 = {}
esn_ca_config2['n_internal_units'] = 100
esn_ca_config2['W_out_mode'] = False
esn_ca_config2['pca_dim'] = None
esn_ca_config2['bidir'] = True
esn_ca_config2['n_epoch'] = 100
esn_ca_config2['embedding'] = 50

rc_config = {}

# * Reservoir cnn_settings
rc_config['n_internal_units_m'] = 20
rc_config['n_dim_m'] = 10
rc_config['n_internal_units'] = 20        # size of the reservoir
rc_config['spectral_radius'] = 0.9        # largest eigenvalue of the reservoir
rc_config['leak'] = 0.6                    # amount of leakage in the reservoir state update (None or 1.0 --> no leakage)
rc_config['connectivity'] = 0.35           # percentage of nonzero connections in the reservoir
rc_config['input_scaling'] = 0.1           # scaling of the input weights
rc_config['noise_level'] = 0.01            # noise in the reservoir state update
rc_config['n_drop'] = 0                    # transient states to be dropped
rc_config['bidir'] = True                  # if True, use bidirectional reservoir
rc_config['circ'] = False                  # use reservoir with circle topology

# Dimensionality reduction
rc_config['dimred_method'] ='tenpca'       # options: {None (no dimensionality reduction), 'pca', 'tenpca'}
# rc_config['dimred_method'] = 'None'
# rc_config['n_dim'] = 10                   # number of resulting dimensions after the dimensionality reduction procedure
rc_config['n_dim'] = 10
# MTS representation
# rc_config['mts_rep'] = 'reservoir'         # MTS representation:  {'last', 'mean', 'output', 'reservoir'}
rc_config['mts_rep'] = 'reservoir'
rc_config['w_ridge_embedding'] = 10.0      # regularization parameter of the ridge regression

# Readout
# rc_config['readout_type'] = 'svm'           # by setting None, the input representations will be stored
rc_config['readout_type'] = 'svm'

cnn_n_epochs = 50
cnn_batch_size = 50

lstm_embedding = 64
lstm_n_epochs = 50
lstm_batch_size = 50

# * cnn cnn_settings
cnn_settings = {
  "conv_filters": [
    128,
    192,
    288,
    432
  ],
  "conv_kernels": [
    3,
    5,
    7,
    9
  ],
  "dense_units": [
    1024,
    256
  ],
  "dropouts": [
    0,
    0,
    0,
    0,
    0,
    0
  ],
  "embedding": 25,
  "reg_lambda": 0,
  "optimizer": "adadelta",
  "learning_rate": 1.0,
  "learning_rate_decay": 0.001,
  "loss": "bce",
  "metrics": [],
  "batch_norm": [
    False,
    False,
    False,
    False,
    False,
    False
  ],
  "batch_norm_input": False,
  "crop_end": False,
}


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

def binary_crossentropy_numpy(y_true, y_pred):
    output = np.clip(y_pred, eps, 1 - eps)

    return np.mean(
        -(y_true * np.log(output) + (1 - y_true) * np.log(1 - output)), axis=-1
    )



def lstmModel(timesteps, n_features):
    opt = SGD(lr=0.01, momentum=0.9, clipvalue=0.5)
    model = Sequential()
    model.add(LSTM(128, activation='relu', input_shape=(timesteps,n_features), return_sequences=True))
    model.add(LSTM(lstm_embedding, activation='relu', return_sequences=False))
    model.add(RepeatVector(timesteps))
    model.add(LSTM(64, activation='relu', return_sequences=True))
    model.add(LSTM(128, activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(n_features, activation='sigmoid')))
    # model.compile(optimizer='adam', loss='mse')
    # model.compile(optimizer='adam', loss='binary_crossentropy')
    model.compile(optimizer=opt, loss='binary_crossentropy')
    
    
    # model.summary()
    encoder = Model(model.input, model.get_layer(index = 1).output )
    return model, encoder

def compute_test_scores(pred_class, Yte):
    """
    Wrapper to compute classification accuracy and F1 score
    """
    true_class = np.argmax(Yte, axis=1)
    
    accuracy = accuracy_score(true_class, pred_class)
    if Yte.shape[1] > 2:
        f1 = f1_score(true_class, pred_class, average='weighted')
        recall = recall_score(true_class, pred_class, average='weighted')
        precision = precision_score(true_class, pred_class, average='weighted')
    else:
        f1 = f1_score(true_class, pred_class, average='weighted')
        recall = recall_score(true_class, pred_class, average='weighted')
        precision = precision_score(true_class, pred_class, average='weighted')
    print("recall")
    print(recall)
    print("precision")
    print(precision)
    return accuracy, recall, f1


def get_embeddings(
        algorithm: str,
        x_train: np.ndarray,
        x_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
    ):
    if(algorithm == "rc"):
        rcm =  RC_model(
            reservoir=None,     
            n_internal_units=rc_config['n_internal_units'],
            spectral_radius=rc_config['spectral_radius'],
            leak=rc_config['leak'],
            connectivity=rc_config['connectivity'],
            input_scaling=rc_config['input_scaling'],
            noise_level=rc_config['noise_level'],
            circle=rc_config['circ'],
            n_drop=rc_config['n_drop'],
            bidir=rc_config['bidir'],
            dimred_method=rc_config['dimred_method'], 
            n_dim=rc_config['n_dim'],
            mts_rep=rc_config['mts_rep'],
            w_ridge_embedding=rc_config['w_ridge_embedding'],
            readout_type=rc_config['readout_type'] 
        )
        input_repr_tr = rcm.train(x_train, y_train)
        accuracy, f1, input_repr_te = rcm.test(x_test, y_test)
        return input_repr_tr, input_repr_te
    
    if(algorithm == "fe-esn"):
        n_internal_units=esn_ca_config['n_internal_units']
        spectral_radius=0.9
        leak=None
        connectivity=0.35
        input_scaling=0.1
        noise_level=0.01
        circle=False
        input_weights = True

        ae = EsnAe(
            n_internal_units=n_internal_units,
            spectral_radius=spectral_radius,
            leak=leak,
            connectivity=connectivity,
            input_scaling=input_scaling,
            noise_level=noise_level,
            circle=circle,
            n_drop=0, 
            bidir=True,
            pca_n_dim = None,
            n_epochs = esn_ca_config['n_epoch'],
            w_out_mode = True,
            embedding_size = esn_ca_config['embedding']
        )
        input_repr_tr = ae.train(x_train)
        input_repr_te = ae.test(x_test)
        return input_repr_tr, input_repr_te

    if(algorithm == "esn-ae"):
        n_internal_units=esn_ca_config['n_internal_units']
        spectral_radius=0.9
        leak=None
        connectivity=0.35
        input_scaling=0.1
        noise_level=0.01
        circle=False
        input_weights = True

        ae = EsnAe(
            n_internal_units=n_internal_units,
            spectral_radius=spectral_radius,
            leak=leak,
            connectivity=connectivity,
            input_scaling=input_scaling,
            noise_level=noise_level,
            circle=circle,
            n_drop=0, 
            bidir=False,
            pca_n_dim = None,
            n_epochs = esn_ca_config['n_epoch'],
            w_out_mode=False,
            embedding_size= esn_ca_config['embedding']
        )
        input_repr_tr = ae.train(x_train)
        input_repr_te = ae.test(x_test)
        return input_repr_tr, input_repr_te
    
    if(algorithm == "besn-ae"):
        n_internal_units=esn_ca_config['n_internal_units']
        spectral_radius=0.9
        leak=None
        connectivity=0.35
        input_scaling=0.1
        noise_level=0.01
        circle=False
        input_weights = True

        ae = EsnAe2(
            n_internal_units=n_internal_units,
            spectral_radius=spectral_radius,
            leak=leak,
            connectivity=connectivity,
            input_scaling=input_scaling,
            noise_level=noise_level,
            circle=circle,
            n_drop=0, 
            bidir=False,
            pca_n_dim = esn_ca_config['pca_dim'],
            n_epochs = esn_ca_config['n_epoch'],
            w_out_mode=False,
            embedding_size= esn_ca_config['embedding']
        )
        input_repr_tr = ae.train(x_train)
        input_repr_te = ae.test(x_test)
        return input_repr_tr, input_repr_te
    
    if(algorithm == "esn-ae2"):
        n_internal_units=esn_ca_config2['n_internal_units']
        spectral_radius=0.9
        leak=None
        connectivity=0.35
        input_scaling=0.1
        noise_level=0.01
        circle=False
        input_weights = True

        ae = EsnAe2(
            n_internal_units=n_internal_units,
            spectral_radius=spectral_radius,
            leak=leak,
            connectivity=connectivity,
            input_scaling=input_scaling,
            noise_level=noise_level,
            circle=circle,
            n_drop=0, 
            bidir=esn_ca_config2['bidir'],
            pca_n_dim = esn_ca_config2['pca_dim'],
            n_epochs = esn_ca_config2['n_epoch'],
            w_out_mode=esn_ca_config2['W_out_mode'],
            embedding_size= esn_ca_config2['embedding']
        )
        input_repr_tr = ae.train(x_train)
        input_repr_te = ae.test(x_test)
        return input_repr_tr, input_repr_te
    
    if(algorithm == "rc-ng"):
        ae = NextGenAE(
            w_out_mode= False,
            n_epochs = 500,
        )
        input_repr_tr = ae.train(x_train, test = False)
        input_repr_te = ae.train(x_test, test=True)
        return input_repr_tr, input_repr_te


    if(algorithm == "rc2"):
        N, T, D = x_train.shape
        encodings_tr = []
        encodings_te = []
        for i in range(D):
            train = np.expand_dims(x_train[:,:,i], axis=2)
            test = np.expand_dims(x_test[:,:,i], axis=2)
            rcm =  RC_model(
                reservoir=None,     
                n_internal_units=rc_config['n_internal_units_m'],
                spectral_radius=rc_config['spectral_radius'],
                leak=rc_config['leak'],
                connectivity=rc_config['connectivity'],
                input_scaling=rc_config['input_scaling'],
                noise_level=rc_config['noise_level'],
                circle=rc_config['circ'],
                n_drop=rc_config['n_drop'],
                bidir=rc_config['bidir'],
                dimred_method=rc_config['dimred_method'], 
                n_dim=rc_config['n_dim_m'],
                mts_rep=rc_config['mts_rep'],
                w_ridge_embedding=rc_config['w_ridge_embedding'],
                readout_type=rc_config['readout_type'] 
            )
            input_repr_tr = rcm.train(x_train, y_train)
            _, _, input_repr_te = rcm.test(x_test, y_test)
            encodings_tr += [input_repr_tr]
            encodings_te += [input_repr_te]

        encodings_tr = np.concatenate(np.array(encodings_tr), axis=1)
        encodings_te = np.concatenate(np.array(encodings_te), axis=1)

        # pca = PCA(n_components=50)
        # pca.fit(encodings_tr)
        # encodings_tr = pca.transform(encodings_tr)
        # encodings_te = pca.transform(encodings_te)

        
        return encodings_tr, encodings_te
    
    if(algorithm == "lstm"):
        N, T, D = x_train.shape
        lstm, encoder = lstmModel(T, D)
        
        lstm.fit(x_train, x_train, epochs=lstm_n_epochs, batch_size=lstm_batch_size, verbose=1)
        repr_tr = encoder.predict(x_train)
        repr_te = encoder.predict(x_test)
        return repr_tr, repr_te

    if(algorithm == "multidr"):
        x_train = np.transpose(x_train, (1, 0, 2))
        x_test = np.transpose(x_test, (1, 0, 2))
        T, N, D = x_train.shape
        N_te = x_test.shape[1]

        n_neighbors = 7
        min_dist = 0.15
        tdr = TDR(first_learner=PCA(n_components=1),
          second_learner=UMAP(n_components=2,
                              n_neighbors=n_neighbors,
                              min_dist=min_dist))
        results = tdr.fit_transform(x_train,
                            first_scaling=True,
                            second_scaling=False,
                            verbose=True)
        train_tn = tdr.Y_tn
        train_nd = tdr.Y_nd;

        X_tn_d = np.zeros((T * N_te, D))
        X_nd_t = np.zeros((N_te * D, T))
        # X_dt_n = np.zeros((D * T, N_te))
        # for d in range(D):
        #     X_dt_n[d * T:(d + 1) * T, :] = x_test[:, :, d]
        for t in range(T):
            X_tn_d[t * N_te:(t + 1) * N_te, :] = x_test[t, :, :]
        for n in range(N_te):
            X_nd_t[n * D:(n + 1) * D, :] = x_test[:, n, :].T
        # y_dt_n = tdr.first_learner['n'].fit_transform(preprocessing.scale(X_dt_n))
        y_nd_t = tdr.first_learner['t'].transform(preprocessing.scale(X_nd_t))
        y_tn_d = tdr.first_learner['d'].transform(preprocessing.scale(X_tn_d))

        # y_dt = y_dt_n.reshape((D, T))
        y_tn = y_tn_d.reshape((T, N_te))
        y_nd = y_nd_t.reshape((N_te, D))

        # return train_tn.transpose(), y_tn.transpose()
        return train_nd, y_nd
    if(algorithm == "cnn"):
        N, T, D = x_train.shape
        encodings_tr = []
        encodings_te = []
        for i in range(D):
            print("Dimension")
            print(i)
            train = np.expand_dims(x_train[:,:,i], axis=2)
            test = np.expand_dims(x_test[:,:,i], axis=2)
            encoder, decoder, autoencoder, history = trainAutoencoder(
                train,
                test,
                test,
                model_name = 'savedData/test.h5', 
                window_size = T,
                batch_size = cnn_batch_size,
                n_epochs = cnn_n_epochs,
                settings = cnn_settings,
            )
            # print("shape")
            # print(encoder.predict(train).shape)
            encodings_tr += [encoder.predict(train)]
            encodings_te += [encoder.predict(test)]
            K.clear_session()
            # print(encoding_i_tr.shape)
        encodings_tr = np.concatenate(np.array(encodings_tr), axis=1)
        encodings_te = np.concatenate(np.array(encodings_te), axis=1)
        # print(encodings_tr.shape)
        # print(encodings_te.shape)

        return encodings_tr, encodings_te


# def get_distance_matrix(
#     algorithm: str,
#     ):
#     if(algorithm == "mds"):
        

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
    parser.add_argument(
        "-a", "--alg",
        type=str,
        required=True,
        help="Algorithm"
    )
    parser.add_argument(
        "-d", "--dataset",
        type=str,
        required=True,
        help="Algorithm"
    )

    args: Dict[str, Any] = vars(parser.parse_args())

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

        rc_config['n_dim'] = 20
        rc_config['n_internal_units'] = 100
        rc_config['n_dim_m'] = 10
        rc_config['n_internal_units_m'] = 20
        # rc_config['n_internal_units'] = 20

        cnn_settings['embedding'] = 25
        cnn_n_epochs = 150
        cnn_batch_size = 50
        
        lstm_embedding = 10
        lstm_n_epochs = 50
        lstm_batch_size = 50

    else:
        if args["dataset"] == "wafer":
            X_train, y_train = load_from_tsfile_to_dataframe('datasets/Wafer/Wafer_TRAIN.ts')
            X_test, y_test = load_from_tsfile_to_dataframe('datasets/Wafer/Wafer_TRAIN.ts')

            rc_config['n_dim'] = 20
            rc_config['n_internal_units'] = 100
            rc_config['n_dim_m'] = 10
            rc_config['n_internal_units_m'] = 100

            cnn_settings['embedding'] = 25
            cnn_n_epochs = 100
            cnn_batch_size = 50

            lstm_embedding = 15
            lstm_n_epochs = 20
            lstm_batch_size = 500
        elif args["dataset"] == "libras":
            X_train, y_train = load_from_tsfile_to_dataframe('datasets/Libras/Libras_TRAIN.ts')
            X_test, y_test = load_from_tsfile_to_dataframe('datasets/Libras/Libras_TEST.ts')

            rc_config['n_dim'] = 20
            rc_config['n_internal_units'] = 100
            rc_config['n_dim_m'] = 10
            rc_config['n_internal_units_m'] = 55

            esn_ca_config['n_internal_units'] = 100
            esn_ca_config['W_out_mode'] = False
            esn_ca_config['pca_dim'] = None
            esn_ca_config['bidir'] = True
            esn_ca_config['n_epoch'] = 800
            esn_ca_config['embedding'] = 100

            esn_ca_config['n_internal_units'] = 100
            esn_ca_config['W_out_mode'] = False
            esn_ca_config['pca_dim'] = None
            esn_ca_config['bidir'] = False
            # esn_ca_config['n_epoch'] = 3500
            esn_ca_config['embedding'] = 100
            # esn_ca_config['embedding'] = 10


            cnn_settings['crop_end'] = True
            cnn_settings['embedding'] = 5
            cnn_n_epochs = 150
            cnn_batch_size = 50

            lstm_embedding = 20
            lstm_n_epochs = 3000
            lstm_batch_size = 100
        elif args["dataset"] == "uwave":
            X_train, y_train = load_from_tsfile_to_dataframe('datasets/UWaveGestureLibraryAll/UWaveGestureLibraryAll_TRAIN.ts')
            X_test, y_test = load_from_tsfile_to_dataframe('datasets/UWaveGestureLibraryAll/UWaveGestureLibraryAll_TEST.ts')

            rc_config['n_dim'] = 20
            rc_config['n_internal_units'] = 100
            rc_config['n_dim_m'] = 20
            rc_config['n_internal_units_m'] = 100

            cnn_settings['crop_end'] = True
            cnn_settings['embedding'] = 25
            cnn_n_epochs = 150
            cnn_batch_size = 50

            lstm_embedding = 64
            lstm_n_epochs = 7
            lstm_batch_size = 200
        elif args["dataset"] == "stand":
            X_train, y_train = load_from_tsfile_to_dataframe('datasets/StandWalkJump/StandWalkJump_TRAIN.ts')
            X_test, y_test = load_from_tsfile_to_dataframe('datasets/StandWalkJump/StandWalkJump_TEST.ts')

            rc_config['n_dim'] = 20
            rc_config['n_internal_units'] = 400
            rc_config['n_dim_m'] = 30
            rc_config['n_internal_units_m'] = 200

            esn_ca_config['n_internal_units'] = 200
            esn_ca_config['W_out_mode'] = False
            esn_ca_config['pca_dim'] = None
            esn_ca_config['bidir'] = True
            esn_ca_config['n_epoch'] = 400
            esn_ca_config['embedding'] = 50

            esn_ca_config['n_internal_units'] = 400
            esn_ca_config['W_out_mode'] = False
            esn_ca_config['pca_dim'] = None
            esn_ca_config['bidir'] = False
            # esn_ca_config['n_epoch'] = 3000
            esn_ca_config['embedding'] = 50



            cnn_settings['embedding'] = 10
            cnn_n_epochs = 5
            cnn_batch_size = 2

            lstm_embedding = 64
            lstm_n_epochs = 200
            lstm_batch_size = 30
        elif args["dataset"] == "handwritting":
            X_train, y_train = load_from_tsfile_to_dataframe('datasets/Handwriting/Handwriting_TRAIN.ts')
            X_test, y_test = load_from_tsfile_to_dataframe('datasets/Handwriting/Handwriting_TEST.ts')

            
            rc_config['n_dim'] = 20
            rc_config['n_internal_units'] = 100
            rc_config['n_dim_m'] = 20
            rc_config['n_internal_units_m'] = 50

            cnn_settings['embedding'] = 25
            cnn_n_epochs = 200
            cnn_batch_size = 50

            lstm_embedding = 25
            lstm_n_epochs = 100
            lstm_batch_size = 15
        
        elif args["dataset"] == "atrialFibrillation":
            X_train, y_train = load_from_tsfile_to_dataframe('datasets/AtrialFibrillation/AtrialFibrillation_TRAIN.ts')
            X_test, y_test = load_from_tsfile_to_dataframe('datasets/AtrialFibrillation/AtrialFibrillation_TEST.ts')

            
            rc_config['n_dim'] = 25
            rc_config['n_internal_units'] = 100

            esn_ca_config['n_internal_units'] = 100
            esn_ca_config['W_out_mode'] = False
            esn_ca_config['pca_dim'] = None
            esn_ca_config['bidir'] = True
            esn_ca_config['n_epoch'] = 1500
            esn_ca_config['embedding'] = 100

            esn_ca_config['n_internal_units'] = 200
            esn_ca_config['W_out_mode'] = False
            esn_ca_config['pca_dim'] = None
            esn_ca_config['bidir'] = False
            esn_ca_config['n_epoch'] = 4000
            esn_ca_config['embedding'] = 100

            cnn_settings['embedding'] = 100
            cnn_n_epochs = 500
            cnn_batch_size = 50

            lstm_embedding = 100
            lstm_n_epochs = 100
            lstm_batch_size = 100
        
        elif args["dataset"] == "fingerMovements":
            X_train, y_train = load_from_tsfile_to_dataframe('datasets/FingerMovements/FingerMovements_TRAIN.ts')
            X_test, y_test = load_from_tsfile_to_dataframe('datasets/FingerMovements/FingerMovements_TEST.ts')

            
            rc_config['n_dim'] = 40
            rc_config['n_internal_units'] = 50
            rc_config['n_dim_m'] = 20
            rc_config['n_internal_units_m'] = 50

            esn_ca_config['n_internal_units'] = 50
            esn_ca_config['W_out_mode'] = False
            esn_ca_config['pca_dim'] = None
            esn_ca_config['bidir'] = True
            esn_ca_config['n_epoch'] = 1000
            esn_ca_config['embedding'] = 100

            esn_ca_config['n_internal_units'] = 200
            esn_ca_config['W_out_mode'] = False
            esn_ca_config['pca_dim'] = None
            esn_ca_config['bidir'] = False
            # esn_ca_config['n_epoch'] = 2000
            esn_ca_config['embedding'] = 100

            cnn_settings['embedding'] = 25
            cnn_n_epochs = 200
            cnn_batch_size = 200

            lstm_embedding = 100
            lstm_n_epochs = 300
            lstm_batch_size = 50
        
        elif args["dataset"] == "handMovementDirection":
            X_train, y_train = load_from_tsfile_to_dataframe('datasets/HandMovementDirection/HandMovementDirection_TRAIN.ts')
            X_test, y_test = load_from_tsfile_to_dataframe('datasets/HandMovementDirection/HandMovementDirection_TEST.ts')
            
            rc_config['n_dim'] = 50
            rc_config['n_internal_units'] = 400
            rc_config['n_dim_m'] = 20
            rc_config['n_internal_units_m'] = 50

            esn_ca_config['n_internal_units'] = 400
            esn_ca_config['W_out_mode'] = False
            esn_ca_config['pca_dim'] = None
            esn_ca_config['bidir'] = True
            esn_ca_config['n_epoch'] = 1200
            esn_ca_config['embedding'] = 200

            esn_ca_config['n_internal_units'] = 100
            esn_ca_config['W_out_mode'] = False
            esn_ca_config['pca_dim'] = None
            esn_ca_config['bidir'] = False
            # esn_ca_config['n_epoch'] = 4000
            esn_ca_config['embedding'] = 100

            cnn_settings['embedding'] = 25
            cnn_n_epochs = 100
            cnn_batch_size = 50

            lstm_embedding = 250
            lstm_n_epochs = 50
            lstm_batch_size = 50
        
        elif args["dataset"] == "natops":
            X_train, y_train = load_from_tsfile_to_dataframe('datasets/NATOPS/NATOPS_TRAIN.ts')
            X_test, y_test = load_from_tsfile_to_dataframe('datasets/NATOPS/NATOPS_TEST.ts')
            
            rc_config['n_dim'] = 50
            rc_config['n_internal_units'] = 100
            rc_config['n_dim_m'] = 20
            rc_config['n_internal_units_m'] = 50

            esn_ca_config['n_internal_units'] = 100
            esn_ca_config['W_out_mode'] = False
            esn_ca_config['pca_dim'] = None
            esn_ca_config['bidir'] = True
            esn_ca_config['n_epoch'] = 1500
            esn_ca_config['embedding'] = 200

            esn_ca_config['n_internal_units'] = 100
            esn_ca_config['W_out_mode'] = False
            esn_ca_config['pca_dim'] = None
            esn_ca_config['bidir'] = False
            # esn_ca_config['n_epoch'] = 3000
            esn_ca_config['embedding'] = 100

            cnn_settings['embedding'] = 10
            cnn_n_epochs = 50
            cnn_batch_size = 100

            lstm_embedding = 50
            lstm_n_epochs = 2000
            lstm_batch_size = 50
        
        elif args["dataset"] == "penDigits":
            X_train, y_train = load_from_tsfile_to_dataframe('datasets/PenDigits/PenDigits_TRAIN.ts')
            X_test, y_test = load_from_tsfile_to_dataframe('datasets/PenDigits/PenDigits_TEST.ts')
            
            rc_config['n_dim'] = 20
            rc_config['n_internal_units'] = 20
            rc_config['n_dim_m'] = 20
            rc_config['n_internal_units_m'] = 50

            esn_ca_config['n_internal_units'] = 20
            esn_ca_config['W_out_mode'] = False
            esn_ca_config['pca_dim'] = None
            esn_ca_config['bidir'] = True
            esn_ca_config['n_epoch'] = 90
            esn_ca_config['embedding'] = 25

            esn_ca_config['n_internal_units'] = 20
            esn_ca_config['W_out_mode'] =False
            esn_ca_config['pca_dim'] = None
            esn_ca_config['bidir'] = True
            esn_ca_config['n_epoch'] = 101
            esn_ca_config['embedding'] = 25
            # esn_ca_config['embedding'] = 10

            cnn_settings['embedding'] = 10
            cnn_n_epochs = 50
            cnn_batch_size = 300

            lstm_embedding = 4
            lstm_n_epochs = 200
            lstm_batch_size = 50
            
        elif args["dataset"] == "selfRegulationSCP2":
            X_train, y_train = load_from_tsfile_to_dataframe('datasets/SelfRegulationSCP2/SelfRegulationSCP2_TRAIN.ts')
            X_test, y_test = load_from_tsfile_to_dataframe('datasets/SelfRegulationSCP2/SelfRegulationSCP2_TEST.ts')
            
            rc_config['n_dim'] = 100
            rc_config['n_internal_units'] = 50
            rc_config['n_dim_m'] = 20
            rc_config['n_internal_units_m'] = 50

            esn_ca_config['n_internal_units'] = 50
            esn_ca_config['W_out_mode'] = False
            esn_ca_config['pca_dim'] = None
            esn_ca_config['bidir'] = True
            esn_ca_config['n_epoch'] = 401
            esn_ca_config['embedding'] = 50

            esn_ca_config['n_internal_units'] = 100
            esn_ca_config['W_out_mode'] = False
            esn_ca_config['pca_dim'] = None
            esn_ca_config['bidir'] = False
            # esn_ca_config['n_epoch'] = 1000
            esn_ca_config['embedding'] = 50

            cnn_settings['embedding'] = 25
            cnn_n_epochs = 100
            cnn_batch_size = 300

            lstm_embedding = 100
            lstm_n_epochs = 200
            lstm_batch_size = 25
        
        elif args["dataset"] == "motorImagery":
            X_train, y_train = load_from_tsfile_to_dataframe('datasets/MotorImagery/MotorImagery_TRAIN.ts')
            X_test, y_test = load_from_tsfile_to_dataframe('datasets/MotorImagery/MotorImagery_TEST.ts')
            
            rc_config['n_dim'] = 50
            rc_config['n_internal_units'] = 100
            rc_config['n_dim_m'] = 20
            rc_config['n_internal_units_m'] = 50

            esn_ca_config['n_internal_units'] = 100
            esn_ca_config['W_out_mode'] = False
            esn_ca_config['pca_dim'] = None
            esn_ca_config['bidir'] = True
            esn_ca_config['n_epoch'] = 200
            esn_ca_config['embedding'] = 100

            esn_ca_config['n_internal_units'] = 100
            esn_ca_config['W_out_mode'] = False
            esn_ca_config['pca_dim'] = None
            esn_ca_config['bidir'] = False
            esn_ca_config['n_epoch'] = 200
            esn_ca_config['embedding'] = 100

            cnn_settings['embedding'] = 10
            cnn_n_epochs = 10
            cnn_batch_size = 1

            lstm_embedding = 100
            lstm_n_epochs = 50
            lstm_batch_size = 100
        elif args["dataset"] == "heartbeat":
            X_train, y_train = load_from_tsfile_to_dataframe('datasets/Heartbeat/Heartbeat_TRAIN.ts')
            X_test, y_test = load_from_tsfile_to_dataframe('datasets/Heartbeat/Heartbeat_TEST.ts')
            
            rc_config['n_dim'] = 50
            rc_config['n_internal_units'] = 100
            rc_config['n_dim_m'] = 20
            rc_config['n_internal_units_m'] = 50

            esn_ca_config['n_internal_units'] = 100
            esn_ca_config['W_out_mode'] = False
            esn_ca_config['pca_dim'] = None
            esn_ca_config['bidir'] = True
            esn_ca_config['n_epoch'] = 400
            esn_ca_config['embedding'] = 50

            esn_ca_config['n_internal_units'] = 100
            esn_ca_config['W_out_mode'] = False
            esn_ca_config['pca_dim'] = None
            esn_ca_config['bidir'] = False
            # esn_ca_config['n_epoch'] = 1000
            esn_ca_config['n_epoch'] = 400
            esn_ca_config['embedding'] = 50

            cnn_settings['embedding'] = 10
            cnn_n_epochs = 10
            cnn_batch_size = 300

            lstm_embedding = 100
            lstm_n_epochs = 200
            lstm_batch_size = 100





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

    start = timer()

    repr_tr, repr_te = get_embeddings(algorithm=args["alg"], x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)

    end = timer()
    print(f"Elapsed time in seconds: {end - start} seconds") 

    print(f"Train shape: {repr_tr.shape} - Test shape: {repr_te.shape}")
    
    # readout = SVC(C=5, kernel='precomputed')
    readout = MLPClassifier(
        hidden_layer_sizes=(20,len(labels)), 
        activation='tanh', 
        alpha=0.001,
        batch_size=32, 
        learning_rate='adaptive', # 'constant' or 'adaptive'
        learning_rate_init=0.001, 
        max_iter=5000, 
        early_stopping=False, # if True, set validation_fraction > 0
        validation_fraction=0.0 # used for early stopping
    )

    # svm_gamma = 1.0
    # Ktr = squareform(pdist(repr_tr, metric='sqeuclidean')) 
    # Ktr = np.exp(-svm_gamma*Ktr)
    # readout.fit(Ktr, np.argmax(y_train,axis=1))
    readout.fit(repr_tr, y_train)


    # Kte = cdist(repr_te, repr_tr, metric='sqeuclidean')
    # Kte = np.exp(-svm_gamma*Kte)
    # pred_class = readout.predict(Kte)
    pred_class = readout.predict(repr_te)
    pred_class = np.argmax(pred_class, axis=1)

    accuracy, recall, f1 = compute_test_scores(pred_class, y_test)
    # print('Accuracy = %.3f, F1 = %.3f'%(accuracy, f1))
    print('-----Accuracy = %.3f'%(accuracy))
    print(f'Labels: {len(labels)}')

    reducer = UMAP()
    coords = reducer.fit_transform(np.concatenate([repr_tr, repr_te]))

    # mts_representations = np.concatenate([repr_tr, repr_te])
    # similarity_matrix = cosine_similarity(mts_representations)
    # similarity_matrix = (similarity_matrix + 1.0)/2.0
    # kpca = KernelPCA(n_components=2, kernel='precomputed')
    # coords = kpca.fit_transform(similarity_matrix)

    cmap = plt.cm.jet
    # extract all colors from the .jet map
    cmaplist = [cmap(i) for i in range(cmap.N)]
    # create the new map
    cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)

    clusters = np.concatenate([y_train_int, y_test_int])
    
    plt.scatter(
        coords[:, 0], coords[:, 1], marker = 'o', cmap=cmap, c =clusters, s=15
    )
    plt.axis('off')
    # plt.savefig('images/' + args['dataset'] + '_' +  args['alg'] + '.png')

    # plt.savefig('images/' + args['dataset'] + '_' +  'multidr_nd' + '.png', bbox_inches='tight')
    # plt.savefig('images/' + args['dataset'] + '_' +  'multidr_nt' + '.png', bbox_inches='tight')
    plt.show()









