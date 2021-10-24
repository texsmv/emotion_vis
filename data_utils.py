import numpy as np
from sktime.datasets import load_basic_motions
from sktime.utils.data_io import load_from_tsfile_to_dataframe
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

def labelInt(labels):
    for i in range(len(labels)):
        if labels[i] == 1:
            return i

def scaleSerie(data, scaler = None):
    if(scaler == None):
        scaler = MinMaxScaler()
        scaler.fit(data)
    scaler_data = scaler.transform(data)
    return np.clip(scaler_data, 0, 1), scaler

def getValues(dataFrame):
    N, D = dataFrame.shape
    T = dataFrame.to_numpy()[0][0].to_numpy().shape[0]
    data = np.zeros((N, D, T), dtype=np.float32)
    # data = np.zeros((N, D, T))
    for i in range(N):
        for j in range(D):
            data[i][j] = dataFrame.to_numpy()[i][j].to_numpy()
    return data

def loadDataset(dataset):
    # * Data loads
    # dataset = "handwritting"
    if dataset == "motions":
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
        variables = ["accelerometer-x", "accelerometer-y", "accelerometer-z", "gyroscope-x", "gyroscope-y", "gyroscope-z"]
    else:
        if dataset == "wafer":
            X_train, y_train = load_from_tsfile_to_dataframe('datasets/Wafer/Wafer_TRAIN.ts')
            X_test, y_test = load_from_tsfile_to_dataframe('datasets/Wafer/Wafer_TRAIN.ts')
            variables = ["sensor"]
        elif dataset == "libras":
            X_train, y_train = load_from_tsfile_to_dataframe('datasets/Libras/Libras_TRAIN.ts')
            X_test, y_test = load_from_tsfile_to_dataframe('datasets/Libras/Libras_TEST.ts')
            variables = ["x", "y"]
        elif dataset == "uwave":
            X_train, y_train = load_from_tsfile_to_dataframe('datasets/UWaveGestureLibraryAll/UWaveGestureLibraryAll_TRAIN.ts')
            X_test, y_test = load_from_tsfile_to_dataframe('datasets/UWaveGestureLibraryAll/UWaveGestureLibraryAll_TEST.ts')
            X_test = X_test[:1000]
            y_test = y_test[:1000]
            variables = ["accelerometer"]
        elif dataset == "stand":
            X_train, y_train = load_from_tsfile_to_dataframe('datasets/StandWalkJump/StandWalkJump_TRAIN.ts')
            X_test, y_test = load_from_tsfile_to_dataframe('datasets/StandWalkJump/StandWalkJump_TEST.ts')
            variables = ["ECG-1", "ECG-2", "ECG-3", "ECG-4"]
        elif dataset == "handwritting":
            X_train, y_train = load_from_tsfile_to_dataframe('datasets/Handwriting/Handwriting_TRAIN.ts')
            X_test, y_test = load_from_tsfile_to_dataframe('datasets/Handwriting/Handwriting_TEST.ts')
            variables = ["accelerometer-x", "accelerometer-y", "accelerometer-z"]

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
    lb.fit(labels)
    y_train = lb.transform(y_train)
    y_test = lb.transform(y_test)
    if(len(labels) == 2):
        y_train = np.hstack((y_train, 1 - y_train))
        y_test = np.hstack((y_test, 1 - y_test))
    y_train_int = [labelInt(label) for label in y_train]
    y_test_int = [labelInt(label) for label in y_test]



    X = np.concatenate([x_train, x_test])
    y = np.concatenate([y_train_int, y_test_int])
    y = np.expand_dims(y, axis=1)

    return X, y