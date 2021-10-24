import numpy as np
import umap
import scipy
import scipy.io
import matplotlib.pyplot as plt
from scipy import sparse
from sklearn.linear_model import Ridge
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import KernelPCA
from sktime.datasets import load_basic_motions
from sktime.utils.data_io import load_from_tsfile_to_dataframe
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import torchvision.transforms as transforms
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from esn_ae import EsnAe


n_internal_units=500
spectral_radius=0.59
leak=None
connectivity=0.3
input_scaling=0.2
noise_level=0.01
circle=False
input_weights = None


import numpy as np
from scipy import sparse
import torch 
import torchvision
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, size, n_hidden ):
        super(Autoencoder, self).__init__()
        # encoder
        self.hidden = nn.Linear(in_features=size, out_features=n_hidden)
        self.output = nn.Linear(in_features=n_hidden, out_features=size)
        
    def forward(self, w):
        x = F.relu(self.hidden(w))
        x = self.output(x)
        return x

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

class Reservoir(object):
    """
    Build a reservoir and evaluate internal states
    
    Parameters:
        n_internal_units = processing units in the reservoir
        spectral_radius = largest eigenvalue of the reservoir matrix of connection weights
        leak = amount of leakage in the reservoir state update (optional)
        connectivity = percentage of nonzero connection weights (unused in circle reservoir)
        input_scaling = scaling of the input connection weights
        noise_level = deviation of the Gaussian noise injected in the state update
        circle = generate determinisitc reservoir with circle topology
    """
    
    def __init__(self, n_internal_units=100, spectral_radius=0.99, leak=None,
                 connectivity=0.3, input_scaling=0.2, noise_level=0.01, circle=False):
        
        # Initialize attributes
        self._n_internal_units = n_internal_units
        self._input_scaling = input_scaling
        self._noise_level = noise_level
        self._leak = leak

        # Input weights depend on input size: they are set when data is provided
        self._input_weights = None

        # Generate internal weights
        if circle:
            self._internal_weights = self._initialize_internal_weights_Circ(
                    n_internal_units,
                    spectral_radius)
        else:
            self._internal_weights = self._initialize_internal_weights(
                n_internal_units,
                connectivity,
                spectral_radius)


    def _initialize_internal_weights_Circ(self, n_internal_units, spectral_radius):
        
        # Construct reservoir with circular topology
        internal_weights = np.zeros((n_internal_units, n_internal_units))
        internal_weights[0,-1] = 1.0
        for i in range(n_internal_units-1):
            internal_weights[i+1,i] = 1.0
            
        # Adjust the spectral radius.
        E, _ = np.linalg.eig(internal_weights)
        e_max = np.max(np.abs(E))
        internal_weights /= np.abs(e_max)/spectral_radius 
                
        return internal_weights
    
    
    def _initialize_internal_weights(self, n_internal_units,
                                     connectivity, spectral_radius):

        # Generate sparse, uniformly distributed weights.
        internal_weights = sparse.rand(n_internal_units,
                                       n_internal_units,
                                       density=connectivity).todense()

        # Ensure that the nonzero values are uniformly distributed in [-0.5, 0.5]
        internal_weights[np.where(internal_weights > 0)] -= 0.5
        
        # Adjust the spectral radius.
        E, _ = np.linalg.eig(internal_weights)
        e_max = np.max(np.abs(E))
        internal_weights /= np.abs(e_max)/spectral_radius       

        return internal_weights


    def _compute_state_matrix(self, X, n_drop=0):
        N, T, _ = X.shape
        previous_state = np.zeros((N, self._n_internal_units), dtype=float)

        # Storage
        state_matrix = np.empty((N, T - n_drop, self._n_internal_units), dtype=float)

        for t in range(T):
            current_input = X[:, t, :]

            # Calculate state
            state_before_tanh = self._internal_weights.dot(previous_state.T) + self._input_weights.dot(current_input.T)

            # Add noise
            state_before_tanh += np.random.rand(self._n_internal_units, N)*self._noise_level

            # Apply nonlinearity and leakage (optional)
            if self._leak is None:
                previous_state = np.tanh(state_before_tanh).T
            else:
                previous_state = (1.0 - self._leak)*previous_state + np.tanh(state_before_tanh).T

            # Store everything after the dropout period
            if (t > n_drop - 1):
                state_matrix[:, t - n_drop, :] = previous_state

        return state_matrix


    def get_states(self, X, n_drop=0, bidir=True):
        N, T, V = X.shape
        if self._input_weights is None:
            self._input_weights = (2.0*np.random.binomial(1, 0.5 , [self._n_internal_units, V]) - 1.0)*self._input_scaling

        # compute sequence of reservoir states
        states = self._compute_state_matrix(X, n_drop)
    
        # reservoir states on time reversed input
        if bidir is True:
            X_r = X[:, ::-1, :]
            states_r = self._compute_state_matrix(X_r, n_drop)
            states = np.concatenate((states, states_r), axis=2)

        return states


#  returns data with shape N, T, D
def load_data():
    data = scipy.io.loadmat('JpVow.mat')
    X = data['X']  # shape is [N,T,V]
    if len(X.shape) < 3:
        X = np.atleast_3d(X)
    Y = data['Y']  # shape is [N,1]
    Xte = data['Xte']
    if len(Xte.shape) < 3:
        Xte = np.atleast_3d(Xte)
    Yte = data['Yte']

    # Since we are doing clustering, we do not need the train/test split
    X = np.concatenate((X, Xte), axis=0)
    Y = np.concatenate((Y, Yte), axis=0)
    return X, Y

def initialize_internal_weights(n_internal_units, connectivity, spectral_radius):
    
    # Generate sparse, uniformly distributed weights.
    internal_weights = sparse.rand(n_internal_units,
                                    n_internal_units,
                                    density=connectivity).todense()

    # Ensure that the nonzero values are uniformly distributed in [-0.5, 0.5]
    internal_weights[np.where(internal_weights > 0)] -= 0.5
    
    # Adjust the spectral radius.
    E, _ = np.linalg.eig(internal_weights)
    e_max = np.max(np.abs(E))
    internal_weights /= np.abs(e_max)/spectral_radius
    return internal_weights

internal_weights = initialize_internal_weights(n_internal_units, connectivity, spectral_radius)






# * Data loads
dataset = "libras"
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
lb.fit(labels)
y_train = lb.transform(y_train)
y_test = lb.transform(y_test)
if(len(labels) == 2):
    y_train = np.hstack((y_train, 1 - y_train))
    y_test = np.hstack((y_test, 1 - y_test))
y_train_int = [labelInt(label) for label in y_train]
y_test_int = [labelInt(label) for label in y_test]



X = np.concatenate([x_train, x_test])
Y = np.concatenate([y_train_int, y_test_int])
Y = np.expand_dims(Y, axis=1)
# X = np.concatenate([x_train, x_test])

X, Y = load_data()

print(X.shape)
print(Y.shape)
# print(Y)

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
    pca_n_dim = 100,
    n_epochs = 500,
    w_out_mode=False,
)

n_internal_units=200
spectral_radius=0.9
leak=None
connectivity=0.35
input_scaling=0.1
noise_level=0.01
circle=False
input_weights = None

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
    pca_n_dim = 100,
    n_epochs = 500,
    w_out_mode=False,
    embedding_size=100,
)
mts_representations = ae.train(X)
print("---")
print(X.shape)
print(mts_representations.shape)
print(Y.shape)
print("---")

# N, T, D = X.shape

# input_weights = (2.0 * np.random.binomial(1, 0.5 , [n_internal_units, D]) - 1.0) * input_scaling


# previous_state = np.zeros((N, n_internal_units), dtype=float)

# R = np.empty((N, T, n_internal_units), dtype=float)

# for t in range(T):
#     current_input = X[:, t, :]

#     # Calculate state
#     state_before_tanh = internal_weights.dot(previous_state.T) + input_weights.dot(current_input.T)

#     # Add noise
#     state_before_tanh += np.random.rand(n_internal_units, N)* noise_level

#     # Apply nonlinearity and leakage (optional)
#     if leak is None:
#         previous_state = np.tanh(state_before_tanh).T
#     else:
#         previous_state = (1.0 - leak)*previous_state + np.tanh(state_before_tanh).T

#     # Store everything after the dropout period
#     R[:, t , :] = previous_state
# n_drop = 0
# bidir = True

# # reservoir = Reservoir(n_internal_units=n_internal_units,
# #     spectral_radius=spectral_radius,
# #     leak=leak,
# #     connectivity=connectivity,
# #     input_scaling=input_scaling,
# #     noise_level=noise_level,
# #     circle=circle)
# # res_states = reservoir.get_states(X, n_drop=n_drop, bidir=bidir)

# # red_states = res_states


# # ridge_embedding = Ridge(alpha=10.0, fit_intercept=True)
# # coeff_tr = []
# # biases_tr = []  
# # # print(red_states)
# # for i in range(X.shape[0]):
# #     ridge_embedding.fit(red_states[i, 0:-1, :], red_states[i, 1:, :])
# #     coeff_tr.append(ridge_embedding.coef_.ravel())
# #     biases_tr.append(ridge_embedding.intercept_.ravel())
# # input_repr = np.concatenate((np.vstack(coeff_tr), np.vstack(biases_tr)), axis=1)

# # R = res_states






# # print(X.shape)
# # print(R.shape)

# # print(R[0,:,:].shape)
# # W_out = 

# G = -1
# W = []
# coeff = []
# interc = []
# for i in range(N):
#     x = X[i,:,:]
#     r = R[i,:,:]
#     clf = Ridge(alpha=1.0)
#     clf.fit(r, x)
#     w = clf.coef_
#     G = w.shape[1]
#     # print(w.shape)
#     # coeff += [clf.coef_]
#     w = w.ravel()
#     W += [w]
#     # print(w.shape)
#     # print(x.shape)
#     # print(r.shape)
#     # print(clf.coef_.shape)
#     # print(x[4])
#     # print(clf.predict(r)[4])
#     # print(clf.coef_.shape)

# W = np.array(W, dtype=float)

# S = W.shape[1]

# print(W.shape)

# print(f"D: {D}")

# print(S)

# NS = 30

# net = Autoencoder(S, NS)
# criterion = nn.MSELoss()
# NUM_EPOCHS = 100
# LEARNING_RATE = 1e-3
# BATCH_SIZE = 64
# optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)


# w_tensor = torch.Tensor(W)
# r_tensor = torch.Tensor(R)
# x_tensor = torch.Tensor(X)

# tensorDataset = TensorDataset(w_tensor, r_tensor, x_tensor)
# trainloader = DataLoader(tensorDataset, batch_size=BATCH_SIZE)

# train_loss = []
# for epoch in range(NUM_EPOCHS):
#     running_loss = 0.0
#     for w, r, x in trainloader:
#         optimizer.zero_grad()
#         # print(f"w shape: {w.shape}")
#         w_r = net(w)
#         w_r = torch.reshape(w_r, (-1 , D, G))
#         w_r = torch.transpose(w_r, 1, 2)
#         # img = img.to(device)
#         # img = img.view(img.size(0), -1)
#         # outputs = net(img)
#         # print(f"r shape: {r.shape}")
#         # print(f"w_r shape: {w_r.shape}")
#         x_r = torch.matmul(r, w_r)
        
        
#         loss = criterion(x, x_r)
#         # print(loss.item())
#         loss.backward()
#         optimizer.step()
#         running_loss += loss.item()
    
#     loss = running_loss / len(trainloader)
#     train_loss.append(loss)
#     print('Epoch {} of {}, Train Loss: {:.3f}'.format(
#         epoch+1, NUM_EPOCHS, loss))
    

# encoded_repr = net.hidden(torch.tensor(W.astype(float)).float())
# print(encoded_repr.shape)
# print(f"G: {G} -> NS: {NS}")

# encoded_repr = encoded_repr.cpu().detach().numpy()





























# # mts_representations = input_repr
# # mts_representations = W
# mts_representations = encoded_repr






        
# Normalize the similarity in [0,1]
similarity_matrix = cosine_similarity(mts_representations)
similarity_matrix = (similarity_matrix + 1.0)/2.0

# Plot similarity matrix
# fig =  plt.figure(figsize=(8,8))
# h = plt.imshow(similarity_matrix)
# plt.title("RC similarity matrix")
# plt.colorbar(h)
# plt.show()

reducer = umap.UMAP()
embeddings_pca = reducer.fit_transform(mts_representations)

# kpca = KernelPCA(n_components=2, kernel='precomputed')
# embeddings_pca = kpca.fit_transform(similarity_matrix)

fig =  plt.figure(figsize=(10,8))
plt.scatter(embeddings_pca[:,0], embeddings_pca[:,1], c=Y[:,0], s=10, cmap='tab20')
plt.title("Kernel PCA embeddings")
plt.show()

# fig =  plt.figure(figsize=(10,8))
# plt.scatter(embeddings_pca[:,0], embeddings_pca[:,1], c=Y[:,0], s=10, cmap='tab20')
# plt.title("Kernel PCA embeddings")
# plt.show()


# n_samples, n_features, d = 10, 5, 3
# rng = np.random.RandomState(0)

# y = rng.randn(n_samples, d)
# X = rng.randn(n_samples, n_features)

# print(X.shape)
# print(y.shape)
# clf = Ridge(alpha=1.0)
# clf.fit(X, y)

# print(clf.coef_.shape)