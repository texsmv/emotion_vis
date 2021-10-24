import numpy as np
from reservoir import Reservoir
from data_utils import loadDataset
from sklearn.linear_model import Ridge
import torch 
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import KernelPCA
from tensorPCA import tensorPCA

DISTRIBUTION_VAL = 0.3
cuda = torch.cuda.is_available()

device = torch.device('cuda:0' if cuda else 'cpu')

SPARSE_REG = 1e-3


class SparseAutoencoderKL(nn.Module):
    def __init__(self, n_input, n_hidden, n_output):
        super(SparseAutoencoderKL, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_input, n_hidden),
            nn.ReLU(inplace=True),
            # nn.Linear(n_hidden, n_hidden//2),
            # nn.ReLU(inplace=True),
            # nn.Linear(128, 64),
            # nn.ReLU(inplace=True),
            # nn.Linear(n_hidden, n_output),
            # nn.ReLU(inplace=True),
        )

        self.decoder = nn.Sequential(
            # nn.Linear(n_hidden//2, n_hidden),
            # nn.ReLU(inplace=True),
            nn.Linear(n_hidden, n_output),
            nn.Sigmoid(),
            # nn.Linear(64, 128),
            # nn.ReLU(inplace=True),
            # nn.Linear(128, 784),
            # nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def kl_divergence(p, p_hat):
    funcs = nn.Sigmoid()
    p_hat = torch.mean(funcs(p_hat), 1)
    p_tensor = torch.Tensor([p] * len(p_hat)).to(device)
    return torch.sum(p_tensor * torch.log(p_tensor) - p_tensor * torch.log(p_hat) + (1 - p_tensor) * torch.log(1 - p_tensor) - (1 - p_tensor) * torch.log(1 - p_hat))

def sparse_loss(autoencoder, images):
    loss = 0
    values = images
    for i in range(1):
        fc_layer = list(autoencoder.encoder.children())[2 * i]
        relu = list(autoencoder.encoder.children())[2 * i + 1]
        values = fc_layer(values)
        loss += kl_divergence(DISTRIBUTION_VAL, values)
    for i in range(1):
        fc_layer = list(autoencoder.decoder.children())[2 * i]
        relu = list(autoencoder.decoder.children())[2 * i + 1]
        values = fc_layer(values)
        loss += kl_divergence(DISTRIBUTION_VAL, values)
    return loss


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


class EsnAe:
    def __init__(self,
            n_internal_units=None,
            spectral_radius=None,
            leak=None,
            connectivity=None,
            input_scaling=None,
            noise_level=None,
            n_drop=None,
            bidir=False,
            circle=False,
            # 
            embedding_size = 50, 
            n_epochs = 10,
            l_rate = 1e-3,
            batch_size = 120,
            pca_n_dim=None,
            w_out_mode=True,
    ):
        self.n_drop = n_drop
        self.bidir = bidir
        self.n_internal_units = n_internal_units
        self.embedding_size = embedding_size
        self.n_epochs = n_epochs
        self.l_rate = l_rate
        self.batch_size = batch_size
        self.pca_n_dim = pca_n_dim
        self.w_out_mode = w_out_mode
        self.reservoirRepr = False

        self.device = torch.device("cuda:" + str(0) if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            print("Using cuda device")
            torch.cuda.set_device(self.device)
        print(self.device)

        if self.pca_n_dim != None:
            self.dim_red = tensorPCA(n_components=pca_n_dim)

        self.reservoir = Reservoir(
            n_internal_units=n_internal_units,
            spectral_radius=spectral_radius,
            leak=leak,
            connectivity=connectivity,
            input_scaling=input_scaling,
            noise_level=noise_level,
            circle=circle
        )
        self.reservoir2 = Reservoir(
            n_internal_units=n_internal_units//2,
            spectral_radius=spectral_radius,
            leak=leak,
            connectivity=connectivity,
            input_scaling=input_scaling,
            noise_level=noise_level,
            circle=circle
        )
        self.reservoir3 = Reservoir(
            n_internal_units=n_internal_units//4,
            spectral_radius=spectral_radius,
            leak=leak,
            connectivity=connectivity,
            input_scaling=input_scaling,
            noise_level=noise_level,
            circle=circle
        )

        self.ridge_embedding = Ridge(alpha=10.0, fit_intercept=True)
        
        # Created in train method
        self.ae = None
    
    # Assumes data has shape N, T, V
    # Outputs feature vectors of shape N, DxV(x2 if bidir is True)
    def describe_weigths(self, X):
        N, T, V = X.shape

        if self.bidir:
            D = self.n_internal_units * 2
        else:
            D = self.n_internal_units


        H = self.reservoir.get_states(X, n_drop = self.n_drop, bidir = self.bidir)

        W_out = np.empty([N, D, V])

        for i in range(N):
            clf = Ridge(alpha=1.0)
            clf.fit(H[i,:,:], X[i,:,:])
            W_out[i] = clf.coef_.transpose()
            
        return W_out.reshape([N, -1])

    # Assumes data has shape N, T, V
    def train(self,
            X, 
            
    ):
        N, T, V = X.shape


        H = self.reservoir.get_states(X, n_drop = self.n_drop, bidir = self.bidir)
        # H2 = self.reservoir2.get_states(H, n_drop = self.n_drop, bidir = self.bidir)
        # H3 = self.reservoir3.get_states(H2, n_drop = self.n_drop, bidir = self.bidir)
        # H = np.concatenate([ H, H2, H3], axis = 2)

        # if self.bidir:
        #     D = self.n_internal_units * 2
        # else:
        D = H.shape[2]
        
        if self.pca_n_dim != None:
            RH = self.dim_red.fit_transform(H)
            R = self.pca_n_dim
        else:
            RH = H
            R = D
        # print(RH.shape)
        # print(R)

        # if self.reservoirRepr:
            
        #     coeff_tr = []
        #     biases_tr = []
        #     for i in range(X.shape[0]):
        #         self.ridge_embedding.fit(RH[i, 0:-1, :], RH[i, 1:, :])
        #         coeff_tr.append(self.ridge_embedding.coef_.ravel())
        #         biases_tr.append(self.ridge_embedding.intercept_.ravel())
        #     input_repr = np.concatenate((np.vstack(coeff_tr), np.vstack(biases_tr)), axis=1)
        #     return input_repr

        W_out = np.empty([N, R, V])
        W_out_i = np.empty([N, V])

        for i in range(N):
            clf = Ridge(alpha=1.0)
            clf.fit(RH[i,:,:], X[i,:,:])
            W_out[i] = clf.coef_.transpose()
            W_out_i[i] = clf.intercept_.transpose()

        # if self.reservoirRepr:
        #     for i in range(Xte.shape[0]):
        #         self._ridge_embedding.fit(red_states_te[i, 0:-1, :], red_states_te[i, 1:, :])
        #         coeff_te.append(self._ridge_embedding.coef_.ravel())
        #         biases_te.append(self._ridge_embedding.intercept_.ravel())
        #     input_repr_te = np.concatenate((np.vstack(coeff_te), np.vstack(biases_te)), axis=1)

        # * Get features from w_out
        # print(W_out.shape)
        # print(np.expand_dims(W_out_i, axis=1).shape)
        # Feat = np.concatenate([W_out, np.expand_dims(W_out_i, axis=1)], axis=1)
        # print(Feat.shape)
        Feat = W_out.reshape([N, -1])


        if self.w_out_mode:
            # return Feat
            return np.concatenate([Feat, W_out_i.reshape(N, -1)], axis=1)

        Feat = np.concatenate([Feat, W_out_i.reshape(N, -1)], axis=1)
        # * Train AE on features
        F = Feat.shape[1]
        print(self.embedding_size)
        self.ae = Autoencoder(F, self.embedding_size).to(self.device)
        # self.ae = SparseAutoencoderKL(F, self.embedding_size, F).to(self.device)
        self.ae.cuda()

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.ae.parameters(), lr=self.l_rate)

        Feat_tensor = torch.Tensor(Feat)
        H_tensor = torch.Tensor(RH)
        X_tensor = torch.Tensor(X)
        W_tensor = torch.Tensor(W_out)
        # W_out_i = np.array([np.repeat(e, T).reshape([T, -1]) for e in W_out_i], dtype=np.float)

        W_i_tensor = torch.Tensor(W_out_i)
        # print(W_i_tensor.shape)

        tensorDataset = TensorDataset(Feat_tensor, H_tensor, X_tensor, W_tensor, W_i_tensor)
        trainloader = DataLoader(tensorDataset, batch_size=self.batch_size)

        # torch.set_printoptions(precision=6)
        train_loss = []
        for epoch in range(self.n_epochs):
            running_loss = 0.0
            for feat, h, x, w, w_i in trainloader:
                feat = feat.to(self.device)
                x = x.to(self.device)
                h = h.to(self.device)
                w = w.to(self.device)
                w_i = w_i.to(self.device)

                optimizer.zero_grad()
                feat_r = self.ae(feat)
                # kl_loss = sparse_loss(self.ae, feat)
                feat_r = torch.reshape(feat_r, (-1 , R + 1, V))
                w_r = feat_r[:,:-1,:]
                w_i_r = feat_r[:,-1:,:]
                # print(w_i.shape)
                # w_i = w_i.repeat(w_i.shape[0], T)
                # w_i = w_i.unsqueeze(1).repeat(1, T, 1)
                # print(w_i.shape)
                # w_i = w_i.reshape(w_i.shape[0], T, V)
                x_r = torch.matmul(h, w_r) + w_i_r
                # print(x_r.shape)
                # print(w_i.shape)

                loss = criterion(x, x_r) 
                # loss = loss + kl_loss * SPARSE_REG

                # + criterion(w, w_r) + criterion(w_i, w_i_r)
                # loss = criterion(x, x_r) 
                # loss =  criterion(w, w_r) + criterion(w_i, w_i_r)
                # print(loss.item())
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            
            loss = running_loss / len(trainloader)
            train_loss.append(loss)
            if epoch % 50 == 0:
                print('Epoch {} of {}, Train Loss: {:.6f}'.format(
                    epoch+1, self.n_epochs, loss))

        encoded_repr = self.ae.hidden(Feat_tensor.to(self.device)).cpu().detach().numpy()
        # encoded_repr = self.ae.encoder(Feat_tensor.to(self.device)).cpu().detach().numpy()
        return encoded_repr

        # Assumes data has shape N, T, V
    def test(self,
            X
    ):
        N, T, V = X.shape

        # if self.bidir:
        #     D = self.n_internal_units * 2
        # else:
        #     D = self.n_internal_units

        H = self.reservoir.get_states(X, n_drop = self.n_drop, bidir = self.bidir)
        # H2 = self.reservoir2.get_states(H, n_drop = self.n_drop, bidir = self.bidir)
        # H3 = self.reservoir3.get_states(H2, n_drop = self.n_drop, bidir = self.bidir)
        # H = np.concatenate([ H, H2, H3], axis = 2)
        # print("H shape")
        # print(H.shape)
        # print(H2.shape)
        # print(H3.shape)

        D = H.shape[2]
        
        if self.pca_n_dim != None:
            RH = self.dim_red.transform(H)
            R = self.pca_n_dim
        else:
            RH = H
            R = D

        if self.reservoirRepr:
            coeff_tr = []
            biases_tr = []   
            for i in range(X.shape[0]):
                self.ridge_embedding.fit(RH[i, 0:-1, :], RH[i, 1:, :])
                coeff_tr.append(self.ridge_embedding.coef_.ravel())
                biases_tr.append(self.ridge_embedding.intercept_.ravel())
            input_repr = np.concatenate((np.vstack(coeff_tr), np.vstack(biases_tr)), axis=1)
            return input_repr

        W_out = np.empty([N, R, V])
        W_out_i = np.empty([N, V])

        for i in range(N):
            clf = Ridge(alpha=1.0)
            clf.fit(RH[i,:,:], X[i,:,:])
            W_out[i] = clf.coef_.transpose()
            W_out_i[i] = clf.intercept_.transpose()

        Feat = W_out.reshape([N, -1])
        # * Get features from w_out
        # Feat = np.concatenate([W_out, np.expand_dims(W_out_i, axis=1)], axis=1)
        # Feat = np.concatenate([Feat, W_out_i.reshape(N, -1)], axis=1)
        # return np.concatenate([Feat, W_out_i], axis=1)

        if self.w_out_mode:
            return np.concatenate([Feat, W_out_i.reshape(N, -1)], axis=1)
            return Feat
        Feat = np.concatenate([Feat, W_out_i.reshape(N, -1)], axis=1)

        Feat_tensor = torch.Tensor(Feat).to(self.device)
        
        encoded_repr = self.ae.hidden(Feat_tensor).cpu().detach().numpy()
        # encoded_repr = self.ae.encoder(Feat_tensor).cpu().detach().numpy()
        return encoded_repr



if __name__ == "__main__":
    n_internal_units=500
    spectral_radius=0.59
    leak=None
    connectivity=0.3
    input_scaling=0.2
    noise_level=0.01
    circle=False
    input_weights = None

    net = EsnAe(
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
        n_epochs=200,
    )


    X, y = loadDataset("motions")
    print(X.shape)

        
    mts_representations = net.train(X)
    # mts_representations = net.test(X)
    print(mts_representations.shape)

    similarity_matrix = cosine_similarity(mts_representations)
        
    # Normalize the similarity in [0,1]
    similarity_matrix = (similarity_matrix + 1.0)/2.0

    kpca = KernelPCA(n_components=2, kernel='precomputed')
    embeddings_pca = kpca.fit_transform(similarity_matrix)

    fig =  plt.figure(figsize=(10,8))
    plt.scatter(embeddings_pca[:,0], embeddings_pca[:,1], c=y[:,0], s=10, cmap='tab20')
    plt.title("Kernel PCA embeddings")
    plt.show()



