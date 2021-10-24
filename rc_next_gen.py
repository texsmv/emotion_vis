import numpy as np
from sklearn.linear_model import Ridge
from data_utils import loadDataset
import torch 
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import KernelPCA
from sklearn.decomposition import PCA
from tensorPCA import tensorPCA

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


class NextGenAE:
    def __init__(self,
            embedding_size = 50, 
            n_epochs = 30,
            l_rate = 1e-3,
            batch_size = 64,
            pca_n_dim=None,
            w_out_mode=True,
    ):
        self.embedding_size = embedding_size
        self.n_epochs = n_epochs
        self.l_rate = l_rate
        self.batch_size = batch_size
        self.pca_n_dim = pca_n_dim
        self.w_out_mode = w_out_mode

        # Created in train method
        self.ae = None
        self.ridge_embedding = Ridge(alpha=10.0, fit_intercept=True)
        self.reservoirRepr = False
        self.dim_red = tensorPCA(n_components=50)
    
    def process_instance(self,X):
        # input dimension
        d = X.shape[0]
        # number of time delay taps
        k = 3
        # number of time steps between taps. skip = 1 means take consecutive points
        skip = 1
        # size of linear part of feature vector (leave out z)
        dlin = k*(d)
        # size of nonlinear part of feature vector
        dnonlin = int(dlin*(dlin+1)/2)
        # total size of feature vector: linear + nonlinear
        dtot = dlin + dnonlin

        maxtime_pts = X.shape[1]
        print("------------")
        # print(X.shape)

        x = np.ones((dlin,maxtime_pts)) * -2

        # out = np.ones((dtot+1,maxtime_pts)) * -1
        out = []

        for delay in range(k):
            for j in range(0,maxtime_pts):
                # only include x and y
                # print(j)
                # print(X[:,j-delay*skip].shape)
                # print(x[(d)*delay:(d)*(delay+1),j].shape)
                x[(d)*delay:(d)*(delay+1),j]=X[:,j-delay*skip]

        # out[1:dlin+1,:]=x[:,:maxtime_pts]
            
        # print(x[:, :])

        # fill in the non-linear part
        cnt=0
        for row in range(dlin):
            for column in range(0,dlin):
                # shift by one for constant
                # out[dlin+1+cnt,:]=x[row]*x[column]
                out += [x[row]*x[column]]
                # cnt += 1
        out = np.array(out)
        print(x.shape)
        print(out.shape)
        out = np.concatenate([x, out], axis=0)

        # print(f"out shape: {out.shape}")

        # print(out)

        clf = Ridge(alpha=1.0)
        clf.fit(out.transpose(), X.transpose())
        w_out = clf.coef_
        i_out = clf.intercept_
        # print(clf.predict(out.transpose())[0])
        # print(x.shape)
        # print(out.shape)
        # print(W_out.shape)
        # print(np.matmul(W_out, out).transpose()[0] + clf.intercept_)
        # print(X.transpose()[0])
        return out, w_out, i_out
    # Assumes data has shape N, T, V
    def train(self,
            X,test: False,
    ):
        N, T, V = X.shape

        res = [self.process_instance(X[i].transpose()) for i in range(N)]
        H = np.array([res[i][0] for i in range(N)])
        W_out = np.array([res[i][1] for i in range(N)])
        I_out = np.array([res[i][2] for i in range(N)])

        R = W_out.shape[2]

        print(H.shape)
        print(W_out.shape)
        # print(f"H shape: {H.shape}")
        # print(X[0].shape)
        # print(H[0].shape)
        if self.reservoirRepr:
            # if test:
            #     H = self.dim_red.transform(H)        
            # else:
            #     H = self.dim_red.fit_transform(H)        
            # print(H)
            # print(f"H: {H.shape}")
            coeff_tr = []
            biases_tr = []   
            for i in range(X.shape[0]):
                self.ridge_embedding.fit(H[i, 0:-1, :], H[i, 1:, :])
                coeff_tr.append(self.ridge_embedding.coef_.ravel())
                biases_tr.append(self.ridge_embedding.intercept_.ravel())
            input_repr = np.concatenate((np.vstack(coeff_tr), np.vstack(biases_tr)), axis=1)
            return input_repr

        # print((np.matmul(H[0].transpose(), W_out[0].transpose()) + I_out[0])[0])
        # print(X[0][0])

        print(f"W_out shape: {W_out.shape}")
        print(f"I_out shape: {I_out.shape}")
        if self.w_out_mode:
            I_out = np.expand_dims(I_out, axis=2)
            # Feat = np.concatenate([W_out, I_out], axis=2)
            # print(f"Feat shape: {Feat.shape}")
            # Feat = Feat.reshape([N, -1])
            Feat = W_out.reshape([N, -1])
            # pca = PCA(n_components=20)
            # return pca.fit_transform(Feat)

            # return Feat

            F = Feat.shape[1]

            if test:
                Feat_tensor = torch.Tensor(Feat)
                encoded_repr = self.ae.hidden(Feat_tensor).detach().numpy()
                return encoded_repr

            self.ae = Autoencoder(F, self.embedding_size)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(self.ae.parameters(), lr=self.l_rate)

            Feat_tensor = torch.Tensor(Feat)
            H_tensor = torch.Tensor(H.transpose([0,2,1]))
            X_tensor = torch.Tensor(X)
            W_tensor = torch.Tensor(W_out.transpose([0,2,1]))

            tensorDataset = TensorDataset(Feat_tensor, H_tensor, X_tensor, W_tensor)
            trainloader = DataLoader(tensorDataset, batch_size=self.batch_size)

            train_loss = []
            for epoch in range(self.n_epochs):
                running_loss = 0.0
                for feat, h, x, w in trainloader:
                    optimizer.zero_grad()
                    feat_r = self.ae(feat)
                    # print(f"R: {R}  - V: {V}")
                    w_r = torch.reshape(feat_r, (-1 , R, V))
                    # print(f"h: {h.shape}  - w: {w_r.shape}")
                    x_r = torch.matmul(h, w_r)
                    
                    loss = criterion(x, x_r) + criterion(w, w_r)
                    # loss = criterion(x, x_r) 
                    # loss =  criterion(w, w_r)
                    # print(loss.item())
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                
                loss = running_loss / len(trainloader)
                train_loss.append(loss)
                if epoch % 10 == 0:
                    print('Epoch {} of {}, Train Loss: {:.3f}'.format(
                        epoch+1, self.n_epochs, loss))

            encoded_repr = self.ae.hidden(Feat_tensor).detach().numpy()
            return encoded_repr
        else:
            print("RAW FEAT")
            return H.reshape([N, -1])
        



        
        # if self.pca_n_dim != None:
        #     RH = self.dim_red.fit_transform(H)
        #     R = self.pca_n_dim
        # else:
        #     RH = H
        #     R = D
        # print(RH.shape)
        # print(R)

        # W_out = np.empty([N, R, V])
        # W_out_i = np.empty([N, V])

        # for i in range(N):
        #     clf = Ridge(alpha=1.0)
        #     clf.fit(RH[i,:,:], X[i,:,:])
        #     W_out[i] = clf.coef_.transpose()
        #     W_out_i[i] = clf.intercept_.transpose()

        # # * Get features from w_out
        # Feat = W_out.reshape([N, -1])
        # print(Feat.shape)
        # print(W_out_i.shape)
        # # return np.concatenate([Feat, W_out_i], axis=1)

        # if self.w_out_mode:
        #     return Feat

        # # * Train AE on features
        # F = Feat.shape[1]

        # self.ae = Autoencoder(F, self.embedding_size)
        # criterion = nn.MSELoss()
        # optimizer = optim.Adam(self.ae.parameters(), lr=self.l_rate)

        # Feat_tensor = torch.Tensor(Feat)
        # H_tensor = torch.Tensor(RH)
        # X_tensor = torch.Tensor(X)

        # tensorDataset = TensorDataset(Feat_tensor, H_tensor, X_tensor)
        # trainloader = DataLoader(tensorDataset, batch_size=self.batch_size)

        # train_loss = []
        # for epoch in range(self.n_epochs):
        #     running_loss = 0.0
        #     for feat, h, x in trainloader:
        #         optimizer.zero_grad()
        #         feat_r = self.ae(feat)
        #         w_r = torch.reshape(feat_r, (-1 , R, V))
        #         x_r = torch.matmul(h, w_r)
                
        #         loss = criterion(x, x_r)
        #         # print(loss.item())
        #         loss.backward()
        #         optimizer.step()
        #         running_loss += loss.item()
            
        #     loss = running_loss / len(trainloader)
        #     train_loss.append(loss)
        #     if epoch % 10 == 0:
        #         print('Epoch {} of {}, Train Loss: {:.3f}'.format(
        #             epoch+1, self.n_epochs, loss))

        # encoded_repr = self.ae.hidden(Feat_tensor).detach().numpy()
        # return encoded_repr

        # Assumes data has shape N, T, V
    # def test(self,
    #         X
    # ):
    #     N, T, V = X.shape

    #     if self.bidir:
    #         D = self.n_internal_units * 2
    #     else:
    #         D = self.n_internal_units

    #     H = self.reservoir.get_states(X, n_drop = self.n_drop, bidir = self.bidir)

        
    #     if self.pca_n_dim != None:
    #         RH = self.dim_red.transform(H)
    #         R = self.pca_n_dim
    #     else:
    #         RH = H
    #         R = D

    #     W_out = np.empty([N, R, V])
    #     W_out_i = np.empty([N, V])

    #     for i in range(N):
    #         clf = Ridge(alpha=1.0)
    #         clf.fit(RH[i,:,:], X[i,:,:])
    #         W_out[i] = clf.coef_.transpose()
    #         W_out_i[i] = clf.intercept_.transpose()

    #     # * Get features from w_out
    #     Feat = W_out.reshape([N, -1])
    #     # return np.concatenate([Feat, W_out_i], axis=1)

    #     if self.w_out_mode:
    #         return Feat

    #     Feat_tensor = torch.Tensor(Feat)
        
    #     encoded_repr = self.ae.hidden(Feat_tensor).detach().numpy()
    #     return encoded_repr



if __name__ == "__main__":
    n_internal_units=500
    spectral_radius=0.59
    leak=None
    connectivity=0.3
    input_scaling=0.2
    noise_level=0.01
    circle=False
    input_weights = None

    net = NextGenAE(
    )


    X, y = loadDataset("motions")
    print(X.shape)

        
    mts_representations = net.train(X)
    # mts_representations = net.test(X)
    print(mts_representations.shape)

    similarity_matrix = cosine_similarity(mts_representations)
        
    similarity_matrix = (similarity_matrix + 1.0)/2.0

    kpca = KernelPCA(n_components=2, kernel='precomputed')
    embeddings_pca = kpca.fit_transform(similarity_matrix)

    fig =  plt.figure(figsize=(10,8))
    plt.scatter(embeddings_pca[:,0], embeddings_pca[:,1], c=y[:,0], s=10, cmap='tab20')
    plt.title("Kernel PCA embeddings")
    plt.show()



# X, y = loadDataset("motions")
# X = X[4]
# X = X.transpose()
# # print(f"X shape: {X.shape}")

# # input dimension
# d = X.shape[0]
# # number of time delay taps
# k = 4
# # number of time steps between taps. skip = 1 means take consecutive points
# skip = 5
# # size of linear part of feature vector (leave out z)
# dlin = k*(d)
# # size of nonlinear part of feature vector
# dnonlin = int(dlin*(dlin+1)/2)
# # total size of feature vector: linear + nonlinear
# dtot = dlin + dnonlin

# maxtime_pts = X.shape[1]

# x = np.zeros((dlin,maxtime_pts))

# out = np.ones((dtot+1,maxtime_pts))

# print(f"x shape: {x.shape}")
# print(f"X shape: {X.shape}")

# # fill in the linear part of the feature vector for all times
# for delay in range(k):
#     for j in range(delay,maxtime_pts):
#         # only include x and y
#         # print(j)
#         x[(d)*delay:(d)*(delay+1),j]=X[:,j-delay*skip]

# out[1:dlin+1,:]=x[:,:maxtime_pts]

# # fill in the non-linear part
# cnt=0
# for row in range(dlin):
#     for column in range(row,dlin):
#         # shift by one for constant
#         out[dlin+1+cnt,:]=x[row,:maxtime_pts]*x[column,:maxtime_pts]
#         cnt += 1

# print(f"out shape: {out.shape}")

# print(out)

# clf = Ridge(alpha=1.0)
# clf.fit(out.transpose(), X.transpose())
# W_out = clf.coef_
# print(W_out.shape)
# print(clf.score(out.transpose(), X.transpose()))

# print("---")
# print(clf.predict(out.transpose())[0])
# print(X.transpose()[0])