import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

from mlp import MultilayerPerceptron
class AutoEncoder():
    def __init__(self, input_dim, encoder_layers, decoder_layers, latent_dim, 
                 n_iter=200, lr=1e-3, tol=None, train_mode='MBGD', 
                 batch_size = 10, type ='regression'):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_num = 0

        self.autoencoder = MultilayerPerceptron(n_iter, lr, tol, train_mode, batch_size , type)
        self.autoencoder.build_layer(self.input_dim, encoder_layers[0][0], encoder_layers[0][1])
        for i in range(len(encoder_layers)-1):
            self.autoencoder.build_layer(encoder_layers[i][0], encoder_layers[i+1][0], encoder_layers[i+1][1])
            self.hidden_num += 1
        self.autoencoder.build_layer(encoder_layers[-1][0], self.latent_dim, 'linear')

        self.autoencoder.build_layer(self.latent_dim, decoder_layers[0][0], decoder_layers[0][1])
        for i in range(len(decoder_layers)-1):
            self.autoencoder.build_layer(decoder_layers[i][0], decoder_layers[i+1][0], decoder_layers[i+1][1])
        self.autoencoder.build_layer(decoder_layers[-1][0], self.input_dim, 'linear')

    def train(self, X):
        self.autoencoder.train(X, X)
    
    def plot_loss(self):
        self.autoencoder.plot_loss()

    def reconstructe(self, X):
        self.autoencoder.forward(X)
        return self.autoencoder.H[-1]
    
    def get_latent(self, X):
        self.autoencoder.forward(X)
        return self.autoencoder.H[self.hidden_num+2]



if __name__ == '__main__':
    df_wine = pd.read_csv('principle_components_analysis/wine.data', header=None)
    df_wine.head()

    X,y = df_wine.iloc[:,1:].values, df_wine.iloc[:,0].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=0)

    sc = StandardScaler()
    X_train_std = sc.fit_transform(X_train)
    X_test_std = sc.transform(X_test)
    # print(X_train_std.shape)
    _, n_feature = X_train_std.shape

    model = AutoEncoder(n_feature,
                        [(32, 'relu'),], [(32, 'relu')], 
                        latent_dim=2, n_iter=10000, lr=0.01, tol=0.00001, 
            train_mode='MBGD',batch_size=124, type='regression')
    model.train(X_train_std)
    model.plot_loss()

    latent = model.get_latent(X_test_std)
    # print(latent.shape)
    colors = ['r', 'b', 'g']
    markers = ['s', 'x', 'o']

    for l, c, m in zip(np.unique(y_test), colors, markers):
        idx = np.where(y_test == l)
        plt.scatter(latent[idx, 0], latent[idx, 1], c=c, label=f'Class {l}', marker=m)

    plt.xlabel('Latent Dimension 1')
    plt.ylabel('Latent Dimension 2')
    # plt.legend(loc='lower left')
    plt.title('Latent Space Representation')
    plt.show()


    reconstructed_std = model.reconstructe(X_train_std)
    X_reconstructed = sc.inverse_transform(reconstructed_std)
    # print(X_reconstructed)
    reconstruction_error = np.mean((X_train_std - reconstructed_std) ** 2, axis=1)
    print("平均重建误差：", np.mean(reconstruction_error))