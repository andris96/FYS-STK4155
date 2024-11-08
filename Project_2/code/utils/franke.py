# Franke function class for generating data and plotting the Franke function
# The code is based on the Franke function from the lecture notes in FYS-STK4155

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

class FrankeFunction():
    def __init__(self, mesh_start=0, mesh_stop=1, n=101, eps=0.1, seed=42):
        self.mesh_start = mesh_start
        self.mesh_stop = mesh_stop
        self.n = n
        self.eps = eps
        self.seed = seed
        self.x1, self.x2 = self.generate_mesh()
        self.y_mesh = self.franke_function()

    def generate_mesh(self):
        x1 = np.linspace(self.mesh_start, self.mesh_stop, self.n)
        x2 = np.linspace(self.mesh_start, self.mesh_stop, self.n)
        return np.meshgrid(x1, x2)
    
    def franke_function(self):
        np.random.seed(self.seed)
        term1 = 0.75 * np.exp(-(0.25*(9*self.x1-2)**2) - 0.25*((9*self.x2-2)**2))
        term2 = 0.75 * np.exp(-((9*self.x1+1)**2)/49.0 - 0.1*(9*self.x2+1))
        term3 = 0.5 * np.exp(-(9*self.x1-7)**2/4.0 - 0.25*((9*self.x2-3)**2))
        term4 = -0.2 * np.exp(-(9*self.x1-4)**2 - (9*self.x2-7)**2)
        return term1 + term2 + term3 + term4 + self.eps*np.random.randn(self.n, self.n)
    
    def generate_data(self):
        self.X = np.c_[self.x1.ravel(), self.x2.ravel()]
        self.y = self.y_mesh.ravel()
        return self.X, self.y

    def create_design_matrix(self, degree=5):
        x1 = self.x1.ravel()
        x2 = self.x2.ravel()
        l = int((degree+1)*(degree+2)/2)
        X = np.ones((len(x1), l))
        for i in range(1, degree+1):
            q = int((i)*(i+1)/2)
            for k in range(i+1):
                X[:, q+k] = (x1**(i-k))*(x2**k)
        return X

    def plot_franke(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(self.x1, self.x2, self.y_mesh, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        plt.show()