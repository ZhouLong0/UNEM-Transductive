import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import torch
import time
from numpy import linalg as LA
import numpy as np
import math
from scipy import sparse
import matplotlib
matplotlib.use('Agg')
from sklearn.neighbors import NearestNeighbors
from ..utils import get_metric, Logger, extract_features


class U_Paddle(nn.Module):
    def __init__(self, n_layers, device, log_file, gamma=1.0, verbose=False, diff_gamma_layers=False):
        """
        params:
            gamma: float, penalty parameter
        
        """
        super(U_Paddle, self).__init__()
        self.verbose = verbose
        self.n_layers = n_layers
        self.diff_gamma_layers = diff_gamma_layers

        if not diff_gamma_layers:
            self.gamma = nn.Parameter(torch.tensor(gamma, device='cuda'))
        else:
            self.gamma = nn.ParameterList([nn.Parameter(torch.tensor(gamma, device='cuda')) for i in range(self.n_layers)])

        #self.last_layer = Pad_last_layer(gamma=self.gamma)

        self.device = device
        self.log_file = log_file
        self.logger = Logger(__name__, self.log_file)
        self.init_info_lists()
    
    def init_info_lists(self):
        self.timestamps = []
        self.test_acc = []


    def forwardold(self, Z_support, Y_support, Z_query, device):
        """"
        params:
            Z_query: torch.tensor of shape (Batch, Q_size, d) 
            Z_support: torch.tensor of shape (Batch, S_size, d)
            Y_support: torch.tensor of shape (Batch, S_size)
        """
        Z_query = Z_query.to(device)
        Z_support = Z_support.to(device)
        Y_support = Y_support.to(device)
        U_support = self.get_one_hot(Y_support).to(device)
        #U_support = F.one_hot(Y_support.long()).to(device)  
        k = len(torch.unique(Y_support))
        n_task, n_ways = Z_support.size(0), U_support.size(2)

        W = self.init_w(Z_support, Y_support).to(device)

        V = torch.zeros(n_task, n_ways).to(device)
        
        for layer in tqdm(self.layers):
            Us, W, V = layer(W, V, Z_support, U_support, Z_query)
        #Us = self.last_layer(W, Pi, Z_support, U_support, Z_query, S, device)
        return Us
    
    def forward(self, Z_support, Y_support, Z_query, device):
        """
        Runs UNEM on a task (support and query set) 

        params:
            Z_query: torch.tensor of shape (Q_size, d) 
            Z_support: torch.tensor of shape (n_classes * n_shots, d)
            Y_support: torch.tensor of shape (n_classes * n_shots)
        """
        Z_query = Z_query.to(device)
        Z_support = Z_support.to(device)
        Y_support = Y_support.to(device)
        U_support = self.get_one_hot(Y_support).to(device)
        #U_support = F.one_hot(Y_support.long()).to(device)  


        k = len(torch.unique(Y_support))
        n_task, n_ways = Z_support.size(0), U_support.size(2)

        # Initialize the centroids with the support set
        W = self.init_w(Z_support, Y_support).to(device)

        # Initialize the class proportions
        V = torch.zeros(n_task, n_ways).to(device)+1

        n_query = Z_query.size(1)
        n_tasks = Z_query.size(0)

        for i in tqdm(range(self.n_layers)):
            # ratio of the class proportions
            gamma = self.gamma[i] if self.diff_gamma_layers else self.gamma

            # update the logits
            logits = (Z_query.matmul(W.transpose(1, 2)) \
                    - 1 / 2 * (W**2).sum(2).view(n_tasks, 1, -1) \
                    - 1 / 2 * (Z_query**2).sum(2).view(n_tasks, -1, 1))

            # update the assignment matrix
            U_new = (logits + gamma * (V.unsqueeze(1).repeat(1, n_query, 1))).softmax(2)

            # update the class proportions
            V = torch.log(U_new.sum(1) / n_query + 1e-6) + 1

            num = torch.einsum('bkq,bqd->bkd',torch.transpose(U_new, 1, 2), Z_query) \
                    + torch.einsum('bkq,bqd->bkd',torch.transpose(U_support, 1, 2), Z_support)
            den  = U_new.sum(1) + U_support.sum(1)

            # update the centroids
            W = torch.div(num, den.unsqueeze(2))
        
        return U_new


    

    
    def PT(self, datas, beta=0.5):
        # ------------------------------------PT-MAP-----------------------------------------------
        nve_idx = np.where(datas.cpu().detach().numpy() < 0)
        datas[nve_idx] *= -1
        datas[:, ] = torch.pow(datas[:, ] + 1e-6, beta)
        datas[nve_idx] *= -1  # return the sign
        return datas
    
    def centerData(self, datas):
        # PT code
        #    datas[:, :] = datas[:, :, :] - datas[:, :].mean(1, keepdim=True)
        #   datas[:, :] = datas[:, :, :] / torch.norm(datas[:, :, :], 2, 2)[:, :, None]
        # centre of mass of all data support + querries
        datas[:, :] -= datas[:, :].mean(1, keepdim=True)  # datas[:, :, :] -
        norma = torch.norm(datas[:, :, :], 2, 2)[:, :, None].detach()
        datas[:, :, :] /= norma

        return datas
    
    def scaleEachUnitaryDatas(self, datas):
        norms = datas.norm(dim=2, keepdim=True)
        return datas / norms
    

    def run_task(self, task_dic, shot):
        # Extract support and query
        y_s, y_q = task_dic['y_s'], task_dic['y_q']
        x_s, x_q = task_dic['x_s'], task_dic['x_q']
        train_mean = task_dic['train_mean'].unsqueeze(0).unsqueeze(0)
        # print(task_dic['train_mean'].shape)
        # Transfer tensors to GPU if needed
        support = x_s.to(self.device)  # [ N * (K_s + K_q), d]
        query = x_q.to(self.device)  # [ N * (K_s + K_q), d]
        y_s = y_s.long().squeeze(2).to(self.device)
        y_q = y_q.long().squeeze(2).to(self.device)
        # Extract features
        
        # scaler = MinMaxScaler(feature_range=(0, 1))
        # query, support = scaler(query, support)

        # predictions
        with torch.no_grad():
            Us = self.forward(support, y_s, query, self.device)
            # Compute accuracy
            preds_q = Us.argmax(2)
            accuracy = (preds_q == y_q).float().mean(1, keepdim=True).cpu().numpy()
            if self.verbose:
                print('PADNET ACCURACY:', accuracy)
            #print('PADNET ACCURACY:', accuracy)

        return {'acc':  accuracy}
    
    def get_one_hot(self, y_s):
        n_ways = torch.unique(y_s).size(0)
        eye = torch.eye(n_ways).to(y_s.device)
        one_hot = []
        for y_task in y_s:
            one_hot.append(eye[y_task].unsqueeze(0))
        one_hot = torch.cat(one_hot, 0)
        return one_hot
    
    def init_w(self, support, y_s):
        """
        inputs:
            support : torch.Tensor of shape [n_task, shot, feature_dim]
            y_s : torch.Tensor of shape [n_task, shot]

        updates :
            self.w : torch.Tensor of shape [n_task, num_class, feature_dim]
        """
        n_tasks = support.size(0)
        one_hot = self.get_one_hot(y_s)
        counts = one_hot.sum(1).view(n_tasks, -1, 1)
        weights = one_hot.transpose(1, 2).matmul(support)
        return weights / counts

        





class MinMaxScaler(object):
    """MinMax Scaler

    Transforms each channel to the range [a, b].

    Parameters
    ----------
    feature_range : tuple
        Desired range of transformed data.
    """

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __call__(self, query, support):

        dist = (query.max(dim=1, keepdim=True)[0] - query.min(dim=1, keepdim=True)[0])
        dist[dist==0.] = 1.
        scale = 1.0 /  dist
        ratio = query.min(dim=1, keepdim=True)[0]
        query.mul_(scale).sub_(ratio)
        support.mul_(scale).sub_(ratio)
        return query, support