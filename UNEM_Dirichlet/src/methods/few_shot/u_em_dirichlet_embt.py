from src.utils import get_one_hot, Logger, clip_weights
from tqdm import tqdm
import torch
import time
from copy import deepcopy
import numpy as np
import torch.nn as nn


class U_EM_DIRICHLET_EMBT(nn.Module):
    def __init__(self, device, log_file, args, gamma=20.0, diff_gamma_layers=False):
        super(U_EM_DIRICHLET_EMBT, self).__init__()
        self.device = device
        self.n_layers = args.iter
        self.log_file = log_file
        self.logger = Logger(__name__, self.log_file)
        self.init_info_lists()
        self.args = args
        self.eps = 1e-15
        self.iter_mm = args.iter_mm
        self.diff_gamma_layers = diff_gamma_layers


        self.emb_t = nn.Parameter(torch.tensor(30., device='cuda'))
        self.gamma = nn.Parameter(torch.tensor(gamma, device='cuda')) if not diff_gamma_layers \
            else nn.ParameterList([nn.Parameter(torch.tensor(gamma, device='cuda')) for i in range(self.n_layers)])
        self.t = nn.Parameter(torch.tensor(1.0, device='cuda')) if not diff_gamma_layers \
            else nn.ParameterList([nn.Parameter(torch.tensor(1.0, device='cuda')) for i in range(self.n_layers)])
       
    def init_info_lists(self):
        self.timestamps = []
        self.criterions = []
        self.test_acc = []

    def record_convergence(self, new_time, criterions):
        """
        inputs:
            new_time : scalar
            criterions : torch.Tensor of shape [n_task]
        """
        self.criterions.append(criterions)
        self.timestamps.append(new_time)

    def compute_acc(self, y_q):
        """
        inputs:
            y_q : torch.Tensor of shape [n_task, n_query] :
        """

        preds_q = self.u.argmax(2)
        accuracy = (preds_q == y_q).float().mean(1, keepdim=True)
        self.test_acc.append(accuracy)

    def get_logs(self):
        self.criterions = torch.stack(self.criterions, dim=0).cpu().numpy()
        self.test_acc = torch.cat(self.test_acc, dim=1).cpu().numpy()
        return {'timestamps': np.array(self.timestamps).mean(), 'criterions': self.criterions,
                'acc': self.test_acc}


    def forward(self, X_support, y_support, X_query):
        # prepare the features
        X_support = (self.emb_t * X_support).softmax(-1)
        X_query = (self.emb_t * X_query).softmax(-1)

        # import pickle
        # with open('support1.pkl', 'wb') as f:
        #     pickle.dump(X_support, f)
        # with open('query1.pkl', 'wb') as f:
        #     pickle.dump(X_query, f)
            

        #torch.autograd.set_detect_anomaly(True)
        self.zero_value = torch.polygamma(1, torch.Tensor([1]).to(self.device)).float()
        self.log_gamma_1 = torch.lgamma(torch.Tensor([1]).to(self.device)).float()
        n_task, n_class = X_query.shape[0], self.args.num_classes_test

       # Initialization
        V = torch.zeros(n_task, n_class).to(self.device)        # dual variable set to zero

        U = X_query.clone()
        
        alpha = torch.ones((n_task, n_class, n_class)).to(self.device)
        y_s_one_hot = self.get_one_hot(y_support, n_class)


        X_support = X_support + self.eps
        X_query = X_query + self.eps
        X_support = X_support.log()
        X_query = X_query.log()

        for i in tqdm(range(self.n_layers)):
            gamma = self.gamma[i] if self.diff_gamma_layers else self.gamma
            t = self.t[i] if self.diff_gamma_layers else self.t

             # update of dirichlet parameter alpha
            y_s_sum = y_s_one_hot.sum(dim=1)  # Shape [n_task, num_class]
            u_sum = U.sum(dim=1)
            y_cst = (1 / (y_s_sum + u_sum)).unsqueeze(-1)
            y_cst = y_cst * ((y_s_one_hot.unsqueeze(-1) * X_support.unsqueeze(2)).sum(dim=1) + (U.unsqueeze(-1) * X_query.unsqueeze(2)).sum(dim=1))
            
            alpha = self.update_alpha(alpha, y_cst)

            # update on dual variable v
            V = torch.log(U.sum(1) / U.size(1) + self.eps) + 1

            # update on assignment variable u
            __, n_query = X_query.size(-1), X_query.size(1)

            l1 = torch.lgamma(alpha.sum(-1)).unsqueeze(1)
            l2 = - torch.lgamma(alpha).sum(-1).unsqueeze(1)
            l3 = ((alpha.unsqueeze(1) - 1) * X_query.unsqueeze(2)).sum(-1)
            logits = l1 + l2 + l3
        
            U = ((1/(nn.Softplus()(t)+1))*(logits + nn.Softplus()(gamma) * V.unsqueeze(1))).softmax(2)

            # compute criterion

        return U


    def get_one_hot(self, y_s, n_class):
        eye = torch.eye(n_class).to(y_s.device)
        one_hot = []
        for y_task in y_s:
            one_hot.append(eye[y_task].unsqueeze(0))
        one_hot = torch.cat(one_hot, 0)
        return one_hot
    

    def update_alpha(self, alpha_0, y_cst):
        alpha = alpha_0

        for l in range(self.iter_mm):
            curv, digam = self.curvature(alpha)
            b = digam - \
                torch.polygamma(0, alpha.sum(-1)).unsqueeze(-1) - curv * alpha
            b = b - y_cst
            a = curv
            delta = b**2 + 4 * a
            alpha_new = (- b + torch.sqrt(delta)) / (2 * a)

            if l > 0 and l % 50 == 0:
                criterion = torch.norm(
                    alpha_new - alpha)**2 / torch.norm(alpha)**2
                if l % 1000 == 0:
                    print('iter', l, 'criterion', criterion)
                if criterion < 1e-11:
                    break
            alpha = alpha_new
        return alpha_new

    def curvature(self, alpha):
        digam = torch.polygamma(0, alpha + 1)
        return torch.where(alpha > 1e-11, abs(2 * (self.log_gamma_1 - torch.lgamma(alpha + 1) + digam * alpha) / alpha**2), self.zero_value), digam




    def run_task(self, task_dic, shot=10):
        """
        inputs:
            task_dic : dictionnary with n_task few-shot tasks
            shot : scalar, number of shots
        """

        # Extract support and query
        y_s = task_dic['y_s']               # [n_task, shot]
        y_q = task_dic['y_q']               # [n_task, n_query]
        support = task_dic['x_s']           # [n_task, shot, feature_dim]
        query = task_dic['x_q']             # [n_task, n_query, feature_dim]

        # Transfer tensors to GPU if needed
        support = support.to(self.device)
        query = query.to(self.device)
        y_s = y_s.long().squeeze(2).to(self.device)
        y_q = y_q.long().squeeze(2).to(self.device)
        del task_dic

        # Run adaptation
        with torch.no_grad():
            Us = self.forward(support, y_s, query)
            # Compute accuracy

            preds_q = Us.argmax(2)
            accuracy = (preds_q == y_q).float().mean(1, keepdim=True).cpu().numpy()
            # if self.verbose:
            #     print('ACCURACY:', accuracy)
            #print('PADNET ACCURACY:', accuracy)

        return {'acc':  accuracy}