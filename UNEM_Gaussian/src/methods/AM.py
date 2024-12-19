import torch.nn.functional as F
from src.utils import get_mi, get_cond_entropy, get_entropy, get_one_hot, Logger, extract_features
from tqdm import tqdm
import torch
import time
import numpy as np
import scipy as sp
from scipy.stats import t


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), sp.stats.sem(a)
    h = se * t._ppf((1+confidence)/2., n-1)
    return m, h
class ALPHA_AM(object):
    def __init__(self, model, device, log_file, args):
        self.device = device
        self.args = args
        self.phi = args.phi
        self.temp = args.temp
        self.k = args.K
        self.alpha = args.alpha
        self.loss_weights = args.loss_weights.copy()
        self.iter = args.iter
        self.lr = float(args.lr_lp)
        self.model = model
        self.log_file = log_file
        self.logger = Logger(__name__, self.log_file)
        self.init_info_lists()
        # self.lr = float(args.lr_alpha_tim)
        self.PLC = args.PLC
        self.entropies = args.entropies.copy()
        self.alpha_values = args.alpha_values
        self.use_tuned_alpha_values = args.use_tuned_alpha_values

    def __del__(self):
        self.logger.del_logger()

    def init_info_lists(self):
        self.timestamps = []
        self.mutual_infos = []
        self.entropy = []
        self.cond_entropy = []
        self.test_acc = []
        self.losses = []

    def get_logits(self, samples):
        """
        inputs:
            samples : torch.Tensor of shape [n_task, shot, feature_dim]

        returns :
            logits : torch.Tensor of shape [n_task, shot, num_class]
        """
        n_tasks = samples.size(0)
        logits = self.temp * (samples.matmul(self.weights.transpose(1, 2)) \
                              - 1 / 2 * (self.weights**2).sum(2).view(n_tasks, 1, -1) \
                              - 1 / 2 * (samples**2).sum(2).view(n_tasks, -1, 1))  #
        return logits

    def get_preds(self, samples):
        """
        inputs:
            samples : torch.Tensor of shape [n_task, s_shot, feature_dim]

        returns :
            preds : torch.Tensor of shape [n_task, shot]
        """
        logits = self.get_logits(samples)
        preds = logits.argmax(2)
        return preds

    def init_weights(self, support, query, y_s, y_q):
        """
        inputs:
            support : torch.Tensor of shape [n_task, s_shot, feature_dim]
            query : torch.Tensor of shape [n_task, q_shot, feature_dim]
            y_s : torch.Tensor of shape [n_task, s_shot]
            y_q : torch.Tensor of shape [n_task, q_shot]
        updates :
            self.weights : torch.Tensor of shape [n_task, num_class, feature_dim]
        """
        pass
        n_tasks = support.size(0)
        one_hot = get_one_hot(y_s)
        counts = one_hot.sum(1).view(n_tasks, -1, 1)
        weights = one_hot.transpose(1, 2).matmul(support)
        weights = weights / counts
        return weights

    def compute_lambda(self, support, query, y_s):
        """
        inputs:
            support : torch.Tensor of shape [n_task, s_shot, feature_dim]
            query : torch.Tensor of shape [n_task, q_shot, feature_dim]
            y_s : torch.Tensor of shape [n_task, s_shot]

        updates :
            self.loss_weights[0] : Scalar
        """
        self.N_s, self.N_q = support.size(1), query.size(1)
        self.num_classes = torch.unique(y_s).size(0)
        if self.loss_weights[0] == 'auto':
            self.loss_weights[0] = (1 + self.loss_weights[2]) * self.N_s / self.N_q

    def get_alpha_values(self, shot):
        if shot == 1:
            self.alpha_values = [2.0, 2.0, 2.0]
        elif shot >= 5:
            self.alpha_values = [5.0, 5.0, 5.0]

    def record_info(self, Z, y_q):
        """
        inputs:
            support : torch.Tensor of shape [n_task, s_shot, feature_dim]
            query : torch.Tensor of shape [n_task, q_shot, feature_dim]
            y_s : torch.Tensor of shape [n_task, s_shot]
            y_q : torch.Tensor of shape [n_task, q_shot] :
        """
        preds_q = Z.argmax(2)
        self.test_acc.append((preds_q == y_q).float().mean(1, keepdim=True))

    def get_logs(self):
        self.test_acc = torch.cat(self.test_acc, dim=1).cpu().numpy()
        return {'acc': self.test_acc}

    def nc_regularizer(self, query, P):
        A = torch.bmm(query, query.permute(0, 2, 1)) - torch.eye(query.size(1)).unsqueeze(0).cuda() * 2
        A = torch.pow(A, 3)  # power transform
        A = torch.clamp(A, min=0)

        _, idx = torch.topk(A, k=self.k, dim=2)
        mask = torch.zeros_like(A)  # (bs, n_way*n_sup+n_way*n_query, n_way*n_sup+n_way*n_query)
        mask = mask.scatter(-1, idx, 1)  # (bs, n_way*n_sup+n_way*n_query, n_way*n_sup+n_way*n_query)
        A = A * mask
        # A = F.normalize(A, p=1, dim=2)
        P_nn = []
        for i in range(A.size(0)):
            P_nn.append(P[i][idx[i]])
        P_nn = torch.stack(P_nn, dim=0)
        dist = torch.pow(P.unsqueeze(2) - P_nn, 2).mean(3)
        dist = torch.bmm(A, dist).mean(2).mean(1).sum(0)
        # print(P.shape, A.shape, idx.shape, P_nn.shape, dist.shape)
        # print(P[idx].shape)
        return dist

    def laplacian_eigenmaps(self, fs, qs, alpha, k, ys, yq):
        batch, sup, dim = fs.shape
        z_all = torch.cat([fs, qs], dim=1)
        sigma = (torch.ones(z_all.size(0), z_all.size(1), z_all.size(1))).cuda()
        sq_dist = torch.cdist(z_all, z_all)
        # mask = sq_dist != 0
        stds = sq_dist.std(dim=(1, 2), keepdim=True)
        # print(stds.shape)
        sq_dist = sq_dist / stds  # sq_dist[mask].std()
        # weights = torch.exp(-sq_dist * 1)
        weights = torch.exp(-torch.mul(sq_dist, sigma))
        mask = torch.eye(weights.size(2), dtype=torch.bool, device=weights.device)
        A = weights * (~mask).float()
        A = torch.clamp(A, min=0)  # non-negative
        # A[:, :n_mus] = 0
        N = A.size(1)  # N = n_way*n_sup+n_way*n_query
        A[:, range(N), range(N)] = 0  # zero diagonal
        # graph construction
        _, indices = torch.topk(A, k, dim=-1)  # (bs, n_way*n_sup+n_way*n_query, topk)
        mask = torch.zeros_like(A)  # (bs, n_way*n_sup+n_way*n_query, n_way*n_sup+n_way*n_query)
        mask = mask.scatter(-1, indices, 1)  # (bs, n_way*n_sup+n_way*n_query, n_way*n_sup+n_way*n_query)
        A = A * mask  # (bs, n_way*n_sup+n_way*n_query, n_way*n_sup+n_way*n_query)
        W = (A + A.permute(0, 2, 1)) * 0.5  # (bs, n_way*n_sup+n_way*n_query, n_way*n_sup+n_way*n_query)
        D = W.sum(-1)  # (bs, n_way*n_sup+n_way*n_query)
        print(D.shape)
        D_sqrt_inv = torch.sqrt(1.0 / (D + 1e-12))  # (bs, n_way*n_sup+n_way*n_query)
        D1 = torch.unsqueeze(D_sqrt_inv, 2).repeat(1, 1, N)  # (bs, n_way*n_sup+n_way*n_query, {1})
        D2 = torch.unsqueeze(D_sqrt_inv, 1).repeat(1, N, 1)  # (bs, {1}, n_way*n_sup+n_way*n_query)
        #-----------------------------calculate the laplacian matrix----------------------------------------------------
        # I = torch.eye(W.size(1))  # (n_way*n_sup+n_way*n_query, n_way*n_sup+n_way*n_query)
        # I = I.unsqueeze(0).repeat(W.size(0), 1, 1).cuda()
        # W = D.unsqueeze(2).repeat(1,1,N)-W
        #---------------------------------------------------------------------------------------------------------------
        W = D1 * W * D2  # (bs, n_way*n_sup+n_way*n_query, n_way*n_sup+n_way*n_query)
        ss, qq = [], []
        from sklearn.manifold import spectral_embedding
        for i in range(len(W)):
            e = spectral_embedding(n_components=4, adjacency=W[i].cpu().numpy(), norm_laplacian = True, drop_first=True, eigen_solver='lobpcg')
            f, q = e[:sup], e[sup:]
            ss.append(f), qq.append(q)
        ss, qq  =np.stack(ss, axis =0), np.stack(qq, axis = 0)
        fs, fq = torch.Tensor(ss).cuda(), torch.Tensor(qq).cuda()
        # label propagation
        # L, V = torch.linalg.eigh(torch.rand(400, 2000,512))
        # L, V = torch.linalg.eigh(W)
        # V = torch.mul(V, L.unsqueeze(2))
        # print(L.shape, V.shape)
        # L, V = torch.linalg.qr(W)
        # V = V[:,:,:32]
        # V = V.permute(0,2,1)
        # z_projected = torch.bmm(V, z_all)#V[:, :sup], V[:, sup:]
        # fs, qs = z_projected[:, :sup], z_projected[:, sup:]
        # print(fs.shape, qs.shape)
        # from sklearn.linear_model import LogisticRegression
        # from sklearn import metrics
        # acc=[]
        # for i in range(fs.size(0)):
        #     clf = LogisticRegression(penalty='l2', random_state=0, C=1.0, solver='lbfgs', max_iter=1000, multi_class='multinomial')
        #     clf.fit(fs[i].cpu(), ys[i].cpu())
        #     query_ys_pred = clf.predict(qs[i].cpu())
        #     acc.append(metrics.accuracy_score(yq[i].cpu(), query_ys_pred))
        # print(mean_confidence_interval(acc))

        return fs, fq #V[:, :sup], V[:, sup:]

    def run_adaptation(self, support, query, y_s, y_q, shot):
        """
        Corresponds to the ALPHA-TIM inference
        inputs:
            support : torch.Tensor of shape [n_task, s_shot, feature_dim]
            query : torch.Tensor of shape [n_task, q_shot, feature_dim]
            y_s : torch.Tensor of shape [n_task, s_shot]
            y_q : torch.Tensor of shape [n_task, q_shot]

        updates :
            self.weights : torch.Tensor of shape [n_task, num_class, feature_dim]
        """
        t0 = time.time()

        if self.use_tuned_alpha_values or self.alpha_values is None:
            self.get_alpha_values(shot)
        if shot ==1:
            self.alpha = 0.8
            self.k = 20
        elif shot>=5:
            self.alpha=0.9
            self.k=10
        # fs, qs = F.normalize(support, p=2, dim=-1, eps=1e-12), F.normalize(query, p=2, dim=-1, eps=1e-12)
        fs, qs = support, query
        batch, n_support, dim = fs.shape
        batch, n_query, dim = qs.shape
        k = self.k
        y_s = get_one_hot(y_s)
        n_label = y_s.size(1)
        print('alpha vals: ', self.alpha_values, 'alpha: ', self.alpha, ' k: ', self.k)
        #------------------------------------------------MUS IDEA-------------------------------------------------------
        n_tasks = y_s.size(0)
        counts = y_s.sum(1).view(n_tasks, -1, 1)
        mus = y_s.transpose(1, 2).matmul(fs)
        mus = (mus / counts)#.cuda()
        mus = mus.cuda()
        y_mus = torch.eye(mus.size(1)).repeat(n_tasks, 1, 1).cuda()
        y_s_zeros = torch.zeros_like(y_s).cuda()
        n_mus = mus.size(1)
        #------------------------------------------------MUS IDEA-------------------------------------------------------

        y_q_pred = torch.zeros(y_q.shape[0], y_q.shape[1], y_s.size(2)).cuda()
        l_all = torch.cat([y_mus, y_s_zeros, y_q_pred], dim=1)
        z_all = torch.cat([mus, fs, qs], dim=1)  # to bypass the error of array not C-contiguous

        Wb = (torch.ones(z_all.size(0), z_all.size(1), z_all.size(1))).cuda()*self.alpha#, 1)).cuda() #
        Wb = Wb.requires_grad_()
        mus = mus.requires_grad_()
        # fs = fs.requires_grad_()
        G = (torch.ones(z_all.size(0), z_all.size(1), z_all.size(1))).cuda()
        G = G.requires_grad_()
        if self.phi=='mus':
            optimizer = torch.optim.Adam([{'params': [mus]}], lr=self.lr)
        elif self.phi=='mus+G':
            optimizer = torch.optim.Adam([{'params': [mus]}, {'params': [G]}], lr=self.lr)
        elif self.phi=='mus+Wb':
            optimizer = torch.optim.Adam([{'params': [mus]}, {'params': [Wb]}], lr=self.lr)
        elif self.phi=='mus+G+Wb':
            optimizer = torch.optim.Adam([{'params': [mus]}, {'params': [Wb]}, {'params': [G]}], lr=self.lr)
        for i in tqdm(range(self.iter)):
            z_all = torch.cat([mus, fs, qs],dim=1).cuda()  # .unsqueeze(0).cuda() # to bypass the error of array not C-contiguous
            # -------------------------RBF graph from embedding propagation-----------------------------------------------
            sq_dist = torch.cdist(z_all, z_all)
            stds = sq_dist.std(dim=(1, 2), keepdim=True)
            sq_dist = sq_dist / stds  # sq_dist[mask].std()
            weights = torch.exp(-torch.mul(sq_dist, G))
            mask = torch.eye(weights.size(2), dtype=torch.bool, device=weights.device)
            A = weights * (~mask).float()
            # ------------------------------------------------------------------------------------------------------------
            A = torch.clamp(A, min=0)  # non-negative
            A[:,:n_mus]=0
            N = A.size(1)  # N = n_way*n_sup+n_way*n_query
            A[:, range(N), range(N)] = 0  # zero diagonal
            # graph construction
            _, indices = torch.topk(A, k, dim=-1)  # (bs, n_way*n_sup+n_way*n_query, topk)
            mask = torch.zeros_like(A)  # (bs, n_way*n_sup+n_way*n_query, n_way*n_sup+n_way*n_query)
            mask = mask.scatter(-1, indices, 1)  # (bs, n_way*n_sup+n_way*n_query, n_way*n_sup+n_way*n_query)
            A = A * mask  # (bs, n_way*n_sup+n_way*n_query, n_way*n_sup+n_way*n_query)
            W = (A + A.permute(0, 2, 1)) * 0.5  # (bs, n_way*n_sup+n_way*n_query, n_way*n_sup+n_way*n_query)
            # #----------------------------------------PROPAGATOR BEFORE NORMALIZATION-----------------------------------
            W = torch.mul(Wb, W)
            # W = (W + W.permute(0, 2, 1)) * 0.5  # (bs, n_way*n_sup+n_way*n_query, n_way*n_sup+n_way*n_query)
            # #----------------------------------------PROPAGATOR BEFORE NORMALIZATION-----------------------------------
            D = W.sum(-1)  # (bs, n_way*n_sup+n_way*n_query)
            D_sqrt_inv = torch.sqrt(1.0 / (D + 1e-12))  # (bs, n_way*n_sup+n_way*n_query)
            D1 = torch.unsqueeze(D_sqrt_inv, 2).repeat(1, 1, N)  # (bs, n_way*n_sup+n_way*n_query, {1})
            D2 = torch.unsqueeze(D_sqrt_inv, 1).repeat(1, N, 1)  # (bs, {1}, n_way*n_sup+n_way*n_query)
            W = D1 * W * D2  # (bs, n_way*n_sup+n_way*n_query, n_way*n_sup+n_way*n_query)
            I = torch.eye(N)  # (n_way*n_sup+n_way*n_query, n_way*n_sup+n_way*n_query)
            I = I.unsqueeze(0).repeat(z_all.size(0), 1, 1).cuda()  # (bs, n_way*n_sup+n_way*n_query, n_way*n_sup+n_way*n_query)
            propagator = torch.inverse(I - self.alpha * W)  # (bs, n_way*n_sup+n_way*n_query, n_way*n_sup+n_way*n_query)
            scores_all = torch.matmul(propagator, l_all) * self.temp  # (bs, n_way*n_sup+n_way*n_query, n_way)
            logits_s = scores_all[:, n_mus:n_label+n_mus, :]
            logits_q = scores_all[:, n_label+n_mus:, :]
            q_probs = logits_q.softmax(2)

            # Cross entropy type
            if self.entropies[0] == 'Shannon':
                ce = - (y_s * torch.log(logits_s.softmax(2) + 1e-12)).sum(2).mean(1).sum(0)
            elif self.entropies[0] == 'Alpha':
                ce = torch.pow(y_s, self.alpha_values[0]) * torch.pow(logits_s.softmax(2) + 1e-12, 1 - self.alpha_values[0])
                ce = ((1 - ce.sum(2))/(self.alpha_values[0] - 1)).mean(1).sum(0)
            else:
                raise ValueError("Entropies must be in ['Shannon', 'Alpha']")

            # Marginal entropy type
            if self.entropies[1] == 'Shannon':
                q_ent = - (q_probs.mean(1) * torch.log(q_probs.mean(1))).sum(1).sum(0)
            elif self.entropies[1] == 'Alpha':
                q_ent = ((1 - (torch.pow(q_probs.mean(1), self.alpha_values[1])).sum(1)) / (self.alpha_values[1] - 1)).sum(0)
            else:
                raise ValueError("Entropies must be in ['Shannon', 'Alpha']")

            # Conditional entropy type
            if self.entropies[2] == 'Shannon':
                q_cond_ent = - (q_probs * torch.log(q_probs + 1e-12)).sum(2).mean(1).sum(0)
            elif self.entropies[2] == 'Alpha':
                q_cond_ent = ((1 - (torch.pow(q_probs + 1e-12, self.alpha_values[2])).sum(2)) / (self.alpha_values[2] - 1)).mean(1).sum(0)
            else:
                raise ValueError("Entropies must be in ['Shannon', 'Alpha']")

            loss = self.loss_weights[0] * ce - (self.loss_weights[1] * q_ent - self.loss_weights[2] * q_cond_ent) #'+ loss_nc

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        Z = scores_all[:, n_mus+n_label:, :]
        Z = F.normalize(Z, 2, dim=-1)
        self.record_info(Z, y_q)

    def scaleEachUnitaryDatas(self, datas):

        norms = datas.norm(dim=2, keepdim=True)
        return datas / norms

    def QRreduction(self, data):

        ndatas = torch.qr(data.permute(0, 2, 1)).R
        ndatas = ndatas.permute(0, 2, 1)
        return ndatas

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

    def run_task(self, task_dic, shot, train_loader=None, reductor=None):
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

        # print(support.shape, query.shape)
        if self.PLC:
            batch, n_sup, dim = support.shape
            datas = torch.cat([support, query], dim=1)
            datas = self.PT(datas)
            datas = self.scaleEachUnitaryDatas(datas)
            datas = self.centerData(datas)
            support, query = datas[:,:n_sup], datas[:,n_sup:]
            # print('PLC pre-processing')
        else:
        # # Perform normalizations required
        #     train_mean = F.normalize(train_mean, dim=2)
            support = F.normalize(support, dim=2)  #- train_mean.cuda()#
            query = F.normalize(query, dim=2)  #- train_mean.cuda()#
            # mix_fs, mix_ys = self.multi_mix(support, F.one_hot(y_s), n=1000)
            # if reductor != None:
            #     fs = []
            #     fq = []
            #     for i in range(len(support)):
            #         # z_all = torch.Tensor(reductor.transform(torch.cat([support[i].cpu(), query[i].cpu()], dim=0).cpu().numpy())).cuda()[:support.size(1)+query.size(1)]
            #         # reductor.fit_model(mix_fs[i].cpu().detach().numpy(), mix_ys[i].cpu().detach().numpy(), 48)
            #         z_all = torch.Tensor(reductor.model.transform(torch.cat([support[i].cpu(), query[i].cpu()], dim=0).cpu().detach().numpy(), from_space='D', to_space='U_model')).cuda()[:support.size(1)+query.size(1)]
            #         # print(z_all.shape)
            #         output_s, output_q = z_all[:support.size(1)], z_all[support.size(1):]
            #         # print(output_q.shape, output_s.shape)
            #         fs.append(output_s), fq.append(output_q)
            #     support = torch.stack(fs).cuda()
            #     query = torch.stack(fq).cuda()
            # mix_fs, _ = self.multi_mix(support, F.one_hot(y_s), n=200)
            # mix_fs, mix_ys = self.multi_mix(support, F.one_hot(y_s), n=200)
            # base_fs = F.normalize(extract_train_features(self.model, train_loader).cpu(), dim=1)
            # from sklearn.manifold import SpectralEmbedding
            # from sklearn.decomposition import PCA
            # embedding = PCA(n_components=100)
            # # embedding = SpectralEmbedding(n_components=10)
            # fs = []
            # fq = []
            # for i in range(len(support)):
            #     z_all = torch.Tensor(embedding.fit_transform(torch.cat([support[i].cpu(), query[i].cpu(), mix_fs[i].cpu()], dim=0).cpu())).cuda()[:support.size(1)+query.size(1)]
            #     # print(z_all.shape)
            #     output_s, output_q = z_all[:support.size(1)], z_all[support.size(1):]
            #     # print(output_q.shape, output_s.shape)
            #     fs.append(output_s), fq.append(output_q)
            # support = torch.stack(fs).cuda()
            # query = torch.stack(fq).cuda()

            # print(support.shape, query.shape)
            # print('NO PLC pre-processing')

        # Initialize weights
        self.compute_lambda(support=support, query=query, y_s=y_s)
        # Init basic prototypes
        self.init_weights(support=support, y_s=y_s, query=query, y_q=y_q)
        # Run adaptation
        self.run_adaptation(support=support, query=query, y_s=y_s, y_q=y_q, shot=shot)

        # Extract adaptation logs
        logs = self.get_logs()

        return logs
    def multi_mix(self, fs, ys, n =100):
        dir = torch.distributions.dirichlet.Dirichlet(torch.ones(fs.shape[1])*10 )#/ (fs.shape[1]*10))
        samples = dir.sample_n(n).unsqueeze(0)
        samples = samples.expand(fs.size(0), samples.size(1), samples.size(2))
        # print(k.shape)
        mix_fs = []
        mix_ys = []
        for i in range(n):
            j = samples[:,i:i+1,:].cuda()
            # print(fs.shape, ys.shape, j.shape)
            tmp = torch.matmul(j, fs)
            tmp_y = torch.matmul(j, ys.float())
            # print(tmp.shape, tmp_y.shape)
            mix_ys.append(tmp_y), mix_fs.append(tmp)
        mix_ys = torch.cat(mix_ys, dim=1)
        mix_fs = torch.cat(mix_fs, dim=1)
        print(mix_ys.shape, mix_fs.shape)
        return mix_fs, mix_ys


class AM(ALPHA_AM):
    def run_adaptation(self, support, query, y_s, y_q, shot):
        """
        Corresponds to the ALPHA-TIM inference
        inputs:
            support : torch.Tensor of shape [n_task, s_shot, feature_dim]
            query : torch.Tensor of shape [n_task, q_shot, feature_dim]
            y_s : torch.Tensor of shape [n_task, s_shot]
            y_q : torch.Tensor of shape [n_task, q_shot]

        updates :
            self.weights : torch.Tensor of shape [n_task, num_class, feature_dim]
        """
        t0 = time.time()

        if self.use_tuned_alpha_values or self.alpha_values is None:
            self.get_alpha_values(shot)

        if shot ==1:
            self.alpha = 0.8
            self.k = 20
        elif shot==5:
            self.alpha=0.9
            self.k=10
        # fs, qs = F.normalize(support, p=2, dim=-1, eps=1e-12), F.normalize(query, p=2, dim=-1, eps=1e-12)
        fs, qs = support, query
        batch, n_support, dim = fs.shape
        batch, n_query, dim = qs.shape
        # mus = self.init_weights(support, query, y_s, y_q).cuda()
        # y_mus = torch.eye(mus.size(1)).repeat(y_s.size(0), 1, 1).cuda()
        k = self.k
        # fs, qs = self.laplacian_eigenmaps(fs, qs, self.alpha, self.k, y_s, y_q)
        y_s = get_one_hot(y_s)
        n_label = y_s.size(1)
        n_tasks = y_s.size(0)
        counts = y_s.sum(1).view(n_tasks, -1, 1)
        mus = y_s.transpose(1, 2).matmul(fs)
        mus = (mus / counts)#.cuda()
        mus = mus.cuda()
        y_mus = torch.eye(mus.size(1)).repeat(n_tasks, 1, 1).cuda()
        y_s_zeros = torch.zeros_like(y_s).cuda()
        n_mus = mus.size(1)
        #------------------------------------------------MUS IDEA-------------------------------------------------------
        y_q_pred = torch.zeros(y_q.shape[0], y_q.shape[1], y_s.size(2)).cuda()
        l_all = torch.cat([y_mus, y_s_zeros, y_q_pred], dim=1)
        z_all = torch.cat([mus, fs, qs], dim=1)  # to bypass the error of array not C-contiguous

        Wb = (torch.ones(z_all.size(0), z_all.size(1), z_all.size(1)) * self.alpha).cuda()
        G = (torch.ones(z_all.size(0), z_all.size(1), z_all.size(1))).cuda()
        #-------- SET FOR GRADIENTS------------------------------------------------------------------------------------
        Wb = Wb.requires_grad_()
        mus = mus.requires_grad_()
        G = G.requires_grad_()
        print('alpha vals: ', self.alpha_values, 'alpha: ', self.alpha, ' k: ', self.k)
        if self.phi=='mus':
            optimizer = torch.optim.Adam([{'params': [mus]}], lr=self.lr)
        elif self.phi=='mus+G':
            optimizer = torch.optim.Adam([{'params': [mus]}, {'params': [G]}], lr=self.lr)
        elif self.phi=='mus+Wb':
            optimizer = torch.optim.Adam([{'params': [mus]}, {'params': [Wb]}], lr=self.lr)
        elif self.phi=='mus+G+Wb':
            optimizer = torch.optim.Adam([{'params': [mus]}, {'params': [Wb]}, {'params': [G]}], lr=self.lr)
        for i in tqdm(range(self.iter)):
            z_all = torch.cat([mus, fs, qs], dim=1)

            sq_dist = torch.cdist(z_all, z_all)
            # mask = sq_dist != 0
            stds = sq_dist.std(dim=(1, 2), keepdim=True)
            # print(stds.shape)
            sq_dist = sq_dist / stds  # sq_dist[mask].std()
            # weights = torch.exp(-sq_dist * 1)
            weights = torch.exp(-torch.mul(sq_dist, G))
            mask = torch.eye(weights.size(2), dtype=torch.bool, device=weights.device)
            A = weights * (~mask).float()

            # A = torch.bmm(z_all, z_all.permute(0, 2, 1))  # (bs, n_way*n_sup+n_way*n_query, n_way*n_sup+n_way*n_query)
            # A = torch.pow(A, 3)  # power transform
            A = torch.clamp(A, min=0)  # non-negative
            A[:,:n_mus]=0
            N = A.size(1)  # N = n_way*n_sup+n_way*n_query
            A[:, range(N), range(N)] = 0  # zero diagonal
            # graph construction
            _, indices = torch.topk(A, k, dim=-1)  # (bs, n_way*n_sup+n_way*n_query, topk)
            mask = torch.zeros_like(A)  # (bs, n_way*n_sup+n_way*n_query, n_way*n_sup+n_way*n_query)
            mask = mask.scatter(-1, indices, 1)  # (bs, n_way*n_sup+n_way*n_query, n_way*n_sup+n_way*n_query)
            # if self.args.nnk:
            A = A * mask  # (bs, n_way*n_sup+n_way*n_query, n_way*n_sup+n_way*n_query)
            W = (A + A.permute(0, 2, 1)) * 0.5  # (bs, n_way*n_sup+n_way*n_query, n_way*n_sup+n_way*n_query)
            #----------------------------------------PROPAGATOR BEFORE NORMALIZATION-----------------------------------
            W = torch.mul(Wb, W)
            # W = (W + W.permute(0, 2, 1)) * 0.5  # (bs, n_way*n_sup+n_way*n_query, n_way*n_sup+n_way*n_query)
            #----------------------------------------PROPAGATOR BEFORE NORMALIZATION-----------------------------------
            D = W.sum(-1)  # (bs, n_way*n_sup+n_way*n_query)
            D_sqrt_inv = torch.sqrt(1.0 / (D + 1e-12))  # (bs, n_way*n_sup+n_way*n_query)
            D1 = torch.unsqueeze(D_sqrt_inv, 2).repeat(1, 1, N)  # (bs, n_way*n_sup+n_way*n_query, {1})
            D2 = torch.unsqueeze(D_sqrt_inv, 1).repeat(1, N, 1)  # (bs, {1}, n_way*n_sup+n_way*n_query)
            W = D1 * W * D2  # (bs, n_way*n_sup+n_way*n_query, n_way*n_sup+n_way*n_query)
            I = torch.eye(N)  # (n_way*n_sup+n_way*n_query, n_way*n_sup+n_way*n_query)
            I = I.unsqueeze(0).repeat(z_all.size(0), 1, 1).cuda()  # (bs, n_way*n_sup+n_way*n_query, n_way*n_sup+n_way*n_query)
            propagator = torch.inverse(I - self.alpha*W)
            scores_all = torch.matmul(propagator, l_all) * self.temp  # (bs, n_way*n_sup+n_way*n_query, n_way)
            logits_s = scores_all[:, n_mus:n_label+n_mus, :]
            logits_q = scores_all[:, n_label+n_mus:, :]

            ce = - (y_s.cuda() * torch.log(logits_s.softmax(2) + 1e-12)).sum(2).mean(1).sum(0)
            q_probs = logits_q.softmax(2)
            q_cond_ent = - (q_probs * torch.log(q_probs + 1e-12)).sum(2).mean(1).sum(0)
            q_ent = - (q_probs.mean(1) * torch.log(q_probs.mean(1))).sum(1).sum(0)
            loss = self.loss_weights[0] * ce - (
                        self.loss_weights[1] * q_ent - self.loss_weights[2] * q_cond_ent)  # '+ loss_nc

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        Z = scores_all[:, n_mus + n_label:, :]
        Z = F.normalize(Z, 2, dim=-1)
        self.record_info(Z, y_q)