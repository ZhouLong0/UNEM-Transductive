import numpy as np
from src.methods.U_Paddle import U_Paddle
from src.methods.U_Paddle_embt import U_Paddle_EMBT
from src.utils import compute_confidence_interval, load_checkpoint, Logger, extract_mean_features, extract_features
from src.methods.tim import ALPHA_TIM, TIM_GD
from src.methods.paddle import PADDLE
from src.methods.soft_km import SOFT_KM
from src.methods.paddle_gd import PADDLE_GD
from src.methods.ici import ICI
from src.methods.laplacianshot import LaplacianShot
from src.methods.bdcspn import BDCSPN
from src.methods.baseline import Baseline
from src.methods.pt_map import PT_MAP
from src.datasets import Tasks_Generator, get_dataset, get_dataloader, SamplerSupport, SamplerQuery, CategoriesSampler
import torch
import os
from src.utils import load_pickle, save_pickle
import random
import matplotlib.pyplot as plt

class U_Trainer:
    def __init__(self, device, args, log_file):
        self.device = device
        self.args = args
        self.log_file = log_file
        self.logger = Logger(__name__, self.log_file)


    def run_training(self, model):
        """
        Run the evaluation over all the tasks
        inputs:
            model : The loaded model containing the feature extractor
            args : All parameters

        returns :
            results : List of the mean accuracy for each number of support shots
        """
        #self.logger.info("=> Runnning full evaluation with method: {}".format(self.args.name_method))
        self.logger.info("=> Training U-PADDLE full evaluation with sampling: ".format(self.args.sampling))
        load_checkpoint(model=model, model_path=self.args.ckpt_path, type=self.args.model_tag)
        dataset = {}
        loader_info = {'aug': False, 'out_name': False}

        support_set = get_dataset('val', args=self.args, **loader_info)
        dataset.update({'support': support_set})
        query_set = get_dataset('val', args=self.args, **loader_info)
        dataset.update({'query': query_set})
        
        ## Compute train mean
        name_file = 'train_mean_' + self.args.dataset + '_' + self.args.arch + '.pt'
        if os.path.isfile(name_file) == False:
            train_set = get_dataset('train', args=self.args, **loader_info)
            dataset['train_loader'] = train_set
            train_loader = get_dataloader(sets=train_set, args=self.args)
            train_mean, _ = extract_mean_features(model=model,  train_loader=train_loader, args=self.args,
                                                logger=self.logger, device=self.device)
            torch.save(train_mean, name_file)
        else:
            train_mean = torch.load(name_file)

        # Extract features (just load them if already in memory)
        extracted_features_dic_support = extract_features(model=model,
                                                        model_path=self.args.ckpt_path, model_tag=self.args.model_tag,
                                                        loaders_dic=dataset, used_set='support',
                                                        used_set_name = 'val',
                                                        fresh_start=self.args.fresh_start)
       
        extracted_features_dic_query = extracted_features_dic_support 

        all_features_support = extracted_features_dic_support['concat_features']
        all_labels_support = extracted_features_dic_support['concat_labels'].long()
        all_features_query = extracted_features_dic_query['concat_features']
        all_labels_query = extracted_features_dic_query['concat_labels'].long()

        results = []
        loss_histories = []
        for shot in self.args.shots:
            u_paddle = self.get_model_new()
            u_paddle = u_paddle.to(self.device)

            #for i in range(int(self.args.train_tasks/self.args.train_batch_size)):
            sampler = CategoriesSampler(all_labels_support, all_labels_query, self.args.train_samples,
                                    self.args.k_eff, self.args.n_ways, shot, self.args.n_query, 
                                    self.args.sampling, self.args.used_set_support, self.args.alpha_dirichlet)
            sampler.create_list_classes(all_labels_support, all_labels_query)
            sampler_support = SamplerSupport(sampler)
            sampler_query = SamplerQuery(sampler)

            test_loader_query = []
            for indices in sampler_query :
                test_loader_query.append((all_features_query[indices,:], all_labels_query[indices]))

            test_loader_support = []
            for indices in sampler_support :
                test_loader_support.append((all_features_support[indices,:], all_labels_support[indices]))

            task_generator = Tasks_Generator(k_eff=self.args.k_eff, n_ways=self.args.n_ways, shot=shot, n_query=self.args.n_query, loader_support=test_loader_support, loader_query=test_loader_query, train_mean=train_mean, log_file=self.log_file)

            tasks = task_generator.generate_tasks()

            criterion = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(u_paddle.parameters(), lr=self.args.train_lr)
            scheduler = StepLR(optimizer, step_size=50, gamma=0.5)

            u_paddle, loss_history, params_history = training2(u_paddle, tasks, criterion, optimizer, scheduler, self.args.train_iter, self.device)
            #torch.cuda.empty_cache()

            results.append(u_paddle)
            loss_histories.append(loss_history)
        
        if self.args.save_results == True:
            path = 'results/u_params/{}/{}/{}'.format(self.args.dataset, self.args.arch, self.args.train_n_layers)
            name_file = path + '/{}.txt'.format(self.args.train_method)
            if not os.path.exists(path):
                os.makedirs(path)
            if os.path.isfile(name_file) == True:
                f = open(name_file, 'a')
                print('Adding to already existing .txt file to avoid overwritting')
            else:
                f = open(name_file, 'w')
    
            f.write(str(self.args.k_eff)+'\t'+'layers: '+str(self.args.train_n_layers))
            for i, shot in enumerate(self.args.shots):
                self.logger.info('{}-shot -- sampling: {}'.format(shot, self.args.sampling))
                f.write('{}-shot -- sampling: {}'.format(shot, self.args.sampling))
                for name, param in results[i].named_parameters():
                    f.write(f"Parameter name: {name}")
                    f.write(f"Parameter shape: {param.shape}")
                    f.write(f"Parameter value: {param}\n")
                model_file = path + '/{}_{}shots.pt'.format(self.args.train_method, shot)
                torch.save(results[i].state_dict(), model_file)
            f.write('\n')
            f.close()

            for i, shot in enumerate(self.args.shots):
                plot_name = path + '/{}_{}shots_loss_history.png'.format(self.args.train_method, shot)
                self.logger.info('Saving the plot of the loss')
                plt.plot(loss_histories[i])
                plt.savefig(plot_name)
                plt.close()
                
        return results
    
    def get_model_new(self):
        if self.args.train_method == 'U_PADDLE':
            model = U_Paddle(n_layers=self.args.train_n_layers, device=self.device, log_file=self.log_file, gamma=1.0, verbose=False, diff_gamma_layers=False)
        elif self.args.train_method == 'U_PADDLE_L':
            model = U_Paddle(n_layers=self.args.train_n_layers, device=self.device, log_file=self.log_file, gamma=1.0, verbose=False, diff_gamma_layers=True)
        elif self.args.train_method == 'U_PADDLE_EMBT':
            model = U_Paddle_EMBT(n_layers=self.args.train_n_layers, device=self.device, log_file=self.log_file, diff_gamma_layers=False)
        elif self.args.train_method == 'U_PADDLE_EMBT_L':
            model = U_Paddle_EMBT(n_layers=self.args.train_n_layers, device=self.device, log_file=self.log_file, diff_gamma_layers=True)
        else:
            raise ValueError('Method not implemented')
        return model



from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

        
        

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
    


def training2(u_padnet, task_dic, criterion, optimizer, scheduler, epochs, device):
    import torch.nn as nn
    y_s, y_q = task_dic['y_s'], task_dic['y_q']
    x_s, x_q = task_dic['x_s'], task_dic['x_q']
    train_mean = task_dic['train_mean'].unsqueeze(0).unsqueeze(0)
    # print(task_dic['train_mean'].shape)
    # Transfer tensors to GPU if needed
    support = x_s.to(device)  # [ N * (K_s + K_q), d]
    query = x_q.to(device)  # [ N * (K_s + K_q), d]
    y_support = y_s.long().squeeze(2).to(device)
    y_query = y_q.long().squeeze(2).to(device)
    # Extract features
    
    # scaler = MinMaxScaler(feature_range=(0, 1))
    # query, support = scaler(query, support)

    # predictions

    u_padnet.train()

    loss_history = []
    params_history = []

    for epoch in tqdm(range(epochs)):
        epoch_loss = 0
        
        optimizer.zero_grad()

        logits = u_padnet(support, y_support, query, device)
        logits = logits.view(-1, logits.size(-1))
        labels = y_query.view(-1)

        loss = criterion(logits, labels)

        loss.backward()

        for name, param in u_padnet.named_parameters():
           if param.grad is not None:
               print(f"Gradient of {name}: {param.grad.norm()}")
           else:
               print(f"No gradient for {name}")

        optimizer.step()

        #loss = loss.detach()

        # print(f"loss: {loss.item()}")
        # print(f"updated param: , {padnet.gamma}, {nn.Softplus()(padnet.gamma)}")
        # print("-----------------")
        scheduler.step()
        
        loss_history.append(loss.item())
        #params_history.append({key: value.detach().clone() for key, value in u_padnet.state_dict().items()})

        print(f"Epoch {epoch} - Loss: {loss.item()}")
        for name, param in u_padnet.named_parameters():
            print(f"Parameter name: {name}")
            print(f"Parameter shape: {param.shape}")
            print(f"Parameter value: {param}\n")
    
    return u_padnet, loss_history, params_history