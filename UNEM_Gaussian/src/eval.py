import copy
import numpy as np
from src.methods.AM import AM, ALPHA_AM
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
from src.methods.U_Paddle import U_Paddle
from src.datasets import Tasks_Generator, get_dataset, get_dataloader, SamplerSupport, SamplerQuery, CategoriesSampler
import torch
import os
from src.utils import load_pickle, save_pickle
import random

class Evaluator:
    def __init__(self, device, args, log_file):
        self.device = device
        self.args = args
        self.log_file = log_file
        self.logger = Logger(__name__, self.log_file)

    def run_full_evaluation(self, model):
        """
        Run the evaluation over all the tasks
        inputs:
            model : The loaded model containing the feature extractor
            args : All parameters

        returns :
            results : List of the mean accuracy for each number of support shots
        """
        self.logger.info("=> Runnning full evaluation with method: {}".format(self.args.name_method))
        load_checkpoint(model=model, model_path=self.args.ckpt_path, type=self.args.model_tag)
        dataset = {}
        loader_info = {'aug': False, 'out_name': False}

        support_set = get_dataset(self.args.used_set_support, args=self.args, **loader_info)
        dataset.update({'support': support_set})
        query_set = get_dataset(self.args.used_set_query, args=self.args, **loader_info)
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
                                                       used_set_name = self.args.used_set_support,
                                                               fresh_start=self.args.fresh_start)
        if self.args.used_set_support != self.args.used_set_query:
            extracted_features_dic_query = extract_features(model=model,
                                                       model_path=self.args.ckpt_path, model_tag=self.args.model_tag,
                                                       loaders_dic=dataset, used_set='query', 
                                                        used_set_name = self.args.used_set_query,
                                                                 fresh_start=self.args.fresh_start)
        else:
            extracted_features_dic_query = extracted_features_dic_support 

        all_features_support = extracted_features_dic_support['concat_features']
        all_labels_support = extracted_features_dic_support['concat_labels'].long()
        all_features_query = extracted_features_dic_query['concat_features']
        all_labels_query = extracted_features_dic_query['concat_labels'].long()

        results = []
        for shot in self.args.shots:
        
            results_task = []
            for i in range(int(self.args.number_tasks/self.args.batch_size)):
                sampler = CategoriesSampler(all_labels_support, all_labels_query, self.args.batch_size,
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
              
                method = self.get_method_builder(model=model, shot=shot)

                tasks = task_generator.generate_tasks()
                #Run task
                # if self.args.name_method == 'ALPHA_AM':
                #     task1, task2 = Tasks_Generator.split_task(tasks)
                #     method1 = self.get_method_builder(model=model, shot=shot)
                #     method2 = self.get_method_builder(model=model, shot=shot)
                #     logs1 = method1.run_task(task_dic=task1, shot=shot)
                #     logs2 = method2.run_task(task_dic=task2, shot=shot)
                #     logs = {'acc': np.concatenate((logs1['acc'], logs2['acc']), axis=1)}
                #     acc_mean, acc_conf = compute_confidence_interval(logs['acc'][:, -1])
                # else:
                logs = method.run_task(task_dic=tasks, shot=shot)
                acc_mean, acc_conf = compute_confidence_interval(logs['acc'][:, -1])

                ##save_pickle(self.log_file.replace('.log', 'convergence{}.plk'.format(shot)), logs['criterions'])

                results_task.append(acc_mean)
                
                del method
                del tasks
            results.append(results_task)
            print(f"mean accuracy shot: {shot} : {np.mean(results_task)}")

            if self.args.acc_vs_gamma_study:
                import pandas as pd
                data = {
                        "alpha": [self.args.alpha],
                        "accuracy": [np.mean(results_task)],
                    }
                df = pd.DataFrame(data)
                # read a csv and append the results
                dir = f'plot/{self.args.arch}/'
                path =  '{}_plot_{}.csv'.format(self.args.dataset, shot)
                # Ensure the directory exists
                if not os.path.exists(dir):
                    os.makedirs(dir)
                # Check if the CSV file exists
                name_file = dir + path
                if os.path.isfile(name_file):
                    # File exists, append without header
                    df.to_csv(name_file, mode="a", header=False, index=False)
                else:
                    # File doesn't exist, write with header
                    df.to_csv(name_file, mode="w", header=True, index=False)
                    
        mean_accuracies = np.asarray(results).mean(1)

        if self.args.name_method == 'ALPHA-TIM':
            param = self.args.alpha_values[1]
        elif self.args.name_method == 'PADDLE':
            param = self.args.alpha
        elif self.args.name_method == 'SOFT-KM':
            param = self.args.alpha
        elif self.args.name_method == 'LaplacianShot':
            param = self.args.lmd
        elif self.args.name_method == 'Baseline':
            param = self.args.iter
        elif self.args.name_method == 'Baseline++':
            param = self.args.temp
        elif self.args.name_method == 'PT-MAP':
            param = self.args.alpha
        elif self.args.name_method == 'TIM-GD':
            param = self.args.loss_weights[1]
        elif self.args.name_method == 'ICI':
            param = self.args.d
        elif self.args.name_method == 'BDCSPN':
            param = self.args.temp
            
        self.logger.info('----- Final test results -----')
        for shot in self.args.shots:
            self.logger.info('{}-shot mean test accuracy over {} tasks: {}'.format(shot, self.args.number_tasks,
                                                                                    mean_accuracies[self.args.shots.index(shot)]))
        if self.args.ablation == True:
            path = 'results/ablation'.format(self.args.dataset, self.args.arch)
            name_file = path + '/{}.txt'.format(self.args.name_method)
            #name_file_criterions = path + '/criterions_{}.plk'.format(self.args.name_method)
            if 'criterions' in logs:
                if not os.path.exists(path):
                    os.makedirs(path)
                np.savetxt(name_file, (np.array(logs['timestamps']), np.array(np.mean(logs['criterions'], axis=1))))            

        ### If in parameter tuning mode ###
        if self.args.tune_parameters == True:
            path = 'results/params/{}/{}'.format(self.args.dataset, self.args.arch)
            name_file = path + '/{}.txt'.format(self.args.name_method)

            if not os.path.exists(path):
                os.makedirs(path)
            if os.path.isfile(name_file) == True:
                f = open(name_file, 'a')
                print('Adding to already existing .txt file to avoid overwritting')
            else:
                f = open(name_file, 'w')
                
            f.write(str(param)+'\t')
            self.logger.info('{}-shot mean test accuracy over {} tasks: {}'.format(shot, self.args.number_tasks,
                                                                                   mean_accuracies[self.args.shots.index(shot)]))
            f.write(str(round(100*mean_accuracies[self.args.shots.index(shot)], 1)) +'\n' )

            f.close()

        ### If in testing mode ###
        
        elif self.args.save_results == True:
            path = 'results/test/{}/{}'.format(self.args.dataset, self.args.arch)
            name_file = path + '/{}.txt'.format(self.args.name_method)
            if not os.path.exists(path):
                os.makedirs(path)
            if os.path.isfile(name_file) == True:
                f = open(name_file, 'a')
                print('Adding to already existing .txt file to avoid overwritting')
            else:
                f = open(name_file, 'w')
    
            f.write(str(self.args.k_eff)+'\t')
            try:
                f.write(str(self.args.iter)+'iter \t')
            except:
                pass
            try:
                f.write(str(self.args.n_layers)+'n_layers \n')
            except:
                pass
            for shot in self.args.shots:
                f.write('shot: {} '.format(shot) + str(round(100*mean_accuracies[self.args.shots.index(shot)], 1)) +'\t' )
            f.write('\n')
            f.close()
                
        return mean_accuracies

    def get_method_builder(self, model, shot):
        # Initialize method classifier builder
        method_info = {'model': model, 'device': self.device, 'log_file': self.log_file, 'args': self.args}
        if self.args.name_method == 'ALPHA-TIM':
            method_builder = ALPHA_TIM(**method_info)
        elif self.args.name_method == 'PADDLE':
            method_builder = PADDLE(**method_info)
        # elif self.args.name_method == 'PADDLE':
        #     method_info.pop('model')
        #     method_info.pop('args')
        #     method_builder = U_PaddleID_tbdu(**method_info, diff_gamma_layers=False, n_layers=self.args.iter, gamma=self.args.gamma)
        elif self.args.name_method == 'SOFT-KM':
            method_builder = SOFT_KM(**method_info)
        elif self.args.name_method == 'TIM-GD':
            method_builder = TIM_GD(**method_info)
        elif self.args.name_method == 'LaplacianShot':
            method_builder = LaplacianShot(**method_info)
        elif self.args.name_method == 'BDCSPN':
            method_builder = BDCSPN(**method_info)
        elif self.args.name_method == 'Baseline':
            method_builder = Baseline(**method_info)
        elif self.args.name_method == 'PT-MAP':
            method_builder = PT_MAP(**method_info)
        elif self.args.name_method == 'ICI':
            method_builder = ICI(**method_info)
        elif self.args.name_method == 'PADDLE-GD':
            method_builder = PADDLE_GD(**method_info)


        # elif self.args.name_method == 'U_PADDLE':
        #     method_info.pop('model')
        #     method_info.pop('args')
        #     method_builder = U_PaddleID(**method_info, diff_gamma_layers=False, n_layers=self.args.n_layers)
        #     path = 'results/u_params/{}/{}/{}'.format(self.args.dataset, self.args.arch, self.args.n_layers)
        #     checkpoint_file = path + '/{}_{}shots.pt'.format(self.args.name_method, shot)
        #     if os.path.isfile(checkpoint_file) == True:
        #         print(f"Loading checkpoint from {checkpoint_file}")
        #         initial_state_dict = copy.deepcopy(method_builder.state_dict())
        #         method_builder.load_state_dict(torch.load(checkpoint_file))
        #         loaded_state_dict = method_builder.state_dict()
        #         # Compare one layer's weights to confirm they've changed
        #         for key in initial_state_dict:
        #             if not torch.equal(initial_state_dict[key], loaded_state_dict[key]):
        #                 print(f"Layer '{key}' has been updated after loading checkpoint.")
        #             else:
        #                 print(f"Layer '{key}' remains unchanged.")
        #     else:
        #         raise ValueError('Checkpoint file for U_PADDLE not found')
        # elif self.args.name_method == 'U_PADDLE_L':
        #     method_info.pop('model')
        #     method_info.pop('args')
        #     method_builder = U_PaddleID(**method_info, diff_gamma_layers=True, n_layers=self.args.n_layers)
        #     path = 'results/u_params/{}/{}/{}'.format(self.args.dataset, self.args.arch, self.args.n_layers)
        #     checkpoint_file = path + '/{}_{}shots.pt'.format(self.args.name_method, shot)
        #     if os.path.isfile(checkpoint_file) == True:
        #         print(f"Loading checkpoint from {checkpoint_file}")
        #         initial_state_dict = copy.deepcopy(method_builder.state_dict())
        #         method_builder.load_state_dict(torch.load(checkpoint_file))
        #         loaded_state_dict = method_builder.state_dict()
        #         # Compare one layer's weights to confirm they've changed
        #         for key in initial_state_dict:
        #             if not torch.equal(initial_state_dict[key], loaded_state_dict[key]):
        #                 print(f"Layer '{key}' has been updated after loading checkpoint.")
        #             else:
        #                 print(f"Layer '{key}' remains unchanged.")
        #     else:
        #         raise ValueError('Checkpoint file for U_PADDLE_L not found')
        elif self.args.name_method == 'ALPHA_AM':
            method_builder = ALPHA_AM(**method_info)

        elif self.args.name_method == 'U_PADDLE':
            method_info.pop('model')
            method_info.pop('args')
            method_builder = U_Paddle(**method_info, diff_gamma_layers=False, n_layers=self.args.n_layers)
            path = 'results/u_params/{}/{}/{}'.format(self.args.dataset, self.args.arch, self.args.n_layers)
            checkpoint_file = path + '/{}_{}shots.pt'.format(self.args.name_method, shot)
            if os.path.isfile(checkpoint_file) == True:
                print(f"Loading checkpoint from {checkpoint_file}")
                initial_state_dict = copy.deepcopy(method_builder.state_dict())
                method_builder.load_state_dict(torch.load(checkpoint_file))
                loaded_state_dict = method_builder.state_dict()
                # Compare one layer's weights to confirm they've changed
                for key in initial_state_dict:
                    if not torch.equal(initial_state_dict[key], loaded_state_dict[key]):
                        print(f"Layer '{key}' has been updated after loading checkpoint.")
                    else:
                        print(f"Layer '{key}' remains unchanged.")
            else:
                raise ValueError('Checkpoint file for U_PADDLE not found')
        elif self.args.name_method == 'U_PADDLE_L':
            method_info.pop('model')
            method_info.pop('args')
            method_builder = U_Paddle(**method_info, diff_gamma_layers=True, n_layers=self.args.n_layers)
            path = 'results/u_params/{}/{}/{}'.format(self.args.dataset, self.args.arch, self.args.n_layers)
            checkpoint_file = path + '/{}_{}shots.pt'.format(self.args.name_method, shot)
            if os.path.isfile(checkpoint_file) == True:
                print(f"Loading checkpoint from {checkpoint_file}")
                initial_state_dict = copy.deepcopy(method_builder.state_dict())
                method_builder.load_state_dict(torch.load(checkpoint_file))
                loaded_state_dict = method_builder.state_dict()
                # Compare one layer's weights to confirm they've changed
                for key in initial_state_dict:
                    if not torch.equal(initial_state_dict[key], loaded_state_dict[key]):
                        print(f"Layer '{key}' has been updated after loading checkpoint.")
                    else:
                        print(f"Layer '{key}' remains unchanged.")
            else:
                raise ValueError('Checkpoint file for U_PADDLE_L not found')
        
            
        elif self.args.name_method == 'U_PADDLE_EMBT':
            method_info.pop('model')
            method_info.pop('args')
            method_builder = U_Paddle_EMBT(**method_info, diff_gamma_layers=False, n_layers=self.args.n_layers)
            path = 'results/u_params/{}/{}/{}'.format(self.args.dataset, self.args.arch, self.args.n_layers)
            checkpoint_file = path + '/{}_{}shots.pt'.format(self.args.name_method, shot)
            if os.path.isfile(checkpoint_file) == True:
                print(f"Loading checkpoint from {checkpoint_file}")
                initial_state_dict = copy.deepcopy(method_builder.state_dict())
                method_builder.load_state_dict(torch.load(checkpoint_file))
                loaded_state_dict = method_builder.state_dict()
                # Compare one layer's weights to confirm they've changed
                for key in initial_state_dict:
                    if not torch.equal(initial_state_dict[key], loaded_state_dict[key]):
                        print(f"Layer '{key}' has been updated after loading checkpoint.")
                    else:
                        print(f"Layer '{key}' remains unchanged.")
            else:
                raise ValueError('Checkpoint file for U_PADDLE_EMBT not found')
        
        elif self.args.name_method == 'U_PADDLE_EMBT_L':
            method_info.pop('model')
            method_info.pop('args')
            method_builder = U_Paddle_EMBT(**method_info, diff_gamma_layers=True, n_layers=self.args.n_layers)
            path = 'results/u_params/{}/{}/{}'.format(self.args.dataset, self.args.arch, self.args.n_layers)
            checkpoint_file = path + '/{}_{}shots.pt'.format(self.args.name_method, shot)
            if os.path.isfile(checkpoint_file) == True:
                print(f"Loading checkpoint from {checkpoint_file}")
                initial_state_dict = copy.deepcopy(method_builder.state_dict())
                method_builder.load_state_dict(torch.load(checkpoint_file))
                loaded_state_dict = method_builder.state_dict()
                # Compare one layer's weights to confirm they've changed
                for key in initial_state_dict:
                    if not torch.equal(initial_state_dict[key], loaded_state_dict[key]):
                        print(f"Layer '{key}' has been updated after loading checkpoint.")
                    else:
                        print(f"Layer '{key}' remains unchanged.")
            else:
                raise ValueError('Checkpoint file for U_PADDLE_EMBT not found')
            
        elif self.args.name_method == 'U_PADDLE_EMBT_T_CUSTOM':
            method_info.pop('model')
            method_info.pop('args')
            method_builder = U_Paddle_EMBT(**method_info, t1=self.args.t1, t2=self.args.t2, gamma=self.args.gamma ,diff_gamma_layers=False, n_layers=self.args.n_layers)

        elif self.args.name_method == 'U_PADDLE_EMBT_L_AVG':
            method_info.pop('model')
            method_info.pop('args')
            method_builder = U_Paddle_EMBT(**method_info, diff_gamma_layers=True, n_layers=self.args.n_layers)
            path = 'results/u_params/average/'
            checkpoint_file = path + 'U_PADDLE_EMBT_L_{}shots.pt'.format(shot)
            if os.path.isfile(checkpoint_file) == True:
                print(f"Loading checkpoint from {checkpoint_file}")
                initial_state_dict = copy.deepcopy(method_builder.state_dict())
                method_builder.load_state_dict(torch.load(checkpoint_file))
                loaded_state_dict = method_builder.state_dict()
                # Compare one layer's weights to confirm they've changed
                for key in initial_state_dict:
                    if not torch.equal(initial_state_dict[key], loaded_state_dict[key]):
                        print(f"Layer '{key}' has been updated after loading checkpoint.")
                    else:
                        print(f"Layer '{key}' remains unchanged.")
            else:
                raise ValueError('Checkpoint file for U_PADDLE_EMBT not found')


        else:
            self.logger.exception("Method must be in ['PADDLE', 'PADDLE_GD', 'SOFT_KM', 'TIM_GD', 'ICI', 'ALPHA_TIM', 'LaplacianShot', 'BDCSPN', 'SimpleShot', 'Baseline', 'Baseline++', 'PT-MAP', 'Entropy_min']")
            raise ValueError("Method must be in ['PADDLE', 'SOFT_KM', 'TIM_GD', 'ICI', ALPHA_TIM', 'LaplacianShot', 'BDCSPN', 'SimpleShot', 'Baseline', 'Baseline++', 'PT-MAP', 'Entropy_min']")
        return method_builder