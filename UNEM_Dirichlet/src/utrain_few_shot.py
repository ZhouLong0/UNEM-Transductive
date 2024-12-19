import os
import numpy as np

# Import utilities and datasets
from src.utils import Logger, load_pickle, extract_features_softmax, extract_features_visual, compute_confidence_interval, extract_features_pre_softmax
from src.datasets import OxfordPets, EuroSAT, UCF101, SUN397, Caltech101, DescribableTextures, FGVCAircraft, Food101, Flowers102, StanfordCars
from src.datasets import build_data_loader
from src.task_generator_few_shot import Tasks_Generator_few_shot
from src.sampler_few_shot import CategoriesSampler_few_shot, SamplerQuery_few_shot, SamplerSupport_few_shot

# Import methods for few-shot learning
from src.methods.few_shot.em_dirichlet import EM_DIRICHLET
from src.methods.few_shot.hard_em_dirichlet import HARD_EM_DIRICHLET
from src.methods.few_shot.paddle import PADDLE
from src.methods.few_shot.bdcspn import BDCSPN
from src.methods.few_shot.tim import ALPHA_TIM
from src.methods.few_shot.laplacian_shot import LAPLACIAN_SHOT
from src.methods.few_shot.u_em_dirichlet import U_EM_DIRICHLET
from src.methods.few_shot.u_em_dirichlet_embt import U_EM_DIRICHLET_EMBT
import torch
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import torch.nn as nn
import matplotlib.pyplot as plt

# Dataset list for few-shot learning tasks
dataset_list = {
    "oxfordpets": OxfordPets,
    "eurosat": EuroSAT,
    "ucf101": UCF101,
    "sun397": SUN397,
    "caltech101": Caltech101,
    "dtd": DescribableTextures,
    "fgvcaircraft": FGVCAircraft,
    "food101": Food101,
    "flowers102": Flowers102,
    "stanfordcars": StanfordCars,
}

class U_Trainer_few_shot:
    def __init__(self, device, args, log_file):
        self.device = device
        self.args = args
        self.log_file = log_file
        self.logger = Logger(__name__, self.log_file)


    
    def run_training(self, model, preprocess):
        model.eval()

        # Initialize dataset and data loaders
        dataset = dataset_list[self.args.dataset](self.args.dataset_path)
        self.args.classnames = dataset.classnames
        self.args.template = dataset.template
        data_loaders = self.initialize_data_loaders(dataset, preprocess)

        # Extract and load features
        extracted_features_dic_support, extracted_features_dic_query = self.extract_and_load_features(
            model, dataset, data_loaders)
        all_features_support = extracted_features_dic_support['concat_features'].to(
            'cpu')
        all_labels_support = extracted_features_dic_support['concat_labels'].long().to(
            'cpu')
        all_features_query = extracted_features_dic_query['concat_features'].to(
            'cpu')
        all_labels_query = extracted_features_dic_query['concat_labels'].long().to(
            'cpu')
        
        self.logger.info("=> Runnning u_training with method {} on {} dataset".format(self.args.name_method, self.args.used_test_set))

        results = []
        results_time = []
        results_task = []
        results_task_time = []

        model.to('cpu')

        u_method = self.get_method_builder(model=model, device=self.device, args=self.args, log_file=self.log_file)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(u_method.parameters(), lr=self.args.u_train_lr)
        scheduler = StepLR(optimizer, step_size=self.args.u_train_drop_step, gamma=self.args.u_train_drop_factor)

        print('==== Training starting ...\n')
        print('==== Initial parameters: ')
        for name, param in u_method.named_parameters():
            print(f"Parameter name: {name}")
            print(f"Parameter value: {param}\n")

        # Start the evaluation over the tasks
        tasks_list = []
        for i in tqdm(range(int(self.args.utrain_number_tasks/self.args.utrain_batch_size)), desc='Tasks generation'):

            # Create sampler for transductive few-shot tasks
            #if imagenet change n_class
            sampler = CategoriesSampler_few_shot(self.args.utrain_batch_size,
                                                 self.args.k_eff, self.args.n_class, self.args.shots, self.args.u_train_n_query, force_query_size=True)
            sampler.create_list_classes(all_labels_support, all_labels_query)
            sampler_support = SamplerSupport_few_shot(sampler)
            sampler_query = SamplerQuery_few_shot(sampler)

            # Get the query and support samples at the indexes given by the samplers
            test_loader_query = []
            for indices in sampler_query:
                test_loader_query.append(
                    (all_features_query[indices, :], all_labels_query[indices]))

            test_loader_support = []
            for indices in sampler_support:
                test_loader_support.append(
                    (all_features_support[indices, :], all_labels_support[indices]))

            # Prepare the tasks
            task_generator = Tasks_Generator_few_shot(k_eff=self.args.k_eff, shot=self.args.shots, n_query=self.args.n_query,
                                                      n_class=self.args.n_class, loader_support=test_loader_support, loader_query=test_loader_query, model=model, args=self.args)
            tasks = task_generator.generate_tasks()

            tasks_list.append(tasks)

        u_method, loss_history, params_history = self.train_list_tasks(u_method, tasks_list, criterion, optimizer, scheduler, self.args.u_train_epochs, self.device)


        if self.args.save_results == True:
            path = 'u_params/{}/{}/{}'.format(self.args.dataset, self.args.backbone, self.args.iter)
            name_file = path + '/{}.txt'.format(self.args.name_method)
            if not os.path.exists(path):
                os.makedirs(path)
            if os.path.isfile(name_file) == True:
                f = open(name_file, 'a')
                print('Adding to already existing .txt file to avoid overwritting')
            else:
                f = open(name_file, 'w')
    
            f.write('layers: '+str(self.args.iter))
            self.logger.info('{}-shot --'.format(self.args.shots))
            f.write('   {}-shot   '.format(self.args.shots))
            for name, param in u_method.named_parameters():
                f.write(f"Parameter name: {name} ")
                f.write(f"Parameter shape: {param.shape} ")
                f.write(f"Parameter value: {param}\n")
            f.write('\n')
            f.close()
            
            # save the parameters
            model_file = path + '/{}_{}shots.pt'.format(self.args.name_method, self.args.shots)
            torch.save(u_method.state_dict(), model_file)

            plot_name = path + '/{}_{}shots_loss_history.png'.format(self.args.name_method, self.args.shots)
            self.logger.info('Saving the plot of the loss')
            plt.plot(loss_history)
            plt.savefig(plot_name)
            plt.close()
            


                

    def initialize_data_loaders(self, dataset, preprocess):
        """
        Initialize data loaders for training, validation, and testing.
        :param dataset: The dataset object.
        :param preprocess: Preprocessing function for data.
        :return: Dictionary of data loaders for train, val, and test.
        """
        batch_size = 1024
        data_loaders = {
            'train': build_data_loader(data_source=dataset.train_x, batch_size=batch_size, is_train=False, shuffle=False, tfm=preprocess),
            'val': build_data_loader(data_source=dataset.val, batch_size=batch_size, is_train=False, shuffle=False, tfm=preprocess),
            'test': build_data_loader(data_source=dataset.test, batch_size=batch_size, is_train=False, shuffle=False, tfm=preprocess)
        }
        return data_loaders

    def extract_and_load_features(self, model, dataset, data_loaders):
        """
        Extract and load features for the evaluation.
        :param model: The model to be evaluated.
        :param dataset: The dataset object.
        :param data_loaders: Data loaders for train, val, and test.
        """

        # Load the features: either the softmax features, either the visual embeddings
         # Load the features: either the softmax features, either the visual embeddings
        if hasattr(self.args, 'use_pre_softmax_feature') and self.args.use_pre_softmax_feature:
            extract_features_pre_softmax(
                model, dataset, data_loaders['test'], 'test', self.args, self.device, list_T=[self.args.T])
            extract_features_pre_softmax(
                model, dataset, data_loaders['val'], 'val', self.args, self.device, list_T=[self.args.T])
            extract_features_pre_softmax(
                model, dataset, data_loaders['train'], 'train', self.args, self.device, list_T=[self.args.T])

            filepath_support = 'data/{}/saved_features/{}_pre_softmax_{}.plk'.format(
                self.args.dataset, self.args.used_utrain_set, self.args.backbone)
            filepath_query = 'data/{}/saved_features/{}_pre_softmax_{}.plk'.format(
                self.args.dataset, self.args.used_utrain_set, self.args.backbone)

        elif self.args.use_softmax_feature == True:
            extract_features_softmax(
                model, dataset, data_loaders['test'], 'test', self.args, self.device, list_T=[self.args.T])
            extract_features_softmax(
                model, dataset, data_loaders['val'], 'val', self.args, self.device, list_T=[self.args.T])
            extract_features_softmax(
                model, dataset, data_loaders['train'], 'train', self.args, self.device, list_T=[self.args.T])

            filepath_support = 'data/{}/saved_features/{}_softmax_{}_T{}.plk'.format(
                self.args.dataset, self.args.used_utrain_set, self.args.backbone, self.args.T)
            filepath_query = 'data/{}/saved_features/{}_softmax_{}_T{}.plk'.format(
                self.args.dataset, self.args.used_utrain_set, self.args.backbone, self.args.T)
        else:
            extract_features_visual(
                model, dataset, data_loaders['test'], 'test', self.args, self.device)
            extract_features_visual(
                model, dataset, data_loaders['val'], 'val', self.args, self.device)
            extract_features_visual(
                model, dataset, data_loaders['train'], 'train', self.args, self.device)

            filepath_support = 'data/{}/saved_features/{}_visual_{}.plk'.format(
                self.args.dataset, self.args.used_utrain_set, self.args.backbone)
            filepath_query = 'data/{}/saved_features/{}_visual_{}.plk'.format(
                self.args.dataset, self.args.used_utrain_set, self.args.backbone)

        extracted_features_dic_support = load_pickle(filepath_support)
        extracted_features_dic_query = load_pickle(filepath_query)

        print('Features loaded from file: {} and {}'.format(filepath_support, filepath_query))

        return extracted_features_dic_support, extracted_features_dic_query
    
    
    
    def train_list_tasks(self, u_model, task_dic_list, criterion, optimizer, scheduler, epochs, device):
        import torch.nn as nn
        u_model.train()

        loss_history = []
        params_history = []

        for epoch in tqdm(range(epochs)):
            epoch_loss = 0
            for task_dic in task_dic_list:
                y_s, y_q = task_dic['y_s'], task_dic['y_q']
                x_s, x_q = task_dic['x_s'], task_dic['x_q']

                # print(task_dic['train_mean'].shape)
                # Transfer tensors to GPU if needed
                support = x_s.to(device)  # [ N * (K_s + K_q), d]
                query = x_q.to(device)  # [ N * (K_s + K_q), d]
                y_support = y_s.long().squeeze(2).to(device)
                y_query = y_q.long().squeeze(2).to(device)
                
                optimizer.zero_grad()

                logits = u_model(support, y_support, query)
                logits = logits.view(-1, logits.size(-1))
                labels = y_query.view(-1)

                loss = criterion(logits, labels)

                loss.backward()

                # for name, param in u_model.named_parameters():
                #     if param.grad is not None:
                #         print(f"Gradient of {name}: {param.grad.norm()}")
                #     else:
                #         print(f"No gradient for {name}")

                optimizer.step()

                #loss = loss.detach()

                # print(f"loss: {loss.item()}")
                # print(f"updated param: , {padnet.gamma}, {nn.Softplus()(padnet.gamma)}")
                # print("-----------------")
                epoch_loss += loss.item()
                
                
                #loss_history.append(epoch_loss)
                #params_history.append({key: value.detach().clone() for key, value in u_padnet.state_dict().items()})

            print(f" ======== Epoch {epoch} ========= Loss: {epoch_loss}")
            loss_history.append(epoch_loss)
            for name, param in u_model.named_parameters():
                print(f"Parameter name: {name}")
                print(f"Parameter shape: {param.shape}")
                print(f"Parameter value: {param}\n")

            scheduler.step()
        
        return u_model, loss_history, params_history
    

    
    def get_method_builder(self, model, device, args, log_file):
        # Initialize method classifier builder
        method_info = {'model': model, 'device': device,
                       'log_file': log_file, 'args': args}
        init_gamma = self.args.num_classes_test/self.args.k_eff
        # few-shot methods
        if args.name_method == 'U_EM_DIRICHLET':
            method_info.pop('model')
            method_builder = U_EM_DIRICHLET(**method_info, gamma=init_gamma)
        elif args.name_method == 'U_EM_DIRICHLET_L':
            method_info.pop('model')
            method_builder = U_EM_DIRICHLET(**method_info, diff_gamma_layers=True, gamma=init_gamma)
        elif args.name_method == 'U_EM_DIRICHLET_EMBT_L':
            method_info.pop('model')
            method_builder = U_EM_DIRICHLET_EMBT(**method_info, gamma=init_gamma, diff_gamma_layers=True)
        elif args.name_method == 'U_EM_DIRICHLET_EMBT':
            method_info.pop('model')
            method_builder = U_EM_DIRICHLET_EMBT(**method_info, gamma=init_gamma)


        else:
            raise ValueError(
                "The method your entered is not unrollable. Please check the spelling")
        return method_builder

        
    



