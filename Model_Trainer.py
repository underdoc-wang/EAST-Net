import time
import numpy as np
import torch
from torch import nn, optim
import STNet, EASTNet
from Metrics import ModelEvaluator



class ModelTrainer(object):
    def __init__(self, params:dict, data:dict, data_container, graph_bool:bool):
        self.params = params
        self.mask = data['mask']    # for evaluation
        self.data_container = data_container
        self.graph_bool = graph_bool
        self.model = self.get_model().to(params['device']) if params['device'].startswith('cuda') else self.get_model()
        self.criterion = self.get_loss()
        self.optimizer = self.get_optimizer()

    def get_model(self):
        if self.params['model'] == 'ST-Net':
            model = STNet.ST_Net(n_node=self.params['H'] * self.params['W'],
                                 c_in=self.params['C'],
                                 h_dim=self.params['hidden_dim'],
                                 cheby_k=self.params['chebyshev_order'],
                                 n_layer=self.params['nn_layers'],
                                 horizon=self.params['pred_len'],
                                 device=self.params['device'],
                                 time_cov=bool(self.params['rectify_ST']=='Tcov'),
                                 tcov_in_dim=int(24*self.params['N_slot_per_hour']+7+self.params['N_months']+1),
                                 st_memo=bool(self.params['rectify_ST']=='Memo'),
                                 st_memo_num=4,
                                 st_memo_dim=128)

        elif self.params['model'] == 'EAST-Net':
            model = EASTNet.EAST_Net(n_node=self.params['H'] * self.params['W'],
                                      c_in=self.params['C'],
                                      h_dim=self.params['hidden_dim'],
                                      cheby_k=self.params['chebyshev_order'],
                                      n_layer=self.params['nn_layers'],
                                      horizon=self.params['pred_len'],
                                      device=self.params['device'],
                                      tcov_in_dim=int(24*self.params['N_slot_per_hour']+7+self.params['N_months']+1))

        else:
            raise NotImplementedError('Invalid model name.')

        return model

    def get_loss(self):
        if self.params['loss'] == 'MAE':
            criterion = nn.L1Loss(reduction='mean')
        else:
            raise NotImplementedError('Invalid loss function.')
        return criterion

    def get_optimizer(self):
        if self.params['optimizer'] == 'Adam':
            optimizer = optim.Adam(params=self.model.parameters(),
                                   lr=self.params['learn_rate'],
                                   weight_decay=self.params['decay_rate'])
        else:
            raise NotImplementedError('Invalid optimizer name.')
        return optimizer


    def train(self, data_loader:dict, modes:list, early_stop_patience=10):
        checkpoint = {'epoch': 0, 'train_loss': np.inf, 'val_loss': np.inf, 'state_dict': self.model.state_dict()}
        val_loss = np.inf       # initialize validation loss
        patience_count = early_stop_patience
        loss_curve = {mode: [] for mode in modes}
        run_time = {mode: [] for mode in modes}

        print('\n', time.ctime())
        print(f'     {self.params["model"]} model training begins:')
        for epoch in range(1, 1 + self.params['num_epochs']):
            running_loss = {mode: 0.0 for mode in modes}
            for mode in modes:
                if mode == 'train':
                    self.model.train()
                else:
                    self.model.eval()

                step = 0
                start_time = time.time()
                if self.params['model'] == 'EAST-Net':
                    ht_pair = (torch.zeros(self.params['batch_size'], self.params['H'] * self.params['W'], self.params['hidden_dim']).to(self.params['device']),
                               torch.zeros(self.params['batch_size'], self.params['C'], self.params['hidden_dim']).to(self.params['device']))
                for x_seq, t_x, t_y, y_true in data_loader[mode]:
                    # print(step)
                    with torch.set_grad_enabled(mode=(mode=='train')):
                        if self.params['model'] == 'ST-Net':
                            y_pred = self.model(x_seq=x_seq, t_x=t_x, t_y=t_y)
                        elif self.params['model'] == 'EAST-Net':
                            y_pred, ht_pair = self.model(x_seq=x_seq, t_x=t_x, t_y=t_y, ht_pair=ht_pair)
                        else:
                            raise NotImplementedError('Invalid model name.')

                        loss = self.criterion(y_pred, y_true)
                        if mode == 'train':
                            self.optimizer.zero_grad()
                            loss.backward(retain_graph=True)
                            self.optimizer.step()

                    running_loss[mode] += loss.data * y_true.shape[0]    # loss reduction='mean': batchwise average
                    step += y_true.shape[0]
                    torch.cuda.empty_cache()

                # epoch mode end
                end_time = time.time()
                run_time[mode].append(end_time - start_time)
                loss_curve[mode].append(running_loss[mode]/step)

            # epoch end
            log = f'Epoch {epoch}: training time: {run_time["train"][-1]:.4} s/epoch, training loss: {loss_curve["train"][-1]:.4}; ' \
                  f'inference time: {run_time["validate"][-1]:.4} s, '
            if loss_curve["validate"][-1] < val_loss:
                add_log = f'validation loss drops from {val_loss:.4} to {loss_curve["validate"][-1]:.4}. Update model checkpoint..'
                val_loss = loss_curve["validate"][-1]
                checkpoint.update(epoch=epoch,
                                  train_loss = loss_curve["train"][-1],
                                  val_loss = loss_curve["validate"][-1],
                                  state_dict=self.model.state_dict())
                torch.save(checkpoint, self.params['output_dir'] + f'/{self.params["model"]}-{self.params["data"]}.pkl')
                patience_count = early_stop_patience
            else:
                add_log = f'validation loss does not improve from {val_loss:.4}.'
                patience_count -= 1
                if patience_count == 0:     # early stopping
                    print('\n', time.ctime())
                    print(f'    Early stopping triggered at epoch {epoch}.')
                    break
            print(log + add_log)

        # training end
        print('\n', time.ctime())
        print(f'     {self.params["model"]} model training ends.')
        # torch.save(checkpoint, self.params['output_dir']+f'/{self.params["model"]}_{self.params["data"]}.pkl')
        print(f'    Average training time: {np.mean(run_time["train"]):.4} s/epoch.')
        print(f'    Average inference time: {np.mean(run_time["validate"]):.4} s.')

        return


    def test(self, data_loader:dict, modes:list):
        trained_checkpoint = torch.load(self.params['output_dir']+f'/{self.params["model"]}-{self.params["data"]}.pkl')
        print(f'Successfully loaded trained {self.params["model"]} model - epoch: {trained_checkpoint["epoch"]}, training loss: {trained_checkpoint["train_loss"]}, validation loss: {trained_checkpoint["val_loss"]}')
        # print([key for key in trained_checkpoint['state_dict']])
        self.model.load_state_dict(trained_checkpoint['state_dict'])        # load model

        self.model.eval()           # for dropout, batchnorm
        with torch.no_grad():       # turn off gradient computing
            model_evaluator = ModelEvaluator(self.params)
            for mode in modes:
                print('\n', time.ctime())
                print(f'     {self.params["model"]} model testing on {mode} data begins:')

                start_time = time.time()
                if self.params['model'] == 'EAST-Net':
                    ht_pair = (torch.zeros(self.params['batch_size'], self.params['H'] * self.params['W'], self.params['hidden_dim']).to(self.params['device']),
                               torch.zeros(self.params['batch_size'], self.params['C'], self.params['hidden_dim']).to(self.params['device']))
                forecast, ground_truth = [], []
                for x_seq, t_x, t_y, y_true in data_loader[mode]:
                    if self.params['model'] == 'ST-Net':
                        y_pred = self.model(x_seq=x_seq, t_x=t_x, t_y=t_y)
                    elif self.params['model'] == 'EAST-Net':
                        y_pred, ht_pair = self.model(x_seq=x_seq, t_x=t_x, t_y=t_y, ht_pair=ht_pair)
                    else:
                        raise NotImplementedError('Invalid model name.')

                    forecast.append(y_pred.cpu().detach().numpy())
                    ground_truth.append(y_true.cpu().detach().numpy())

                forecast = np.concatenate(forecast, axis=0)
                ground_truth = np.concatenate(ground_truth, axis=0)

                # denormalize
                forecast = self.data_container.channel_wise_denormalize(forecast)
                ground_truth = self.data_container.channel_wise_denormalize(ground_truth)
                # save results
                # if mode == 'test':
                #     np.save(f'{self.params["output_dir"]}/{self.params["model"]}_pred.npy', forecast)
                #     np.save(f'{self.params["output_dir"]}/{self.params["model"]}_true.npy', ground_truth)
                # evaluate on metrics
                model_evaluator.evaluate_numeric(y_pred=forecast, y_true=ground_truth, mode=mode, mask=self.mask)

                # inference time
                end_time = time.time()
                inf_time = end_time - start_time
                print(f'{mode} inference time: {inf_time}')

        print('\n', time.ctime())
        print(f'     {self.params["model"]} model testing ends.')
        return
