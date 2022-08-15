import time
import numpy as np



def mask_data(x:np.array, H:int, W:int, mask):       # for evaluation only on selected grids
    assert (len(x.shape) == 4)|(len(x.shape) == 5)        # graph: (num_samples, horizon, N, C) / grid: (num_samples, horizon, C, H, W)
    if len(x.shape) == 4:   # graph
        assert x.shape[-2]==H*W
        x = x.reshape(x.shape[0], x.shape[1], H, W, x.shape[-1])
    else:   # len=5
        if (x.shape[-2]==H)&(x.shape[-1]==W):   # grid
            x = x.transpose((0, 1, 3, 4, 2))    # switch to channel last
        else:
            assert (x.shape[2]==H)&(x.shape[3]==W)
            pass

    if mask != []:
        mask_count = 0
        x_masked = list()
        for h in range(H):
            for w in range(W):
                if (h, w) in mask:
                    mask_count += 1
                    continue
                x_masked.append(x[:, :, h, w, :])
        # print('    Number of masked grids:', mask_count)
        x_masked = np.array(x_masked).transpose((1, 2, 0, 3))  # (num_samples, horizon, masked_N, C)
    else:
        x_masked = x.reshape(x.shape[0], x.shape[1], -1, x.shape[-1])       # unmasked
    return x_masked



class ModelEvaluator(object):
    def __init__(self, params:dict, precision=4, epsilon=1e-5):
        self.params = params
        self.precision = precision      # digits behind decimal
        self.epsilon = epsilon      # avoid zero division

    def evaluate_numeric(self, y_pred: np.array, y_true: np.array, mode:str, mask:list=None):
        assert y_pred.shape == y_true.shape

        with open(self.params['output_dir'] + f'/{self.params["data"]}-{self.params["model"]}-eval-metrics.csv', 'a') as cf:
            print(' '.join(['*' * 10, f'Evaluation on {mode} set started at', time.ctime(), '*' * 10]))
            cf.write(f'*****, Evaluation starts, {mode}, {time.ctime()}, ***** \n')
            for param in self.params.keys():        # write model parameters
                cf.write(f'{param}: {self.params[param]},')
            cf.write('\n')

        # mask data
        y_pred_masked, y_true_masked = mask_data(y_pred, self.params["H"], self.params["W"], mask), mask_data(y_true, self.params["H"], self.params["W"], mask)
        # stepwise through horizon
        multistep_metrics = list()
        for step in range(self.params['pred_len']):
            print(f'Evaluating step {step}:')
            step_metrics = self.one_step_eval_num(y_pred_masked[:, step,...], y_true_masked[:, step,...])
            print('Overall: \n'
                  f'   MSE: {step_metrics["MSE"]:10.4f}, RMSE: {step_metrics["RMSE"]:9.4f} \n'
                  f'   MAE: {step_metrics["MAE"]:10.4f}, MAPE: {step_metrics["MAPE"]:9.4%} \n')
            multistep_metrics.append(step_metrics)
        # horizon avg
        horizon_avg = list()
        for measure in list(step_metrics.keys()):
            step_measures = list()
            for step in range(self.params['pred_len']):
                step_measures.append(multistep_metrics[step][measure])
            horizon_avg.append(np.mean(step_measures))
            print(f'Horizon avg. {measure}: {horizon_avg[-1]:9.4f}')

        with open(self.params['output_dir'] + f'/{self.params["data"]}-{self.params["model"]}-eval-metrics.csv', 'a') as cf:
            col_names = [' '] + list(step_metrics.keys())
            cf.write(','.join(col_names) + '\n')
            for step in range(self.params['pred_len']):
                row_items = [f'Step {step}'] + list(multistep_metrics[step].values())
                cf.write(','.join([str(item) for item in row_items]) + '\n')
            row_items = [f'Horizon avg.'] + horizon_avg
            cf.write(','.join([str(item) for item in row_items]) + '\n')
            cf.write(f'*****, Evaluation ends, {mode}, {time.ctime()}, ***** \n \n')
            print(' '.join(['*' * 10, f'Evaluation on {mode} set ended at', time.ctime(), '*' * 10]))

        return

    def one_step_eval_num(self, y_pred_step: np.array, y_true_step: np.array):
        assert y_pred_step.shape == y_true_step.shape

        MSEs, RMSEs, MAEs, MAPEs = [], [], [], []
        # loop channels
        for c in range(self.params['C']):
            y_pred_step_c = y_pred_step[...,c]
            y_true_step_c = y_true_step[...,c]

            MSE_c = self.MSE(y_pred_step_c, y_true_step_c)
            RMSE_c = self.RMSE(y_pred_step_c, y_true_step_c)
            MAE_c = self.MAE(y_pred_step_c, y_true_step_c)
            MAPE_c = self.MAPE(y_pred_step_c, y_true_step_c)
            print(f'  {self.params["channels"][c]}  MSE: {MSE_c:11.4f}, RMSE: {RMSE_c:9.4f}, MAE: {MAE_c:9.4f}, MAPE: {MAPE_c:9.4%}')
            MSEs.append(MSE_c)
            RMSEs.append(RMSE_c)
            MAEs.append(MAE_c)
            MAPEs.append(MAPE_c)

        # dict of metrics
        step_metrics = dict()
        step_metrics['MSE'] = np.mean(MSEs)
        step_metrics['RMSE'] = np.mean(RMSEs)
        step_metrics['MAE'] = np.mean(MAEs)
        step_metrics['MAPE'] = np.nanmean(MAPEs)
        return step_metrics


    @staticmethod
    def MSE(y_pred: np.array, y_true: np.array):
        return np.mean(np.square(y_pred - y_true))

    @staticmethod
    def RMSE(y_pred: np.array, y_true: np.array):
        return np.sqrt(np.mean(np.square(y_pred - y_true)))

    @staticmethod
    def MAE(y_pred: np.array, y_true: np.array):
        return np.mean(np.abs(y_pred - y_true))

    #def MAPE(self, y_pred: np.array, y_true: np.array):
    #    return np.mean(np.abs(y_pred - y_true) / (y_true + self.epsilon))    # avoid zero division

    @staticmethod
    def MAPE(y_pred: np.array, y_true: np.array):
        greater_than_ = y_true > 10
        y_pred, y_true = y_pred[greater_than_], y_true[greater_than_]
        return np.mean(np.abs(y_pred - y_true) / np.abs(y_true))
