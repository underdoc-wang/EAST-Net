import os
import time
import argparse
import Data_Container, Model_Trainer



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run multimodal mobility nowcasting.')

    # command line arguments
    parser.add_argument('-device', '--device', type=str, help='Specify device usage', default='cuda:0',
                        choices=['cpu', 'cuda:0', 'cuda:1', 'cuda:2', 'cuda:3'])
    parser.add_argument('-in', '--input_dir', type=str, default='./data')
    # dataset, model
    parser.add_argument('-data', '--data', type=str, help='Specify data', choices=['JONAS-NYC', 'JONAS-DC', 'COVID-CHI', 'COVID-US'])
    parser.add_argument('-model', '--model', type=str, help='Specify model', choices=['ST-Net', 'EAST-Net'])
    parser.add_argument('-rect', '--rectify_ST', type=str, help='Rectification for ST-Net', choices=['None', 'Tcov', 'Memo'], default='None')
    # hyper-parameters
    parser.add_argument('-obs', '--obs_len', type=int, help='Length of observation sequence', default=8)
    parser.add_argument('-pred', '--pred_len', type=int, help='Length of prediction sequence', default=8)
    parser.add_argument('-split', '--split_ratio', type=int, nargs='+',
                        help='Data split ratio in train : validate : test. Example: -split 7 1 2', default=[7, 1, 2])
    parser.add_argument('-batch', '--batch_size', type=int, default=32)
    parser.add_argument('-hidden', '--hidden_dim', type=int, default=32)
    parser.add_argument('-K', '--chebyshev_order', type=int, default=3)
    parser.add_argument('-nn', '--nn_layers', type=int, default=2)
    # training
    parser.add_argument('-loss', '--loss', type=str, help='Specify loss function', choices=['MSE', 'MAE'], default='MAE')
    parser.add_argument('-optim', '--optimizer', type=str, help='Specify optimizer', default='Adam')
    parser.add_argument('-lr', '--learn_rate', type=float, default=5e-4)
    parser.add_argument('-dr', '--decay_rate', type=float, default=0)       # weight decay: L2regularization
    parser.add_argument('-epoch', '--num_epochs', type=int, default=100)
    parser.add_argument('-test', '--test_only', action='store_true')

    params = parser.parse_args().__dict__       # save in dict

    if params['data'] == 'JONAS-NYC':
        params['channels'] = ['taxi_demand', 'taxi_supply', 'bike_demand', 'bike_supply']
        params['C'] = len(params['channels'])
        params['H'], params['W'] = (16, 8)
        geo_graph = False
        params['N_slot_per_hour'] = 2
        start, end = (2015, 10, 24), (2016, 1, 31)      # 100 days; 0.5 hour interval -> (4800, 16, 8, 4)
        params['N_months'] = 4
    elif params['data'] == 'JONAS-DC':
        params['channels'] = ['taxi_demand', 'taxi_supply', 'bike_demand', 'bike_supply']
        params['C'] = len(params['channels'])
        params['H'], params['W'] = (9, 12)
        geo_graph = False
        params['N_slot_per_hour'] = 1
        start, end = (2015, 10, 24), (2016, 1, 31)      # 100 days; 1 hour interval -> (2400, 9, 12, 4)
        params['N_months'] = 4
    elif params['data'] == 'COVID-CHI':
        params['channels'] = ['taxi_demand', 'taxi_supply', 'bike_demand', 'bike_supply', 'scooter_demand', 'scooter_supply']
        params['C'] = len(params['channels'])
        params['H'], params['W'] = (14, 8)
        geo_graph = False
        params['N_slot_per_hour'] = 1/2
        start, end = (2019, 7, 1), (2020, 12, 31)      # 550 days; 2 hour interval -> (6600, 14, 8, 6)
        params['N_months'] = 12
    elif params['data'] == 'COVID-US':
        params['channels'] = ['grocery_store', 'other_retailer', 'transportation', 'office', 'school', 'healthcare', 'entertainment', 'hotel', 'restaurant', 'service']
        params['C'] = len(params['channels'])
        params['H'], params['W'] = (51, 1)
        geo_graph = True
        params['N_slot_per_hour'] = 1
        start, end = (2019, 11, 14), (2020, 5, 31)        # 200 days; 1 hour interval -> (4800, 51, 10)
        params['N_months'] = 7
    else:
        raise ValueError('Invalid input data.')

    # paths
    params['data_dir'] = os.path.join(params['input_dir'],
                    f'{params["data"]}-{params["H"]}x{params["W"]}-{start[0]}{str(start[1]).zfill(2)}{str(start[2]).zfill(2)}-{end[0]}{str(end[1]).zfill(2)}{str(end[2]).zfill(2)}.npz')
    params['output_dir'] = os.path.join('.', params['data'])
    os.makedirs(params['output_dir'], exist_ok=True)

    # load data
    data_input = Data_Container.DataInput(norm_bool=True, graph_bool=geo_graph)
    data = data_input.load_data(data_dir=params['data_dir'])
    print('\n', time.ctime())
    print(f'    Loading {params["data"]} data: {start}~{end} on {round(1/params["N_slot_per_hour"], 2)} hour timeslice, '
          f'shape: {data["mob"].shape}, time_covariate: {data["tcov"].shape}')

    # get data loader
    data_generator = Data_Container.DataGenerator(obs_len=params['obs_len'],
                                                  pred_len=params['pred_len'],
                                                  data_split_ratio=params['split_ratio'])
    data_loader = data_generator.get_data_loader(data=data,
                                                 params=params)
    # get model
    trainer = Model_Trainer.ModelTrainer(params=params,
                                         data=data,
                                         data_container=data_input,
                                         graph_bool=geo_graph)

    if not params['test_only']:
        trainer.train(data_loader=data_loader, modes=['train', 'validate'])
    trainer.test(data_loader=data_loader, modes=['train', 'test'])
