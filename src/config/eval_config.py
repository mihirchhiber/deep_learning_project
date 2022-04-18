import os
import argparse
import torch
from easydict import EasyDict as edict

def parse_eval_configs():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--working_dir', type=str, default='deep_learning_project',
                        help='The ROOT working directory')

    ####################################################################
    ###################     Model configs     ##########################
    ####################################################################
    parser.add_argument('-a', '--arch', type=str, default='cnn',
                            help='The architecture of model')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Train test split ')
    parser.add_argument('--num_workers', type=int, default=0,
                            help='Number of threads for loading data')
    parser.add_argument('--trg_batch_size', type=int, default=16,
                            help='Batch size')
    parser.add_argument('--val_batch_size', type=int, default=16,
                            help='Batch size')
    parser.add_argument('--test_batch_size', type=int, default=8,
                            help='Batch size')
    parser.add_argument('--no_cuda', action='store_true',
                        help='If true, cuda is not used.')

    configs = edict(vars(parser.parse_args()))

    ####################################################################
    ############## Hardware configurations #############################
    ####################################################################
    configs.device = torch.device('cpu' if configs.no_cuda else 'cuda')
    configs.ngpus_per_node = torch.cuda.device_count()

    configs.pin_memory = True

    # ####################################################################
    # ############## Dataset, logs, Checkpoints dir ######################
    # ####################################################################
    configs.dataset_dir = os.path.join(configs.working_dir, 'data')
    configs.checkpoints_dir = os.path.join(configs.working_dir, 'checkpoints', configs.arch)

    if not os.path.isdir(configs.checkpoints_dir):
        os.makedirs(configs.checkpoints_dir)

    # print(configs)
    return configs