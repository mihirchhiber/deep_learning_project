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

    configs = edict(vars(parser.parse_args()))

    ####################################################################
    ############## Hardware configurations #############################
    ####################################################################
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