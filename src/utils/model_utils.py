from models.CustomCNN import CustomCNN
from models.InceptionModule import InceptionModule_t1, InceptionModule_t2, InceptionModule_t3_1, InceptionModule_t3_2, InceptionModule_t3_3, conv_layer_averpl
from models.ResNet import ResNet, ResidualBlock
from models.RecurrentNet import RecurrentNet
import torch.optim as optim
import sys
import torch.nn as nn

def create_model(configs):

    # Create model based on architecture name
    if configs.arch == "cnn":
        print(f"------Using CNN------")
        model = CustomCNN(configs)
    elif configs.arch == "incp":
        print(f"------Using InceptionModule------")
        # # Trial 1: only 1 Inception module
        # model = nn.Sequential(
        #             conv_layer_averpl(1, 128),
        #             conv_layer_averpl(128, 256),
        #             conv_layer_averpl(256, 512),
        #             InceptionModule_t1(configs),
        #             nn.AdaptiveAvgPool2d((2,2)),
        #             nn.Flatten(),
        #             nn.Linear(2048, 40),
        #             nn.Linear(40, 10)
        #         )

        # # Trial 2: 3 Inception modules with skip connections
        # model = nn.Sequential(
        #             conv_layer_averpl(1, 128),
        #             conv_layer_averpl(128, 256),
        #             InceptionModule_t2(configs),
        #             InceptionModule_t2(configs),
        #             InceptionModule_t2(configs),
        #             nn.AdaptiveAvgPool2d((2,2)),
        #             nn.Flatten(),
        #             nn.Linear(1024, 40),
        #             nn.Linear(40, 10)
        #         )

        # Trial 3: 3 Inception modules 
        model = nn.Sequential(
                    InceptionModule_t3_1(configs),
                    InceptionModule_t3_2(configs),
                    InceptionModule_t3_3(configs),
                    nn.AdaptiveAvgPool2d((2,2)),
                    nn.Flatten(),
                    nn.Linear(2048, 40),
                    nn.Linear(40, 10)
                )

    elif configs.arch == "resnet":
        print(f"------Using ResNet------")
        model = ResNet(ResidualBlock, [2, 2, 2], configs)
    elif configs.arch in ["rnn", "gru", "lstm"]:
        print(f"------Using RecurrentNet------")
        model = RecurrentNet(input_size=221, hidden_dim=128, num_layers=1, output_size = 10, configs=configs, arch=configs.arch)
    else:
        assert False, "Undefined Model Backbone"

    print("# of trainable parameters:",sum(p.numel() for p in model.parameters() if p.requires_grad))

    return model

def create_optimizer(configs, model):

    if configs.optimizer_type == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=configs.lr, momentum=configs.momentum)
    elif configs.optimizer_type == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=configs.lr)
    else:
        assert False, "Invalid Optimizer type"

    return optimizer


if __name__ == "__main__":
    pass