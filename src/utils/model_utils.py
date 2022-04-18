from models import CustomCNN, InceptionModule, ResNet
import torch.optim as optim

def create_model(configs):

    # Create model based on architecture name
    if configs.arch == "cnn":
        print(f"Using CNN")
        model = CustomCNN()
    elif configs.arch == "incp":
        print(f"Using InceptionModule")
        model = InceptionModule()
    elif configs.arch == "resnet":
        print(f"Using ResNet")
        model = ResNet()
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