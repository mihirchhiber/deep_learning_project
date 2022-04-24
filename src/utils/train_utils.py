import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import time
import copy
import torch
from tqdm import tqdm
from time import sleep
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from sklearn.model_selection import train_test_split, StratifiedKFold
from itertools import product

def train_model(model, dataloaders, dataset_sizes, configs, criterion, optimizer, scheduler, num_epochs, patience=5):
    losses = {'train':[], 'val':[]}
    accuracies = {'train':[], 'val':[]}

    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    stop_trigger = 0

    for epoch in range(num_epochs):
        # print('Epoch {}/{}'.format(epoch+1, num_epochs))
        # print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            with tqdm(dataloaders[phase], unit="batch") as tepoch:
                for inputs, labels in tepoch:
                    # Iterate over data.
                    # for inputs, labels in dataloaders[phase]:
                    if phase == 'train':
                        tepoch.set_description(f"Epoch {epoch+1}")
                    elif phase == 'val':
                        tepoch.set_description(f"Val {epoch+1}")

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):

                        if configs.arch in ['rnn', 'gru', 'lstm']:
                            # Reshape inputs to (batch_size, seq_length, input_size)
                            inputs = inputs.reshape(-1, 338, 219).to(device=configs.device)
                        else:
                            inputs = inputs.type(torch.DoubleTensor)
                            inputs = inputs.to(device=configs.device)

                        labels = labels.to(device=configs.device)

                        # zero the parameter gradients
                        optimizer.zero_grad()

                        outputs = model(inputs.float())
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                    
                    sleep(0.1)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase.capitalize(), epoch_loss, epoch_acc))
            
            losses[phase] += [epoch_loss]
            accuracies[phase] += [epoch_acc.cpu().detach().numpy()]

            # deep copy the model
            if phase == 'val' and epoch_acc >= best_acc:
              best_acc = epoch_acc
              best_model_wts = copy.deepcopy(model.state_dict())
              stop_trigger = 0 # Reset count

            elif phase == 'val' and epoch_acc < best_acc:
              stop_trigger += 1
              print(epoch_acc, best_acc)
              print("Triggered! --> ", stop_trigger , "/", patience)
        
        if stop_trigger == patience:
            print("Early Stopped!!!")
            break

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # save model weights
    torch.save(best_model_wts,f"{configs.checkpoints_dir}/{configs.arch}_weights_best.pth")
    torch.save(model.state_dict(),f"{configs.checkpoints_dir}/{configs.arch}_weights_last.pth") # Last

    # load best model weights
    model.load_state_dict(best_model_wts)

    return model, losses, accuracies

def cross_validate(model, dataset, k_folds, hyperparams, dataloaders, dataset_sizes, configs, criterion, optimizer, scheduler, num_epochs, patience=5):
    results = {}

    for fold, (train_ids, test_ids) in enumerate(k_folds):
        print(f'FOLD {fold+1}')
        
        # Define data loaders for training and testing data in this fold
        train_subsampler = SubsetRandomSampler(train_ids)
        train_loader = DataLoader(dataset, batch_size=16, sampler = train_subsampler)
        test_subsampler = SubsetRandomSampler(test_ids)
        test_loader = DataLoader(dataset, batch_size=16, sampler = test_subsampler)
        
        # Reset weights for each fold
        model.apply(reset_weights)

        # Training configuration
        optimizer = get_optimizer(hyperparams['optimizer'], model.parameters(), hyperparams['lr'])
        scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

        # Train for given epochs
        total = len(train_ids)
        model.train()
        for epoch in range(num_epochs):
            epoch_loss, epoch_acc = 0.0, 0

            # Iterate over data
            for inputs, labels in train_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                with torch.set_grad_enabled(True):
                  outputs = model(inputs.float())
                  _, preds = torch.max(outputs, 1)
                  loss = criterion(outputs, labels)
                  # backward
                  loss.backward()
                  optimizer.step()

                scheduler.step()
                # statistics
                epoch_loss += loss.item() * inputs.size(0)
                epoch_acc += torch.sum(preds == labels.data)

            epoch_loss = epoch_loss/total
            epoch_acc = epoch_acc.double()/total
            if (epoch+1)%5 == 0: 
              print('Epoch {}/{} --- Train Loss: {:.3f} Acc: {:.3f}' \
                    .format(epoch+1, num_epochs, epoch_loss, epoch_acc))

        # Evaluation for this fold
        correct, total = 0, len(test_ids)
        model.eval()
        with torch.no_grad():
          for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs.float())
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels.data)

        correct = correct.cpu().detach().numpy()
        eval_acc = correct/total
        results[fold+1] = {'train':epoch_acc, 'eval':eval_acc}
      
    # Print results for all folds
    print('\nCross validation results')
    avg = 0.0
    for key, value in results.items():
      print('Fold {}: Train acc = {:.3f} Val acc = {:.3f}'.format(key, value['train'], value['eval']))
      avg += value['eval']
    avg = avg/len(results)
    print(hyperparams)
    print('Average val acc: {:.3f} \n'.format(avg))
    return avg

def plot_performance(metric, values, configs):
    plt.plot(values['train'])
    plt.plot(values['val'])
    plt.xlabel("epochs")
    plt.ylabel(metric)
    plt.legend(values.keys())

    if not os.path.isdir(f"{configs.working_dir}/plots/{configs.arch}"):
        os.makedirs(f"{configs.working_dir}/plots/{configs.arch}")

    plt.savefig(f"{configs.working_dir}/plots/{configs.arch}/{metric}.png")
    plt.show()