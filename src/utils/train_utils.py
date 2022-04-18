import matplotlib.pyplot as plt
import time
import copy
import torch
from tqdm import tqdm
from time import sleep

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
                    # # Iterate over data.
                    # for inputs, labels in dataloaders[phase]:
                    if phase == 'train':
                        tepoch.set_description(f"Epoch {epoch+1}")
                    elif phase == 'val':
                        tepoch.set_description(f"Val {epoch+1}")

                    inputs = inputs.type(torch.DoubleTensor)
                    inputs = inputs.to(device=configs.device)
                    labels = labels.to(device=configs.device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
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

    # load best model weights
    model.load_state_dict(best_model_wts)

    # save model weights
    torch.save(model.state_dict(),f"{configs.checkpoints_dir}/{configs.arch}_weights.pth")

    return model, losses, accuracies

def plot_performance(metric, values):
    plt.plot(values['train'])
    plt.plot(values['val'])
    plt.xlabel("epochs")
    plt.ylabel(metric)
    plt.legend(values.keys())
    plt.show()