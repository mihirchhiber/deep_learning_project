import torch
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

def eval_model(model, dataloaders, configs):

    model.eval() # Set model to evaluate mode

    t_output = []
    t_pred = []
    y_actual = []
    top_k = []
    # Iterate over data.
    i = 1
    for inputs, labels in dataloaders['test']:
        
        if configs.arch in ['rnn', 'gru', 'lstm']:
            # Reshape inputs to (batch_size, seq_length, input_size)
            inputs = inputs.reshape(-1, 338, 219).to(device=configs.device)
        else:
            inputs = inputs.to(device=configs.device)

        labels = labels.to(device=configs.device)
        y_actual.append(labels)

        # track history if only in train
        with torch.no_grad():
            outputs = model(inputs.float())
            _, preds = torch.max(outputs, 1)
            t_output.append(outputs)
            t_pred.append(preds)
            _, temp2 = outputs.topk(5)
            top_k.append(temp2)

    y_actual = torch.cat(y_actual).cpu().detach().numpy() 
    y_test = torch.cat(t_pred).cpu().detach().numpy() 
    # y_pred = torch.cat(top_k).cpu().detach().numpy() 

    print('\nConfusion Matrix')
    conf_mt = confusion_matrix(y_actual, y_test)
    print(conf_mt)
    plt.matshow(conf_mt)
    plt.show()
    print('\nClassification Report')
    print(classification_report(y_actual, y_test, zero_division=0))