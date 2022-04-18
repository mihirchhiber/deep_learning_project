import torch
from sklearn.metrics import roc_curve, auc, roc_auc_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

def eval_model(model, dataloaders, configs):

    model.eval() # Set model to evaluate mode

    t_output = []
    t_pred = []
    y_test = []
    top_k = []
    # Iterate over data.
    i = 1
    for inputs, labels in dataloaders['test']:
        
        inputs = inputs.to(device=configs.device)
        labels = labels.to(device=configs.device)
        y_test.append(labels)

        # forward
        # track history if only in train
        with torch.set_grad_enabled(True):
            outputs = model(inputs.float())
            _, preds = torch.max(outputs, 1)
            t_output.append(outputs)
            t_pred.append(preds)
            _, temp2 = outputs.topk(5)
            top_k.append(temp2)

    y_test = torch.cat(y_test).cpu().detach().numpy() 
    y_test_num = torch.cat(t_pred).cpu().detach().numpy() 
    y_pred = torch.cat(top_k).cpu().detach().numpy() 

    print('\nConfusion Matrix')
    conf_mt = confusion_matrix(y_test_num, y_test)
    print(conf_mt)
    plt.matshow(conf_mt)
    plt.show()
    print('\nClassification Report')
    print(classification_report(y_test_num, y_test, zero_division=0))