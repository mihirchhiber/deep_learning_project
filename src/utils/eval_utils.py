import torch
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from models.CustomCNN import CustomCNN
from inference_utils import songRecomendation

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
            inputs = inputs.reshape(-1, 339, 221).to(device=configs.device)
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

def eval_embed(song_name, song_embed):
    score = 0
    dc= {}
    dc_count = {}
    for name, embed in zip(song_name, song_embed):
        ans = songRecomendation(song_name, song_embed, embed, k=6)
        ans = [i[:-5] for i in ans]
        name = name[:-5]
        if dc.get(name,False) == False:
            dc[name] = 0
            dc_count[name] = 0
        dc_count[name] +=1
        dc[name] += (ans.count(name)-1)/5
        score += ans.count(name)/5
    for i in dc.keys():
        print(i,dc[i]/dc_count[i])
    score = score/(len(song_name))
    print("average score :",score)
