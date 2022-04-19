import torch.nn as nn
import torch

class RecurrentNet(nn.Module):
    def __init__(self, input_size, hidden_dim, num_layers, output_size, configs, arch='gru'):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.arch = arch
        self.configs = configs
        
        # batch_first = True -> (batch, seq_dim, input_features)

        if arch == 'rnn':
            self.rnn = nn.RNN(input_size, hidden_dim, num_layers, batch_first=True)
        elif arch == 'gru':
            self.gru = nn.GRU(input_size, hidden_dim, num_layers, batch_first=True)
        else:
            self.lstm = nn.LSTM(input_size, hidden_dim, num_layers, batch_first=True)
        
        self.fc = nn.Linear(hidden_dim, output_size)
        
        # self.fc1 = nn.Linear(hidden_dim, 64)
        # self.fc2 = nn.Linear(64, 32)
        # self.fc3 = nn.Linear(32, output_size)
        # self.dropout = nn.Dropout(p=0.5)
        
    def forward(self, x):
        
        h_init = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device=self.configs.device)

        if self.arch == 'lstm':
            c_init = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device=self.configs.device)
            out, (_,_) = self.lstm(x, (h_init,c_init))
        elif self.arch == 'rnn':
            out, _ = self.rnn(x, h_init)
        else:
            out, _ = self.gru(x, h_init)
        
        out = self.fc(out[:,-1,:])

        # out = self.fc1(out[:,-1,:])
        # out = torch.sigmoid(out)
        # out = self.dropout(out)
        # out = self.fc2(out)
        # out = torch.sigmoid(out)
        # out = self.dropout(out)
        # out = self.fc3(out)

        # Place only the output from last step into the FC layer
        # out -> (batch_size, hidden_size)
        return out