import torch
import torch.nn as nn
import math

class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        self.x2h = nn.Linear(input_size, 4 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 4 * hidden_size, bias=bias)
        
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, input, hx=None):
        if hx is None:
            hx = input.new_zeros(input.size(0), self.hidden_size, requires_grad=False)
            hx = (hx, hx)
            
        # We used hx to pack both the hidden and cell states
        hx, cx = hx
        
        hi = self.x2h(input) + self.h2h(hx)
        i, f, o, g = torch.chunk(hi, 4, dim=-1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        cy = f * cx + i * g  
        hy = o * torch.tanh(cy)

        return (hy, cy)

    
class GRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super(GRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        self.x2h = nn.Linear(input_size, 2 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 2 * hidden_size, bias=bias)

        self.x2r = nn.Linear(input_size, hidden_size, bias=bias)
        self.h2r = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.reset_parameters()
        

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, input, hx=None):
        if hx is None:
            hx = input.new_zeros(self.hidden_size, requires_grad=False)

        z, r = torch.chunk(self.x2h(input) + self.h2h(hx), 2, -1)
        z = torch.sigmoid(z)
        r = torch.sigmoid(r)
        g = torch.tanh(self.h2r(hx)*r + self.x2r(input))
        hy = z * hx + (1 - z) * g
        
        return hy


class RNNModel(nn.Module):
    def __init__(self, mode, output_size, num_layers, bias):
        super(RNNModel, self).__init__()
        self.mode = mode
        self.input_size = output_size
        self.hidden_size = output_size
        self.num_layers = num_layers
        self.bias = bias
        self.output_size = output_size
        
        self.rnn_cell_list = nn.ModuleList()
        
        if mode == 'LSTM':
    
            self.rnn_cell_list.append(LSTMCell(self.input_size,
                                              self.hidden_size,
                                              self.bias))
            for l in range(1, self.num_layers):
                self.rnn_cell_list.append(LSTMCell(self.hidden_size,
                                                  self.hidden_size,
                                                  self.bias))
            

        elif mode == 'GRU':
    
            self.rnn_cell_list.append(GRUCell(self.input_size,
                                              self.hidden_size,
                                              self.bias))
            for l in range(1, self.num_layers):
                self.rnn_cell_list.append(GRUCell(self.hidden_size,
                                                  self.hidden_size,
                                                  self.bias))

        else:
            raise ValueError("Invalid RNN mode selected.")


        self.att_fc = nn.Linear(self.hidden_size, 1)
        self.fc = nn.Linear(self.hidden_size, self.output_size)

        
    def forward(self, input, time_steps=10, hx=None):
        # The 'input' is the input x into the first timestep
        # I think this should be a random vector
        assert input.shape == (self.output_size, )

        outs = []
        h0 = [None] * self.num_layers if hx is None else list(hx)
    

        X = [None] * time_steps
        X[0] = input # first input is 'input'
        for layer_idx, layer_cell in enumerate(self.rnn_cell_list):
            hx = h0[layer_idx]
            for i in range(time_steps):
                hx = layer_cell(X[i], hx)
                
                # we feed in this timestep's output into the next timestep's input
                # except if we are at the last timestep
                if i != time_steps-1:
                    X[i+1] = hx if self.mode == 'GRU' else hx[0]
                
        outs = X

        return outs
    

class BidirRecurrentModel(nn.Module):
    def __init__(self, mode, input_size, hidden_size, num_layers, bias, output_size):
        super(BidirRecurrentModel, self).__init__()
        self.mode = mode
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.output_size = output_size
        
        self.rnn_cell_list = nn.ModuleList()
        self.rnn_cell_list_rev = nn.ModuleList()
        
        if mode == 'LSTM':
            self.rnn_cell_list.append(LSTMCell(self.input_size,
                                               self.hidden_size,
                                               self.bias))
            for l in range(1, self.num_layers):
                self.rnn_cell_list.append(LSTMCell(self.hidden_size,
                                                  self.hidden_size,
                                                  self.bias))
                                                  
            self.rnn_cell_list_rev.append(LSTMCell(self.input_size,
                                                   self.hidden_size,
                                                   self.bias))
            for l in range(1, self.num_layers):
                self.rnn_cell_list_rev.append(LSTMCell(self.hidden_size,
                                                       self.hidden_size,
                                                       self.bias))
                                                  
        elif mode == 'GRU':
            self.rnn_cell_list.append(GRUCell(self.input_size,
                                              self.hidden_size,
                                              self.bias))
            for l in range(1, self.num_layers):
                self.rnn_cell_list.append(GRUCell(self.hidden_size,
                                                  self.hidden_size,
                                                  self.bias))
            
            self.rnn_cell_list_rev.append(GRUCell(self.input_size,
                                                  self.hidden_size,
                                                  self.bias))
            for l in range(1, self.num_layers):
                self.rnn_cell_list_rev.append(GRUCell(self.hidden_size,
                                                      self.hidden_size,
                                                      self.bias))
 
        else:
            raise ValueError("Invalid RNN mode selected.")


        self.fc = nn.Linear(2 * self.hidden_size, self.output_size)
        
        
    def forward(self, input, hx=None):
        assert NotImplementedError('right now this forward function is written for classification. \
                                You should modify it for our purpose, like the RNNModel was.')
        outs = []
        outs_rev = []
        
        X = list(input.permute(1, 0, 2))
        X_rev = list(input.permute(1, 0, 2))
        X_rev.reverse()
        hi = [None] * self.num_layers if hx is None else list(hx)
        hi_rev = [None] * self.num_layers if hx is None else list(hx)
        for j in range(self.num_layers):
            hx = hi[j]
            hx_rev = hi_rev[j]
            for i in range(input.shape[1]):
                hx = self.rnn_cell_list[j](X[i], hx)
                X[i] = hx if self.mode != 'LSTM' else hx[0]
                hx_rev = self.rnn_cell_list_rev[j](X_rev[i], hx_rev)
                X_rev[i] = hx_rev if self.mode != 'LSTM' else hx_rev[0]
        outs = X 
        outs_rev = X_rev

        out = outs[-1].squeeze()
        out_rev = outs_rev[0].squeeze()
        out = torch.cat((out, out_rev), 1)

        out = self.fc(out)
        return out