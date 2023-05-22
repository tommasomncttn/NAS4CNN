# ===========================================
# ||                                       ||
# ||       Section 1: Importing modules    ||
# ||                                       ||
# ===========================================

import torch
import torch.nn as nn


# ===========================================
# ||                                       ||
# ||       Section 2: Model                ||
# ||                                       ||
# ===========================================

class MultiConfigCNN(nn.Module):
    '''Conv2d → f(.) → Pooling → Flatten → Linear 1 → f(.) → Linear 2 → Softmax
    '''
    def __init__(self, config, cnn_i_N = 1, pool_k_size = 2, fnn_o_N = 10):
        super(MultiConfigCNN, self).__init__()

        self.config = config 

        self.cnn_i_N = cnn_i_N
        self.cnn_o_N = self.config["conv_filters"]
        self.cnn_k_size = self.config["cnn_architectures"]["kernel_size"]
        self.stride = self.config["cnn_architectures"]["stride"]
        self.padding = self.config["cnn_architectures"]["padding"]
        self.pool_k_size = self.config["pooling"]["kernel_size"]
        self.fnn_o_N = self.config["linear_1_neurons"]
        self.pooling_type = self.config["pooling"]["pooling_type"]
        
        self.cnn =  nn.Conv2d(in_channels = cnn_i_N, out_channels = self.cnn_o_N, kernel_size = self.cnn_k_size, stride = self.stride, padding = self.padding)
        self.activation1 = self.config["activation"] 
        if self.pooling_type != "Average":
          self.pool = nn.MaxPool2d(kernel_size = self.pool_k_size) 
        else:
          self.pool = nn.AvgPool2d(kernel_size = self.pool_k_size) 
        self.flatten = nn.Flatten()

        self.linear1 = nn.Linear(in_features = self.compute_input_2_linear(), out_features = self.fnn_o_N)
        self.activation2 = self.config["activation"] 
        self.linear2 = nn.Linear(in_features = self.fnn_o_N, out_features = 10)
        self.softmax = nn.LogSoftmax(dim=1)

        self.nll = nn.NLLLoss(reduction="none") 
    
    def compute_input_2_linear(self):

        # computing after convolution => [(W-K+2P)/S]+1
        after_cnn_channels = self.cnn_o_N
        after_cnn_height = after_cnn_width = ((8 - self.cnn_k_size + 2 * self.padding) / self.stride) + 1
        
        # computing after pooling => fixed values stride=kernel_dimension, padding=0, dilation=1 => formula at end of https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html
        after_pool_height = after_pool_width = ((after_cnn_height - self.pool_k_size) / self.pool_k_size ) + 1

        # computing after flattening 

        return int(after_pool_height * after_pool_width * after_cnn_channels)

    def classify(self, log_prob):
        
        y_pred = torch.argmax(log_prob, dim = 1).long()        
        return y_pred

    def forward(self, x):
        
        x = self.cnn(x)
        x = self.activation1(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.activation2(x)
        x = self.linear2(x)
        log_prob = self.softmax(x)

        return log_prob


    def compute_loss(self, log_prob, y, reduction="avg"):
        
        y = y.to(torch.int64)
        loss = self.nll(log_prob, y)

        if reduction == "sum":
            return loss.sum()

        else:
            return loss.mean()

    def count_misclassified(self, predictions, targets):

        e = 1.0 * (predictions == targets)
        misclassified = (1.0 - e).sum().item()
        
        return misclassified

    @staticmethod
    def possible_config():
      possible_config = {
    "conv_filters": [8, 16, 32],
    "cnn_architectures": [
        {"kernel_size": 3, "stride" : 1, "padding" : 1},
     {"kernel_size": 5, "stride" : 1, "padding" : 2}
     ],
    "activation": [nn.ReLU(), nn.Sigmoid(), nn.Tanh(), nn.Softplus(), nn.ELU],
    "pooling": [
        {"kernel_size": 2, "pooling_type": "Average"},
        {"kernel_size": 2, "pooling_type": "Maximum"},
        {"kernel_size": 1, "pooling_type": "Average"},
        {"kernel_size": 1, "pooling_type": "Maximum"}
    ],
    "linear_1_neurons": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]}
      return possible_config
