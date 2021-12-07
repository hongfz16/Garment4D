import math

import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        # stdv = 0.0001
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
        # self.weight.data.fill_(0)
        # if self.bias is not None:
        #     self.bias.data.fill_(0)

    def forward(self, input, adj, ismlp=False):
        # support = torch.mm(input, self.weight)
        # output = torch.spmm(adj, support)
        if len(input.shape) == 3:
            B = input.shape[0]
            N = input.shape[1]
            support = torch.matmul(input, self.weight)
            if ismlp:
                return support if self.bias is None else support + self.bias
            support = support.transpose(0, 1).reshape(N, B * self.out_features)
            output = torch.spmm(adj, support)
            output = output.reshape(N, B, self.out_features).transpose(0, 1)
        else:
            support = torch.mm(input, self.weight)
            if ismlp:
                return support if self.bias is None else support + self.bias
            output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GraphConvolutionChebyshev(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolutionChebyshev, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight1 = Parameter(torch.FloatTensor(in_features, out_features))
        self.weight2 = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        # stdv = 0.0001
        self.weight1.data.uniform_(-stdv, stdv)
        self.weight2.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
        # self.weight.data.fill_(0)
        # if self.bias is not None:
        #     self.bias.data.fill_(0)

    def forward(self, input, adj, ismlp=False):
        assert ismlp == False
        # support = torch.mm(input, self.weight)
        # output = torch.spmm(adj, support)
        if len(input.shape) == 3:
            B = input.shape[0]
            N = input.shape[1]
            support = torch.matmul(input, self.weight2)
            support = support.transpose(0, 1).reshape(N, B * self.out_features)
            output = torch.spmm(adj, support)
            output = output.reshape(N, B, self.out_features).transpose(0, 1) + torch.matmul(input, self.weight1)
        else:
            support = torch.mm(input, self.weight)
            output = torch.spmm(adj, support) + torch.matmul(input, self.weight1)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GraphConvolutionChebyshev(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolutionChebyshev, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight1 = Parameter(torch.FloatTensor(in_features, out_features))
        self.weight2 = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        # stdv = 0.0001
        self.weight1.data.uniform_(-stdv, stdv)
        self.weight2.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
        # self.weight.data.fill_(0)
        # if self.bias is not None:
        #     self.bias.data.fill_(0)

    def forward(self, input, adj, ismlp=False):
        assert ismlp == False
        # support = torch.mm(input, self.weight)
        # output = torch.spmm(adj, support)
        if len(input.shape) == 3:
            B = input.shape[0]
            N = input.shape[1]
            support = torch.matmul(input, self.weight2)
            support = support.transpose(0, 1).reshape(N, B * self.out_features)
            output = torch.spmm(adj, support)
            output = output.reshape(N, B, self.out_features).transpose(0, 1) + torch.matmul(input, self.weight1)
        else:
            support = torch.mm(input, self.weight)
            output = torch.spmm(adj, support) + torch.matmul(input, self.weight1)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
