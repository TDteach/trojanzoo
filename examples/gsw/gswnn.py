import numpy as np
import torch
from torch import optim
from .mlp import MLP


class GSW_NN():
    def __init__(self, din=2, nofprojections=1, model_depth=1, num_filters=32, use_cuda=True):

        self.nofprojections = nofprojections

        if torch.cuda.is_available() and use_cuda:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.parameters = None  # This is for max-GSW
        self.din = din
        self.dout = nofprojections
        self.model_depth = model_depth
        self.num_filters = num_filters
        self.model = MLP(din=self.din, dout=self.dout, num_filters=self.num_filters, depth=self.model_depth)
        if torch.cuda.is_available() and use_cuda:
            self.model.cuda()

        # z = din * num_filters**model_depth
        # self.weight_cliping_limit = np.sqrt(din)/np.sqrt(z**2 * self.dout)
        # self.weight_cliping_limit = 1/np.sqrt(z)

    def gsw(self, X, Y, random=True):
        '''
        Calculates GSW between two empirical distributions.
        Note that the number of samples is assumed to be equal
        (This is however not necessary and could be easily extended
        for empirical distributions with different number of samples)
        '''
        N, dn = X.shape
        M, dm = Y.shape
        assert dn == dm and M == N

        if random:
            self.model.reset()

        Xslices = self.model(X.to(self.device))
        Yslices = self.model(Y.to(self.device))

        # Xslices = (torch.tanh(Xslices * 2.0) + 1.0) * 0.5
        # Yslices = (torch.tanh(Yslices * 2.0) + 1.0) * 0.5

        return torch.nn.functional.l1_loss(torch.mean(Xslices), torch.mean(Yslices))

        # Xslices_sorted = torch.sort(Xslices, dim=0)[0]
        # Yslices_sorted = torch.sort(Yslices, dim=0)[0]

        # return torch.sqrt(torch.mean((Xslices_sorted - Yslices_sorted) ** 2))
        # return torch.mean((Xslices_sorted - Yslices_sorted) ** 2)
        # return torch.nn.functional.l1_loss(Xslices_sorted, Yslices_sorted)

    def max_gsw(self, X, Y, iterations=50, lr=1e-4):
        N, dn = X.shape
        M, dm = Y.shape
        assert dn == dm and M == N

        self.model.reset()

        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        total_loss = np.zeros((iterations,))
        for i in range(iterations):

            optimizer.zero_grad()
            loss = -self.gsw(X.to(self.device), Y.to(self.device), random=False)
            total_loss[i] = loss.item()
            loss.backward(retain_graph=True)
            optimizer.step()

            # '''
            for p in self.model.parameters():
                #p.data.clamp_(-self.weight_cliping_limit, self.weight_cliping_limit)
                if len(p.shape) > 1:
                    fn = torch.norm(p).item()
                    if fn > 1:
                        p.data /= fn
            # '''

        return self.gsw(X.to(self.device), Y.to(self.device), random=False)
