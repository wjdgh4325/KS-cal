import torch
import torch.nn as nn
import torch.nn.functional as F
import util

def RELU(x):
    return x.relu()

def POOL(x):
    return F.max_pool2d(x, 2)

def POOL1d(x):
    return F.max_pool1d(x, 2)

class DeepNet(nn.Module):
    def __init__(self, D_in, D_out, HS, dropout_rate, batchnorm=False):
        super(DeepNet, self).__init__()
        self.fc1 = nn.Linear(D_in, HS[0])
        self.bn1 = nn.BatchNorm1d(HS[0])

        self.fc2 = nn.Linear(HS[0], HS[1])
        self.bn2 = nn.BatchNorm1d(HS[1])

        self.fc3 = nn.Linear(HS[1], HS[2])
        self.bn3 = nn.BatchNorm1d(HS[2])

        self.fc4 = nn.Linear(HS[2], D_out)
        self.drop1 = nn.Dropout(dropout_rate)
        self.drop2 = nn.Dropout(dropout_rate)

        self.batchnorm = batchnorm

    def forward(self, x):
        if self.batchnorm:
            x = self.drop1(RELU(self.bn1(self.fc1(x))))
            x = self.drop2(RELU(self.bn2(self.fc2(x))))
            x = RELU(self.bn3(self.fc3(x)))

        else:
            x = self.drop1(RELU(self.fc1(x)))
            x = self.drop2(RELU(self.fc2(x)))
            x = RELU(self.fc3(x))

        pred = self.fc4(x)
        
        return pred
    
class MTLRNN(nn.Module):
    def __init__(self, D_in, num_cat_bins, **kwargs):
        super(MTLRNN, self).__init__()
        # IN MTLR, the prob of first bin is exp(0)
        self.fc = nn.Linear(D_in, num_cat_bins-1)
        self.kwargs = kwargs
        self.D_in = D_in

    def forward(self, x):
        preds = self.fc(x)
        preds = util.cumsum_reverse(preds, dim=1)
        
        return preds
    
    def get_weight(self):
        weight = self.fc.weight
        
        return weight

class AFTNN(nn.Module):
    def __init__(self, data_dir, D_in, model_dist, big_gamma_nn, num_cat_bins, hiddensize, dropout_rate, **kwargs):
        super(AFTNN, self).__init__()

        self.kwargs = kwargs
        self.data_dir = data_dir
        self.D_in = D_in
        self.beta = nn.Parameter(torch.randn(D_in))
        self.pre_k = nn.Parameter(torch.tensor([1.0]))

    def smart_init_2(self):
        self.beta.data = torch.load('models/beta.pt')
        self.pre_k.data = torch.load('models/shape.pt')

    def smart_init(self, train_loader):
        """
        fit a lifelines AFT model
        """
        from lifelines import WeibullAFTFitter
        import pandas as pd

        dataset = train_loader.dataset

        x = dataset.src
        tgt = dataset.tgt

        t = tgt[:, 0]
        c = tgt[:, 1]

        print("t", t.size())
        print("c", c.size())
        print("x", x.size())

        tcx = torch.cat((t.unsqueeze(-1), c.unsqueeze(-1), x), dim=-1)
        print("tcx", tcx.size())
        tcx_numpy = tcx.numpy()

        col_names = ["T", "C"]
        for i in range(x.size()[-1]):
            col_names.append("f" + str(i))

        df = pd.DataFrame(data=tcx_numpy, columns=col_names)
        print("df shape", df.shape)

        print("initializing aft model with lifelines aft fit")

        aft = WeibullAFTFitter()
        aft.fit(df, 'T', 'C')
        aft.print_summary()

        lam = aft.params_.lambda_
        rho = aft.params_.rho_

        lam_list = [float(lam.loc['f{}'.format(i)]) for i in range(x.size()[-1])]

        lam = torch.tensor(lam_list)
        rho = torch.tensor([rho.to_numpy()])

        self.beta.data = lam
        self.pre_k.data = torch.tensor([0.3])
        print("Initialized AFT model with lifelines AFT fit")

    def forward(self, x):
        batch_sz = x.size()[0]
        expbetax = (self.beta*x).sum(-1).exp().view(-1, 1)
        pre_k = self.pre_k.view(-1, 1).repeat(batch_sz, 1)
        
        return torch.cat([expbetax, pre_k], dim=1)

class LogNormalhelper(nn.Module):
    def __init__(self, D_in, D_out, hidden_sizes, dropout_rate, batchnorm=False):
        super(LogNormalhelper, self).__init__()
        self.mu_model = DeepNet(D_in, D_out, hidden_sizes, dropout_rate, batchnorm)
        self.sigma_model = DeepNet(D_in, D_out, hidden_sizes, dropout_rate, batchnorm)
        
    def forward(self, src):
        mu = self.mu_model(src).view(-1, 1)
        pre_log_sigma = self.sigma_model(src).view(-1, 1)
        pred = torch.cat([mu, pre_log_sigma], dim=1)
        
        return pred

class CoxNN(nn.Module):
    def __init__(self, D_in, D_out, hidden_sizes, dropout_rate, batchnorm=False):
        super(CoxNN, self).__init__()
        self.risk_model = DeepNet(D_in, D_out, hidden_sizes, dropout_rate, batchnorm)

    def forward(self, src):
        pred = self.risk_model(src).view(-1, 1)

        return pred

class SyntheticNN(nn.Module):
    def __init__(self, data_dir, D_in, model_dist, dataset, num_cat_bins, hiddensize, dropout_rate, **kwargs):
        super(SyntheticNN, self).__init__()
        self.kwargs = kwargs
        self.data_dir = data_dir
        self.D_in = D_in

        assert model_dist in ['lognormal', 'cox', 'cat']
        
        self.model_dist = model_dist
        self.dataset = dataset
        # recent experiments
        #hidden_sizes = [1024, 1024, 1024]

        # re-create original paper results for gamma data with cat model
        if self.dataset == 'support':
            hidden_sizes = [56, 28, 28]

        elif self.dataset == 'sequence':
            hidden_sizes = [96, 48, 48]
        
        elif self.dataset in ['mimic', 'liver', 'stomach', 'lung', 'breast']:
            hidden_sizes = [60, 30, 30]
        
        elif self.dataset == 'dlbcl':
            hidden_sizes = [4096, 2048, 2048, 1024, 1024]

        elif self.dataset == 'nacd':
            hidden_sizes = [51, 51, 51]

        elif self.dataset == 'whas':
            hidden_sizes = [24, 12, 12]
            
        elif self.dataset == 'gbsg':
            hidden_sizes = [28, 14, 14]
            
        else:
            hidden_sizes = [128, 64, 64]

        if model_dist == 'lognormal':
            self.model = LogNormalhelper(D_in, 1, hidden_sizes, dropout_rate, batchnorm=True)
        
        elif model_dist == 'cox':
            self.model = CoxNN(D_in, 1, hidden_sizes, dropout_rate, batchnorm=True)

        elif model_dist == 'cat':
            self.model = DeepNet(D_in, num_cat_bins, hidden_sizes, dropout_rate, batchnorm=True)

        else:
            assert False

    def init_weights(self, m):
            if type(m) == nn.Linear:
                nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)

    def forward(self, src):
        pred = self.model(src)

        return pred

    def args_dict(self):
        """
        Get a dictionary of args that can be used to reconstruct this architecture.
        To use the returned dict, initialize the model with `ModelName(**model_args)`.
        """
        model_args = {'D_in': self.D_in}

        return model_args

class MIMICNN(nn.Module):
    def __init__(self, data_dir, dataset, D_in, model_dist, num_cat_bins, dropout_rate, **kwargs):
        super(MIMICNN, self).__init__()
        self.kwargs = kwargs
        self.data_dir = data_dir
        self.D_in = D_in
        self.dataset = dataset

        # assert model_dist in ['cat', 'mtlr'], "need to update this class for non-cat"
        if self.dataset == 'gbsg':
            hidden_sizes = [28, 14, 14]

        else:
            hidden_sizes = [128, 64, 64]
        #hidden_sizes = [2048, 1024, 1024]
        # self.model = DeepNet(self.D_in, D_out, ff_hidden_sizes, dropout_rate)

        if model_dist == 'cat':
            self.model = DeepNet(D_in, num_cat_bins, hidden_sizes, dropout_rate)

        elif model_dist == 'mtlr':
            self.model = nn.Sequential(DeepNet(D_in, num_cat_bins, hidden_sizes, dropout_rate), MTLRNN(num_cat_bins, num_cat_bins))

        elif model_dist == 'lognormal':
            self.model = LogNormalhelper(D_in, 1, hidden_sizes, dropout_rate, batchnorm=True)

        else:
            assert False

    def forward(self, src):
        pred = self.model(src)

        return pred

    def args_dict(self):
        """
        Get a dictionary of args that can be used to reconstruct this architecture.
        To use the returned dict, initialize the model with `ModelName(**model_args)`.
        """
        model_args = {'D_in': self.D_in}

        return model_args
    

class SurvMNISTNN(nn.Module):
    def __init__(self, data_dir, D_in, model_dist, num_cat_bins, dropout_rate, **kwargs):
        super(SurvMNISTNN, self).__init__()
        self.kwargs = kwargs
        self.data_dir = data_dir
        self.D_in = D_in
        self.model_dist = model_dist
        hidden_sizes = [512, 1024, 1024]
        last_filters = 256

        assert model_dist in ['cox', 'mtlr', 'lognormal']

        #self.conv = ConvHelper(F1=32,F2=64,F3=128,F4=last_filters,dr=dropout_rate)
        #DeepNet(last_filters,num_cat_bins,hidden_sizes,dropout_rate)

        if model_dist == 'mtlr':
            #self.ff = LeNetFF(last_filters, num_cat_bins)
            self.conv = LeNetConv(dropout_rate)
            self.ff = MTLRNN(last_filters, num_cat_bins)
        
        elif model_dist == 'lognormal':
            self.conv1 = LeNetConv(dropout_rate)
            self.conv2 = LeNetConv(dropout_rate)
            self.ff = LogNormalMNISTHelper(last_filters, 1, hidden_sizes, dropout_rate)

        elif model_dist == 'cox':
            self.conv = LeNetConv(dropout_rate)
            self.ff = CoxNN(last_filters, 1, hidden_sizes, dropout_rate)

        else:
            assert False

    def forward(self, src):
        if self.model_dist in ['cox', 'mtlr']:
            x = self.conv(src) # conv flattens
            pred = self.ff(x)

        elif self.model_dist == 'lognormal':
            x1 = self.conv1(src)
            x2 = self.conv2(src)
            pred = self.ff(x1, x2)

        return pred

    def args_dict(self):
        """Get a dictionary of args that can be used to reconstruct this architecture.
        To use the returned dict, initialize the model with `ModelName(**model_args)`.
        """
        model_args = {
            'D_in': self.D_in,
        }

        return model_args
    
class LeNetConv(nn.Module):
    def __init__(self, dropout_rate):
        super(LeNetConv, self).__init__()

        CH1=1
        CH2=64
        CH3=128
        CH4=256

        self.conv1 = nn.Conv2d(in_channels=CH1, out_channels=CH2, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(in_channels=CH2, out_channels=CH3, kernel_size=5, stride=1)
        self.conv3 = nn.Conv2d(in_channels=CH3, out_channels=CH4, kernel_size=4, stride=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.drop1 = nn.Dropout2d(dropout_rate)
        self.drop2 = nn.Dropout2d(dropout_rate)

    def forward(self, x, _print=False):
        x = self.conv1(x).relu()
        
        if _print:
            print("x.size()", x.size())

        x = self.drop1(x)

        x = self.pool1(x)
        
        if _print:
            print("x.size()", x.size())
        
        x = self.conv2(x).relu()
        
        if _print:
            print("x.size()", x.size())

        x = self.drop2(x)

        x = self.pool2(x)
        
        if _print:
            print("x.size()", x.size())
        
        x = self.conv3(x).relu()
        
        if _print:
            print("x.size()", x.size())
        
        x = torch.flatten(x, 1)
        
        if _print:
            print("x.size()", x.size())

        return x

class LogNormalMNISTHelper(nn.Module):
    def __init__(self, D_in, D_out, hidden_sizes, dropout_rate):
        super(LogNormalMNISTHelper,self).__init__()
        self.mu_model = DeepNet(D_in, D_out, hidden_sizes, dropout_rate)
        self.sigma_model = DeepNet(D_in, D_out, hidden_sizes, dropout_rate)

    def forward(self, src1, src2):
        mu = self.mu_model(src1).view(-1,1)
        pre_log_sigma = self.sigma_model(src2).view(-1,1)
        pred = torch.cat([mu, pre_log_sigma],dim=1)
        return pred