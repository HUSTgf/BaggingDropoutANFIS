'''
    DropoutANFIS in torch
    @author: Fei Guo
    Acknowledgement: James Power' implementation of ANFIS in Pytorch:
    https://github.com/jfpower/anfis-pytorch
'''

import itertools

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.datasets import make_classification
from torch.utils.data import TensorDataset, DataLoader

dtype = torch.float


# These hooks are handy for debugging:
def module_hook(label):
    ''' Use this module hook like this:
        m = AnfisNet()
        m.layer.fuzzify.register_backward_hook(module_hook('fuzzify'))
        m.layer.consequent.register_backward_hook(modul_hook('consequent'))
    '''
    return (lambda module, grad_input, grad_output:
            print('BP for module', label,
                  'with out grad:', grad_output,
                  'and in grad:', grad_input))


def tensor_hook(label):
    '''
        If you want something more fine-graned, attach this to a tensor.
    '''
    return (lambda grad:
            print('BP for', label, 'with grad:', grad))


def is_pandas_ndframe(x):
    # the sklearn way of determining this
    return hasattr(x, 'iloc')


def to_numpy(X):
    """Generic function to convert a pytorch tensor to numpy.

    Returns X when it already is a numpy array.

    """
    if isinstance(X, np.ndarray):
        return X

    if is_pandas_ndframe(X):
        return X.values

    if X.is_cuda:
        X = X.cpu()

    if X.requires_grad:
        X = X.detach()

    return X.numpy()


def lsesq(A,B):
    initialGamma = 1000.
    coeffMat = A
    rhsMat = B
    S = np.eye(coeffMat.shape[1]) * initialGamma
    x = np.zeros((coeffMat.shape[1], 1))
    for i in range(len(coeffMat[:, 0])):
        a = coeffMat[i, :]
        b = np.array(rhsMat[i])
        S = S - (np.array(np.dot(np.dot(np.dot(S, np.matrix(a).transpose()), np.matrix(a)), S))) / (
                    1 + (np.dot(np.dot(S, a), a)))
        x = x + (np.dot(S, np.dot(np.matrix(a).transpose(), (np.matrix(b) - np.dot(np.matrix(a), x)))))
    return torch.Tensor(x)


def sinc(x, y):
    '''
        Sinc is a simple two-input non-linear function
        used by Jang in section V of his paper (equation 30).
    '''
    def s(z):
        return (1 if z == 0 else np.sin(z) / z)
    return s(x) * s(y)


def make_sinc_xy(batch_size=1024):
    '''
        Generates a set of (x, y) values for the sync function.
        Use the range (-10,10) that was used in sec. V of Jang's paper.
    '''
    pts = torch.arange(-10, 11, 2)
    x = torch.tensor(list(itertools.product(pts, pts)), dtype=dtype)
    y = torch.tensor([[sinc(*p)] for p in x], dtype=dtype)
    td = TensorDataset(x, y)
    return DataLoader(td, batch_size=batch_size, shuffle=True)



def coverage_score(model):

    firing_strength = model.weights.numpy()
    print(firing_strength.shape)
    # epsilon = 1. / (model.num_in * model.num_mfs)
    # print(epsilon)
    # print(torch.mean(model.weights, dim=0))
    def identity(arr):
        return any(np.where(arr > 0.2, True, False))
    # new_tmp = firing_strength[:8].copy()
    # select = [1,3,5,7,15,17,19,20,21,22,23,25,27,29,30,31]
    # new_tmp[:,select] = 0
    # np.savetxt('fig6_b.txt', new_tmp,fmt='%.4e')
    # np.savetxt('fig6_a.txt', firing_strength[:8],fmt='%.4e')
    result = np.apply_along_axis(identity, 1, firing_strength)
    return result.sum()/result.shape[0]


def consistency_score(model, y_pred):
    n_rules = model.num_rules
    n_samples = y_pred.shape[0]
    cats = y_pred.long().unsqueeze(1)
    # Now convert this to one-hot representation:
    y = torch.zeros((n_samples, 2), dtype=torch.float32)\
        .scatter(1, cats, 1)
    c = 0
    with torch.no_grad():
        for i in range(n_samples):
            tmp = model.rule_tsk[i].transpose(1,0)*model.weights[i].unsqueeze(1)
            tmp = F.softmax(tmp,dim=1)
            tmp -= y[i]
            tmp = torch.sum(tmp**2,dim=1)**(0.5)
            # if i <8:
            #     print(tmp)
            #     print(1-tmp/(2**0.5))
            r = np.where(tmp<(2**0.5)/2,1,0)
            c += r.sum()/n_rules
    return c/n_samples

def distinguishability_score(model):
    n_rules = model.num_rules
    tmp = model.layer['consequent'].coeff.detach().numpy()
    tmp = tmp.reshape(n_rules,-1)
    dist = np.sum(tmp**2, axis=1).reshape(n_rules,1) + np.sum(tmp**2, axis=1).reshape(1,n_rules) - \
        2 * np.matmul(tmp, tmp.T)

    dist = np.abs(dist)**0.5
    tmp = dist/ np.max(dist)

    return np.sum(tmp)/(n_rules**2)


def data_generated(in_feat:int , num_instances:int):
    # raw data
    num_features = in_feat
    # num_inf = num_features - 1
    x, y = make_classification(num_instances, num_features,\
                                random_state=0, n_classes=2)
    return x, y


def dataset_generated(x, y, batch_size=1024):
    # Torch datasets
    if type(x) != torch.Tensor:
        x = torch.tensor(x, dtype=torch.float32)
    if type(y) != torch.Tensor:
        y = torch.tensor(y,dtype=torch.float32).unsqueeze(1)
    td = TensorDataset(x, y)
    print(x.shape, y.shape)
    return DataLoader(td, batch_size=batch_size, shuffle=False)