#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
    DropoutANFIS in torch
    @author: Fei Guo
    Acknowledgement: James Power' implementation of ANFIS in Pytorch:
    https://github.com/jfpower/anfis-pytorch
'''

import itertools
import numbers
from abc import ABCMeta, abstractmethod
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.base import ClassifierMixin
from sklearn.ensemble.bagging import _generate_bagging_indices, _parallel_build_estimators, _parallel_predict_proba
from sklearn.metrics import accuracy_score, f1_score
from sklearn.multiclass import check_classification_targets
from sklearn.utils import check_array, check_random_state, check_consistent_length, column_or_1d, indices_to_mask
from sklearn.utils._joblib import effective_n_jobs, Parallel, delayed
from sklearn.utils.validation import has_fit_parameter, check_is_fitted
from skorch.callbacks import Callback

from fuzzynet.membership import make_mfs, make_mfs_clustering_fcm, make_mfs_clustering_essc
from fuzzynet.utils import dataset_generated

MAX_INT = np.iinfo(np.int32).max

dtype = torch.float


class FuzzifyVariable(torch.nn.Module):
    '''
        Represents a single fuzzy variable, holds a list of its MFs.
        Forward pass will then fuzzify the input (value for each MF).
    '''

    def __init__(self, mfdefs):
        super(FuzzifyVariable, self).__init__()
        if isinstance(mfdefs, list):  # No MF names supplied
            mfnames = ['mf{}'.format(i) for i in range(len(mfdefs))]
            mfdefs = OrderedDict(zip(mfnames, mfdefs))
        self.mfdefs = torch.nn.ModuleDict(mfdefs)
        self.padding = 0

    @property
    def num_mfs(self):
        '''Return the actual number of MFs (ignoring any padding)'''
        return len(self.mfdefs)

    def members(self):
        '''
            Return an iterator over this variables's membership functions.
            Yields tuples of the form (mf-name, MembFunc-object)
        '''
        return self.mfdefs.items()

    def pad_to(self, new_size):
        '''
            Will pad result of forward-pass (with zeros) so it has new_size,
            i.e. as if it had new_size MFs.
        '''
        self.padding = new_size - len(self.mfdefs)

    def fuzzify(self, x):
        '''
            Yield a list of (mf-name, fuzzy values) for these input values.
        '''
        for mfname, mfdef in self.mfdefs.items():
            yvals = mfdef(x)
            yield (mfname, yvals)

    def forward(self, x):
        '''
            Return a tensor giving the membership value for each MF.
            x.shape: n_cases
            y.shape: n_cases * n_mfs
        '''
        y_pred = torch.cat([mf(x) for mf in self.mfdefs.values()], dim=1)
        if self.padding > 0:
            y_pred = torch.cat([y_pred,
                                torch.zeros(x.shape[0], self.padding)], dim=1)
        return y_pred


class FuzzifyLayer(torch.nn.Module):
    '''
        A list of fuzzy variables, representing the inputs to the FIS.
        Forward pass will fuzzify each variable individually.
        We pad the variables so they all seem to have the same number of MFs,
        as this allows us to put all results in the same tensor.
    '''

    def __init__(self, varmfs, varnames=None):
        super(FuzzifyLayer, self).__init__()
        if not varnames:
            self.varnames = ['x{}'.format(i) for i in range(len(varmfs))]
        else:
            self.varnames = list(varnames)
        maxmfs = max([var.num_mfs for var in varmfs])
        for var in varmfs:
            var.pad_to(maxmfs)
        self.varmfs = torch.nn.ModuleDict(zip(self.varnames, varmfs))
        self.raw_parameters = self.get_parameters()

    @property
    def num_in(self):
        '''Return the number of input variables'''
        return len(self.varmfs)

    @property
    def max_mfs(self):
        ''' Return the max number of MFs in any variable'''
        return max([var.num_mfs for var in self.varmfs.values()])

    def get_parameters(self):
        """
        :return: [num_features, num_mfs, parameters(mean, sigma)]
        """
        # res = np.zeros(shape=(self.num_in,self.max_mfs,2))
        res = [[[v.detach().numpy() for v in mfdef.parameters()] for mfdef in members.mfdefs.values()] for members in
               self.varmfs.values()]
        res = np.array(res)
        return res

    def get_raw_parameters(self):
        """
        :return: the initial MFs' parameters
        """
        return self.raw_parameters

    def __repr__(self):
        """
            Print the variables, MFS and their parameters (for info only)
        """
        r = ['Input variables']
        for varname, members in self.varmfs.items():
            r.append('Variable {}'.format(varname))
            for mfname, mfdef in members.mfdefs.items():
                r.append('- {}: {}({})'.format(mfname,
                                               mfdef.__class__.__name__,
                                               ', '.join(['{}={}'.format(n, p.item())
                                                          for n, p in mfdef.named_parameters()])))
        return '\n'.join(r)

    def forward(self, x):
        ''' Fuzzyify each variable's value using each of its corresponding mfs.
            x.shape = n_cases * n_in
            y.shape = n_cases * n_in * n_mfs
        '''
        assert x.shape[1] == self.num_in, \
            '{} is wrong no. of input values'.format(self.num_in)
        y_pred = torch.stack([var(x[:, i:i + 1])
                              for i, var in enumerate(self.varmfs.values())],
                             dim=1)
        return y_pred


class AntecedentLayer(torch.nn.Module):
    '''
        Form the 'rules' by taking all possible combinations of the MFs
        for each variable. Forward pass then calculates the fire-strengths.
    '''

    def __init__(self, varlist, init_method):

        super(AntecedentLayer, self).__init__()
        if init_method == 'no clustering':
            # Count the (actual) mfs for each variable:
            mf_count = [var.num_mfs for var in varlist]
            # Now make the MF indices for each rule:
            # mf_indices.shape is n_rules * n_in
            mf_indices = itertools.product(*[range(n) for n in mf_count])
            self.mf_indices = torch.tensor(list(mf_indices)).long()

        else:
            # mf_indices.shape is n_rules(n_clusters) * n_in
            n_clusters = varlist[0].num_mfs
            n_in = len(varlist)
            mf_indices = [[c for i in range(n_in)] for c in range(n_clusters)]
            self.mf_indices = torch.tensor(list(mf_indices)).long()

        self.init_method = init_method

    def num_rules(self):
        return self.mf_indices.shape[0]

    def extra_repr(self, varlist=None):
        rule_set = []
        for rule_idx in range(self.mf_indices.shape[0]):
            thisrule = []
            for f_idx, mf_idx in enumerate(self.mf_indices[rule_idx]):
                thisrule.append("X{}_mf{}".format(f_idx + 1, mf_idx))
            rule_set.append(' and '.join(thisrule))
        return '\n'.join(rule_set)
        # if not varlist:
        #     return None
        # row_ants = []
        # mf_count = [len(fv.mfdefs) for fv in varlist.values()]
        # for rule_idx in itertools.product(*[range(n) for n in mf_count]):
        #     thisrule = []
        #     for (varname, fv), i in zip(varlist.items(), rule_idx):
        #         thisrule.append('{} is {}'
        #                         .format(varname, list(fv.mfdefs.keys())[i]))
        #     row_ants.append(' and '.join(thisrule))
        # return '\n'.join(row_ants)

    def forward(self, x):
        ''' Calculate the fire-strength for (the antecedent of) each rule
            x.shape = n_cases * n_in * n_mfs
            y.shape = n_cases * n_rules
        '''
        # Expand (repeat) the rule indices to equal the batch size:
        batch_indices = self.mf_indices.expand((x.shape[0], -1, -1))
        # Then use these indices to populate the rule-antecedents
        ants = torch.gather(x.transpose(1, 2), 1, batch_indices) + 1e-12
        # ants.shape is n_cases * n_rules * n_in
        # Last, take the AND (= product) for each rule-antecedent
        # rules = torch.min(ants, dim=2)
        # rules = torch.min(ants,dim=2)[0]
        rules = torch.prod(ants, dim=2)
        return rules


class ConsequentLayer(torch.nn.Module):
    '''
        A simple linear layer to represent the TSK consequents.
        Hybrid learning, so use MSE (not BP) to adjust coefficients.
        Hence, coeffs are no longer parameters for backprop.
    '''

    def __init__(self, d_in, d_rule, d_out):
        super(ConsequentLayer, self).__init__()
        c_shape = torch.Size([d_rule, d_out, d_in + 1])
        self._coeff = torch.zeros(c_shape, dtype=dtype, requires_grad=True)
        torch.nn.init.kaiming_normal_(self._coeff)

    @property
    def coeff(self):
        '''
            Record the (current) coefficients for all the rules
            coeff.shape: n_rules * n_out * (n_in+1)
        '''
        return self._coeff

    @coeff.setter
    def coeff(self, new_coeff):
        '''
            Record new coefficients for all the rules
            coeff: for each rule, for each output variable:
                   a coefficient for each input variable, plus a constant
        '''
        assert new_coeff.shape == self.coeff.shape, \
            'Coeff shape should be {}, but is actually {}' \
                .format(self.coeff.shape, new_coeff.shape)
        self._coeff = new_coeff

    def fit_coeff(self, x, weights, y_actual):
        '''
            Use LSE to solve for coeff: y_actual = coeff * (weighted)x
                  x.shape: n_cases * n_in
            weights.shape: n_cases * n_rules
            [ coeff.shape: n_rules * n_out * (n_in+1) ]
                  y.shape: n_cases * n_out
        '''
        # Append 1 to each list of input vals, for the constant term:
        x_plus = torch.cat([x, torch.ones(x.shape[0], 1)], dim=1)
        # Shape of weighted_x is n_cases * n_rules * (n_in+1)
        weighted_x = torch.einsum('bp, bq -> bpq', weights, x_plus)
        # Can't have value 0 for weights, or LSE won't work:
        # Value < 1e-12 are clamped to 1e-12
        weighted_x[weighted_x == 0] = 1e-12
        # Squash x and y down to 2D matrices for gels:
        weighted_x_2d = weighted_x.view(weighted_x.shape[0], -1)
        y_actual_2d = y_actual.view(y_actual.shape[0], -1)
        # Use gels to do LSE, then pick out the solution rows:
        try:
            coeff_2d, _ = torch.lstsq(y_actual_2d, weighted_x_2d)
            # coeff_2d = lsesq(weighted_x_2d, y_actual_2d)
        except RuntimeError as e:
            print('Internal error in gels', e)
            print('Weights are:', weighted_x)
            raise e
        coeff_2d = coeff_2d[0:weighted_x_2d.shape[1]]
        # Reshape to 3D tensor: divide by rules, n_in+1, then swap last 2 dims
        self.coeff = coeff_2d.view(weights.shape[1], x.shape[1] + 1, -1) \
            .transpose(1, 2)
        # coeff dim is thus: n_rules * n_out * (n_in+1)

    def forward(self, x):
        '''
            Calculate: y = coeff * x + const   [NB: no weights yet]
                  x.shape: n_cases * n_in
              coeff.shape: n_rules * n_out * (n_in+1)
                  y.shape: n_cases * n_out * n_rules
        '''
        # Append 1 to each list of input vals, for the constant term:
        x_plus = torch.cat([x, torch.ones(x.shape[0], 1)], dim=1)
        # Need to switch dimansion for the multipy, then switch back:
        y_pred = torch.matmul(self.coeff, x_plus.t())
        return y_pred.transpose(0, 2)  # swaps cases and rules


class PlainConsequentLayer(ConsequentLayer):
    '''
        A linear layer to represent the TSK consequents.
        Not hybrid learning, so coefficients are backprop-learnable parameters.
    '''

    def __init__(self, *params):
        super(PlainConsequentLayer, self).__init__(*params)
        self.register_parameter('coefficients',
                                torch.nn.Parameter(self._coeff))

    @property
    def coeff(self):
        '''
            Record the (current) coefficients for all the rules
            coeff.shape: n_rules * n_out * (n_in+1)
        '''
        return self.coefficients

    def fit_coeff(self, x, weights, y_actual):
        '''
        '''
        assert False, \
            'Not hybrid learning: I\'m using BP to learn coefficients'


class WeightedSumLayer(torch.nn.Module):
    '''
        Sum the TSK for each outvar over rules, weighted by fire strengths.
        This could/should be layer 5 of the Anfis net.
        I don't actually use this class, since it's just one line of code.
    '''

    def __init__(self):
        super(WeightedSumLayer, self).__init__()

    def forward(self, weights, tsk):
        '''
            weights.shape: n_cases * n_rules
                tsk.shape: n_cases * n_out * n_rules
             y_pred.shape: n_cases * n_out
        '''
        # Add a dimension to weights to get the bmm to work:
        y_pred = torch.bmm(tsk, weights.unsqueeze(2))
        return y_pred.squeeze(2)


class AnfisNet(torch.nn.Module):
    '''
        This is a container for the 5 layers of the ANFIS net.
        The forward pass maps inputs to outputs based on current settings,
        and then fit_coeff will adjust the TSK coeff using LSE.
    '''

    def __init__(self, description, invardefs, outvarnames, hybrid=True):
        super(AnfisNet, self).__init__()
        self.description = description
        self.outvarnames = outvarnames
        self.num_mfs = len(invardefs[0][1])
        self.mf_type = invardefs[0][1][0].mf_type
        self.dropout = False  # version flag
        self.hybrid = hybrid
        varnames = [v for v, _ in invardefs]
        mfdefs = [FuzzifyVariable(mfs) for _, mfs in invardefs]
        self.num_in = len(invardefs)
        if self.num_in > 6:
            self.num_rules = 100
        else:
            self.num_rules = np.prod([len(mfs) for _, mfs in invardefs])
        if self.hybrid:
            cl = ConsequentLayer(self.num_in, self.num_rules, self.num_out)
        else:
            cl = PlainConsequentLayer(self.num_in, self.num_rules, self.num_out)
        self.layer = torch.nn.ModuleDict(OrderedDict([
            ('fuzzify', FuzzifyLayer(mfdefs, varnames)),
            ('rules', AntecedentLayer(mfdefs)),
            # normalisation layer is just implemented as a function.
            ('consequent', cl),
            # weighted-sum layer is just implemented as a function.
        ]))

    @property
    def num_out(self):
        return len(self.outvarnames)

    @property
    def coeff(self):
        return self.layer['consequent'].coeff

    @coeff.setter
    def coeff(self, new_coeff):
        self.layer['consequent'].coeff = new_coeff

    def fit_coeff(self, x, y_actual):
        '''
            Do a forward pass (to get weights), then fit to y_actual.
            Does nothing for a non-hybrid ANFIS, so we have same interface.
        '''
        if self.hybrid:
            self(x)
            self.layer['consequent'].fit_coeff(x, self.weights, y_actual)

    def input_variables(self):
        '''
            Return an iterator over this system's input variables.
            Yields tuples of the form (var-name, FuzzifyVariable-object)
        '''
        return self.layer['fuzzify'].varmfs.items()

    def output_variables(self):
        '''
            Return an list of the names of the system's output variables.
        '''
        return self.outvarnames

    def extra_repr(self):
        rstr = []
        vardefs = self.layer['fuzzify'].varmfs
        rule_ants = self.layer['rules'].extra_repr(vardefs).split('\n')
        for i, crow in enumerate(self.layer['consequent'].coeff):
            rstr.append('Rule {:2d}: IF {}'.format(i, rule_ants[i]))
            rstr.append(' ' * 9 + 'THEN {}'.format(crow.tolist()))
        return '\n'.join(rstr)

    def forward(self, x):
        '''
            Forward pass: run x thru the five layers and return the y values.
            I save the outputs from each layer to an instance variable,
            as this might be useful for comprehension/debugging.
        '''
        self.fuzzified = self.layer['fuzzify'](x)
        self.raw_weights = self.layer['rules'](self.fuzzified)
        self.weights = F.normalize(self.raw_weights, p=1, dim=1)
        self.rule_tsk = self.layer['consequent'](x)
        # y_pred = self.layer['weighted_sum'](self.weights, self.rule_tsk)
        y_pred = torch.bmm(self.rule_tsk, self.weights.unsqueeze(2))
        self.y_pred = y_pred.squeeze(2)
        return self.y_pred


class DropoutANFIS(torch.nn.Module):
    '''
        This is a container for the 6(5+relu) layers of the ANFIS net.
        The forward pass maps inputs to outputs based on current settings,
        and then fit_coeff will adjust the TSK coeff using LSE.
        init_method: can be each of blow "no clustering", "fcm clustering", or "sc clustering"
    '''

    def __init__(self, description, invardefs, outvarnames, hybrid=False, proba=0.5, init_method="no clustering"):
        super(DropoutANFIS, self).__init__()
        self.description = description
        self.outvarnames = outvarnames
        self.num_mfs = len(invardefs[0][1])
        self.mf_type = invardefs[0][1][0].mf_type
        self.hybrid = hybrid
        self.dropout = True  # version flag
        varnames = [v for v, _ in invardefs]
        mfdefs = [FuzzifyVariable(mfs) for _, mfs in invardefs]
        self.num_in = len(invardefs)
        if init_method == 'no clustering':
            self.num_rules = np.prod([len(mfs) for _, mfs in invardefs])
            print("init method: {}".format(init_method))

        else:
            print("init method: {}".format(init_method))
            self.num_rules = len(invardefs[0][1])
        print("Rules in the model :{}".format(self.num_rules))
        if self.hybrid:
            cl = ConsequentLayer(self.num_in, self.num_rules, self.num_out)
        else:
            cl = PlainConsequentLayer(self.num_in, self.num_rules, self.num_out)
        self.layer = torch.nn.ModuleDict(OrderedDict([
            ('fuzzify', FuzzifyLayer(mfdefs, varnames)),
            ('rules', AntecedentLayer(mfdefs, init_method)),
            ('dropout', torch.nn.Dropout(p=proba)),
            # normalisation layer is just implemented as a function.
            ('bn', torch.nn.BatchNorm1d(self.num_in)),
            ('consequent', cl),
            # weighted-sum layer is just implemented as a function.
        ]))

        self.own_rules_index = torch.ones(self.num_rules)

    @property
    def num_out(self):
        return len(self.outvarnames)

    @property
    def coeff(self):
        return self.layer['consequent'].coeff

    @coeff.setter
    def coeff(self, new_coeff):
        self.layer['consequent'].coeff = new_coeff

    def fit_coeff(self, x, y_actual):
        '''
            Do a forward pass (to get weights), then fit to y_actual.
            Does nothing for a non-hybrid ANFIS, so we have same interface.
        '''
        if self.hybrid:
            self(x)
            self.layer['consequent'].fit_coeff(x, self.weights, y_actual)

    def input_variables(self):
        '''
            Return an iterator over this system's input variables.
            Yields tuples of the form (var-name, FuzzifyVariable-object)
        '''
        return self.layer['fuzzify'].varmfs.items()

    def output_variables(self):
        '''
            Return an list of the names of the system's output variables.
        '''
        return self.outvarnames

    def extra_repr(self):
        rstr = []
        vardefs = self.layer['fuzzify'].varmfs
        rule_ants = self.layer['rules'].extra_repr(vardefs).split('\n')
        for i, crow in enumerate(self.layer['consequent'].coeff):
            rstr.append('Rule {:2d}: IF {}'.format(i, rule_ants[i]))
            rstr.append(' ' * 9 + 'THEN {}'.format(crow.tolist()))
        return '\n'.join(rstr)

    def set_rules(self, index):
        print(f"raw rules: {self.own_rules_index.sum()}")
        for i in index:
            self.own_rules_index[i] = 0.
        print(f"rules after drop: {self.own_rules_index.sum()}")

    def reset_rule(self):
        self.own_rules_index = torch.ones(self.num_rules)

    def forward(self, x):
        '''
            Forward pass: run x thru the five layers and return the y values.
            I save the outputs from each layer to an instance variable,
            as this might be useful for comprehension/debugging.
        '''
        self.fuzzified = self.layer['fuzzify'](x)
        self.raw_weights = self.layer['rules'](self.fuzzified) + 1e-08
        self.raw_weights[torch.isnan(self.raw_weights)] = 1e-12
        self.dropout_weights = self.layer['dropout'](self.raw_weights)
        self.weights = F.normalize(self.dropout_weights, p=1, dim=1)
        # print(self.weights[0,:])
        # print(torch.sum(self.weights[0]))
        x = self.layer['bn'](x)
        self.rule_tsk = self.layer['consequent'](x)
        # y_pred = self.layer['weighted_sum'](self.weights, self.rule_tsk)
        y_pred = torch.bmm(self.rule_tsk, self.weights.unsqueeze(2))
        self.y_pred = y_pred.squeeze(2)
        # self.y_pred.sigmoid()
        return self.y_pred

    def infer(self, x):
        self.fuzzified = self.layer['fuzzify'](x)
        self.raw_weights = self.layer['rules'](self.fuzzified) + 1e-08
        self.raw_weights[torch.isnan(self.raw_weights)] = 1e-12
        self.dropout_weights = self.raw_weights * self.own_rules_index
        self.weights = F.normalize(self.dropout_weights, p=1, dim=1)
        x = self.layer['bn'](x)
        self.rule_tsk = self.layer['consequent'](x)
        y_pred = torch.bmm(self.rule_tsk, self.weights.unsqueeze(2))
        self.y_pred = y_pred.squeeze(2)
        return self.y_pred


class FittingCallback(Callback):
    '''
        In order to use ANFIS-style hybrid learning with sklearn/skorch,
        we need to add a callback to do the LSE step after each epoch.
        This class contains that callback hook.
    '''

    def __init__(self):
        super(FittingCallback, self).__init__()

    def on_epoch_end(self, net, dataset_train=None,
                     dataset_valid=None, **kwargs):
        # Get the dataset: different if we're train or train/test
        # In the latter case, we have a Subset that contains the data...
        if isinstance(dataset_train, torch.utils.data.dataset.Subset):
            dataset_train = dataset_train.dataset
        with torch.no_grad():
            net.module.fit_coeff(dataset_train.X, dataset_train.y)


class SkAnfis:
    def __init__(self, num_mfs, num_out, mf_type, dropout, proba, labels, hybrid=False, penalty=False,
                 init_method='no clustering'):
        self.anfis = None
        self.num_mfs = num_mfs
        self.num_out = num_out
        self.mf_type = mf_type
        self.dropout = dropout
        self.drop_proba = proba
        self.hybrid = hybrid
        self.penalty = penalty
        self.labels_ = labels
        self.classes_ = []
        self.init_method = init_method

        self.thresh_list = []
        self.percentiles = np.array([50, 70, 80, 90, 95, 97.5])

    def fit(self, X, y):
        train_data = dataset_generated(X, y)
        class_k = np.unique(y)
        self.classes_.append(class_k)
        self.classes_ = self.classes_[0]
        if self.init_method == 'fcm clustering':
            print('using fcm')
            invars, outvars = make_mfs_clustering_fcm(X, 20, self.num_out, self.mf_type)
        elif self.init_method == 'essc clustering':
            print('using essc')
            invars, outvars = make_mfs_clustering_essc(X, 20, self.num_out, self.mf_type)
        else:
            print('using no')
            invars, outvars = make_mfs(X, self.num_mfs, self.num_out, self.mf_type)
        if self.dropout:
            print('Using Dropout ANFIS')
            self.anfis = DropoutANFIS('Dropout ANFIS', invars, outvars, hybrid=self.hybrid, proba=self.drop_proba,
                                      init_method=self.init_method)
        else:
            print('Using Original ANFIS')
            self.anfis = AnfisNet('Original ANFIS', invars, outvars, hybrid=self.hybrid)

        optimizer = torch.optim.Adam(self.anfis.parameters(), lr=1e-3, amsgrad=True)

        def criterion(input, target):  # change the dim and type
            tmp = 0.
            if self.penalty:
                tmp = torch.clamp(torch.log(self.anfis.weights), min=-1e10)
                tmp = torch.sum(0.3 * self.anfis.weights * tmp)
            return torch.nn.CrossEntropyLoss()(input, target.squeeze().long()) - tmp

        train_anfis_with(self.anfis, train_data, optimizer, criterion, 200)

    def predict(self, X):
        print('predict')
        with torch.no_grad():
            # self.anfis.eval()
            y_p_test = F.softmax(self.anfis.infer(X), dim=1)

            self.y_p_test = torch.argmax(y_p_test, dim=1).squeeze()
            return self.y_p_test

    def predict_proba(self, X):

        with torch.no_grad():
            self.anfis.eval()
            y_p_praba = F.softmax(self.anfis.infer(X), dim=1)
            return y_p_praba.data.numpy()

    def drop_rule(self, x, threshold_index):
        self.anfis.reset_rule()

        res = self.get_avg_firing_strength(x)
        res = res.data.numpy()

        self.thresh_list = np.percentile(res, self.percentiles)
        print(f"Threshold values: {self.thresh_list}")
        drop_index = np.where(res < self.thresh_list[threshold_index])
        self.anfis.set_rules(drop_index)
        # print(f"Drop rules: {drop_index}")

    def evaluate(self, x_test, y_test):
        test_acc = accuracy_score(y_test.squeeze(), self.predict(x_test))
        test_f1 = f1_score(y_test.squeeze(), self.predict(x_test), average='micro')
        results = [test_acc, test_f1]
        print(results)
        return results

    def get_NOR(self):
        """
        :return: the number of rules in an ANFIS
        """
        return self.anfis.num_rules

    def get_ARL(self):
        """
        :return: the length of rules in an ANFIS
        """
        return self.anfis.num_in

    def get_CSR(self, X):
        with torch.no_grad():
            # print(self.anfis.layer['fuzzify'])
            # print(self.anfis.layer['rules'].extra_repr())
            # print the normalized firing strength of a sample
            # print the rule output of a sample
            y_p_test = F.softmax(self.anfis(X), dim=1)
            y_p_idx = torch.argmax(y_p_test, dim=1).squeeze()

            tmp = self.anfis.layer['fuzzify'](X)
            tmp = self.anfis.layer['rules'](tmp) + 1e-08
            fs_rules = F.normalize(tmp, p=1, dim=1)

            tmp_1 = self.anfis.layer['consequent'](self.anfis.layer['bn'](X))
            # tmp_1.transpose()
            consequent_rules = tmp_1[range(len(y_p_idx)), y_p_idx]
            consequent_rules[consequent_rules < 0] = 0
            normalized_consequent_rules = F.normalize(consequent_rules, p=1, dim=1)

            CSR_rules = 1.0 - torch.mean(torch.abs(fs_rules - normalized_consequent_rules), dim=0).numpy()

            return np.mean(CSR_rules)

    def get_avg_firing_strength(self, X):
        with torch.no_grad():
            self.anfis.eval()
            tmp = self.anfis.layer['fuzzify'](X)
            tmp = self.anfis.layer['rules'](tmp) + 1e-08
            tmp = F.normalize(tmp, p=1, dim=1)
            # given input's firing strengths
            return tmp.mean(dim=0)

    def get_firing_strength_of_given_inputs(self, X):
        with torch.no_grad():
            tmp = self.anfis.layer['fuzzify'](X)
            tmp = self.anfis.layer['rules'](tmp) + 1e-08
            tmp = F.normalize(tmp, p=1, dim=1)
            # given input's firing strengths
            return tmp

    def get_co_firing_strength(self, X):
        with torch.no_grad():
            tmp = self.anfis.layer['fuzzify'](X)
            tmp = self.anfis.layer['rules'](tmp) + 1e-08
            tmp = F.normalize(tmp, p=1, dim=1)
            return tmp

    def get_DR(self):
        with torch.no_grad():
            mf_idx = self.anfis.layer['rules'].mf_indices
            n_rules = mf_idx.shape[0]
            cp_rules = self.anfis.layer['consequent'].coeff.detach().numpy().reshape(n_rules, -1)
            dist = np.sum(cp_rules ** 2, axis=1).reshape(n_rules, 1) + np.sum(cp_rules ** 2, axis=1).reshape(1,
                                                                                                             n_rules) - \
                   2 * np.matmul(cp_rules, cp_rules.T)
            dist = np.abs(dist) ** 0.5
            dist = dist / np.max(dist)

            return np.mean(dist)

    def get_MFD(self):
        with torch.no_grad():
            raw_ap = self.anfis.layer['fuzzify'].get_raw_parameters()
            ap = self.anfis.layer['fuzzify'].get_parameters()

            raw_ap_center = raw_ap[:, :, 0].ravel()
            ap_center = ap[:, :, 0].ravel()

            left_value = np.min([np.min(raw_ap_center), np.min(ap_center)])
            right_value = np.max([np.max(raw_ap_center), np.max(ap_center)])

            displacement = np.abs(raw_ap_center - ap_center) / (right_value - left_value)
            mfd = 1 - np.max(displacement)
            return mfd

    def get_MFAS(self):
        with torch.no_grad():
            raw_ap = self.anfis.layer['fuzzify'].get_raw_parameters().reshape(-1, 2)
            ap = self.anfis.layer['fuzzify'].get_parameters().reshape(-1, 2)

            raw_mf_area = np.array([np.abs(vals[1]) * np.power(2 * np.pi, 0.5) for vals in raw_ap])
            mf_area = np.array([np.abs(vals[1]) * np.power(2 * np.pi, 0.5) for vals in ap])
            area_dissimilarity = np.vstack([raw_mf_area, mf_area])
            area_dissimilarity.sort(axis=0)
            min_rho = np.mean(area_dissimilarity[0] / area_dissimilarity[1])
            mfas = min_rho

            return mfas

    def plot_mf(self):
        raw_ap = self.anfis.layer['fuzzify'].get_raw_parameters().reshape(-1, 2)
        ap = self.anfis.layer['fuzzify'].get_parameters().reshape(-1, 2)

        x = np.linspace(-2, 2, 1000)
        plot_data = np.linspace(-2, 2, 1000).reshape(1000, -1)
        plt.figure(1)
        for i in range(5):
            plt.subplot(510 + i + 1)
            for j in range(2):
                yvals = np.exp(-np.square(x - raw_ap[2 * i + j, 0]) / (2 * raw_ap[2 * i + j, 1] ** 2))
                plot_data = np.hstack((plot_data, yvals.reshape(1000, -1)))
                plt.plot(x, yvals, label=f"mf{j}")
            plt.xlabel('Values for variable x{} ({} MFs)'.format(i, 2))
            plt.ylabel('Membership')
            plt.legend(bbox_to_anchor=(1., 0.95))
        plt.show()
        plt.figure(2)
        for i in range(5):
            plt.subplot(510 + i + 1)
            for j in range(2):
                yvals = np.exp(-np.square(x - ap[2 * i + j, 0]) / (2 * ap[2 * i + j, 1] ** 2))
                plot_data = np.hstack((plot_data, yvals.reshape(1000, -1)))
                plt.plot(x, yvals, label=f"mf{j}")
            plt.xlabel('Values for variable x{} ({} MFs)'.format(i, 2))
            plt.ylabel('Membership')
            plt.legend(bbox_to_anchor=(1., 0.95))
        plt.show()


def _partition_estimators(n_estimators, n_jobs):
    """Private function used to partition estimators between jobs."""
    # Compute the number of jobs
    n_jobs = min(effective_n_jobs(n_jobs), n_estimators)

    # Partition estimators between jobs
    n_estimators_per_job = np.full(n_jobs, n_estimators // n_jobs,
                                   dtype=np.int)
    n_estimators_per_job[:n_estimators % n_jobs] += 1
    starts = np.cumsum(n_estimators_per_job)

    return n_jobs, n_estimators_per_job.tolist(), [0] + starts.tolist()


def _parallel_build_estimators(n_estimators, ensemble, X, y, sample_weight,
                               seeds, total_n_estimators, verbose):
    """Private function used to build a batch of estimators within a job."""
    # Retrieve settings
    n_samples, n_features = X.shape
    max_features = ensemble._max_features
    max_samples = ensemble._max_samples
    bootstrap = ensemble.bootstrap
    bootstrap_features = ensemble.bootstrap_features
    support_sample_weight = has_fit_parameter(ensemble.base_estimator_,
                                              "sample_weight")
    if not support_sample_weight and sample_weight is not None:
        raise ValueError("The base estimator doesn't support sample weight")

    # Build estimators
    estimators = []
    estimators_features = []

    for i in range(n_estimators):
        if verbose > 1:
            print("Building estimator %d of %d for this parallel run "
                  "(total %d)..." % (i + 1, n_estimators, total_n_estimators))

        random_state = np.random.RandomState(seeds[i])

        # Draw random feature, sample indices
        features, indices = _generate_bagging_indices(random_state,
                                                      bootstrap_features,
                                                      bootstrap, n_features,
                                                      n_samples, max_features,
                                                      max_samples)

        # Draw samples, using sample weights, and then fit
        if support_sample_weight:
            if sample_weight is None:
                curr_sample_weight = np.ones((n_samples,))
            else:
                curr_sample_weight = sample_weight.copy()

            if bootstrap:
                sample_counts = np.bincount(indices, minlength=n_samples)
                curr_sample_weight *= sample_counts
            else:
                not_indices_mask = ~indices_to_mask(indices, n_samples)
                curr_sample_weight[not_indices_mask] = 0

            estimator.fit(X[:, features], y, sample_weight=curr_sample_weight)

        else:
            estimator = ensemble._make_estimator((X[indices])[:, features], append=False,
                                                 random_state=random_state)
            print(features)
            estimator.fit((X[indices])[:, features], y[indices])

        estimators.append(estimator)
        estimators_features.append(features)

    return estimators, estimators_features


class BaseEnsemble(metaclass=ABCMeta):
    # overwrite _required_parameters from MetaEstimatorMixin
    _required_parameters = []

    @abstractmethod
    def __init__(self, base_estimator, n_estimators=10,
                 estimator_params=tuple()):
        # Set parameters
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.estimator_params = estimator_params

        # Don't instantiate estimators now! Parameters of base_estimator might
        # still change. Eg., when grid-searching with the nested object syntax.
        # self.estimators_ needs to be filled by the derived classes in fit.

    def _validate_estimator(self, default=None):
        """Check the estimator and the n_estimator attribute, set the
        `base_estimator_` attribute."""
        if not isinstance(self.n_estimators, (numbers.Integral, np.integer)):
            raise ValueError("n_estimators must be an integer, "
                             "got {0}.".format(type(self.n_estimators)))

        if self.n_estimators <= 0:
            raise ValueError("n_estimators must be greater than zero, "
                             "got {0}.".format(self.n_estimators))

        if self.base_estimator is not None:
            self.base_estimator_ = self.base_estimator
        else:
            self.base_estimator_ = default

        if self.base_estimator_ is None:
            raise ValueError("base_estimator cannot be None")

    def _make_estimator(self, X, append=True, random_state=None):
        """Make and configure a copy of the `base_estimator_` attribute.

        Warning: This method should be used to properly instantiate new
        sub-estimators.
        """
        num_mfs = self.base_estimator.num_mfs
        num_out = self.base_estimator.num_out
        mf_type = self.base_estimator.mf_type
        dropout = self.base_estimator.dropout
        drop_proba = self.base_estimator.drop_proba
        labels = self.base_estimator.labels_
        init_method = self.base_estimator.init_method
        estimator = SkAnfis(num_mfs=num_mfs, num_out=num_out, mf_type=mf_type, dropout=dropout, proba=drop_proba,
                            labels=labels, init_method=init_method)

        if append:
            self.estimators_.append(estimator)
        return estimator

    def __len__(self):
        """Returns the number of estimators in the ensemble."""
        return len(self.estimators_)

    def __getitem__(self, index):
        """Returns the index'th estimator in the ensemble."""
        return self.estimators_[index]

    def __iter__(self):
        """Returns iterator over estimators in the ensemble."""
        return iter(self.estimators_)


class BaseBagging(BaseEnsemble, metaclass=ABCMeta):
    """Base class for Bagging meta-estimator.

    Warning: This class should not be used directly. Use derived classes
    instead.
    """

    @abstractmethod
    def __init__(self,
                 base_estimator=None,
                 n_estimators=10,
                 max_samples=1.0,
                 max_features=1.0,
                 bootstrap=True,
                 bootstrap_features=False,
                 oob_score=False,
                 warm_start=False,
                 n_jobs=None,
                 random_state=None,
                 verbose=0):
        super().__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators)

        self.max_samples = max_samples
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.bootstrap_features = bootstrap_features
        self.oob_score = oob_score
        self.warm_start = warm_start
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y, sample_weight=None):
        """Build a Bagging ensemble of estimators from the training
           set (X, y).

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrices are accepted only if
            they are supported by the base estimator.

        y : array-like, shape = [n_samples]
            The target values (class labels in classification, real numbers in
            regression).

        sample_weight : array-like, shape = [n_samples] or None
            Sample weights. If None, then samples are equally weighted.
            Note that this is supported only if the base estimator supports
            sample weighting.

        Returns
        -------
        self : object
        """
        return self._fit(X, y, self.max_samples, sample_weight=sample_weight)

    def _parallel_args(self):
        return {}

    def _fit(self, X, y, max_samples=None, max_depth=None, sample_weight=None):
        """Build a Bagging ensemble of estimators from the training
           set (X, y).

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrices are accepted only if
            they are supported by the base estimator.

        y : array-like, shape = [n_samples]
            The target values (class labels in classification, real numbers in
            regression).

        max_samples : int or float, optional (default=None)
            Argument to use instead of self.max_samples.

        max_depth : int, optional (default=None)
            Override value used when constructing base estimator. Only
            supported if the base estimator has a max_depth parameter.

        sample_weight : array-like, shape = [n_samples] or None
            Sample weights. If None, then samples are equally weighted.
            Note that this is supported only if the base estimator supports
            sample weighting.

        Returns
        -------
        self : object
        """
        random_state = check_random_state(self.random_state)

        # Convert data (X is required to be 2d and indexable)

        if sample_weight is not None:
            sample_weight = check_array(sample_weight, ensure_2d=False)
            check_consistent_length(y, sample_weight)

        # Remap output
        n_samples, self.n_features_ = X.shape
        self._n_samples = n_samples
        y = self._validate_y(y)

        # Check parameters
        self._validate_estimator()

        if max_depth is not None:
            self.base_estimator_.max_depth = max_depth

        # Validate max_samples
        if max_samples is None:
            max_samples = self.max_samples
        elif not isinstance(max_samples, (numbers.Integral, np.integer)):
            max_samples = int(max_samples * X.shape[0])

        if not (0 < max_samples <= X.shape[0]):
            raise ValueError("max_samples must be in (0, n_samples]")

        # Store validated integer row sampling value
        self._max_samples = max_samples

        # Validate max_features
        if isinstance(self.max_features, (numbers.Integral, np.integer)):
            max_features = self.max_features
        elif isinstance(self.max_features, np.float):
            max_features = self.max_features * self.n_features_
        else:
            raise ValueError("max_features must be int or float")

        if not (0 < max_features <= self.n_features_):
            raise ValueError("max_features must be in (0, n_features]")

        max_features = max(1, int(max_features))

        # Store validated integer feature sampling value
        self._max_features = max_features

        # Other checks
        if not self.bootstrap and self.oob_score:
            raise ValueError("Out of bag estimation only available"
                             " if bootstrap=True")

        if self.warm_start and self.oob_score:
            raise ValueError("Out of bag estimate only available"
                             " if warm_start=False")

        if hasattr(self, "oob_score_") and self.warm_start:
            del self.oob_score_

        if not self.warm_start or not hasattr(self, 'estimators_'):
            # Free allocated memory, if any
            self.estimators_ = []
            self.estimators_features_ = []

        n_more_estimators = self.n_estimators - len(self.estimators_)

        if n_more_estimators < 0:
            raise ValueError('n_estimators=%d must be larger or equal to '
                             'len(estimators_)=%d when warm_start==True'
                             % (self.n_estimators, len(self.estimators_)))

        elif n_more_estimators == 0:
            print("Warm-start fitting without increasing n_estimators does not "
                  "fit new trees.")
            return self

        # Parallel loop
        n_jobs, n_estimators, starts = _partition_estimators(n_more_estimators,
                                                             self.n_jobs)
        total_n_estimators = sum(n_estimators)

        # Advance random state to state after training
        # the first n_estimators
        if self.warm_start and len(self.estimators_) > 0:
            random_state.randint(MAX_INT, size=len(self.estimators_))

        seeds = random_state.randint(MAX_INT, size=n_more_estimators)
        self._seeds = seeds

        all_results = Parallel(n_jobs=n_jobs, verbose=self.verbose,
                               **self._parallel_args())(
            delayed(_parallel_build_estimators)(
                n_estimators[i],
                self,
                X,
                y,
                sample_weight,
                seeds[starts[i]:starts[i + 1]],
                total_n_estimators,
                verbose=self.verbose)
            for i in range(n_jobs))

        # Reduce
        self.estimators_ += list(itertools.chain.from_iterable(
            t[0] for t in all_results))
        self.estimators_features_ += list(itertools.chain.from_iterable(
            t[1] for t in all_results))

        if self.oob_score:
            self._set_oob_score(X, y)

        return self

    @abstractmethod
    def _set_oob_score(self, X, y):
        """Calculate out of bag predictions and score."""

    def _validate_y(self, y):
        if len(y.shape) == 1 or y.shape[1] == 1:
            return column_or_1d(y, warn=True)
        else:
            return y

    def _get_estimators_indices(self):
        # Get drawn indices along both sample and feature axes
        for seed in self._seeds:
            # Operations accessing random_state must be performed identically
            # to those in `_parallel_build_estimators()`
            random_state = np.random.RandomState(seed)
            feature_indices, sample_indices = _generate_bagging_indices(
                random_state, self.bootstrap_features, self.bootstrap,
                self.n_features_, self._n_samples, self._max_features,
                self._max_samples)

            yield feature_indices, sample_indices

    @property
    def estimators_samples_(self):
        """The subset of drawn samples for each base estimator.

        Returns a dynamically generated list of indices identifying
        the samples used for fitting each member of the ensemble, i.e.,
        the in-bag samples.

        Note: the list is re-created at each call to the property in order
        to reduce the object memory footprint by not storing the sampling
        data. Thus fetching the property may be slower than expected.
        """
        return [sample_indices
                for _, sample_indices in self._get_estimators_indices()]


class MyBaggingClassifier(BaseBagging, ClassifierMixin):
    """A Bagging classifier for Dropout ANFIS."""

    def __init__(self,
                 base_estimator=None,
                 n_estimators=10,
                 max_samples=1.0,
                 max_features=1.0,
                 bootstrap=True,
                 bootstrap_features=False,
                 oob_score=False,
                 warm_start=False,
                 n_jobs=None,
                 random_state=None,
                 verbose=0):

        super().__init__(
            base_estimator,
            n_estimators=n_estimators,
            max_samples=max_samples,
            max_features=max_features,
            bootstrap=bootstrap,
            bootstrap_features=bootstrap_features,
            oob_score=oob_score,
            warm_start=warm_start,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose)

    def _validate_estimator(self):
        """Check the estimator and set the base_estimator_ attribute."""
        super()._validate_estimator()

    def _set_oob_score(self, X, y):
        n_samples = y.shape[0]
        n_classes_ = self.n_classes_

        predictions = np.zeros((n_samples, n_classes_))

        for estimator, samples, features in zip(self.estimators_,
                                                self.estimators_samples_,
                                                self.estimators_features_):
            # Create mask for OOB samples
            mask = ~indices_to_mask(samples, n_samples)

            if hasattr(estimator, "predict_proba"):
                predictions[mask, :] += estimator.predict_proba(
                    (X[mask, :])[:, features])

            else:
                p = estimator.predict((X[mask, :])[:, features])
                j = 0

                for i in range(n_samples):
                    if mask[i]:
                        predictions[i, p[j]] += 1
                        j += 1

        if (predictions.sum(axis=1) == 0).any():
            print("Some inputs do not have OOB scores. "
                  "This probably means too few estimators were used "
                  "to compute any reliable oob estimates.")

        oob_decision_function = (predictions /
                                 predictions.sum(axis=1)[:, np.newaxis])
        oob_score = accuracy_score(y, np.argmax(predictions, axis=1))

        self.oob_decision_function_ = oob_decision_function
        self.oob_score_ = oob_score

    def _validate_y(self, y):
        y = column_or_1d(y, warn=True)
        check_classification_targets(y)
        self.classes_, y = np.unique(y, return_inverse=True)
        self.n_classes_ = len(self.classes_)

        return y

    def predict(self, X):
        """Predict class for X.

        The predicted class of an input sample is computed as the class with
        the highest mean predicted probability. If base estimators do not
        implement a ``predict_proba`` method, then it resorts to voting.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrices are accepted only if
            they are supported by the base estimator.

        Returns
        -------
        y : array of shape = [n_samples]
            The predicted classes.
        """

        predicted_probabilitiy = self.predict_proba(X)
        return self.classes_.take((np.argmax(predicted_probabilitiy, axis=1)),
                                  axis=0)

    def predict_proba(self, X):
        check_is_fitted(self, "classes_")
        # Parallel loop
        n_jobs, n_estimators, starts = _partition_estimators(self.n_estimators,
                                                             self.n_jobs)

        all_proba = Parallel(n_jobs=n_jobs, verbose=self.verbose,
                             **self._parallel_args())(
            delayed(_parallel_predict_proba)(
                self.estimators_[starts[i]:starts[i + 1]],
                self.estimators_features_[starts[i]:starts[i + 1]],
                X,
                self.n_classes_)
            for i in range(n_jobs))

        # Reduce
        proba = sum(all_proba) / self.n_estimators
        return proba

    def evaluate(self, x_test, y_test):

        test_acc = accuracy_score(y_test.squeeze(), self.predict(x_test))
        test_f1 = f1_score(y_test.squeeze(), self.predict(x_test), average='weighted')
        results = [test_acc, test_f1]
        print(results)
        return results

    def get_NOR(self):
        """
        :return: the number of rules in an ANFIS
        """
        return self.estimators_[0].get_NOR()

    def get_ARL(self):
        """
        :return: the length of rules in an ANFIS
        """
        return self.estimators_[0].get_ARL()

    def get_avg_firing_strength(self, X):
        avg_fs = 0
        for i in range(self.n_estimators):
            avg_fs += self.estimators_[i].get_avg_firing_strength(X[:, self.estimators_features_[0]])
        avg_fs /= self.n_estimators
        inputs_fs = 0
        for i in range(self.n_estimators):
            inputs_fs += self.estimators_[i].get_firing_strength_of_given_inputs(X[:8, self.estimators_features_[0]])
        inputs_fs /= self.n_estimators

        return avg_fs

    def get_co_firing_strength(self, X):
        # print(self.estimators_[0].anfis.layer['rules'].extra_repr())
        avg_fs = 0
        for i in range(self.n_estimators):
            avg_fs += self.estimators_[i].get_avg_firing_strength(X[:, self.estimators_features_[0]])
        avg_fs /= self.n_estimators
        rule_index = torch.argsort(avg_fs, descending=True)
        rule_index = rule_index[:16]
        rule_index = torch.sort(rule_index).values
        tmp = self.estimators_[0].get_co_firing_strength(X[:, self.estimators_features_[0]])
        tmp = tmp[:, rule_index]
        tmp = F.normalize(tmp, p=1, dim=1)

        res = np.zeros(shape=tmp.shape)

        res[tmp > (1 / 16)] = 1

        cf_matrix = np.zeros(shape=(16, 16))
        for i in range(16):
            for j in range(16):
                ri = 0
                rj = 0
                sfr = 0
                for k in range(569):
                    if res[k, i] == 1:
                        ri += 1
                    if res[k, j] == 1:
                        rj += 1
                    if (res[k, i] + res[k, j]) == 2:
                        sfr += 1
                cf_matrix[i, j] = 0 if ri * rj == 0 else sfr / (ri * rj) ** 0.5
                if i == j: cf_matrix[i, j] = 1
        cf_matrix = np.around(cf_matrix, 2)

    def get_CSR(self, X):
        CSR = 0.0
        for estimator, features in zip(self.estimators_, self.estimators_features_):
            CSR += estimator.get_CSR(X[:, features])
        return CSR / self.n_estimators

    def get_DR(self):
        DR = 0.0
        for estimator in self.estimators_:
            DR += estimator.get_DR()
        return DR / self.n_estimators

    def get_MFD(self):
        MFD = 0.0
        for estimator in self.estimators_:
            MFD += estimator.get_MFD()
        return MFD / self.n_estimators

    def get_MFAS(self):
        MFAS = 0.0
        for estimator in self.estimators_:
            MFAS += estimator.get_MFAS()
        return MFAS / self.n_estimators

    def get_base_results(self, x_test, y_test):
        base_results = []
        base_features = []
        for estimator, features in zip(self.estimators_, self.estimators_features_):
            test_acc = accuracy_score(y_test.squeeze(), estimator.predict(x_test[:, features]))
            test_f1 = f1_score(y_test.squeeze(), estimator.predict(x_test[:, features]), average='weighted')
            base_results.append([test_acc, test_f1])
            base_features.append(features)
        return base_results, base_features

    def drop_rule(self, x, threshold):
        for estimator, features in zip(self.estimators_, self.estimators_features_):
            estimator.drop_rule(x[:, features], threshold)

    def plot_mf(self):
        print(f"features : {self.estimators_features_[0]}")
        self.estimators_[0].plot_mf()


def train_anfis_with(model, data, optimizer, criterion,
                     epochs=500, show_plots=False):
    '''
        Train the given model using the given (x,y) data.
    '''
    errors = []  # Keep a list of these for plotting afterwards
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    print('### Training for {} epochs, training size = {} cases'.
          format(epochs, data.dataset.tensors[0].shape[0]))
    for t in range(epochs):
        # Process each mini-batch in turn:
        loss = 1000.
        for x, y_actual in data:
            model.train()
            y_pred = model(x)
            # Compute and print loss
            loss = criterion(y_pred, y_actual)
            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5, norm_type=2)
            optimizer.step()
            if epochs < 30 or t % 10 == 0:
                print('epoch {:4d}: loss={:.5f}'.format(t, loss))
        if loss < 0.05:
            break
        if t >= 80:
            if loss < 0.3:
                break
        # Epoch ending, so now fit the coefficients based on all data:
        x, y_actual = data.dataset.tensors
        with torch.no_grad():
            model.fit_coeff(x, y_actual)