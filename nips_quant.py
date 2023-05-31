#!/usr/bin/python
# -*- coding: utf-8 -*-
###########################################################################
# Pytorch - My manual Learning Algorithm based on Pytorch
###########################################################################
_description = '''\
====================================================
nips_quant.py : Based on torch module
                    Written by ****** @ 2023-04-24
====================================================
Example : python nips_quant.py
'''
#-------------------------------------------------------------
# Description of Optimizer
#-------------------------------------------------------------
import torch
import yaml
import my_debug as DBG
#-------------------------------------------------------------
# Quantization based on pytorch
# Q_p(\tau) = \eta b^{p(\tau)}
# \eta : self.q_eta
# b    : self.q_base
# p    : def p_function(self, _tau):
# \tau : function input
#-------------------------------------------------------------
class Quantization:
    def __init__(self, _config_data, total_epoch=200, batch_size=600):
        # Quantization
        self.q_base = torch.tensor(_config_data['base'])
        self.q_eta  = torch.tensor(_config_data['eta'])
        self.q_param= 0
        self.q_C    = torch.div(1.0, self.q_eta)
        #r_function
        self._kappa = torch.tensor(_config_data['kappa'])
        warmp_up_period = torch.tensor(_config_data['warmp_up_period'])
        self._tau_0 = torch.mul(total_epoch, warmp_up_period)
        #Set_processing parameters
        # batch_size :  Here, It means that Total Number of batch per dataset,
        #               e.g. Fashion MNIST involves 600 batches per 1 minibatch.
        #               minibatch contains 100 samples. That is a general meaning of minibatch size
        self.total_epoch    = total_epoch
        self.batch_size     = batch_size
        self.epoch_cnt      = 0                                 # No Need
        self._supQP         = torch.tensor(0)
        # Debug Parameter
        self._debug         = _config_data['debug']
        self._test          = _config_data['test']
        self.h_data         = torch.tensor(0)
        self.r_data         = torch.tensor(0)
        self._cost          = torch.tensor(0)
        self._avg_cost      = torch.tensor(0)

        # Set Initial QP
        self.eval_QP(torch.tensor(0))
    # -------------------------------------------------------------
    # Service Function
    # -------------------------------------------------------------
    def set_cost(self, _cost, _avg_cost):
        self._cost      = _cost
        self._avg_cost  = _avg_cost

    def chk_tensor(self, _x):
        _res = _x if torch.is_tensor(_x) else torch.tensor(_x)
        return _res
    def get_QP(self):
        return self.q_param
    def get_invQP(self):
        return 1.0/self.q_param

    def get_log_b(self, _x):
        _res = torch.log(_x)/torch.log(self.q_base)
        return _res

    # -------------------------------------------------------------
    # Main Processing Function
    # -------------------------------------------------------------
    # Input : Epoch time index
    def sup_QP(self, _te):
        self._supQP = torch.mul(torch.div(1.0, self.q_C), torch.log(torch.add(_te, 2.0)))
        return self._supQP

    # Time Scaling : minibatch -> Epoch
    def eval_batchtime_to_epochtime(self, _tau):
        #_te    = _tau/self.batch_size
        _te     = torch.div(_tau, self.batch_size)
        return _te

    # Input : minibatch time index, Output : Power for Qp and Epoch time index
    def p_function(self, _tau):
        _te = self.eval_batchtime_to_epochtime(_tau)
        _pcore = torch.log(torch.add(_te, 2.0))
        # evaluation of supQP = 1/C * log(te + 2)
        self._supQP = torch.mul(torch.div(1.0, self.q_C), _pcore)
        # evaluation p = floor(1/eta * supQP)
        _p = self.get_log_b(_pcore)
        _res = torch.floor(_p)

        return _res, _te

    # Input : minibatch time index
    def eval_QP(self, _tau):
        _p , _te    = self.p_function(_tau=_tau)
        self.q_param= torch.mul(self.q_eta, torch.pow(self.q_base, _p))
        return self.q_param, _te

    # Time unit for r_function is Epoch unit
    def r_function(self, _h, _lr, _te):
        # Normalized search vector
        _v      = torch.div(_h, torch.norm(_h))
        # -1.0 * self._kappa * (_te - self._tau_0)
        _powv   = torch.mul(-1.0, torch.mul(self._kappa, torch.sub(_te, self._tau_0)))
        # tanh(x)+1 = 2 exp(x)/(1+exp(x))
        if self._test:
            _expv = torch.exp(_powv)
            _res1 = torch.div(_expv, torch.add(1.0, _expv))
        else:
            _tanh   = torch.add(torch.tanh(_powv), 1.0)
            _res1   = _tanh if self._test else torch.mul(0.5, _tanh)
        # lr * exp(x)/(1+exp(x)) * v
        _res2   = torch.mul(_lr, _v)
        _res    = torch.mul(_res1, _res2)
        return _res
    # _h = \lambda * h(x_t), _lr : Learning_rate, _tau : minibatch unit
    def get_Quantization(self, _h, _lr, _tau):
        if self._debug:
            # Debug --------------------------------------------------------------
            h_data  = torch.norm(_h)
            # Debug --------------------------------------------------------------
            # Processing
            Q_p, _te= self.eval_QP(_tau)
            r_func  = self.r_function(_h, _lr, _te)
            # Debug --------------------------------------------------------------
            r_data  = torch.norm(r_func)
            # Debug --------------------------------------------------------------
            _hq_cr  = torch.add(_h, r_func)
            _hq_mQp = torch.add(torch.mul(Q_p, _hq_cr), 0.5)
            _hq     = torch.div(torch.floor(_hq_mQp), Q_p)
            _res    = _hq
            # Debug --------------------------------------------------------------
            hq_data = torch.norm(_hq)
            print("Batch Index (tau) %d / %d te : %2.3f |h|: %2.3f  |r|: %2.3f |hq|: %2.3f cost: %2.3f Qp: %f supQp: %f"
                  %(_tau, self.batch_size, _te, h_data, r_data, hq_data, self._cost, Q_p, self.sup_QP(_te)))
            # Debug --------------------------------------------------------------
        else:
            Q_p, _te= self.eval_QP(_tau)
            r_func  = self.r_function(_h, _lr, _te)
            _hq_cr  = torch.add(_h, r_func)
            _hq_mQp = torch.mul(Q_p, _hq_cr)
            _hq     = torch.div(torch.floor(_hq_mQp), Q_p)
            _res    = _hq
        return _res, _h
#-------------------------------------------------------------
# Quantization
#-------------------------------------------------------------
class Q_process:
    def __init__(self, _config_data, total_epoch=200, batch_size=100):
        #_config_data = self.read_config_file()
        self.config_data    = _config_data['Q_process']
        self.quantization_config= _config_data['Quantization']
        self.bQuantization  = self.config_data['bQuantization']
        #self.index_limit    = self.config_data['index_limit']   # No Need
        #self.QuantMethod    = self.config_data['QuantMethod']   # 1 or 2, True ....or delete, For debug No Need
        self.c_qtz          = Quantization(self.quantization_config,
                                           total_epoch      =total_epoch,
                                           batch_size       =batch_size)
        #self.c_tmp          = Temperature(self.temperature_config)  No Need
        #self.nBatchpEpoch   = batch_size    # No need
        self.l_index_trend  = []
        self.l_supindextrend= []
        # Store Data to List
        self.l_index_trend.append(self.c_qtz.get_QP())

    def Quantization(self, x, _lr, _t):
        Xq, x = self.c_qtz.get_Quantization(x, _lr, _t) if self.bQuantization else x
        return Xq, x

    def Get_QPIndex(self):
        _QP     = self.c_qtz.get_QP()
        _QP_idx = torch.log2(_QP)
        self.l_index_trend.append(_QP_idx)
        _res = self.l_index_trend[-1]
        return _res

    ## For debug and Development
    def Get_lengthofIndexTrend(self):
        _res = len(self.l_index_trend)
        return _res

    def Get_supQP(self, b_log=True):
        supQP = self.c_qtz._supQP
        self.l_supindextrend.append(supQP)
        _res = torch.log_(self.l_supindextrend[-1]) if b_log else self.l_supindextrend[-1]
        return _res

    # Debug --------------------------------------------------------------
    def set_cost(self, _cost, _avg_cost):
        self.c_qtz.set_cost(_cost=_cost, _avg_cost=_avg_cost)
    # Debug --------------------------------------------------------------