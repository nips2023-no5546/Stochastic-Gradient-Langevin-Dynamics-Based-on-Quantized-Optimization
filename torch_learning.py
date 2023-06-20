#!/usr/bin/python
# -*- coding: utf-8 -*-
###########################################################################
# Pytorch - My manual Learning Algorithm based on Pytorch
#
###########################################################################
_description = '''\
====================================================
torch_nn02.py : Based on torch module
                    Written by ******** @ 2021-03-10
====================================================
Example : python torch_nn02.py
'''
#-------------------------------------------------------------
# Description of Optimizer
#-------------------------------------------------------------
import math
import torch
from typing import Callable, Iterable, Optional, Tuple, Union
from torch.optim import Optimizer

from torch_customcosineannealingwarmrestarts import CustomCosineAnnealingWarmUpRestarts
from nips_quant import Q_process
import yaml
import my_debug as DBG

#-------------------------------------------------------------
# Learning Algorithm based on Pytorch 1. AdamW
#-------------------------------------------------------------
class AdamW(Optimizer):
    """
    Parameters:
        params (:obj:`Iterable[torch.nn.parameter.Parameter]`):
            Iterable of parameters to optimize or dictionaries defining parameter groups.
        lr (:obj:`float`, `optional`, defaults to 1e-3):
            The learning rate to use.
        betas (:obj:`Tuple[float,float]`, `optional`, defaults to (0.9, 0.999)):
            Adam's betas parameters (b1, b2).
        eps (:obj:`float`, `optional`, defaults to 1e-6):
            Adam's epsilon for numerical stability.
        weight_decay (:obj:`float`, `optional`, defaults to 0):
            Decoupled weight decay to apply.
        correct_bias (:obj:`bool`, `optional`, defaults to `True`):
            Whether or not to correct bias in Adam (for instance, in Bert TF repository they use :obj:`False`).
    """
    def __init__(
        self,
        params: Iterable[torch.nn.parameter.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.01,
        correct_bias: bool = True,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr} - should be >= 0.0")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[0]} - should be in [0.0, 1.0[")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[1]} - should be in [0.0, 1.0[")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps} - should be >= 0.0")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
        super().__init__(params, defaults)

    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values : m_t
                    state["exp_avg"] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values : v_t
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)               # m_t = beta1*m_t + (1 - beta1)*grad
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)  # v_t = beta2*v_t + (1 - beta2)*v_t^2
                denom = exp_avg_sq.sqrt().add_(group["eps"])                    # sqrt(v_t^2 + epsilon)

                step_size = group["lr"]
                if group["correct_bias"]:  # No bias correction for Bert
                    bias_correction1 = 1.0 - beta1 ** state["step"]             # b1 = 1 - beta1^t
                    bias_correction2 = 1.0 - beta2 ** state["step"]             # b2 = 1 - beta2^t
                    step_size = step_size * math.sqrt(bias_correction2) / bias_correction1  # lr * sqrt(b2)/b1

                # Original
                #p.data.addcdiv_(exp_avg, denom, value=-step_size)
                # Alternative
                hc      = torch.div(exp_avg, denom)     #   hc = m_t/sqrt(v_t^2 + epsilon)
                hf      = torch.mul(hc, -step_size)     #   hf = -lr * sqrt(1 - beta2^t)/(1 - beta1^t) * m_t/sqrt(v_t^2 + epsilon)
                p.data.add_(hf)                         #   w <- w + hf

                if group["weight_decay"] > 0.0:         # Active AdamW weight_decay is 0.01 (Default)
                    p.data.add_(p.data, alpha=-group["lr"] * group["weight_decay"])

        return loss

#-------------------------------------------------------------
# Learning Algorithm based on Pytorch 2. QtAdamW
#-------------------------------------------------------------
class QtAdamW(Optimizer):
    def __init__(
        self,
        params: Iterable[torch.nn.parameter.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.01,
        correct_bias: bool = True,
        total_epoch=200,
        batch_size=600,
        Qparam=0
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr} - should be >= 0.0")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[0]} - should be in [0.0, 1.0[")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[1]} - should be in [0.0, 1.0[")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps} - should be >= 0.0")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)

        # Quantization
        self.Q_proc = Q_process(_config_data=Qparam, total_epoch=total_epoch, batch_size=batch_size)
        self.Q_proc.l_index_trend.append(Qparam)
        self.Q_proc.l_supindextrend.append(Qparam)

        super().__init__(params, defaults)

    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)               # m_t = beta1*m_t + (1 - beta1)*grad
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)  # v_t = beta2*v_t + (1 - beta2)*v_t^2
                denom = exp_avg_sq.sqrt().add_(group["eps"])                    # sqrt(v_t^2 + epsilon)

                step_size = group["lr"]
                if group["correct_bias"]:  # No bias correction for Bert
                    bias_correction1 = 1.0 - beta1 ** state["step"]             # b1 = 1 - beta1^t
                    bias_correction2 = 1.0 - beta2 ** state["step"]             # b2 = 1 - beta2^t
                    step_size = step_size * math.sqrt(bias_correction2) / bias_correction1  # lr * sqrt(b2)/b1

                #p.data.addcdiv_(exp_avg, denom, value=-step_size)
                hc      = torch.div(exp_avg, denom)                             # hc = m_t/sqrt(v_t^2 + epsilon)
                h       = torch.mul(hc, -step_size)                             # hf = -lr * sqrt(1 - beta2^t)/(1 - beta1^t) * m_t/sqrt(v_t^2 + epsilon)
                # Quantization
                hq,_ = self.Q_proc.Quantization(h, step_size, state["step"])    # hq = Quantization(hf)
                # Weight update
                p.data.add_(hq)                                                 #  w <- w + hf

                if group["weight_decay"] > 0.0: # Active AdamW weight_decay is 0.01 (Default)
                    p.data.add_(p.data, alpha=-group["lr"] * group["weight_decay"])

        return loss

#-------------------------------------------------------------
# Learning Algorithm based on Pytorch 2. QtAdam
#-------------------------------------------------------------
class QtAdam(Optimizer):
    def __init__(
        self,
        params: Iterable[torch.nn.parameter.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        correct_bias: bool = True,
        total_epoch=200,
        batch_size=600,
        Qparam=0
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr} - should be >= 0.0")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[0]} - should be in [0.0, 1.0[")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[1]} - should be in [0.0, 1.0[")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps} - should be >= 0.0")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)

        # Quantization
        self.Q_proc = Q_process(_config_data=Qparam, total_epoch=total_epoch, batch_size=batch_size)
        self.Q_proc.l_index_trend.append(Qparam)
        self.Q_proc.l_supindextrend.append(Qparam)

        super().__init__(params, defaults)

    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p.data)     # m_t : Exponential moving average of gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p.data)  # v_t : Exponential moving average of squared gradient values

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)               # m_t = beta1*m_t + (1 - beta1)*grad
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)  # v_t = beta2*v_t + (1 - beta2)*v_t^2
                denom = exp_avg_sq.sqrt().add_(group["eps"])                    # sqrt(v_t^2 + epsilon)

                step_size = group["lr"]
                if group["correct_bias"]:  # No bias correction for Bert
                    bias_correction1 = 1.0 - beta1 ** state["step"]             # b1 = 1 - beta1^t
                    bias_correction2 = 1.0 - beta2 ** state["step"]             # b2 = 1 - beta2^t
                    step_size = step_size * math.sqrt(bias_correction2) / bias_correction1  # lr * sqrt(b2)/b1

                #p.data.addcdiv_(exp_avg, denom, value=-step_size)
                hc      = torch.div(exp_avg, denom)                             # hc = m_t/sqrt(v_t^2 + epsilon)
                h       = torch.mul(hc, -step_size)                             # hf = -lr * sqrt(1 - beta2^t)/(1 - beta1^t) * m_t/sqrt(v_t^2 + epsilon)
                # Quantization
                hq,_ = self.Q_proc.Quantization(h, step_size, state["step"])    # hq = Quantization(hf)
                # Weight update
                p.data.add_(hq)                                                 # w <- w + hf

                if group["weight_decay"] > 0.0: # No-active (Default)
                    p.data.add_(p.data, alpha=-group["lr"] * group["weight_decay"])

        return loss

#-------------------------------------------------------------
# Learning Algorithm based on Pytorch 2. QtSGD
#-------------------------------------------------------------
class QtSGD(Optimizer):
    def __init__(
            self,
            params: Iterable[torch.nn.parameter.Parameter],
            lr: float   = 1e-3,
            momentum    = 0, dampening   = 0, weight_decay= 0, Qparam=0,
            total_epoch = 200, batch_size=100, nestrov    = False
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening, weight_decay=weight_decay, Qparam=Qparam, nesterov=nestrov)
        if nestrov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nestrov momentum requires a momentum and zero dampening")

        # Quantization
        self.Q_proc = Q_process(_config_data=Qparam, total_epoch=total_epoch, batch_size=batch_size)
        self.Q_proc.l_index_trend.append(Qparam)
        self.Q_proc.l_supindextrend.append(Qparam)

        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            step_size = group['lr']
            for _k, p in enumerate(group["params"]):
                # get Grad and state
                if p.grad is None: continue
                grad = p.grad.data
                state = self.state[p]

                # State initialization and update
                state["step"] = 0 if len(state) == 0 else (state["step"] + 1)

                # Debug --------------------------------------------------------------
                #DBG.dbg("Start Debug") if state["step"] == 1 else pass
                # Debug --------------------------------------------------------------
                # get Directional Derivation
                h = -step_size * grad

                # Quantization
                hq,_ = self.Q_proc.Quantization(h, step_size, state["step"])

                # Update weight by Learning equation #w_t - step_size * g_t
                p.data.add_(hq, alpha=1.0)

        return loss

#-------------------------------------------------------------
# Learning Algorithm based on Pytorch 2. QtSGD
#-------------------------------------------------------------
class stdSGD(Optimizer):
    def __init__(
            self,
            params: Iterable[torch.nn.parameter.Parameter],
            lr: float   = 1e-3,
            momentum    = 0, dampening   = 0, weight_decay= 0, Qparam=0, nestrov    = False
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening, weight_decay=weight_decay, Qparam=Qparam, nesterov=nestrov)
        if nestrov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nestrov momentum requires a momentum and zero dampening")

        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            step_size = group['lr']
            for p in group["params"]:

                # get Grad and state
                if p.grad is None: continue
                grad = p.grad.data
                state = self.state[p]

                # State initialization and update
                state["step"] = 0 if len(state) == 0 else (state["step"] + 1)

                # Set Learning equation #w_t - step_size * g_t
                p.data.add_(grad, alpha=-step_size)

        return loss



#-------------------------------------------------------------
# Learning Module (top)
#-------------------------------------------------------------
class learning_module:
    def __init__(self, model, args, total_batch):
        self.model      = model
        self.args       = args
        self.epoch      = 0
        self.total_batch= total_batch
        self.config_data    = args.LrnParam
        #self.config_data    = self.read_config_file()
        # Scheduler Parameter
        self.scheduler_param= self.config_data['scheduler_param']
        self.CAWR_param     = self.scheduler_param['CosineAnnealingWarmRestarts']
        self.CCAWR_param    = self.scheduler_param['CustomCosineAnnealingWarmRestarts']
        # Note. self.args.learning_rate = 0.01 for All Learning rate except constantLR (0.001)
        # 0. Fundamental Scaling
        _num_of_upwards = self.scheduler_param['num_of_upwards']
        _fund_scaling   = self.args.training_epochs * 1.0/_num_of_upwards
        # 1. CosineAnnealingWarmRestarts
        self._T_0       = self.CAWR_param['T_0']
        self._T_mult    = self.CAWR_param['T_mult']
        self._eta_min   = self.args.learning_rate * self.CAWR_param['eta_min_param']
        # 2. CustomCosineAnnealingWarmRestarts
        self._eta_max   = self.args.learning_rate
        self._T_up      = self.CCAWR_param['T_up']
        self._gamma     = self.CCAWR_param['gamma']
        self.args.learning_rate = 0 if self.args.scheduler_name == 'CCAWR' or self.args.scheduler_name == 'CustomCosineAnnealingWarmRestarts' else self.args.learning_rate
        # 3. CyclicLR
        self.step_size_up = int(_fund_scaling)

        #4 Miscellious parameters for Qunatization
        #self.total_epoch = self.args.training_epochs
        #self.batch_size  = self.args.batch_size

        #5. Optimizer and Scheduler
        self.optimizer  = self.set_optimizer()
        self.scheduler  = self.set_scheduler()

        print("-----------------------------------------------------------------")
        print("model \n ", self.model)
        print("-----------------------------------------------------------------")

    def set_optimizer(self):
        _parameters     = self.model.parameters()
        _learning_rate  = self.args.learning_rate
        _total_epoch    = self.args.training_epochs
        _batch_size     = self.total_batch

        if self.args.model_name == 'stdSGD' or self.args.model_name == 'sgd':
            _optimizer = stdSGD(_parameters, lr=_learning_rate)
        elif self.args.model_name == 'Adam' or self.args.model_name == 'adam':
            _optimizer = torch.optim.Adam(_parameters, lr=_learning_rate)
        elif self.args.model_name == 'AdamW' or self.args.model_name == 'adamw':
            _optimizer = AdamW(_parameters, lr=_learning_rate)
        elif self.args.model_name == 'ASGD' or self.args.model_name == 'asgd':
            _optimizer = torch.optim.ASGD(_parameters, lr=_learning_rate)
        elif self.args.model_name == 'NAdam' or self.args.model_name == 'nadam':
            _optimizer = torch.optim.NAdam(_parameters, lr=_learning_rate)
        elif self.args.model_name == 'RAdam' or self.args.model_name == 'radam':
            _optimizer = torch.optim.RAdam(_parameters, lr=_learning_rate)
        elif self.args.model_name == 'QSGD' or self.args.model_name == 'qsgd':
            _optimizer = QtSGD(_parameters, lr=_learning_rate, total_epoch=_total_epoch, batch_size=_batch_size, Qparam=self.args.QParam)
        elif self.args.model_name == 'QtAdamW' or self.args.model_name == 'qtadamw':
            _optimizer = QtAdamW(_parameters, lr=_learning_rate, total_epoch=_total_epoch, batch_size=_batch_size, Qparam=self.args.QParam)
        elif self.args.model_name == 'QtAdam' or self.args.model_name == 'qtadam':
            _optimizer = QtAdam(_parameters, lr=_learning_rate, total_epoch=_total_epoch, batch_size=_batch_size, Qparam=self.args.QParam)
        else:
            _optimizer = torch.optim.SGD(_parameters, lr=_learning_rate)
        return _optimizer

    def set_scheduler(self):
        _target_epoch   = self.args.training_epochs
        _epoch          = self.epoch
        if self.args.scheduler_name == 'LambdaLR':
            _scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=self.optimizer, lr_lambda=self.QuadStepfunc_for_LambdaLR)
        elif self.args.scheduler_name == 'exp':
            _scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=self._gamma)
        elif self.args.scheduler_name == 'cyclicLR':
            _scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer=self.optimizer,
                                                           base_lr=self._eta_min,
                                                           max_lr= self._eta_max,
                                                           step_size_up=self.step_size_up,
                                                           gamma=self._gamma,
                                                           cycle_momentum=False,
                                                           mode='exp_range')
        elif self.args.scheduler_name == 'CAWR' or self.args.scheduler_name =='CosineAnnealingWarmRestarts':
            _scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer =self.optimizer,
                                                                              T_0       =self._T_0,
                                                                              T_mult    =self._T_mult,
                                                                              eta_min   =self._eta_min)
        elif self.args.scheduler_name == 'CCAWR' or self.args.scheduler_name == 'CustomCosineAnnealingWarmRestarts':
            _scheduler = CustomCosineAnnealingWarmUpRestarts(optimizer  =self.optimizer,
                                                             T_0        =self._T_0,
                                                             T_mult     =self._T_mult,
                                                             eta_max    =self._eta_max,
                                                             T_up       =self._T_up,
                                                             gamma      =self._gamma/2.0)
        else:
            _scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer=self.optimizer, factor=1.0, total_iters=_target_epoch)

        return _scheduler

    def zero_grad(self):
        return self.optimizer.zero_grad()

    def learning(self, epoch):
        self.epoch = epoch
        self.optimizer.step()

    def get_optimizer_parameter(self, _param='lr'):
        return self.optimizer.param_groups[0][_param]

    # -------------------------------------------------------------
    # Service Functions
    # -------------------------------------------------------------
    def set_CosineAnnealingWarmRestarts_Params(self, _T_0, _T_mult, _eta_min):
        self._T_0       = _T_0
        self._T_mult    = _T_mult
        self._eta_min   = _eta_min

    def QuadStepfunc_for_LambdaLR(self, _epoch, _scale=0.618):
        _total_epoch    = self.args.training_epochs
        if _epoch < (_total_epoch * 0.25):
            _rtvalue    = 1.0
        elif _epoch < (_total_epoch * 0.5):
            _rtvalue    = _scale ** 2
        elif _epoch < (_total_epoch * 0.75):
            _rtvalue    = _scale ** 4
        else:
            _rtvalue = _scale ** 6

        return _rtvalue


    # Debug --------------------------------------------------------------
    def Set_cost_info(self, _cost, _avg_cost, b_active=False):
        if b_active:
            self.optimizer.Q_proc.set_cost(_cost=_cost, _avg_cost=_avg_cost)
        else: return
    # Debug --------------------------------------------------------------