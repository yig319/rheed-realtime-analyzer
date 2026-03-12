import torch.nn as nn
import torch
# from ...optimizers.AdaHessian import AdaHessian
"""
Created on Sun Feb 26 16:34:00 2021
@author: Amir Gholami
@coauthor: David Samuel
"""

import numpy as np
import torch

class AdaHessian(torch.optim.Optimizer):
    """
    Implements the AdaHessian algorithm from "ADAHESSIAN: An Adaptive Second OrderOptimizer for Machine Learning"
    Arguments:
        params (iterable) -- iterable of parameters to optimize or dicts defining parameter groups
        lr (float, optional) -- learning rate (default: 0.1)
        betas ((float, float), optional) -- coefficients used for computing running averages of gradient and the squared hessian trace (default: (0.9, 0.999))
        eps (float, optional) -- term added to the denominator to improve numerical stability (default: 1e-8)
        weight_decay (float, optional) -- weight decay (L2 penalty) (default: 0.0)
        hessian_power (float, optional) -- exponent of the hessian trace (default: 1.0)
        update_each (int, optional) -- compute the hessian trace approximation only after *this* number of steps (to save time) (default: 1)
        n_samples (int, optional) -- how many times to sample `z` for the approximation of the hessian trace (default: 1)
    """

    def __init__(self, params, lr=0.1, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, 
                 hessian_power=1.0, update_each=1, n_samples=1, average_conv_kernel=False):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= hessian_power <= 1.0:
            raise ValueError(f"Invalid Hessian power value: {hessian_power}")

        self.n_samples = n_samples
        self.update_each = update_each
        self.average_conv_kernel = average_conv_kernel

        # use a separate generator that deterministically generates the same `z`s across all GPUs in case of distributed training
        self.generator = torch.Generator().manual_seed(2147483647)

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, hessian_power=hessian_power)
        super(AdaHessian, self).__init__(params, defaults)

        for p in self.get_params():
            p.hess = 0.0
            self.state[p]["hessian step"] = 0

    def get_params(self):
        """
        Gets all parameters in all param_groups with gradients
        """

        return (p for group in self.param_groups for p in group['params'] if p.requires_grad)

    def zero_hessian(self):
        """
        Zeros out the accumalated hessian traces.
        """

        for p in self.get_params():
            if not isinstance(p.hess, float) and self.state[p]["hessian step"] % self.update_each == 0:
                p.hess.zero_()

    @torch.no_grad()
    def set_hessian(self):
        """
        Computes the Hutchinson approximation of the hessian trace and accumulates it for each trainable parameter.
        """

        params = []
        for p in filter(lambda p: p.grad is not None, self.get_params()):
            if self.state[p]["hessian step"] % self.update_each == 0:  # compute the trace only each `update_each` step
                params.append(p)
            self.state[p]["hessian step"] += 1

        if len(params) == 0:
            return

        if self.generator.device != params[0].device:  # hackish way of casting the generator to the right device
            self.generator = torch.Generator(params[0].device).manual_seed(2147483647)

        grads = [p.grad for p in params]

        for i in range(self.n_samples):
            zs = [torch.randint(0, 2, p.size(), generator=self.generator, device=p.device) * 2.0 - 1.0 for p in params]  # Rademacher distribution {-1.0, 1.0}
            h_zs = torch.autograd.grad(grads, params, grad_outputs=zs, only_inputs=True, retain_graph=i < self.n_samples - 1)
            for h_z, z, p in zip(h_zs, zs, params):
                p.hess += h_z * z / self.n_samples  # approximate the expected values of z*(H@z)

    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step.
        Arguments:
            closure (callable, optional) -- a closure that reevaluates the model and returns the loss (default: None)
        """

        loss = None
        if closure is not None:
            loss = closure()

        self.zero_hessian()
        self.set_hessian()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None or p.hess is None:
                    continue

                if self.average_conv_kernel and p.dim() == 4:
                    p.hess = torch.abs(p.hess).mean(dim=[2, 3], keepdim=True).expand_as(p.hess).clone()

                # Perform correct stepweight decay as in AdamW
                p.mul_(1 - group['lr'] * group['weight_decay'])

                state = self.state[p]

                # State initialization
                if len(state) == 1:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)  # Exponential moving average of gradient values
                    state['exp_hessian_diag_sq'] = torch.zeros_like(p.data)  # Exponential moving average of Hessian diagonal square values

                exp_avg, exp_hessian_diag_sq = state['exp_avg'], state['exp_hessian_diag_sq']
                beta1, beta2 = group['betas']
                state['step'] += 1

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(p.grad, alpha=1 - beta1)
                exp_hessian_diag_sq.mul_(beta2).addcmul_(p.hess, p.hess, value=1 - beta2)

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                k = group['hessian_power']
                denom = (exp_hessian_diag_sq / bias_correction2).pow_(k / 2).add_(group['eps'])

                # make update
                step_size = group['lr'] / bias_correction1
                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss



# from ...nn.random import random_seed


from m3_learning.nn.benchmarks.inference import computeTime
# from ...viz.layout import get_axis_range, set_axis, Axis_Ratio
from torch.utils.data import DataLoader
import time
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from scipy.signal import resample
from m3_learning.util.file_IO import make_folder, append_to_csv
import itertools
# from m3_learning.optimizers.TrustRegion import TRCG
import torch
from torch.optim.optimizer import Optimizer, _use_grad_for_differentiable
import numpy as np


# extended from Zheng Shi zhs310@lehigh.edu and Majid Jahani maj316@lehigh.edu
# https://github.com/Optimization-and-Machine-Learning-Lab/TRCG
# BSD 3-Clause License

# Copyright (c) 2023, Optimization-and-Machine-Learning-Lab

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.



class TRCG(Optimizer):


    def __init__(self, model, 
                 radius, device,
                   closure_size = 1,  # specifies how many parts the 
#                  lr=required, momentum=0, dampening=0,
#                  weight_decay=0, nesterov=False, *, maximize: bool = False,
#                  foreach: Optional[bool] = None,
                 cgopttol=1e-7,c0tr=0.2,c1tr=0.25,c2tr=0.75,t1tr=0.25,t2tr=2.0,
                 radius_max=5.0,
                 radius_initial=0.1,
                 differentiable: bool = False
                ):
        
        
        self.model = model
        self.device = device
        self.cgopttol = cgopttol
        self.c0tr = c0tr
        self.c1tr = c1tr
        self.c2tr = c2tr
        self.t1tr = t1tr
        self.t2tr = t2tr
        self.radius_max = radius_max
        self.radius_initial = radius_initial
        self.radius = radius
        self.cgmaxiter = 60
        
        
        
#         if lr is not required and lr < 0.0:
#             raise ValueError("Invalid learning rate: {}".format(lr))
#         if momentum < 0.0:
#             raise ValueError("Invalid momentum value: {}".format(momentum))
#         if weight_decay < 0.0:
#             raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        self.closure_size = closure_size
    
    
        defaults = dict(
#             lr=lr, momentum=momentum, dampening=dampening,
#                         weight_decay=weight_decay, nesterov=nesterov,
#                         maximize=maximize, foreach=foreach,
                        differentiable=differentiable
                       )
#         if nesterov and (momentum <= 0 or dampening != 0):
#             raise ValueError("Nesterov momentum requires a momentum and zero dampening")

        self.params = list(model.parameters())


        super(TRCG, self).__init__(self.params, defaults)

    def findroot(self,x,p):
        aa = 0.0; bb = 0.0; cc = 0.0
        for pi, xi in zip(p,x):
            aa += (pi*pi).sum()
            bb += (pi*xi).sum()
            cc += (xi*xi).sum()
        bb = bb*2.0
        cc = cc - self.radius**2
        alpha = (-2.0*cc)/(bb+(bb**2-(4.0*aa*cc)).sqrt())
        return alpha.data.item()    
    
    def _init_group(self, group, params_with_grad, d_p_list, momentum_buffer_list):
         
        for p in group['params']:
            if p.grad is not None:
                params_with_grad.append(p)
                d_p_list.append(p.grad)
                 
                state = self.state[p]
                if 'pk' not in state:
                    momentum_buffer_list.append(None)
                else:
                    momentum_buffer_list.append(state['ph'])

          
        
    def CGSolver(self,loss_grad,cnt_compute, closure):
    
        cg_iter = 0 # iteration counter
        x0 = [] # define x_0 as a list
        for i in self.model.parameters():
            x0.append(torch.zeros(i.shape).to(self.device))
    
        r0 = [] # set initial residual to gradient
        p0 = [] # set initial conjugate direction to -r0
        self.cgopttol = 0.0
        
        for i in loss_grad:
            r0.append(i.data+0.0)     
            p0.append(0.0-i.data)
            self.cgopttol+=torch.norm(i.data)**2
        
        self.cgopttol = self.cgopttol.data.item()**0.5
        self.cgopttol = (min(0.5,self.cgopttol**0.5))*self.cgopttol
    
        cg_term = 0
        j = 0

        while 1:
            j+=1
    
            # if CG does not solve model within max allowable iterations
            if j > self.cgmaxiter:
                j=j-1
                p1 = x0
                print ('\n\nCG has issues !!!\n\n')
                break
            # hessian vector product
            
            
            
            Hp = self.computeHessianVector(closure, p0)            
            cnt_compute+=1
            
            
            pHp = self.computeDotProduct(Hp, p0) # quadratic term
    
            # if nonpositive curvature detected, go for the boundary of trust region
            if pHp.data.item() <= 0:
                tau = self.findroot(x0,p0)
                p1 = []
                for e in range(len(x0)):
                    p1.append(x0[e]+tau*p0[e])
                cg_term = 1
                break
            
            # if positive curvature
            # vector product
            rr0 = 0.0
            for i in r0:
                rr0 += (i*i).sum()
            
            # update alpha
            alpha = (rr0/pHp).data.item()
        
            x1 = []
            norm_x1 = 0.0
            for e in range(len(x0)):
                x1.append(x0[e]+alpha*p0[e])
                norm_x1 += torch.norm(x0[e]+alpha*p0[e])**2
            norm_x1 = norm_x1**0.5
            
            # if norm of the updated x1 > radius
            if norm_x1.data.item() >= self.radius:
                tau = self.findroot(x0,p0)
                p1 = []
                for e in range(len(x0)):
                    p1.append(x0[e]+tau*p0[e])
                cg_term = 2
                break
    
            # update residual
            r1 = []
            norm_r1 = 0.0
            for e in range(len(r0)):
                r1.append(r0[e]+alpha*Hp[e])
                norm_r1 += torch.norm(r0[e]+alpha*Hp[e])**2
            norm_r1 = norm_r1**0.5
    
            if norm_r1.data.item() < self.cgopttol:
                p1 = x1
                cg_term = 3
                break
    
            rr1 = 0.0
            for i in r1:
                rr1 += (i*i).sum()
    
            beta = (rr1/rr0).data.item()
    
            # update conjugate direction for next iterate
            p1 = []
            for e in range(len(r1)):
                p1.append(-r1[e]+beta*p0[e])
    
            p0 = p1
            x0 = x1
            r0 = r1
    

        cg_iter = j
        d = p1

        return d,cg_iter,cg_term,cnt_compute        
        
        
        
    def computeHessianVector(self, closure, p):

        
        with torch.enable_grad():
            if self.closure_size == 1 and self.gradient_cache is not None:
                # we reuse the gradient computation 
                 
                Hpp = torch.autograd.grad(self.gradient_cache,
                                              self.params,
                                              grad_outputs=p, 
                                              retain_graph=True) # hessian-vector in tuple
                Hp = [Hpi.data+0.0 for Hpi in Hpp]
                    
            
                
            
            else:
        
                for part in range(self.closure_size):
                    loss = closure(part,self.closure_size, self.device)
                    loss_grad_v = torch.autograd.grad(loss,self.params,create_graph=True) 
                    Hpp = torch.autograd.grad(loss_grad_v,
                                              self.params,
                                              grad_outputs=p, 
                                              retain_graph=False) # hessian-vector in tuple
                    if part == 0:
                        Hp = [Hpi.data+0.0 for Hpi in Hpp]
                    else:
                        for Hpi, Hppi in zip(Hp, Hpp):
                            Hpi.add_(Hppi)
                    
                    
        return Hp
        
    def computeLoss(self, closure):
        lossVal = 0.0
        with torch.no_grad():
            for part in range(self.closure_size):
                loss = closure(part,self.closure_size, self.device)
                lossVal+= loss.item()
                    
                    
        return lossVal        

        
    def computeGradientAndLoss(self, closure):
        lossVal = 0.0
        with torch.enable_grad():
            for part in range(self.closure_size):
                loss = closure(part,self.closure_size, self.device)
                lossVal+= loss.item()
                if  self.closure_size == 1 and self.gradient_cache is None:
                     
                    loss_grad = torch.autograd.grad(loss,self.params,retain_graph=True,create_graph=True) 
                    self.gradient_cache =  loss_grad
                else:
                    
                    loss_grad = torch.autograd.grad(loss,self.params,create_graph=False) 
                
                if part == 0:
                    grad = [p.data+0.0 for p in loss_grad]
                else:
                    for gi, gip in zip(grad, loss_grad):
                        gi.add_(gip) 
                    
                    
        return lossVal, grad        
        
    def computeGradient(self, closure):
        return self.computeGradientAndLoss(closure)[1]
                    
                    
        return grad
        
    def computeDotProduct(self,v,z):
        return torch.sum(torch.vstack([ (vi*zi).sum() for vi, zi in zip(v, z)  ]))
        
    def computeNorm(self,v):
        return torch.sqrt(torch.sum(torch.vstack([ (p**2).sum() for p in v])))
        
    @_use_grad_for_differentiable
    def step(self, closure):
        """Performs a single optimization step.
        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """

        self.gradient_cache = None
        
        # store the initial weights 
        wInit = [w+0.0 for w in self.params]
        
        update = 2
        
        lossInit, loss_grad = self.computeGradientAndLoss(closure)
        NormG = self.computeNorm(loss_grad)
        
        cnt_compute=1

        
       
        # Conjugate Gradient Method
        d, cg_iter, cg_term, cnt_compute = self.CGSolver(loss_grad,cnt_compute, closure)

        
        Hd = self.computeHessianVector(closure, d)
        dHd = self.computeDotProduct(Hd, d)
        
        
        # update solution
        for wi, di in zip(self.params, d):
            with torch.no_grad():
                wi.add_(di)
        
        loss_new = self.computeLoss(closure)
        numerator = lossInit - loss_new

        gd = self.computeDotProduct(loss_grad, d)

        norm_d = self.computeNorm(d)
        
        denominator = -gd.data.item() - 0.5*(dHd.data.item())

        # ratio
        rho = numerator/denominator

        
        outFval = loss_new
        if rho < self.c1tr: # shrink radius
            self.radius = self.t1tr*self.radius
            update = 0
        elif rho > self.c2tr and np.abs(norm_d.data.item() - self.radius) < 1e-10: # enlarge radius
            self.radius = min(self.t2tr*self.radius,self.radius_max)
            update = 1
        # otherwise, radius remains the same
        if rho <= self.c0tr or numerator < 0: # reject d
            update = 3
            self.radius = self.t1tr*self.radius
            for wi, di in zip(self.params, d):
                with torch.no_grad():
                    wi.sub_(di)  
            outFval = lossInit
        return outFval, self.radius, cnt_compute, cg_iter    

from m3_learning.util.rand_util import save_list_to_txt
import pandas as pd

def static_state_decorator(func):
    """Decorator that stops the function from changing the state

    Args:
        func (method): any method
    """
    def wrapper(*args, **kwargs):
        current_state = args[0].get_state
        out = func(*args, **kwargs)
        args[0].set_attributes(**current_state)
        return out
    return wrapper


def write_csv(write_CSV,
              path,
              model_name,
              optimizer_name,
              epochs,
              total_time,
              train_loss,
              batch_size,
              loss_func,
              seed,
              stoppage_early,
              model_updates):

    if write_CSV is not None:
        headers = ["Model Name",
                   "Optimizer",
                   "Epochs",
                   "Training_Time",
                   "Train Loss",
                   "Batch Size",
                   "Loss Function",
                   "Seed",
                   "filename",
                   "early_stoppage",
                   "model updates"]
        data = [model_name,
                optimizer_name,
                epochs,
                total_time,
                train_loss,
                batch_size,
                loss_func,
                seed,
                f"{path}/{model_name}_model_epoch_{epochs}_train_loss_{train_loss}.pth",
                f"{stoppage_early}",
                f"{model_updates}"]
        append_to_csv(f"{path}/{write_CSV}", data, headers)


class Multiscale1DFitter(nn.Module):
    
    def __init__(self, 
                 function, # function to fit
                 x_data, # x_data to generate
                 input_channels, # number of input channels
                 num_params, # number of parameters to fit
                 scaler = None, # scaler object
                 post_processing = None, 
                 device = "cuda",
                 **kwargs):
        
        self.input_channels = input_channels
        self.scaler = scaler
        self.function = function
        self.x_data = x_data
        self.post_processing = post_processing
        self.device = device
        self.num_params = num_params

        super().__init__()

        # Input block of 1d convolution
        self.hidden_x1 = nn.Sequential(
            nn.Conv1d(in_channels=self.input_channels, out_channels=8, kernel_size=7),
            nn.SELU(),
            nn.Conv1d(in_channels=8, out_channels=6, kernel_size=7),
            nn.SELU(),
            nn.Conv1d(in_channels=6, out_channels=4, kernel_size=5),
            nn.SELU(),
            nn.AdaptiveAvgPool1d(64)
        )

        # fully connected block
        self.hidden_xfc = nn.Sequential(
            nn.Linear(256, 20),
            nn.SELU(),
            nn.Linear(20, 20),
            nn.SELU(),
        )

        # 2nd block of 1d-conv layers
        self.hidden_x2 = nn.Sequential(
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=2, out_channels=4, kernel_size=5),
            nn.SELU(),
            nn.Conv1d(in_channels=4, out_channels=4, kernel_size=5),
            nn.SELU(),
            nn.Conv1d(in_channels=4, out_channels=4, kernel_size=5),
            nn.SELU(),
            nn.Conv1d(in_channels=4, out_channels=4, kernel_size=5),
            nn.SELU(),
            nn.Conv1d(in_channels=4, out_channels=4, kernel_size=5),
            nn.SELU(),
            nn.Conv1d(in_channels=4, out_channels=4, kernel_size=5),
            nn.SELU(),
            nn.AdaptiveAvgPool1d(16),  # Adaptive pooling layer
            nn.Conv1d(in_channels=4, out_channels=2, kernel_size=3),
            nn.SELU(),
            nn.AdaptiveAvgPool1d(8),  # Adaptive pooling layer
            nn.Conv1d(in_channels=2, out_channels=2, kernel_size=3),
            nn.SELU(),
            nn.AdaptiveAvgPool1d(4),  # Adaptive pooling layer
        )

        # Flatten layer
        self.flatten_layer = nn.Flatten()

        # Final embedding block - Output 4 values - linear
        self.hidden_embedding = nn.Sequential(
            nn.Linear(28, 16),
            nn.SELU(),
            nn.Linear(16, 8),
            nn.SELU(),
            nn.Linear(8, self.num_params),
        )

    def forward(self, x, n=-1):
        # print(x.dtype)
        # print('1', self.x_data.shape, x.shape)
        # output shape - samples, (real, imag), frequency
        # x = torch.swapaxes(x, 1, 2)
        x = self.hidden_x1(x)
        # print(x.dtype)

        xfc = torch.reshape(x, (n, 256))  # batch size, features
        xfc = self.hidden_xfc(xfc)

        # batch size, (real, imag), timesteps
        x = torch.reshape(x, (n, 2, 128))
        x = self.hidden_x2(x)
        cnn_flat = self.flatten_layer(x)

        encoded = torch.cat((cnn_flat, xfc), 1)  # merge dense and 1d conv.
        embedding = self.hidden_embedding(encoded)  # output is 4 parameters

        if self.scaler is not None:
            # corrects the scaling of the parameters
            unscaled_param = (
                embedding *
                torch.tensor(self.scaler.var_ ** 0.5).to(self.device)
                + torch.tensor(self.scaler.mean_).to(self.device)
            )
        else:
            unscaled_param = embedding
        # print(unscaled_param.shape)
        # unscaled_param[:,0] = torch.relu(unscaled_param[:,0])
        unscaled_param[:,0] = torch.tanh(unscaled_param[:,0])
        # unscaled_param[:,1] = torch.tanh(unscaled_param[:,2])
        unscaled_param[:,1] = torch.relu(unscaled_param[:,1])+1e-3

        # frequency_bins = resample(self.dataset.frequency_bin,
        #                           self.dataset.resampled_bins)
        
        # print(unscaled_param.shape, self.x_data.shape)
        # passes to the pytorch fitting function
        fits = self.function(
            unscaled_param, self.x_data, device=self.device)
        
        # Does the post processing if required
        if self.post_processing is not None:
            out = self.post_processing.compute(fits)
        else:
            out = fits
        return out, unscaled_param

        # if self.training == True:
        #     return out, unscaled_param
        # if self.training == False:
        #     # this is a scaling that includes the corrections for shifts in the data
        #     embeddings = (unscaled_param.to(self.device) - torch.tensor(self.scaler.mean_).to(self.device)
        #                   )/torch.tensor(self.scaler.var_ ** 0.5).to(self.device)
        #     return out, embeddings, unscaled_param


class Model(nn.Module):

    def __init__(self,
                 model,
                 dataset,
                 model_basename='',
                 training=True,
                 path='Trained Models/SHO Fitter/',
                 device=None,
                 **kwargs):
        
        super().__init__()

        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
                print(f"Using GPU {torch.cuda.get_device_name(0)}")
            else:
                self.device = "cpu"
                print("Using CPU")

        self.model = model
        self.model.dataset = dataset
        self.model.training = True
        self.model_name = model_basename
        self.path = make_folder(path)

    def fit(self,
            data_train,
            batch_size=200,
            epochs=5,
            loss_func=torch.nn.MSELoss(),
            optimizer='Adam',
            seed=42,
            datatype=torch.float32,
            save_all=False,
            write_CSV=None,
            closure=None,
            basepath=None,
            early_stopping_loss=None,
            early_stopping_count=None,
            early_stopping_time=None,
            save_training_loss=True,
            **kwargs):

        loss_ = []

        if basepath is not None:
            path = f"{self.path}/{basepath}/"
            make_folder(path)
            print(f"Saving to {path}")
        else:
            path = self.path

        # sets the model to be a specific datatype and on cuda
        self.to(datatype).to(self.device)

        # Note that the seed will behave differently on different hardware targets (GPUs)
        # random_seed(seed=seed)

        torch.cuda.empty_cache()

        # selects the optimizer
        if optimizer == 'Adam':
            optimizer_ = torch.optim.Adam(self.model.parameters())
        elif optimizer == "AdaHessian":
            optimizer_ = AdaHessian(self.model.parameters(), lr=.5)
        elif isinstance(optimizer, dict):
            if optimizer['name'] == "TRCG":
                optimizer_ = optimizer['optimizer'](
                    self.model, optimizer['radius'], optimizer['device'])
        elif isinstance(optimizer, dict):
            if optimizer['name'] == "TRCG":
                optimizer_ = optimizer['optimizer'](
                    self.model, optimizer['radius'], optimizer['device'])
        else:
            try:
                optimizer = optimizer(self.model.parameters())
            except:
                raise ValueError("Optimizer not recognized")

        # instantiate the dataloader
        train_dataloader = DataLoader(
            data_train, batch_size=batch_size, shuffle=True)

        # if trust region optimizers stores the TR optimizer as an object and instantiates the ADAM optimizer
        if isinstance(optimizer_, TRCG):
            TRCG_OP = optimizer_
            optimizer_ = torch.optim.Adam(self.model.parameters(), **kwargs)

        total_time = 0
        low_loss_count = 0

        # says if the model have already stopped early
        already_stopped = False

        model_updates = 0

        # loops around each epoch
        for epoch in range(epochs):

            train_loss = 0.0
            total_num = 0
            epoch_time = 0

            # sets the model to training mode
            self.model.train()

            for train_batch in train_dataloader:

                model_updates += 1

                # starts the timer
                start_time = time.time()

                train_batch = train_batch.to(datatype).to(self.device)

                if "TRCG_OP" in locals() and epoch > optimizer.get("ADAM_epochs", -1):

                    def closure(part, total, device):
                        pred, embedding = self.model(train_batch)
                        pred = pred.to(torch.float32)
                        embedding = embedding.to(torch.float32)
                        loss = loss_func(train_batch, pred)
                        return loss

                    # if closure is not None:
                    loss, radius, cnt_compute, cg_iter = TRCG_OP.step(
                        closure)
                    train_loss += loss * train_batch.shape[0]
                    total_num += train_batch.shape[0]
                    optimizer_name = "Trust Region CG"
                else:
                    pred, embedding = self.model(train_batch)
                    pred = pred.to(torch.float32)
                    pred = torch.atleast_3d(pred)
                    embedding = embedding.to(torch.float32)
                    optimizer_.zero_grad()
                    loss = loss_func(train_batch, pred)
                    loss.backward(create_graph=True)
                    train_loss += loss.item() * pred.shape[0]
                    total_num += pred.shape[0]
                    optimizer_.step()
                    if isinstance(optimizer_, torch.optim.Adam):
                        optimizer_name = "Adam"
                    elif isinstance(optimizer_, AdaHessian):
                        optimizer_name = "AdaHessian"

                epoch_time += (time.time() - start_time)

                total_time += (time.time() - start_time)

                try:
                    loss_.append(loss.item())
                except:
                    loss_.append(loss)

                if early_stopping_loss is not None and already_stopped == False:
                    if loss < early_stopping_loss:
                        low_loss_count += train_batch.shape[0]
                        if low_loss_count >= early_stopping_count:
                            torch.save(self.model.state_dict(),
                                       f"{path}/Early_Stoppage_at_{total_time}_{self.model_name}_model_optimizer_{optimizer_name}_epoch_{epoch}_train_loss_{train_loss/total_num}.pth")

                            write_csv(write_CSV,
                                      path,
                                      self.model_name,
                                      optimizer_name,
                                      epoch,
                                      total_time,
                                      train_loss/total_num,
                                      batch_size,
                                      loss_func,
                                      seed,
                                      True,
                                      model_updates)

                            already_stopped = True
                    else:
                        low_loss_count -= (train_batch.shape[0]*5)

            if "verbose" in kwargs:
                if kwargs["verbose"] == True:
                    print(f"Loss = {loss.item()}")

            train_loss /= total_num

            print(optimizer_name)
            print("epoch : {}/{}, recon loss = {:.8f}".format(epoch +
                                                              1, epochs, train_loss))
            print("--- %s seconds ---" % (epoch_time))

            if save_all:
                torch.save(self.model.state_dict(),
                           f"{path}/{self.model_name}_model_optimizer_{optimizer_name}_epoch_{epoch}_train_loss_{train_loss}.pth")

            if early_stopping_time is not None:
                if total_time > early_stopping_time:
                    torch.save(self.model.state_dict(),
                               f"{path}/Early_Stoppage_at_{total_time}_{self.model_name}_model_optimizer_{optimizer_name}_epoch_{epoch}_train_loss_{train_loss}.pth")

                    write_csv(write_CSV,
                              path,
                              self.model_name,
                              optimizer_name,
                              epoch,
                              total_time,
                              train_loss,  # already divided by total_num
                              batch_size,
                              loss_func,
                              seed,
                              True,
                              model_updates)
                    break

        torch.save(self.model.state_dict(),
                   f"{path}/{self.model_name}_model_optimizer_{optimizer_name}_epoch_{epoch}_train_loss_{train_loss}.pth")
        write_csv(write_CSV,
                  path,
                  self.model_name,
                  optimizer_name,
                  epoch,
                  total_time,
                  train_loss,  # already divided by total_num
                  batch_size,
                  loss_func,
                  seed,
                  False,
                  model_updates)

        if save_training_loss:
            save_list_to_txt(
                loss_, f"{path}/Training_loss_{self.model_name}_model_optimizer_{optimizer_name}_epoch_{epoch}_train_loss_{train_loss}.txt")

        self.model.eval()

    def load(self, model_path):
        self.model.load_state_dict(torch.load(model_path))
        self.model.to(self.device)

    def inference_timer(self, data, batch_size=.5e4):
        torch.cuda.empty_cache()

        batch_size = int(batch_size)

        dataloader = DataLoader(data, batch_size)

        # Computes the inference time
        computeTime(self.model, dataloader, batch_size, device=self.device)

    def predict(self, data, batch_size=10000,
                single=False,
                translate_params=True):

        self.model.eval()

        dataloader = DataLoader(data, batch_size=batch_size)

        # preallocate the predictions
        num_elements = len(dataloader.dataset)
        num_batches = len(dataloader)
        data = data.clone().detach().requires_grad_(True)
        predictions = torch.zeros_like(data.clone().detach())
        params_scaled = torch.zeros((data.shape[0], 4))
        params = torch.zeros((data.shape[0], 4))

        # compute the predictions
        for i, train_batch in enumerate(dataloader):
            start = i * batch_size
            end = start + batch_size

            if i == num_batches - 1:
                end = num_elements

            pred_batch, params_scaled_, params_ = self.model(
                train_batch.to(self.device))

            predictions[start:end] = pred_batch.cpu().detach()
            params_scaled[start:end] = params_scaled_.cpu().detach()
            params[start:end] = params_.cpu().detach()

            torch.cuda.empty_cache()

        # converts negative ampltiudes to positive and shifts the phase to compensate
        if translate_params:
            params[params[:, 0] < 0, 3] = params[params[:, 0] < 0, 3] - np.pi
            params[params[:, 0] < 0, 0] = np.abs(params[params[:, 0] < 0, 0])

        if self.model.dataset.NN_phase_shift is not None:
            params_scaled[:, 3] = torch.Tensor(self.model.dataset.shift_phase(
                params_scaled[:, 3].detach().numpy(), self.model.dataset.NN_phase_shift))
            params[:, 3] = torch.Tensor(self.model.dataset.shift_phase(
                params[:, 3].detach().numpy(), self.model.dataset.NN_phase_shift))

        return predictions, params_scaled, params

    @staticmethod
    def mse_rankings(true, prediction, curves=False):

        def type_conversion(data):

            data = np.array(data)
            data = np.rollaxis(data, 0, data.ndim-1)

            return data

        true = type_conversion(true)
        prediction = type_conversion(prediction)

        errors = Model.MSE(prediction, true)

        index = np.argsort(errors)

        if curves:
            # true will be in the form [ranked error, channel, timestep]
            return index, errors[index], true[index], prediction[index]

        return index, errors[index]

    @staticmethod
    def MSE(true, prediction):

        # calculates the mse
        mse = np.mean((true.reshape(
            true.shape[0], -1) - prediction.reshape(true.shape[0], -1))**2, axis=1)

        # converts to a scalar if there is only one value
        if mse.shape[0] == 1:
            return mse.item()

        return mse

    @staticmethod
    def get_rankings(raw_data, pred, n=1, curves=True):
        """simple function to get the best, median and worst reconstructions

        Args:
            raw_data (np.array): array of the true values
            pred (np.array): array of the predictions
            n (int, optional): number of values for each. Defaults to 1.
            curves (bool, optional): whether to return the curves or not. Defaults to True.

        Returns:
            ind: indices of the best, median and worst reconstructions
            mse: mse of the best, median and worst reconstructions
        """
        index, mse, d1, d2 = Model.mse_rankings(
            raw_data, pred, curves=curves)
        middle_index = len(index) // 2
        start_index = middle_index - n // 2
        end_index = start_index + n

        ind = np.hstack(
            (index[:n], index[start_index:end_index], index[-n:])).flatten().astype(int)
        mse = np.hstack(
            (mse[:n], mse[start_index:end_index], mse[-n:]))

        d1 = np.stack(
            (d1[:n], d1[start_index:end_index], d1[-n:])).squeeze()
        d2 = np.stack(
            (d2[:n], d2[start_index:end_index], d2[-n:])).squeeze()

        # return ind, mse, np.swapaxes(d1[ind], 1, d1.ndim-1), np.swapaxes(d2[ind], 1, d2.ndim-1)
        return ind, mse, d1, d2

    def print_mse(self, data, labels):
        """prints the MSE of the model

        Args:
            data (tuple): tuple of datasets to calculate the MSE
            labels (list): List of strings with the names of the datasets
        """

        # loops around the dataset and labels and prints the MSE for each
        for data, label in zip(data, labels):

            if isinstance(data, torch.Tensor):
                # computes the predictions
                pred_data, scaled_param, parm = self.predict(data)
            elif isinstance(data, dict):
                pred_data, _ = self.model.dataset.get_raw_data_from_LSQF_SHO(
                    data)
                data, _ = self.model.dataset.NN_data()
                pred_data = torch.from_numpy(pred_data)

            # Computes the MSE
            out = nn.MSELoss()(data, pred_data)

            # prints the MSE
            print(f"{label} Mean Squared Error: {out:0.4f}")


@static_state_decorator
def batch_training(dataset, optimizers, noise, batch_size, epochs, seed, write_CSV="Batch_Training_Noisy_Data.csv",
                   basepath=None, early_stopping_loss=None, early_stopping_count=None, early_stopping_time=None, skip=-1, **kwargs,
                   ):

    # Generate all combinations
    combinations = list(itertools.product(
        optimizers, noise, batch_size, epochs, seed))

    for i, training in enumerate(combinations):
        if i < skip:
            print(
                f"Skipping combination {i}: {training[0]} {training[1]} {training[2]}  {training[3]}  {training[4]}")
            continue

        optimizer = training[0]
        noise = training[1]
        batch_size = training[2]
        epochs = training[3]
        seed = training[4]

        print(f"The type is {type(training[0])}")

        if isinstance(optimizer, dict):
            optimizer_name = optimizer['name']
        else:
            optimizer_name = optimizer

        dataset.noise = noise

        # random_seed(seed=seed)

        # constructs a test train split
        X_train, X_test, y_train, y_test = dataset.test_train_split_(
            shuffle=True)

        model_name = f"SHO_{optimizer_name}_noise_{training[1]}_batch_size_{training[2]}_seed_{training[4]}"

        print(f'Working on combination: {model_name}')

        # instantiate the model
        model = Model(dataset, training=True, model_basename=model_name)

        # fits the model
        model.fit(
            X_train,
            batch_size=batch_size,
            optimizer=optimizer,

            epochs=epochs,
            write_CSV=write_CSV,
            seed=seed,
            basepath=basepath,
            early_stopping_loss=early_stopping_loss,
            early_stopping_count=early_stopping_count,
            early_stopping_time=early_stopping_time,
            **kwargs,
        )
        
        del model

def find_best_model(basepath, filename):
    
    # Read the CSV
    df = pd.read_csv(basepath + '/' + filename)

    # Extract noise level from the 'Model Name' column
    df['Noise Level'] = df['Model Name'].apply(lambda x: float(x.split('_')[3]))

    # Create an empty dictionary to store the results
    results = {}

    # Loop over each unique combination of noise level and optimizer
    for noise_level in df['Noise Level'].unique():
        for optimizer in df['Optimizer'].unique():
            # Create a mask for the current combination
            mask = (df['Noise Level'] == noise_level) & (df['Optimizer'] == optimizer)
            
            # If there's any row with this combination
            if df[mask].shape[0] > 0:
                # Find the index of the minimum 'Train Loss'
                min_loss_index = df.loc[mask, 'Train Loss'].idxmin()
                
                # Store the result
                results[(noise_level, optimizer)] = df.loc[min_loss_index].to_dict()

    return results