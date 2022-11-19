import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer, required
from typing import List, Optional, Callable
import levy
import numpy as np

class SGD_TC(Optimizer):

    def __init__(self, params, lr=required, momentum=0,
                 func = None, device: str = 'cpu', 
                 height: float=1, width: float=required, 
                 scale_annealer: Callable=required, n_epochs: int=required,
                 adjust_dir: bool=False):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))

        self.record = {'lr': lr, 'height': height, 'width': width, 'momentum': momentum}
        self.scale_annealer = scale_annealer
        self.objfunc = func
        self.history = {}
        self.alpha_record = {}
        self.height = height
        self.width_denom = -0.5*(1/width)**2
        self.n_epochs = n_epochs
        self.step_count = 0
        self.device = device
        self.adjust_dir = adjust_dir

        defaults = dict(lr=lr, momentum=momentum)
        super(SGD_TC, self).__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)
            group.setdefault('maximize', False)
            group.setdefault('foreach', None)

    @torch.no_grad()
    def step(self, batch=None, ising=False, model=None):
        """
        Performs a single optimization step.
        """

        # MLP - 1 group, 3 params in group
        param_idx = 0 # tracks index of list of params with grad
        for group in self.param_groups:
            params_with_grad = []
            grad_list = []
            momentum_buffer_list = []

            for p in group['params']:
                if p.grad is not None:

                    params_with_grad.append(p)
                    grad_list.append(p.grad)

                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        momentum_buffer_list.append(None)
                    else:
                        momentum_buffer_list.append(state['momentum_buffer'])

                    # Update history to calculate Vbias
                    curr_param = p.detach().clone().unsqueeze(dim=0)
                    if param_idx not in self.history:
                        self.history[param_idx] =  curr_param
                    else:
                        temp = [self.history[param_idx], curr_param]
                        self.history[param_idx] = torch.cat(temp, dim=0)

                    param_idx += 1
            
            if not ising:

                self.sgd(params_with_grad,
                        grad_list,
                        momentum_buffer_list,
                        momentum=group['momentum'],
                        lr=group['lr'],
                        batch=batch,
                        model=model)

                # update momentum_buffers in state
                for p, momentum_buffer in zip(params_with_grad, momentum_buffer_list):
                    state = self.state[p]
                    state['momentum_buffer'] = momentum_buffer
        
        self.step_count += 1

    def sgd(self,
            params: List[Tensor],
            grad_list: List[Tensor],
            momentum_buffer_list: List[Optional[Tensor]],
            momentum: float,
            lr: float,
            batch,
            model):

        for param_idx, param in enumerate(params):
            grad = grad_list[param_idx]

            if momentum != 0:
                buf = momentum_buffer_list[param_idx]

                if buf is None:
                    buf = torch.clone(grad).detach()
                    momentum_buffer_list[param_idx] = buf
                else:
                    buf.mul_(momentum).add_(grad, alpha=1)

                grad = buf

            # Adapt levy noise properties then calculate noise vector
            alpha = self.adapt_alpha(param, param_idx)
            param.add_(grad, alpha=-lr)
            levy_noise_vector = self.levy_noise(param, alpha, grad, batch, model)
            param.add_(levy_noise_vector, alpha=lr)

            # Enforce periodic boundary conditions
            if self.objfunc is not None:
                param = self.objfunc.pbc(param)
    
    def levy_noise(self, param, alpha, grad, batch=None, model=None):
        dim = param.size()
        direction = hypersphere_sample(dim, self.device)
        if batch is not None:
            with torch.enable_grad():
                loss = model(data=batch)
                loss.backward()
                grad = param.grad

        grad_norm = float(torch.norm(grad))

        factor = self.scale_annealer(self.step_count/self.n_epochs)
        scale = factor*grad_norm
        if scale == 0:
            levy_r = 0
        else:   
            levy_r = abs(float(levy.random(alpha=alpha, beta=0, sigma=scale)))
        # Truncate the noise distributions to avoid unreasonably large steps.
        # if levy_r > 10*grad_norm:
        #     levy_r = 10*grad_norm

        #print("ratio", float(levy_r/torch.norm(grad)), factor, scale)
                
        if self.adjust_dir:
            direction = direction.type_as(grad)
            param_number = param.numel()
            if param_number > 1:
                adjust_direction = direction.view(-1, 1).squeeze().dot(-grad.view(-1,1).squeeze())/grad_norm
            else:
                adjust_direction = direction.dot(-grad)/grad_norm
            adjust_direction = min(torch.exp(adjust_direction  * 0.25 * np.sqrt(param_number) - 1), 1)
            noise = levy_r * direction * adjust_direction
            #print('wow', alpha, adjust_direction, levy_r, grad_norm)
        else:   
            noise = levy_r * direction
        
        return noise

    def adapt_alpha(self, param, param_idx):
        
        v = self.history[param_idx] - param
        dim_count = len(v.size())
        if dim_count == 3:          # parameters are matrices
            Vbias = v.norm(dim=(-2,-1))
        elif dim_count == 2:        # parameters are vectors
            Vbias = v.norm(dim=-1)
        Vbias = torch.exp(self.width_denom * Vbias).sum(0)
        Vbias = float(self.height * Vbias)
        alpha = 1 + Vbias/(1+self.step_count)

        if param_idx not in self.alpha_record:
            self.alpha_record[param_idx] = [alpha]
        else:
            self.alpha_record[param_idx].append(alpha)
        
        return alpha

class SimulatedAnnealing(Optimizer):
    """
    Metropolis-Hastings with Simulated Annealing using list of temperatures (betas).
    """
    def __init__(self, params, betas: List=required, n_epochs: int=required, 
                 func=required, device: str = 'cpu'):

        self.record = {'betas': (betas[0], betas[-1])}
        self.objfunc = func
        assert len(betas) == n_epochs
        self.betas = betas
        self.n_epochs = n_epochs
        self.step_count = 0
        self.device = device

        defaults = dict()
        super(SimulatedAnnealing, self).__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)

    @torch.no_grad()
    def step(self, int_matrices, curr_energy):
        """Performs a single optimization step.
        """

        for group in self.param_groups:
            params_with_grad = []
            for p in group['params']:
                if p.grad is not None:

                    params_with_grad.append(p)

            dE, site = self.simulated_annealing(params_with_grad, int_matrices, curr_energy)
        
        self.step_count += 1
    
        return dE, site

    def simulated_annealing(self, params: List[Tensor], int_matrices, curr_energy):

        for param in params:
            site = self.propose(param)
            temp_param = param.detach().clone()
            temp_param[site] = -1*temp_param[site]
            beta = self.betas[self.step_count]
            dE = float(self.objfunc(int_matrices=int_matrices, spins=temp_param)) - curr_energy

            if self.accept(beta, dE):
                param[site] = temp_param[site]
        
        return dE, site

    def propose(self, param):
        N = param.shape[0]
        return tuple(np.random.choice(N, 2))
    
    def accept(self, beta, dE):
        if dE <= 0:
            return True
        else:
            p = np.exp(-beta*dE)
            return np.random.random() < p

class SimulatedTempering(Optimizer):
    """
    Simulated Tempering implementation based on Li, Protopopescu and Gorin (2004)
    """
    def __init__(self, params, betas: List=required, n_epochs: int=required, 
                 func=required, device: str = 'cpu'):

        self.record = {'betas': (betas[0], betas[-1])}
        self.objfunc = func
        self.betas = betas
        self.beta_idx = 0
        self.n_epochs = n_epochs
        self.step_count = 0
        self.device = device

        defaults = dict()
        super(SimulatedTempering, self).__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)

    @torch.no_grad()
    def step(self, int_matrices, curr_energy):
        """Performs a single optimization step.
        """

        for group in self.param_groups:
            params_with_grad = []
            for p in group['params']:
                if p.grad is not None:

                    params_with_grad.append(p)

            dE, site = self.metropolis_hastings(params_with_grad, int_matrices, curr_energy)
            self.tempering(curr_energy)
        self.step_count += 1
    
        return dE, site

    def metropolis_hastings(self, params: List[Tensor], int_matrices, curr_energy):

        for param in params:
            site = self.propose(param)
            temp_param = param.detach().clone()
            temp_param[site] = -1*temp_param[site]
            beta = self.betas[self.beta_idx]
            dE = float(self.objfunc(int_matrices=int_matrices, spins=temp_param)) - curr_energy

            if self.accept(beta, dE):
                param[site] = temp_param[site]
        
        return dE, site
    
    def tempering(self, curr_energy):
        if self.beta_idx == 0:
            self.beta_idx += 1
        elif self.beta_idx == len(self.betas) - 1:
            self.beta_idx -= 1
        else:
            p = np.random.rand(1)
            if p < 0.5:
                prop_temp = self.beta_idx + 1 # decrease temperature
            else:
                prop_temp = self.beta_idx - 1 # increase temperature
            temp_change = self.betas[prop_temp] - self.betas[self.beta_idx]
            p = np.random.rand(1)
            acc_prob = np.exp(-temp_change*curr_energy)
            if p < acc_prob:
                self.beta_idx = prop_temp

        
    def propose(self, param):
        N = param.shape[0]
        return tuple(np.random.choice(N, 2))
    
    def accept(self, beta, dE):
        if dE <= 0:
            return True
        else:
            p = np.exp(-beta*dE)
            return np.random.random() < p

class Metadynamics(Optimizer):
    def __init__(self, params, n_epochs: int, lr=required, func = None, device: str = 'cpu', 
                 height: float = 1, width: float = 1, scale_annealer: Callable = lambda x: 1,):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))

        self.record = {'lr': lr, 'height': height, 'width': width}
        self.scale_annealer = scale_annealer
        self.objfunc = func
        self.history = {}
        self.height = height
        self.width_denom = -0.5*(1/width)**2
        self.n_epochs = n_epochs
        self.step_count = 0
        self.device = device

        defaults = dict(lr=lr)
        super(Metadynamics, self).__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # MLP - 1 group, 3 params in group
        param_idx = 0 # tracks index of list of params with grad
        for group in self.param_groups:
            params_with_grad = []
            grad_list = []

            for p in group['params']:
                if p.grad is not None:

                    params_with_grad.append(p)
                    grad_list.append(p.grad)

                    # Update history to calculate Vbias
                    curr_param = p.detach().clone().unsqueeze(dim=0)
                    if param_idx not in self.history:
                        self.history[param_idx] =  curr_param
                    else:
                        temp = [self.history[param_idx], curr_param]
                        self.history[param_idx] = torch.cat(temp, dim=0)
                    param_idx += 1  

            self.metadynamics(params_with_grad,
                     grad_list,
                     lr=group['lr']
            )
        
        self.step_count += 1
        return loss


    def metadynamics(self, params: List[Tensor], grad_list: List[Tensor], lr: float):

        for param_idx, param in enumerate(params):
            grad = grad_list[param_idx]

            lr = self.scale_annealer(self.step_count/self.n_epochs) * lr
            param.add_(grad, alpha=-lr)

            # compute Vbias step
            
            md_step = self.Vbias_step(param, param_idx)
            param.add_(md_step, alpha=-lr)

            # Enforce periodic boundary conditions
            if self.objfunc is not None:
                param = self.objfunc.pbc(param)

    def Vbias_step(self, param, param_idx):
        
        with torch.enable_grad():
            param = param.detach().clone()
            history = self.history[param_idx].detach().clone()
            param.requires_grad = True
            history.requires_grad = True
            v = history - param
            dim_count = len(v.size())
            if dim_count == 3:          # parameters are matrices
                exp_arg = v.norm(dim=(-2,-1))
            elif dim_count == 2:        # parameters are vectors
                exp_arg = v.norm(dim=-1)
            Vbias = torch.exp(self.width_denom * exp_arg).sum(0)
            Vbias.backward()
            
            return param.grad

class SGLD(Optimizer):

    def __init__(self, params, lr=required, momentum=0,
                 func = None, device: str = 'cpu', 
                 height: float=required, width: float=required, 
                 scale_annealer: Callable=required, n_epochs: int=required):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))

        self.record = {'lr': lr, 'height': height, 'width': width, 'momentum': momentum}
        self.scale_annealer = scale_annealer
        self.objfunc = func
        self.height = height
        self.width_denom = -0.5*(1/width)**2
        self.n_epochs = n_epochs
        self.step_count = 0
        self.device = device

        defaults = dict(lr=lr, momentum=momentum)
        super(SGLD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)
            group.setdefault('maximize', False)
            group.setdefault('foreach', None)

    @torch.no_grad()
    def step(self, ising=False, closure=None):
        """Performs a single optimization step.
        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # MLP - 1 group, 3 params in group
        for group in self.param_groups:
            params_with_grad = []
            grad_list = []
            momentum_buffer_list = []

            for p in group['params']:
                if p.grad is not None:

                    params_with_grad.append(p)
                    grad_list.append(p.grad)

                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        momentum_buffer_list.append(None)
                    else:
                        momentum_buffer_list.append(state['momentum_buffer'])
            
            if not ising:

                self.sgd(params_with_grad,
                        grad_list,
                        momentum_buffer_list,
                        momentum=group['momentum'],
                        lr=group['lr'])

                # update momentum_buffers in state
                for p, momentum_buffer in zip(params_with_grad, momentum_buffer_list):
                    state = self.state[p]
                    state['momentum_buffer'] = momentum_buffer
        
        self.step_count += 1
        return loss

    def sgd(self,
            params: List[Tensor],
            grad_list: List[Tensor],
            momentum_buffer_list: List[Optional[Tensor]],
            momentum: float,
            lr: float):

        for param_idx, param in enumerate(params):
            grad = grad_list[param_idx]

            if momentum != 0:
                buf = momentum_buffer_list[param_idx]

                if buf is None:
                    buf = torch.clone(grad).detach()
                    momentum_buffer_list[param_idx] = buf
                else:
                    buf.mul_(momentum).add_(grad, alpha=1)

                grad = buf

            #lr = self.scale_annealer(self.step_count/self.n_epochs) * lr
            gaussian_noise_vector = self.gaussian_noise(param, grad)
            param.add_(gaussian_noise_vector, alpha=lr)
            param.add_(grad, alpha=-lr)

            # # Enforce periodic boundary conditions
            if self.objfunc is not None:
                param = self.objfunc.pbc(param)
    
    def gaussian_noise(self, param, grad):
        dim = param.size()
        direction = hypersphere_sample(dim, self.device)
        grad_norm = float(torch.norm(grad))

        factor = self.scale_annealer(self.step_count/self.n_epochs)
        scale = factor*grad_norm
        gaussian_r = float(levy.random(alpha=2, beta=0, sigma=scale))

        noise = gaussian_r * direction
        
        return noise

def hypersphere_sample(dim, device):

    if device == 'cpu':
        direction = torch.normal(mean=0, std=1, size=dim)
    else:
        direction = torch.normal(mean=0, std=1, size=dim).to(device)
    direction = torch.sqrt(sum(direction**2))**(-1) * direction

    return direction

    #https://mathworld.wolfram.com/HyperspherePointPicking.html

def powlaw_samp(x_min, alpha, size=1):
    """
    Samples from powerlaw dist with min value x_min.
    """
    r = np.random.random(size=size)
    samp = x_min * (1 - r) ** (1 / (1-alpha))
    
    if size == 1:
        return float(samp)
    else:
        return samp

    # https://stats.stackexchange.com/questions/173242/random-sample-from-power-law-distribution
    # https://arxiv.org/pdf/0706.1062.pdf