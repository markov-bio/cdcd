import logging
import numpy as np
import torch
from torch import nn
from matplotlib import pyplot as plt

from scipy.optimize import curve_fit
from torch.distributions import Uniform, TransformedDistribution
from torch.distributions.transforms import SigmoidTransform, AffineTransform



class AdaptiveSchedule(nn.Module):
    def __init__(self, tmin, tmax, mu, sigma, height, offset):
        super().__init__()
        
        self.tmin = tmin
        self.tmax = tmax
        
        self.mu = mu
        self.sigma = sigma
        self.height = height
        self.offset = offset

        self.set_parameters([mu,sigma,height,offset])

        self.transforms = None  # To be defined in child classes
        self.distribution = None  # To be instantiated in child classes

        self.times=[]
        self.entropy=[]
    
    def set_parameters(self, params):
        self.optimal_parameters = nn.Parameter(torch.tensor(params,device='cpu'), requires_grad=False)
    
    def sample(self, shape):
        return self.distribution.sample(shape)
    
    # is history in terms of batches? what happens if we change to our update frequency, e.g. 100 batches?
    def update_optimal_parameters(self, history=500):
        #is it possible that over time it saturates the RAM?
        times=self.times[-history:] if len(self.times)>history else self.times
        entropy=self.entropy[-history:] if len(self.entropy)>history else self.entropy

        #i'm not using pytorch to find optimal parameters to fit the sigmoid because it might mess with the optimizer state
        try:
            optimal_parameters, _ = curve_fit(self.cdf, times, entropy, self.optimal_parameters.cpu().numpy(), bounds=((0,0,0,-np.inf),(np.inf,np.inf,np.inf,np.inf)))
        except RuntimeError as e:
            optimal_parameters = [.5,.5,np.log(self.vocab_size),0]
            print(e,'reverting to self.optimal_parameters =',optimal_parameters)
        
        self.mu, self.sigma, self.height, self.offset = optimal_parameters.tolist()
        self.set_parameters(optimal_parameters.tolist())
        logging.info(f"Updated optimal parameters: mu={self.mu}, sigma={self.sigma}, height={self.height}, offset={self.offset}")    

        self.update_distribution() 

    def make_timesteps(self, steps, tmin=None, tmax=None, device='cpu'):
        tmin=self.tmin if tmin is None else tmin
        tmax=self.tmax if tmax is None else tmax


        imin,imax=get_codomain(self.cdf, tmin, tmax, self.mu, self.sigma, self.height, self.offset)
        #indexes[0] is the highest noise level and indexes[-1] the lowest
        indexes=torch.linspace(imax,imin,steps, device=device)

        # if there are some numerical instabilities try using torch.log(indexes)-torch.log(1-indexes) instead of torch.log(indexes/(1-indexes))
        timesteps = self.transforms(indexes)
        return torch.clamp(timesteps,self.tmin, self.tmax)

    def add_data(self,entropy:torch.Tensor, times:torch.Tensor):
        times=times.detach().to('cpu').tolist()
        entropy=entropy.detach().to('cpu').tolist()

        for t,s in zip(times,entropy):
            self.times.append(t)
            self.entropy.append(s)

    def update_distribution(self): 
        pass
            
    def cdf(self, x, mu, sigma, height, offset):
        pass 

    def plot_entropy_time_curve(self, filename='et.png'):
        plt.close()
        plt.figure(figsize=(16, 4)) 
        # Calculate logarithmic indices for coloring
        indices = np.arange(1, len(self.times) + 1)
        log_indices = np.log(indices)[::-1]  # Reverse to give more weight to recent points
        log_indices = (log_indices - np.min(log_indices)) / (np.max(log_indices) - np.min(log_indices))

        # Scatter plot of entropy vs. time
        plt.scatter(self.times, self.entropy, c=log_indices, cmap='viridis', label='datapoints')


        # Plot the best fit function
        t = np.logspace(np.log10(self.tmin), np.log10(self.tmax), 500, base=10.)
        s = self.cdf(t, *self.optimal_parameters.detach().cpu().tolist())

        plt.plot(t, s, color='purple', label='Learnt unnormalized CFD')
        plt.title('CrossEntropy-Sigma Curve')
        plt.xlabel('Sigma')
        plt.ylabel('CrossEntropy')
        plt.xscale('log')
        plt.ylim(-0.2,7.2)
        plt.xlim(0.7,200)
        # plt.yscale('log')
        plt.legend()

        # Save the plot to a file
        plt.savefig(filename)
        plt.show()


class LogisticSchedule(AdaptiveSchedule):
    def __init__(self, tmin, tmax, mu, sigma, height, offset):
        super().__init__(tmin, tmax, mu, sigma, height, offset)

        self.update_distribution()

    def update_distribution(self):
        self.transforms = ComposeTransform([
            AffineTransform(loc=-self.offset / self.height, scale=1 / self.height),
            SigmoidTransform().inv,
            AffineTransform(loc=self.mu, scale=self.sigma)
        ])

        imin, imax = get_codomain(self.cdf, self.tmin, self.tmax, self.mu, self.sigma, self.height, self.offset)
        self.distribution = TransformedDistribution(Uniform(imin, imax), self.transforms)

    def cdf(self, x, mu, sigma, height, offset):
        return height / (1 + np.exp(-(x - mu) / sigma)) + offset


class CauchySchedule(AdaptiveSchedule):
    def __init__(self, tmin, tmax, mu, sigma, height, offset):
        super().__init__(tmin, tmax, mu, sigma, height, offset)

        self.update_distribution()

    def update_distribution(self):
        self.transforms = ComposeTransform([
            AffineTransform(loc=-self.offset / self.height, scale=1 / self.height),
            AffineTransform(loc=-np.pi / 2, scale=np.pi),
            TanTransform(),
            AffineTransform(loc=self.mu, scale=self.sigma)
        ])

        imin, imax = get_codomain(self.cdf, self.tmin, self.tmax, self.mu, self.sigma, self.height, self.offset)
        self.distribution = TransformedDistribution(Uniform(imin, imax), self.transforms)

    def cdf(self, x, mu, sigma, height, offset):
        return (np.arctan((x - mu) / sigma) / np.pi + 0.5) * height + offset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def get_codomain(function,tmin,tmax,*args):
    
    imax=function(tmax,*args)
    imin=function(tmin,*args)

    return imin,imax
    
    
    
from torch.distributions import Transform, ComposeTransform
from torch.distributions import constraints
class TanTransform(Transform):
    bijective = True
    domain = constraints.real
    codomain = constraints.real
    sign = 1

    def _call(self, x):
        return torch.tan(x)

    def _inverse(self, y):
        return torch.atan(y)

    def log_abs_det_jacobian(self, x, y):
        return 2 * torch.log(torch.abs(torch.cos(x)))