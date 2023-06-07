# Loss Function for Diffusion Model
# Original Source: https://github.com/acids-ircam/diffusion_models

import torch
import numpy as np

def make_beta_schedule(schedule='linear', n_timesteps=1000, start=1e-5, end=1e-2):
    '''This function allows to pick a beta schedule and its parameters.'''
    if schedule == 'linear':
        betas = torch.linspace(start, end, n_timesteps)
    elif schedule == "quad":
        betas = torch.linspace(start ** 0.5, end ** 0.5, n_timesteps) ** 2
    elif schedule == "sigmoid":
        betas = torch.linspace(-6, 6, n_timesteps)
        betas = torch.sigmoid(betas) * (end - start) + start
    return betas

def extract(input, t, x):
    '''This is a convenience function to pick a value at time <t> and reshape it
     to the batch size.'''
    shape = x.shape
    out = torch.gather(input, 0, t.to(input.device))
    reshape = [t.shape[0]] + [1] * (len(shape) - 1)
    return out.reshape(*reshape)

def p_sample(model, x, t, alphas, betas, one_minus_alphas_bar_sqrt):
    '''This function samples x_{t-1} given x and t, and the noise schedule
    parameters.
    Basically, it represents p_{theta}(x_{t-1} | x_t).
        
    In practice, it just executes the Equation at step 4 in Algorithm 2.'''
    # Just converting timestep <t> into a 1D tensor:
    t = torch.tensor([t])
    # Multiplier in front of the whole Equation:
    eps_factor = ((1 - extract(alphas, t, x)) / extract(one_minus_alphas_bar_sqrt, t, x))
    eps_theta = model(x, t)
    mean = (1 / extract(alphas, t, x).sqrt()) * (x - (eps_factor * eps_theta))
    z = torch.randn_like(x)
    sigma_t = extract(betas, t, x).sqrt()
    sample = mean + sigma_t * z
    return (sample)

def p_sample_loop(model, shape,n_steps,alphas,betas,one_minus_alphas_bar_sqrt):
    '''This function loops through the backward process, through the timesteps:
        T, ... t, t-1, ..1,
    and each time it samples the data from the distribution of that timestep
    using the data obtained from the previous iteration.
    
    In practice, it just executes steps 2--5 in Algorithm 2.'''    
    cur_x = torch.randn(shape)
    x_seq = [cur_x]
    for i in reversed(range(n_steps)):
        cur_x = p_sample(model, cur_x, i,alphas,betas,one_minus_alphas_bar_sqrt)
        x_seq.append(cur_x)
    return x_seq
