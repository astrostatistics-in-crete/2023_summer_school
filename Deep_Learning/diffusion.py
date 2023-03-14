import torch
import numpy as np


def q_sample(x_start, t, list_bar_alphas, device):
    """
    Diffuse the data (t == 0 means diffused for 1 step)
    """
    alpha_bar_t = list_bar_alphas[t]
    
    mean = alpha_bar_t*x_start
    cov = torch.eye(x_start.shape[0]).to(device)
    cov = cov*(1-alpha_bar_t)
    return torch.distributions.MultivariateNormal(loc=mean,covariance_matrix=cov).sample().to(device)


def denoise_with_mu(denoise_model, x_t, t, list_alpha, list_alpha_bar, DATA_SIZE, device):
    """
    Denoising function considering the denoising models tries to model the posterior mean
    """
    alpha_t = list_alpha[t]
    beta_t = 1 - alpha_t
    alpha_bar_t = list_alpha_bar[t]
    
    mu_theta = denoise_model(x_t,t)
    
    x_t_before = torch.distributions.MultivariateNormal(loc=mu_theta,covariance_matrix=torch.diag(beta_t.repeat(DATA_SIZE))).sample().to(device)
        
    return x_t_before


def posterior_q(x_start, x_t, t, list_alpha, list_alpha_bar, device):
    """
    calculate the parameters of the posterior distribution of q
    """
    beta_t = 1 - list_alpha[t]
    alpha_t = list_alpha[t]
    alpha_bar_t = list_alpha_bar[t]
    # alpha_bar_{t-1}
    alpha_bar_t_before = list_alpha_bar[t-1]
    
    # calculate mu_tilde
    first_term = x_start * torch.sqrt(alpha_bar_t_before) * beta_t / (1 - alpha_bar_t)
    second_term = x_t * torch.sqrt(alpha_t)*(1- alpha_bar_t_before)/ (1 - alpha_bar_t)
    mu_tilde = first_term + second_term
    
    # beta_t_tilde
    beta_t_tilde = beta_t*(1 - alpha_bar_t_before)/(1 - alpha_bar_t)
    
    cov = torch.eye(x_start.shape[0]).to(device)*(1-alpha_bar_t)
      
    return mu_tilde, cov


    
def position_encoding_init(n_position, d_pos_vec):
    ''' 
    Init the sinusoid position encoding table 
    n_position in num_timesteps and d_pos_vec is the embedding dimension
    '''
    # keep dim 0 for padding token position encoding zero vector
    position_enc = np.array([
        [pos / np.power(10000, 2*i/d_pos_vec) for i in range(d_pos_vec)]
        if pos != 0 else np.zeros(d_pos_vec) for pos in range(n_position)])

    position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2]) # dim 2i
    position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2]) # dim 2i+1
    return torch.from_numpy(position_enc).to(torch.float32)


class Denoising(torch.nn.Module):

    def __init__(self, x_dim, num_diffusion_timesteps):
        super(Denoising, self).__init__()

        self.linear1 = torch.nn.Linear(x_dim, x_dim)
        self.emb = position_encoding_init(num_diffusion_timesteps,x_dim)
        self.linear2 = torch.nn.Linear(x_dim, x_dim)
        self.linear3 = torch.nn.Linear(x_dim, x_dim)
        self.relu = torch.nn.ReLU()

    def forward(self, x_input, t):
        emb_t = self.emb[t]
        x = self.linear1(x_input+emb_t)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        return x
