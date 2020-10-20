import torch
import torch.nn as nn
import numpy as np
from warnings import warn
import time

class RNN(nn.Module):
    def __init__(self, dims, noise_std, dt=0.5, 
                 nonlinearity='tanh', readout_nonlinearity=None,
                 g=None, wi_init=None, wrec_init=None, wo_init=None, brec_init=None, h0_init=None,
                 train_wi=False, train_wrec=True, train_wo=False, train_brec=False, train_h0=False, 
                 ML_RNN=False,
                ):
        """
        :param dims: list = [input_size, hidden_size, output_size]
        :param noise_std: float
        :param dt: float, integration time step
        :param nonlinearity: str, nonlinearity. Choose 'tanh' or 'id'
        :param readout_nonlinearity: str, nonlinearity. Choose 'tanh' or 'id'
        :param g: float, std of gaussian distribution for initialization
        :param wi_init: torch tensor of shape (input_dim, hidden_size)
        :param wo_init: torch tensor of shape (hidden_size, output_dim)
        :param wrec_init: torch tensor of shape (hidden_size, hidden_size)
        :param brec_init: torch tensor of shape (hidden_size)
        :param h0_init: torch tensor of shape (hidden_size)
        :param train_wi: bool
        :param train_wo: bool
        :param train_wrec: bool
        :param train_brec: bool
        :param train_h0: bool
        :param ML_RNN: bool; whether forward pass is ML convention f(Wr)
        """
        super(RNN, self).__init__()
        self.dims = dims
        input_size, hidden_size, output_size = dims
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.noise_std = noise_std
        self.dt = dt
        self.train_wi = train_wi
        self.train_wo = train_wo
        self.train_wrec = train_wrec
        self.train_brec = train_brec
        self.train_h0 = train_h0
        self.ML_RNN = ML_RNN
        
        # Either set g or choose initial parameters. Otherwise, there's a conflict!
        assert (g is not None) or (wrec_init is not None), "Choose g or initial wrec!"
        if (g is not None) and (wrec_init is not None):
            g_wrec = wrec_init.std() * np.sqrt(hidden_size)
            tol_g = 0.01
            if np.abs(g_wrec - g) > tol_g:
                warn("Nominal g and wrec_init disagree: g = %.2f, g_wrec = %.2f" % (g, g_wrec))
        self.g = g
        
        # Nonlinearity
        if nonlinearity == 'tanh':
            self.nonlinearity = torch.tanh
        elif nonlinearity == 'id':
            self.nonlinearity = lambda x: x
            if g is not None:
                if g > 1:
                    warn("g > 1. For a linear network, we need stable dynamics!")
        elif nonlinearity.lower() == 'relu':
            self.nonlinearity = nn.ReLU()
        elif nonlinearity == 'softplus':
            softplus_scale = 1 # Note that scale 1 is quite far from relu
            self.nonlinearity = lambda x: torch.log(1. + torch.exp(softplus_scale * x)) / softplus_scale
        elif type(nonlinearity) == str:
            raise NotImplementedError("Nonlinearity not yet implemented.")
        else:
            self.nonlinearity = nonlinearity
            
        # Readout nonlinearity
        if readout_nonlinearity is None:
            # Same as recurrent nonlinearity
            self.readout_nonlinearity = self.nonlinearity
        elif readout_nonlinearity == 'tanh':
            self.readout_nonlinearity = torch.tanh
        elif readout_nonlinearity == 'logistic':
            # Note that the range is [0, 1]. otherwise, 'logistic' is a scaled and shifted tanh
            self.readout_nonlinearity = lambda x: 1. / (1. + torch.exp(-x))
        elif readout_nonlinearity == 'id':
            self.readout_nonlinearity = lambda x: x
        elif type(readout_nonlinearity) == str:
            raise NotImplementedError("readout_nonlinearity not yet implemented.")
        else:
            self.readout_nonlinearity = readout_nonlinearity

        # Define parameters
        self.wi = nn.Parameter(torch.Tensor(input_size, hidden_size))
        if not train_wi:
            self.wi.requires_grad = False
        self.wrec = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        if not train_wrec:
            self.wrec.requires_grad = False
        self.wo = nn.Parameter(torch.Tensor(hidden_size, output_size))
        if not train_wo:
            self.wo.requires_grad = False
        self.brec = nn.Parameter(torch.Tensor(hidden_size))
        if not train_brec:
            self.brec.requires_grad = False
        self.h0 = nn.Parameter(torch.Tensor(hidden_size))
        if not train_h0:
            self.h0.requires_grad = False

        # Initialize parameters
        with torch.no_grad():
            if wi_init is None:
                self.wi.normal_()
            else:
                if type(wi_init) == np.ndarray:
                    wi_init = torch.from_numpy(wi_init)
                self.wi.copy_(wi_init)
            if wrec_init is None:
                self.wrec.normal_(std=g / np.sqrt(hidden_size))
            else:
                if type(wrec_init) == np.ndarray:
                    wrec_init = torch.from_numpy(wrec_init)
                self.wrec.copy_(wrec_init)
            if wo_init is None:
                self.wo.normal_(std=1 / hidden_size)
            else:
                if type(wo_init) == np.ndarray:
                    wo_init = torch.from_numpy(wo_init)
                self.wo.copy_(wo_init)
            if brec_init is None:
                self.brec.zero_()
            else:
                if type(brec_init) == np.ndarray:
                    brec_init = torch.from_numpy(brec_init)
                self.brec.copy_(brec_init)
            if h0_init is None:
                self.h0.zero_()
            else:
                if type(h0_init) == np.ndarray:
                    h0_init = torch.from_numpy(h0_init)
                self.h0.copy_(h0_init)
            
            
    def forward(self, input, return_dynamics=False, h_init=None):
        """
        :param input: tensor of shape (batch_size, #timesteps, input_dimension)
        Important: the 3 dimensions need to be present, even if they are of size 1.
        :param return_dynamics: bool
        :return: if return_dynamics=False, output tensor of shape (batch_size, #timesteps, output_dimension)
                 if return_dynamics=True, (output tensor, trajectories tensor of shape (batch_size, #timesteps, #hidden_units))
        """
        batch_size = input.shape[0]
        seq_len = input.shape[1]
        if h_init is None:
            h = self.h0
        else:
            h_init_torch = nn.Parameter(torch.Tensor(batch_size, self.hidden_size))
            h_init_torch.requires_grad = False
            # Initialize parameters
            with torch.no_grad():
                h = h_init_torch.copy_(torch.from_numpy(h_init))
        noise = torch.randn(batch_size, seq_len, self.hidden_size, device=self.wrec.device)
        output = torch.zeros(batch_size, seq_len, self.output_size, device=self.wrec.device)
        if return_dynamics:
            trajectories = torch.zeros(batch_size, seq_len, self.hidden_size, device=self.wrec.device)

        # simulation loop
        for i in range(seq_len):
            if self.ML_RNN:
                rec_input = self.nonlinearity(
                    h.matmul(self.wrec.t()) 
                    + input[:, i, :].matmul(self.wi)
                    + self.brec)
                     # Note that if noise is added inside the nonlinearity, the amplitude should be adapted to the slope...
                     # + np.sqrt(2. / self.dt) * self.noise_std * noise[:, i, :])
                h = ((1 - self.dt) * h 
                     + self.dt * rec_input
                     + np.sqrt(self.dt) * self.noise_std * noise[:, i, :])
                out_i = self.readout_nonlinearity(h.matmul(self.wo))
                
            else:
                rec_input = (
                    self.nonlinearity(h).matmul(self.wrec.t()) 
                    + input[:, i, :].matmul(self.wi) 
                    + self.brec)
                h = ((1 - self.dt) * h 
                     + self.dt * rec_input
                     + np.sqrt(self.dt) * self.noise_std * noise[:, i, :])
                out_i = self.readout_nonlinearity(h).matmul(self.wo)

            output[:, i, :] = out_i

            if return_dynamics:
                trajectories[:, i, :] = h

        if not return_dynamics:
            return output
        else:
            return output, trajectories

    
    

def loss_mse(output, target, mask):
    """
    Mean squared error loss
    :param output: torch tensor of shape (num_trials, num_timesteps, output_dim)
    :param target: idem
    :param mask: idem -- or torch tensor of shape (num_trials, num_timesteps, 1)
    :return: float
    """
    # If mask has the same shape as output:
    if output.shape == mask.shape:
        loss = (mask * (target - output).pow(2)).sum() / mask.sum()
    else:
        raise Exception("This is problematic...")
        output_dim = output.shape[-1]
        loss = (mask * (target - output).pow(2)).sum() / (mask.sum() * output_dim)
    # Take half:
    loss = 0.5 * loss
    return loss

def train(net, task, n_epochs, batch_size=32, learning_rate=1e-2, clip_gradient=None, cuda=False, rec_step=1, 
          optimizer='sgd', h_init=None, verbose=True):
    """
    Train a network
    :param net: nn.Module
    :param task: function; generates input, target, mask for a single batch
    :param n_epochs: int
    :param batch_size: int
    :param learning_rate: float
    :param clip_gradient: None or float, if not None the value at which gradient norm is clipped
    :param cuda: bool
    :param rec_step: int; record weights after these steps
    :return: res
    """
    # CUDA management
    if cuda:
        if not torch.cuda.is_available():
            print("Warning: CUDA not available on this machine, switching to CPU")
            device = torch.device('cpu')
        else:
            device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    net.to(device=device)
    
    # Optimizer
    if optimizer == 'sgd':
        optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)
    elif optimizer == 'adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    else:
        raise Exception("Optimizer not known.")
    
    # Save initial weights
    wi_init = net.wi.detach().numpy().copy()
    wrec_init = net.wrec.detach().numpy().copy()
    wo_init = net.wo.detach().numpy().copy()
    brec_init = net.brec.detach().numpy().copy()
    weights_init = [wi_init, wrec_init, wo_init, brec_init]
    
    # Record
    dim_rec = net.hidden_size
    dim_in = net.input_size
    dim_out = net.output_size
    n_rec_epochs = n_epochs // rec_step
    
    losses = np.zeros((n_epochs), dtype=np.float32)
    gradient_norm_sqs = np.zeros((n_epochs), dtype=np.float32)
    epochs = np.zeros((n_epochs))
    rec_epochs = np.zeros((n_rec_epochs))
    if net.train_wi:
        wis = np.zeros((n_rec_epochs, dim_in, dim_rec), dtype=np.float32)
    if net.train_wrec:
        wrecs = np.zeros((n_rec_epochs, dim_rec, dim_rec), dtype=np.float32)
    if net.train_wo:
        wos = np.zeros((n_rec_epochs, dim_rec, dim_out), dtype=np.float32)
    if net.train_brec:
        brecs = np.zeros((n_rec_epochs, dim_rec), dtype=np.float32)

    time0 = time.time()
    if verbose:
        print("Training...")
    for i in range(n_epochs):
        # Save weights (before update)
        if i % rec_step == 0:
            k = i // rec_step
            rec_epochs[k] = i
            if net.train_wi:
                wis[k] = net.wi.detach().numpy()
            if net.train_wrec:
                wrecs[k] = net.wrec.detach().numpy()
            if net.train_wo:
                wos[k] = net.wo.detach().numpy()
            if net.train_brec:
                brecs[k] = net.brec.detach().numpy()
        
        # Generate batch
        _input, _target, _mask = task(batch_size)
        # Convert training data to pytorch tensors
        _input = torch.from_numpy(_input)
        _target = torch.from_numpy(_target)
        _mask = torch.from_numpy(_mask)
        # Allocate
        input = _input.to(device=device)
        target = _target.to(device=device)
        mask = _mask.to(device=device)
        
        optimizer.zero_grad()
        output = net(input, h_init=h_init)
        loss = loss_mse(output, target, mask)
        
        # Gradient descent
        loss.backward()
        if clip_gradient is not None:
            torch.nn.utils.clip_grad_norm_(net.parameters(), clip_gradient)
        gradient_norm_sq = sum([(p.grad ** 2).sum() for p in net.parameters() if p.requires_grad])
        
        # Update weights
        optimizer.step()
        
        # These 2 lines important to prevent memory leaks
        loss.detach_()
        output.detach_()
        
        # Save
        epochs[i] = i
        losses[i] = loss.item()
        gradient_norm_sqs[i] = gradient_norm_sq
        
        if verbose:
            print("epoch %d / %d:  loss=%.6f, time=%.1f sec." % (i+1, n_epochs, np.mean(losses), time.time() - time0), end="\r")
    if verbose:
        print("\nDone. Training took %.1f sec." % (time.time() - time0))
    
    # Obtain gradient norm
    gradient_norms = np.sqrt(gradient_norm_sqs)
    
    # Final weights
    wi_last = net.wi.detach().numpy().copy()
    wrec_last = net.wrec.detach().numpy().copy()
    wo_last = net.wo.detach().numpy().copy()
    brec_last = net.brec.detach().numpy().copy()
    weights_last = [wi_last, wrec_last, wo_last, brec_last]
    
    # Weights throughout training: 
    weights_train = {}
    if net.train_wi:
        weights_train["wi"] = wis
    if net.train_wrec:
        weights_train["wrec"] = wrecs
    if net.train_wo:
        weights_train["wo"] = wos
    if net.train_brec:
        weights_train["brec"] = brecs
    
    res = [losses, gradient_norms, weights_init, weights_last, weights_train, epochs, rec_epochs]
    return res

        
def run_net(net, task, batch_size=32, return_dynamics=False, h_init=None):
    # Generate batch
    input, target, mask = task(batch_size)
    # Convert training data to pytorch tensors
    input = torch.from_numpy(input)
    target = torch.from_numpy(target)
    mask = torch.from_numpy(mask)
    with torch.no_grad():
        # Run dynamics
        if return_dynamics:
            output, trajectories = net(input, return_dynamics, h_init=h_init)
        else:
            output = net(input, h_init=h_init)
        loss = loss_mse(output, target, mask)
    res = [input, target, mask, output, loss]
    if return_dynamics:
        res.append(trajectories)
    res = [r.numpy() for r in res]
    return res