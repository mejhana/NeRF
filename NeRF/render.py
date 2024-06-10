import torch

def normalize(arr):
    min_val = arr.min()
    max_val = arr.max()
    normalized_arr = (arr - min_val) / (max_val - min_val)
    return normalized_arr

def accumulated_transmittance(sigma, delta):
    """
    Compute the accumulated transmittance along each ray.
    params:
      sigma: [batch_size, nb_bins] tensor of sigma; 
      delta: [batch_size, nb_bins] tensor of deltas; δ_i = t_{i+1} - t_i
    returns:
        [batch_size, nb_bins] tensor of accumulated transmittances of t_i; t_i = Σ_{j=0}^{i-1} a_j
    """
    # compute the alpha values; a_i = 1 - e^(-σ_i * δ_i)
    alphas = 1 - torch.exp(-sigma * delta)  # [batch_size, nb_bins]

    # Compute the accumulated transmittance along each ray
    accumulated_transmittance = torch.cumprod(alphas, 1)
    T = torch.cat((torch.ones((accumulated_transmittance.shape[0], 1), device=alphas.device),
                      accumulated_transmittance[:, :-1]), dim=-1)
    return T.unsqueeze(2) * alphas.unsqueeze(2)

def render_rays(nerf_model, ray_origins, ray_directions, hn=0, hf=0.5, nb_bins=192):
    """
    Render the scene by sampling along each ray.
    params:
      nerf_model: NeRF model
      ray_origins: [batch_size, 3] tensor of ray origins
      ray_directions: [batch_size, 3] tensor of ray directions
      hn: lower bound of the near plane distance
      hf: upper bound of the far plane distance
      nb_bins: number of bins to sample along each ray
      
    returns:
        [batch_size, 3] tensor of pixel values
    """
    device = ray_origins.device
    
    # sample time along each ray
    t = torch.linspace(hn, hf, nb_bins, device=device).expand(ray_origins.shape[0], nb_bins)
    # Perturb sampling along each ray.
    mid = (t[:, :-1] + t[:, 1:]) / 2.
    lower = torch.cat((t[:, :1], mid), -1)
    upper = torch.cat((mid, t[:, -1:]), -1)
    u = torch.rand(t.shape, device=device)
    t = lower + (upper - lower) * u  # [batch_size, nb_bins]
    delta = torch.cat((t[:, 1:] - t[:, :-1], torch.tensor([1e10], device=device).expand(ray_origins.shape[0], 1)), -1)

    # Compute the 3D points along each ray
    # x = origin + t * direction
    x = ray_origins.unsqueeze(1) + t.unsqueeze(2) * ray_directions.unsqueeze(1)   # [batch_size, nb_bins, 3]
    # Expand the ray_directions tensor to match the shape of x
    ray_directions = ray_directions.expand(nb_bins, ray_directions.shape[0], 3).transpose(0, 1) 

    # Compute the colors and densities at each point along each ray
    colors, sigma = nerf_model(x.reshape(-1, 3), ray_directions.reshape(-1, 3))
    colors = colors.reshape(x.shape)
    sigma = sigma.reshape(x.shape[:-1])

    # Compute the accumulated transmittance along each ray
    weights = accumulated_transmittance(sigma, delta)

    # Compute the pixel values as a weighted sum of colors along each ray
    col = (weights * colors).sum(dim=1)
    weight_sum = weights.sum(-1).sum(-1)  # Regularisation for white background 
    # TODO: normalize the pixel values to [0, 1]
    col = col + 1- weight_sum.unsqueeze(-1)
    return normalize(col)