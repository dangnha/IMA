import torch
import numpy as np
import torch.fft as fft
import torch.distributions.transforms as transforms
from torch.distributions.transforms import SigmoidTransform
from torch.distributions import TransformedDistribution
from scipy.stats import truncnorm

def none_aug(x):
    return x


def jitter(x, sigma=0.03):
    return x + np.random.normal(loc=0., scale=sigma, size=x.shape)


def hflip(x):    
    return x[::-1, :]


def vflip(x):
    x = x.reshape((1, x.shape[0], x.shape[1]))    
    mean_per_feature = x.mean(axis=1, keepdims=True)
    x = 2 * mean_per_feature - x
    return x[0]


def scaling(x, sigma=0.1):
    x = x.reshape((1, x.shape[0], x.shape[1]))
    factor = np.random.normal(loc=1., scale=sigma, size=(x.shape[0],x.shape[2]))
    return np.multiply(x, factor[:,np.newaxis,:])[0]


def window_warp(x, window_ratio=0.3, scales=[0.5, 2.0]):
    x = x.reshape((1, x.shape[0], x.shape[1]))
    warp_scales = np.random.choice(scales, x.shape[0])
    warp_size = np.ceil(window_ratio*x.shape[1]).astype(int)
    window_steps = np.arange(warp_size)
        
    window_starts = np.random.randint(low=1, high=x.shape[1]-warp_size-1, size=(x.shape[0])).astype(int)
    window_ends = (window_starts + warp_size).astype(int)
            
    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        for dim in range(x.shape[2]):
            start_seg = pat[:window_starts[i],dim]
            window_seg = np.interp(np.linspace(0, warp_size-1, num=int(warp_size*warp_scales[i])), window_steps, pat[window_starts[i]:window_ends[i],dim])
            end_seg = pat[window_ends[i]:,dim]
            warped = np.concatenate((start_seg, window_seg, end_seg))                
            ret[i,:,dim] = np.interp(np.arange(x.shape[1]), np.linspace(0, x.shape[1]-1., num=warped.size), warped).T
    return ret[0]


def window_slice(x, reduce_ratio=0.9):
    x = x.reshape((1, x.shape[0], x.shape[1]))
    target_len = np.ceil(reduce_ratio*x.shape[1]).astype(int)
    if target_len >= x.shape[1]:
        return x
    starts = np.random.randint(low=0, high=x.shape[1]-target_len, size=(x.shape[0])).astype(int)
    ends = (target_len + starts).astype(int)
    
    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        for dim in range(x.shape[2]):
            ret[i,:,dim] = np.interp(np.linspace(0, target_len, num=x.shape[1]), np.arange(target_len), pat[starts[i]:ends[i],dim]).T
    return ret[0]


def permutation(x, max_segments=5, seg_mode="equal"):
    x = x.reshape((1, x.shape[0], x.shape[1]))
    orig_steps = np.arange(x.shape[1])
    
    num_segs = np.random.randint(1, max_segments, size=(x.shape[0]))
    
    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        if num_segs[i] > 1:
            if seg_mode == "random":
                split_points = np.random.choice(x.shape[1]-2, num_segs[i]-1, replace=False)
                split_points.sort()
                splits = np.split(orig_steps, split_points)
            else:
                splits = np.array_split(orig_steps, num_segs[i])
            warp = np.concatenate(np.random.permutation(splits)).ravel()
            ret[i] = pat[warp]
        else:
            ret[i] = pat
    return ret[0]


# # # Original Mixup
def orig_mixup(batch_x_a, label_a, alpha):
    N, T, F = batch_x_a.shape 
    lamb = np.random.beta(alpha, alpha) 
    idx = torch.randperm(N).cuda()
    batch_x_b = batch_x_a[idx].clone()
    mix_batch = lamb*batch_x_a + (1-lamb)*batch_x_b
    return mix_batch, label_a, lamb, idx


def noise_function(P0, P1, P2, lamb):
    diff_P0_P1 = torch.norm(P1 - P0, dim=1, keepdim=True)
    diff_P0_P2 = torch.norm(P2 - P0, dim=1, keepdim=True)
    diff_P1_P2 = torch.norm(P2 - P1, dim=1, keepdim=True)

    noise_amplitude = 0.001 
    noise = noise_amplitude * (
        diff_P0_P1 * (1 - lamb) + diff_P0_P2 * lamb + diff_P1_P2 * lamb * (1 - lamb)
    )
    return noise * torch.randn_like(P0)


def fit_gaussian_and_sample(batch_data, mask_percentage=0.10):
    mean = batch_data.mean()
    std = batch_data.std()

    mask = torch.rand_like(batch_data, device=batch_data.device) < mask_percentage
    num_samples = mask.sum()

    sampled_values = torch.normal(mean, std, size=(num_samples,))
    sampled_values = sampled_values.to(batch_data.device)

    augmented_batch = batch_data.clone()
    augmented_batch[mask] = sampled_values
    return augmented_batch


def new_mixup(batch_x_a, label_a, alpha, emixup_vflip_rate=0.15, is_classification=False):
    N, T, F = batch_x_a.shape 
    lamb = 1-truncated_normal_()
    lamb = torch.from_numpy(lamb).to(batch_x_a.device)
    lamb = lamb.unsqueeze(0).unsqueeze(0)
    lamb = lamb.type_as(batch_x_a)
    idx = torch.randperm(N).to(batch_x_a.device)
    idx2 = torch.randperm(N).to(batch_x_a.device)

    batch_x_b = batch_x_a[idx].clone()
    batch_x_c = batch_x_a[idx2].clone()
    batch_x_a = fit_gaussian_and_sample(batch_x_a, mask_percentage=0.02) #? 0.01 and 0.005 
    batch_x_b = fit_gaussian_and_sample(batch_x_b, mask_percentage=0.02)
    batch_x_c = fit_gaussian_and_sample(batch_x_c, mask_percentage=0.02)

    mix_batch = (1-lamb)**2*batch_x_a + 2*(1-lamb)*lamb*batch_x_b + lamb**2*batch_x_c
    mix_batch = mix_batch + noise_function(batch_x_a, batch_x_b, batch_x_c, lamb)

    label_a = fit_gaussian_and_sample(label_a, mask_percentage=0.005)
    idx = (idx, idx2)
    return mix_batch, label_a, lamb, idx

def truncated_normal_():
    # Parameters for truncated normal distribution
    mean = 0.98
    std = 0.01
    lower, upper = 0.96, 1

    # Calculate the normalized bounds
    a, b = (lower - mean) / std, (upper - mean) / std

    # Generate random samples
    samples = truncnorm.rvs(a, b, loc=mean, scale=std, size=1)
    return samples