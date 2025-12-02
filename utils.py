import math
import torch
import torch.nn as nn
from functools import partial
from torch.autograd import Variable
from nested_dict import nested_dict
from collections import OrderedDict

if torch.cuda.is_available():
    import torch.cuda.comm as comm
    from torch.nn.parallel._functions import Broadcast
    from torch.nn.parallel import scatter, parallel_apply, gather
else:
    comm = Broadcast = scatter = parallel_apply = gather = None


def resolve_device(preferred='auto'):
    """Choose a torch.device, preferring CUDA, then MPS, then CPU."""
    if preferred != 'auto':
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


DEFAULT_DEVICE = resolve_device()


def set_default_device(device):
    global DEFAULT_DEVICE
    DEFAULT_DEVICE = torch.device(device)


def get_default_device():
    return DEFAULT_DEVICE

def cast(params, dtype='float', device=None):
    device = device or get_default_device()
    if isinstance(params, dict):
        return {k: cast(v, dtype, device) for k,v in params.items()}
    else:
        return getattr(params.to(device), dtype)()
        

def conv_params(ni,no,k=1,g=1):
    assert ni % g == 0
    return cast(nn.init.orthogonal_(torch.Tensor(no,ni//g,k,k)))

def linear_params(ni,no):
    return cast(dict(
        weight=torch.Tensor(no,ni).normal_(0,2/math.sqrt(ni)),
        bias=torch.zeros(no)))

def bnparams(n):
    return cast(dict(
#        weight=torch.Tensor(n).uniform_(),
        weight=torch.ones(n),
        bias=torch.zeros(n)))

def bnstats(n):
    return cast(dict(
        running_mean=torch.zeros(n),
        running_var=torch.ones(n)))

def data_parallel(f, input, params, stats, mode, device_ids, output_device=None):
    if output_device is None and device_ids:
        output_device = device_ids[0]

    if len(device_ids) == 1 or not torch.cuda.is_available():
        return f(input, params, stats, mode)

    def replicate(param_dict, g):
        replicas = [{} for d in device_ids]
        for k,v in param_dict.iteritems():
            for i,u in enumerate(g(v)):
                replicas[i][k] = u
        return replicas

    params_replicas = replicate(params, lambda x: Broadcast(device_ids)(x))
    stats_replicas = replicate(stats, lambda x: comm.broadcast(x, device_ids))

    replicas = [partial(f, params=p, stats=s, mode=mode)
                for p,s in zip(params_replicas, stats_replicas)]
    inputs = scatter([input], device_ids)
    outputs = parallel_apply(replicas, inputs)
    return gather(outputs, output_device)


def flatten_params(params):
    flat_params = OrderedDict()
    for keys, v in nested_dict(params).iteritems_flat():
        if v is not None:
            flat_params['.'.join(keys)] = Variable(v, requires_grad=True)
    return flat_params


def flatten_stats(stats):
    flat_stats = OrderedDict()
    for keys, v in nested_dict(stats).iteritems_flat():
        flat_stats['.'.join(keys)] = v
    return flat_stats


def matrix_norm_one(W):
    out = torch.abs(W)
    out = torch.sum(out, dim=0)
    out = torch.max(out)
    return out
