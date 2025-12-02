import random
import numpy as np
import torch
from torch.optim.optimizer import Optimizer, required

from gutils import qr_retraction

epsilon = 1e-8


def explicit3(A, B):
    e = ((A ** 2 + A * B + B ** 2) / 3) ** 0.5
    a = 2 / (2 * e ** 3 + A ** 2 * B + B ** 2 * A)
    p = [a * (A ** 2 + A * B + B ** 2), -a]
    err = (2 * e ** 3 - A ** 2 * B - B ** 2 * A) / (2 * e ** 3 + A ** 2 * B + B ** 2 * A)
    return p, err


def cans_retraction(X, iters=1):
    """Polar retraction based on CANS orthogonalization."""
    shape = X.shape
    assert X.ndim >= 2
    transposed = False
    if shape[-2] > shape[-1]:
        X = X.mT
        transposed = True

    norm = (torch.norm(X) ** 2 - X.shape[-2] + 1) ** 0.5
    X = X / norm
    left = 1.0 / norm
    right = 1.0
    for _ in range(iters):
        coefs, err = explicit3(left, right)
        XXT = X @ X.mT
        XXTX = XXT @ X
        c1, c3 = coefs
        X = c1 * X + c3 * XXTX
        left, right = 1 - err, 1 + err

    if transposed:
        X = X.mT
    return X


def _stiefel_like(X):
    X_flat = X.view(X.size()[0], -1)
    X_flat = X_flat / (X_flat.norm(dim=1, keepdim=True) + 1e-7)
    return X_flat.t()


class CANS_SGD(Optimizer):
    r"""SGD variant with CANS retraction for Stiefel-constrained params."""

    def __init__(
        self,
        params,
        lr=required,
        momentum=0,
        dampening=0,
        weight_decay=0,
        nesterov=False,
        stiefel=True,
        omega=0,
        grad_clip=None,
        use_qr=False,
        retraction_iters=1,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
            stiefel=stiefel,
            omega=omega,
            grad_clip=grad_clip,
            retraction_iters=retraction_iters,
            use_qr=use_qr,
        )
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            momentum = group['momentum']
            stiefel = group['stiefel']

            for p in group['params']:
                if p.grad is None:
                    continue

                X = _stiefel_like(p.data)
                if stiefel and X.size()[1] <= X.size()[0]:
                    weight_decay = group['weight_decay']
                    dampening = group['dampening']
                    nesterov = group['nesterov']

                    if random.randint(1, 101) == 1:
                        X = qr_retraction(X.T).T

                    g = p.grad.data.view(p.size()[0], -1)
                    lr = group['lr']

                    param_state = self.state[p]
                    V = param_state.get('momentum_buffer')
                    if V is None:
                        V = torch.zeros_like(g.T, device=p.device)
                        param_state['momentum_buffer'] = V
                    V.mul_(momentum).add_(-g.T)  # update momentum
                    VX = V.T @ X
                    V_new = V - 0.5 * X @ (VX + VX.T)

                    if group['use_qr']:
                        p_new = qr_retraction((X + lr * V_new).T)
                    else:
                        p_new = cans_retraction(X + lr * V_new, iters=group['retraction_iters']).T

                    p.data.copy_(p_new.view_as(p))
                    V.copy_(V_new)

                else:
                    d_p = p.grad.data
                    if group['weight_decay'] != 0:
                        d_p = d_p.add(group['weight_decay'], p.data)
                    if momentum != 0:
                        param_state = self.state[p]
                        buf = param_state.get('momentum_buffer')
                        if buf is None:
                            buf = d_p.clone().detach()
                            param_state['momentum_buffer'] = buf
                        else:
                            buf.mul_(momentum).add_(1 - dampening, d_p)
                        if nesterov:
                            d_p = d_p.add(momentum, buf)
                        else:
                            d_p = buf

                    p.data.add_(-group['lr'], d_p)

        return loss


class CANS_Adam(Optimizer):
    r"""Adam variant with CANS retraction for Stiefel-constrained params."""

    def __init__(
        self,
        params,
        lr=required,
        momentum=0,
        dampening=0,
        weight_decay=0,
        nesterov=False,
        stiefel=True,
        beta2=0.99,
        epsilon=1e-8,
        omega=0,
        grad_clip=None,
        retraction_iters=1,
        use_qr=False,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
            stiefel=stiefel,
            beta2=beta2,
            epsilon=epsilon,
            omega=omega,
            grad_clip=grad_clip,
            retraction_iters=retraction_iters,
            use_qr=use_qr,
        )
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            stiefel = group['stiefel']
            beta1 = group['momentum']
            beta2 = group['beta2']
            epsilon = group['epsilon']

            for p in group['params']:
                if p.grad is None:
                    continue

                X = _stiefel_like(p.data)
                if stiefel and X.size()[1] <= X.size()[0]:
                    if random.randint(1, 101) == 1:
                        X = qr_retraction(X.T).T

                    g = p.grad.data.view(p.size()[0], -1)

                    param_state = self.state[p]
                    m = param_state.get('m_buffer')
                    v = param_state.get('v_buffer')
                    if m is None or v is None:
                        size = p.size()
                        m = torch.zeros((int(np.prod(size[1:])), size[0]), device=p.device, dtype=p.dtype)
                        v = torch.zeros(1, device=p.device, dtype=p.dtype)
                        param_state['m_buffer'] = m
                        param_state['v_buffer'] = v
                        param_state['beta1_power'] = beta1
                        param_state['beta2_power'] = beta2

                    beta1_power = param_state['beta1_power']
                    beta2_power = param_state['beta2_power']

                    mnew = beta1 * m + (1.0 - beta1) * g.T
                    vnew = beta2 * v + (1.0 - beta2) * (torch.norm(g) ** 2)

                    mnew_hat = mnew / (1 - beta1_power)
                    vnew_hat = vnew / (1 - beta2_power)

                    MX = mnew_hat.T @ X
                    mnew = mnew_hat - 0.5 * X @ (MX + MX.T)
                    lr = group['lr']

                    if group['use_qr']:
                        p_new = qr_retraction((X - lr * mnew / vnew_hat.add(epsilon).sqrt()).T)
                    else:
                        p_new = cans_retraction(
                            X - lr * mnew / vnew_hat.add(epsilon).sqrt(),
                            iters=group['retraction_iters'],
                        ).T

                    p.data.copy_(p_new.view_as(p))
                    m.copy_(mnew * (1 - beta1_power))
                    v.copy_(vnew)

                    param_state['beta1_power'] *= beta1
                    param_state['beta2_power'] *= beta2

                else:
                    momentum = group['momentum']
                    weight_decay = group['weight_decay']
                    dampening = group['dampening']
                    nesterov = group['nesterov']
                    d_p = p.grad.data
                    if weight_decay != 0:
                        d_p = d_p.add(weight_decay, p.data)
                    if momentum != 0:
                        param_state = self.state[p]
                        buf = param_state.get('momentum_buffer')
                        if buf is None:
                            buf = d_p.clone().detach()
                            param_state['momentum_buffer'] = buf
                        else:
                            buf.mul_(momentum).add_(1 - dampening, d_p)
                        if nesterov:
                            d_p = d_p.add(momentum, buf)
                        else:
                            d_p = buf

                    p.data.add_(-group['lr'], d_p)

        return loss
