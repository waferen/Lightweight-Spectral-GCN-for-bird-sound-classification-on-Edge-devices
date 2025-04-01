import math
import torch
from torch.optim.optimizer import Optimizer, required
import math
import torch
from torch.optim.optimizer import Optimizer

class RAdam(Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.buffer = [[None, None, None] for _ in range(10)]  # Buffer for SMA values
        super(RAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RAdam, self).__setstate__(state)

    def step(self, closure=None):

        loss = closure() if closure is not None else None

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('RAdam does not support sparse gradients')

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=(1 - beta2))
                exp_avg.mul_(beta1).add_(grad, alpha=(1 - beta1))

                # Compute SMA (Smooth Moving Average)
                buffered = self.buffer[int(state['step'] % 10)]
                if state['step'] == buffered[0]:
                    N_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state['step']
                    beta2_t = beta2 ** state['step']
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma

                    # Update the step size based on SMA
                    if N_sma >= 5:
                        step_size = group['lr'] * math.sqrt((1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (N_sma_max - 2)) / (1 - beta1 ** state['step'])
                    else:
                        step_size = group['lr'] / (1 - beta1 ** state['step'])
                    buffered[2] = step_size

                # Apply weight decay
                if group['weight_decay'] != 0:
                    p_data_fp32.add_(-group['weight_decay'] * group['lr'])

                # Apply RAdam update rule
                if N_sma >= 5:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p_data_fp32.addcdiv_(exp_avg, denom, value=-step_size)
                else:
                    p_data_fp32.add_(-step_size, exp_avg)

                p.data.copy_(p_data_fp32)

        return loss


class RAdam_4step(Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, update_all=False, additional_four=False):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.update_all = update_all # whether update the first 4 steps
        self.additional_four = additional_four # whether use additional 4 steps for SGD
        self.buffer = [[None, None] for ind in range(10)]
        super(RAdam_4step, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RAdam_4step, self).__setstate__(state)

    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('RAdam_4step does not support sparse gradients')

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = -4 if self.additional_four else 0 #since this exp requires exactly 4 step, it is hard coded 
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=(1 - beta2))
                exp_avg.mul_(beta1).add_(grad, alpha=(1 - beta1))

                if state['step'] > 0:

                    state_step = state['step'] + 4 if self.additional_four else state['step'] #since this exp requires exactly 4 step, it is hard coded 

                    buffered = self.buffer[int(state_step % 10)]
                    if state_step == buffered[0]:
                        step_size = buffered[1]
                    else:
                        buffered[0] = state_step
                        beta2_t = beta2 ** state['step']

                        if state['step'] > 4: #since this exp requires exactly 4 step, it is hard coded 
                            N_sma_max = 2 / (1 - beta2) - 1
                            N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                            step_size = group['lr'] * math.sqrt((N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (N_sma_max - 2)) / (1 - beta1 ** state_step)
                        elif  self.update_all:
                            step_size = group['lr'] / (1 - beta1 ** state_step)
                        else:
                            step_size = 0
                        buffered[1] = step_size

                    if state['step'] > 4: #since this exp requires exactly 4 step, it is hard coded 
                        if group['weight_decay'] != 0:
                            p_data_fp32.add_(-group['weight_decay'] * group['lr'])
                        denom = (exp_avg_sq.sqrt() / math.sqrt(1 - beta2 ** state_step)).add_(group['eps'])
                        p_data_fp32.addcdiv_(exp_avg, denom, value=-step_size)
                        p.data.copy_(p_data_fp32)
                    elif self.update_all:
                        if group['weight_decay'] != 0:
                            p_data_fp32.add_(-group['weight_decay'] * group['lr'])
                        denom = (exp_avg_sq.sqrt() / math.sqrt(1 - beta2 ** state_step)) 
                        p_data_fp32.addcdiv_(exp_avg, denom, value=-step_size)
                        p.data.copy_(p_data_fp32)
                else:
                    state_step = state['step'] + 4 if self.additional_four else state['step'] #since this exp requires exactly 4 step, it is hard coded 

                    if group['weight_decay'] != 0:
                        p_data_fp32.add_(-group['weight_decay'] * 0.1)

                    step_size = 0.1 / (1 - beta1 ** state_step)
                    p_data_fp32.add_(-step_size, exp_avg)
                    p.data.copy_(p_data_fp32)

        return loss

class AdamW(Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, use_variance=True, warmup=4000):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, use_variance=use_variance, warmup=warmup)
        self.iter_idx = 0  # 初始化迭代次数
        print(f'======== Warmup: {warmup} =========')
        super(AdamW, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AdamW, self).__setstate__(state)

    def step(self, closure=None):
        self.iter_idx += 1
        grad_list = []
        mom_list = []
        mom_2rd_list = []

        loss = closure() if closure is not None else None

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('AdamW does not support sparse gradients, please consider SparseAdam instead')

                p_data_fp32 = p.data.float()
                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=(1 - beta2))
                exp_avg.mul_(beta1).add_(grad, alpha=(1 - beta1))

                denom = exp_avg_sq.sqrt().add_(group['eps'])
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                # Warmup Learning Rate Scheduling
                if group['warmup'] > state['step']:
                    scheduled_lr = 1e-6 + state['step'] * (group['lr'] - 1e-6) / group['warmup']
                else:
                    scheduled_lr = group['lr']

                step_size = scheduled_lr * math.sqrt(bias_correction2) / bias_correction1
                if group['weight_decay'] != 0:
                    p_data_fp32.add_(-group['weight_decay'] * scheduled_lr)

                p_data_fp32.addcdiv_(exp_avg, denom, value=-step_size)
                p.data.copy_(p_data_fp32)

        return loss
