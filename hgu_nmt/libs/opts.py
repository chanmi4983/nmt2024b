import numpy as np

class InverseRootSquareScheduler(object):
    # Referenced from 
    # https://github.com/pytorch/fairseq/blob/master/fairseq/optim/lr_scheduler/inverse_square_root_schedule.py

    def __init__(self, optimizer, init_lr, n_warmup_steps):
        self._optimizer = optimizer
        self.warmup_end_lr = init_lr
        self.warmup_init_lr = 1e-7
        self.n_warmup_steps = n_warmup_steps
        self.n_steps = 0

        self.lr_step = (self.warmup_end_lr - self.warmup_init_lr) / self.n_warmup_steps

        self.decay_factor = self.warmup_end_lr * (self.n_warmup_steps ** 0.5)

        self.set_lr(self.warmup_init_lr)

    def step_and_update_lr(self):
        "Step with the inner optimizer"
        self._update_learning_rate()
        self._optimizer.step()

    def step(self):
        self.step_and_update_lr()

    def zero_grad(self):
        "Zero out the gradients with the inner optimizer"
        self._optimizer.zero_grad()

    def _update_learning_rate(self):
        ''' Learning rate scheduling per step '''

        self.n_steps += 1
        lr = self.get_lr()
        self.set_lr(lr)

    def get_lr(self):
        if self.n_steps < self.n_warmup_steps:
            return self.warmup_init_lr + self.n_steps * self.lr_step
        else:
            return self.decay_factor * (self.n_steps ** -0.5)

    def set_lr(self, new_lr):
        for param_group in self._optimizer.param_groups:
            param_group['lr'] = new_lr

    def state_dict(self):
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)


class ScheduledOptim(object):
    '''A simple wrapper class for learning rate scheduling'''

    def __init__(self, optimizer, init_lr, d_model, n_warmup_steps):
        self._optimizer = optimizer
        self.init_lr = init_lr
        self.d_model = d_model
        self.n_warmup_steps = n_warmup_steps
        self.n_steps = 0

    def step_and_update_lr(self):
        "Step with the inner optimizer"
        self._update_learning_rate()
        self._optimizer.step()

    def step(self):
        self.step_and_update_lr()

    def zero_grad(self):
        "Zero out the gradients with the inner optimizer"
        self._optimizer.zero_grad()
        
# 이 부분 논문과 비교 확인
    def _get_lr_scale(self):
        d_model = self.d_model
        n_steps, n_warmup_steps = self.n_steps, self.n_warmup_steps
        return (d_model ** -0.5) * min(n_steps ** (-0.5), n_steps * n_warmup_steps ** (-1.5))

    def _update_learning_rate(self):
        ''' Learning rate scheduling per step '''

        self.n_steps += 1
        # 이 부분 확인(11/28)
        lr = self.init_lr * self._get_lr_scale()
        #lr = self._get_lr_scale()
        
        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr

    def state_dict(self):
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)


class ScheduledOptim_old(object):
    def __init__(self, optimizer, start_steps=10000, slope=1e-7):
        self.optimizer = optimizer
        self.slope = slope 
        self.start_steps = start_steps
        self.n_current_steps = 0

    def step(self):
        self.optimizer.step()
        self.update_learning_rate()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def update_learning_rate(self):
        self.n_current_steps += 1

        over_steps = self.n_current_steps - self.start_steps
        if over_steps > 0 and over_steps * self.slope < 0.01:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * (1-self.slope)
