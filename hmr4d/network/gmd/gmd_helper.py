import torch
import torch.nn as nn
import torch.nn.functional as F
import inspect

from tqdm import tqdm
from pytorch_lightning.utilities import rank_zero_only
from diffusers.schedulers import DDIMScheduler, DDPMScheduler
from hmr4d.network.mas.schedulers import DDPM3DScheduler


class GmdHelper:
    def sch_get_scheduler(self, scheduler_type=None, scheduler_opt=None):
        scheduler_type = scheduler_type or self.args.scheduler_type
        scheduler_opt = scheduler_opt or self.args.scheduler_opt
        if scheduler_type == "ddim":
            return DDIMScheduler(**scheduler_opt)
        elif scheduler_type == "ddpm":
            return DDPMScheduler(**scheduler_opt)
        elif scheduler_type == "ddpm3d":
            return DDPM3DScheduler(**scheduler_opt)

    def sch_get_var(self, t):
        scheduler = self.te_scheduler
        if self.args.scheduler_type == "ddim":
            # "t_prev < 0" is handled in the DDIM scheduler
            t_prev = t - scheduler.config.num_train_timesteps // scheduler.num_inference_steps
            return scheduler._get_variance(t, t_prev)
        elif self.args.scheduler_type == "ddpm":
            return scheduler._get_variance(t)

    # ========== Sample ========== #

    def prepare_extra_step_kwargs(self, generator, eta=0.0):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.te_scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.te_scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def randlike_shape(self, shape, generator):
        batch_size = shape[0]
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if isinstance(generator, list):
            shape = (1,) + shape[1:]
            x = torch.cat(
                [torch.randn(shape, generator=generator[i], device=generator[i].device) for i in range(batch_size)]
            )
        else:
            x = torch.randn(shape, generator=generator, device=generator.device)
        return x

    def prepare_x(self, shape, generator):
        x = self.randlike_shape(shape, generator)

        # scale the initial noise by the standard deviation required by the scheduler
        assert self.te_scheduler.init_noise_sigma == 1.0, "Do not match the original implementaiton."
        x = x * self.te_scheduler.init_noise_sigma
        return x

    @rank_zero_only
    def get_prog_bar(self, total):
        return tqdm(total=total, leave=False)
