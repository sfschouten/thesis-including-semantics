
from torch.optim.adagrad import Adagrad
from torch.optim.lr_scheduler import ReduceLROnPlateau


def add_constraints_to_job(job, mdmm_module):

    lambdas = [c.lmbda for c in mdmm_module]
    slacks = [c.slack for c in mdmm_module if hasattr(c, 'slack')]

    lr = next(g['lr'] for g in job.optimizer.param_groups if g['name'] == 'default')
    job.optimizer.add_param_group({'params': lambdas, 'lr': -lr})
    if len(slacks) > 0:
        job.optimizer.add_param_group({'params': slacks, 'lr': lr})
    
    # TODO: remove everything below once fixed in pytorch
    if isinstance(job.optimizer, Adagrad):
        pass #TODO
    scheduler = job.kge_lr_scheduler._lr_scheduler
    if isinstance(scheduler, ReduceLROnPlateau):
        scheduler.min_lrs += [scheduler.min_lrs[0]] * 2
