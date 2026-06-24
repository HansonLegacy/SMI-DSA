from torch.optim.lr_scheduler import *


def make_lr_scheduler(optimizer, lr_scheduler_spec):
    scheduler_args = dict(lr_scheduler_spec['args'])
    if lr_scheduler_spec['name'] == 'one_cycle_lr' and isinstance(scheduler_args.get('max_lr'), dict):
        max_lr_cfg = scheduler_args['max_lr']
        group_lrs = max_lr_cfg.get('groups', {})
        default_lr = max_lr_cfg.get('default')
        max_lr = []
        for param_group in optimizer.param_groups:
            group_name = param_group.get('group_name', 'default')
            if group_name in group_lrs:
                max_lr.append(group_lrs[group_name])
            elif default_lr is not None:
                max_lr.append(default_lr)
            else:
                max_lr.append(param_group['lr'])
        scheduler_args['max_lr'] = max_lr

    lr_scheduler = {
        'step_lr': StepLR,
        'one_cycle_lr': OneCycleLR,
        'cosine_lr': CosineAnnealingLR,
        'constant_lr': ConstantLR,

    }[lr_scheduler_spec['name']](optimizer, **scheduler_args)
    return lr_scheduler
