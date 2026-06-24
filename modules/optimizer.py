from torch.optim import *


def _build_param_groups(model, optimizer_spec):
    named_params = list(model.named_parameters())
    group_specs = optimizer_spec.get('param_groups', [])
    if not group_specs:
        return model.parameters()

    matched_names = set()
    param_groups = []
    default_group = []

    for group_idx, group_spec in enumerate(group_specs):
        prefixes = tuple(group_spec.get('prefixes', []))
        if not prefixes:
            continue

        group_params = []
        for name, param in named_params:
            if name in matched_names:
                continue
            if name.startswith(prefixes):
                group_params.append(param)
                matched_names.add(name)

        if not group_params:
            continue

        group_cfg = {
            k: v for k, v in group_spec.items()
            if k not in {'prefixes', 'name'}
        }
        group_cfg['params'] = group_params
        group_cfg['group_name'] = group_spec.get('name', f'group_{group_idx}')
        param_groups.append(group_cfg)

    for name, param in named_params:
        if name not in matched_names:
            default_group.append(param)

    if default_group:
        param_groups.append({
            'params': default_group,
            'group_name': 'default',
        })

    return param_groups


def make_optimizer(model, optimizer_spec):
    params = _build_param_groups(model, optimizer_spec)
    optimizer = {
        'sgd': SGD,
        'adam': Adam,
        'adamW': AdamW
    }[optimizer_spec['name']](params, **optimizer_spec['args'])
    return optimizer
