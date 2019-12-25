import torch
from copy import deepcopy
from itertools import chain
from collections import defaultdict, Iterable
import os
import logging

logger = logging.getLogger(__name__)


def save_ckpt(output_dir, state, step):
    """Save checkpoint"""
    ckpt_dir = os.path.join(output_dir, 'ckpt')
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    save_name = os.path.join(ckpt_dir, 'model_{}.pth'.format(step))

    torch.save(state, save_name)
    logger.info('save model: %s', save_name)


def load_ckpt(model, ckpt):
    """Load checkpoint"""
    state_dict = {}
    for name in ckpt:
        state_dict[name] = ckpt[name]
    model.load_state_dict(state_dict, strict=False)


def load_weights(model, checkpoint_path, gpu_id=0):
    # print('Loading checkpoint %s', checkpoint_path)
    # checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage.cuda(gpu_id))
    checkpoint = torch.load(checkpoint_path)
    model.module.load_state_dict(checkpoint['model'],strict=False)
    del checkpoint
    torch.cuda.empty_cache()
    return model


def freeze_params(m):
    """Freeze all the weights by setting requires_grad to False
    """
    for p in m.parameters():
        p.requires_grad = False


def load_optimizer_state_dict(optimizer, state_dict):
    # deepcopy, to be consistent with module API
    state_dict = deepcopy(state_dict)
    # Validate the state_dict
    groups = optimizer.param_groups
    saved_groups = state_dict['param_groups']

    if len(groups) != len(saved_groups):
        raise ValueError("loaded state dict has a different number of "
                         "parameter groups")
    param_lens = (len(g['params']) for g in groups)
    saved_lens = (len(g['params']) for g in saved_groups)
    if any(p_len != s_len for p_len, s_len in zip(param_lens, saved_lens)):
        raise ValueError("loaded state dict contains a parameter group "
                         "that doesn't match the size of optimizer's group")

    # Update the state
    id_map = {old_id: p for old_id, p in
              zip(chain(*(g['params'] for g in saved_groups)),
                  chain(*(g['params'] for g in groups)))}

    def cast(param, value):
        """Make a deep copy of value, casting all tensors to device of param."""
        if torch.is_tensor(value):
            # Floating-point types are a bit special here. They are the only ones
            # that are assumed to always match the type of params.
            if isinstance(param.data, (torch.FloatTensor, torch.cuda.FloatTensor,
                                       torch.DoubleTensor, torch.cuda.DoubleTensor,
                                       torch.HalfTensor, torch.cuda.HalfTensor)):  # param.is_floating_point():
                value = value.type_as(param.data)
            value = value.cuda(param.get_device()) if param.is_cuda else value.cpu()
            return value
        elif isinstance(value, dict):
            return {k: cast(param, v) for k, v in value.items()}
        elif isinstance(value, Iterable):
            return type(value)(cast(param, v) for v in value)
        else:
            return value

    # Copy state assigned to params (and cast tensors to appropriate types).
    # State that is not assigned to params is copied as is (needed for
    # backward compatibility).
    state = defaultdict(dict)
    for k, v in state_dict['state'].items():
        if k in id_map:
            param = id_map[k]
            state[param] = cast(param, v)
        else:
            state[k] = v

    # Update parameter groups, setting their 'params' value
    def update_group(group, new_group):
        new_group['params'] = group['params']
        return new_group

    param_groups = [
        update_group(g, ng) for g, ng in zip(groups, saved_groups)]
    optimizer.__setstate__({'state': state, 'param_groups': param_groups})
