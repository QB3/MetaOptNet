import torch
from ot.utils import proj_simplex

# TODO add test grad inner loss
def grad_inner_loss(params_dual, inputs, targets_one_hot, lambda2, dual_reg=0):
    result = inputs.mT @ (params_dual - targets_one_hot)
    result = inputs @ result
    result += dual_reg * (params_dual - targets_one_hot)
    result /= lambda2
    result += targets_one_hot
    return result

# @torch.jit.script
def grad_i_inner_loss(
        params_dual, inputs, inputsT_params, inputsT_targets_one_hot, targets_one_hot,
        lambda2, dual_reg, i: int):
    result = inputsT_params - inputsT_targets_one_hot
    result = inputs[i, :] @ result
    result += dual_reg * (params_dual[i, :] - targets_one_hot[i, :])
    result /= lambda2
    result += targets_one_hot[i, :]
    return result

def prox(params):
    return proj_simplex(params.mT).mT

def inner_step_pgd_(
        params, inputs, targets_one_hot, stepsizes, lambda2, dual_reg=0):
    """One step on proximal gradient descent."""
    grads = grad_inner_loss(params, inputs, targets_one_hot, lambda2, dual_reg)
    res = prox(params - stepsizes * grads)
    return res


# @torch.jit.script
def inner_step_pcd_(
        params, inputs, inputsT_params, inputsT_targets_one_hot, targets_one_hot, stepsizes, lambda2,
        dual_reg):
    """One epochs of proximal gradient descent."""
    n_samples = inputs.shape[0]
    # TODO jit this?
    for i in range(n_samples):
        grads_i = grad_i_inner_loss(
            params, inputs, inputsT_params, inputsT_targets_one_hot,
            targets_one_hot, lambda2, dual_reg, i)
        params_i_old = params[i, :].clone()
        # TODO chose better stepsizes !!
        params[i, :] = proj_simplex(params[i, :] - stepsizes[i] * grads_i)
        inputsT_params += torch.outer(
            inputs[i, :], params[i, :] - params_i_old)
    return params, inputsT_params
