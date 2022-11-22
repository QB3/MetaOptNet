from ot.utils import proj_simplex

# TODO add test grad inner loss
def grad_inner_loss(params_dual, inputs, targets_one_hot, lambda2):
    result = inputs.mT @ (params_dual - targets_one_hot)
    result = inputs @ result / lambda2
    result += targets_one_hot
    return result

def prox(params):
    return proj_simplex(params.mT).mT

def inner_step_pgd_(params, inputs, targets_one_hot, stepsizes, lambda2):
    """One step on proximal gradient descent."""
    grads = grad_inner_loss(params, inputs, targets_one_hot, lambda2)
    res = prox(params - stepsizes * grads)
    return res
