import tensorflow as tf

def huber_loss(x, delta=1.0):
    """Reference: https://en.wikipedia.org/wiki/Huber_loss"""
    return tf.where(
        tf.abs(x) < delta,
        tf.square(x) * 0.5,
        delta * (tf.abs(x) - 0.5 * delta)
    )

def l2_loss(x):
    return tf.reduce_sum(tf.square(x))

def minimize_and_clip(optimizer, objective, var_list, clip_val=10):
    """Minimized `objective` using `optimizer` w.r.t. variables in
    `var_list` while ensure the norm of the gradients for each
    variable is clipped to `clip_val`
    """
    gradients = optimizer.compute_gradients(objective, var_list=var_list)
    for i, (grad, var) in enumerate(gradients):
        if grad is not None:
            gradients[i] = (tf.clip_by_norm(grad, clip_val), var)
    return gradients, optimizer.apply_gradients(gradients)

def minimize(optimizer, objective, var_list):
    """Minimized `objective` using `optimizer` w.r.t. variables in
    `var_list`"""
    gradients = optimizer.compute_gradients(objective, var_list=var_list)
    return gradients, optimizer.apply_gradients(gradients)
