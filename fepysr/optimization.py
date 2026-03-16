import torch
import random

def data_iter(batch_size, features, labels):
    """
    Mini-batch data generator.
    """
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)

    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(
            indices[i: min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]
    
def squared_loss(y_hat, y):  
    """
    Calculate the un-summed version of Mean Squared Error (MSE).
    Note: This function returns an independent loss tensor for each sample, it does not calculate the mean.
    """
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

def sgd(params, lr, batch_size):  
    """Mini-batch stochastic gradient descent."""
    with torch.no_grad():
        for param in params:
            grad_max_abs =param.grad.abs().max()
            # print(grad_max_abs.data,(param.grad).data,i)
            # i+=1
            param -= lr * param.grad / batch_size/grad_max_abs
            param.grad.zero_()

def max_norm_sgd(params, lr, batch_size, eps=1e-9):  
    """
    Mini-batch stochastic gradient descent with gradient normalization.
    """
    with torch.no_grad():
        for param in params:
            if param.grad is not None:
                # Get the maximum absolute value of the current gradient
                grad_max_abs = param.grad.abs().max()
                param -= lr * param.grad / (batch_size * (grad_max_abs + eps))
                param.grad.zero_()
            
def svd_ridge_regression(X, y, alpha):
    """
    Calculate the closed-form solution of Ridge Regression using Singular Value Decomposition (SVD).
    
    Args:
        X (Tensor): Feature matrix
        y (Tensor): Label matrix
        alpha (float): L2 regularization coefficient (penalty term)
        
    Returns:
        Tensor: Calculated weight coefficient Xi
    """
    # Singular value decomposition: X = U * S * V^T
    U, s, Vt = torch.linalg.svd(X, full_matrices=False)
    
    # Extract singular values significantly greater than 0 to avoid floating-point truncation errors
    idx = s > 1e-15
    s_nnz = s[idx][:, None]
    
    UTy = torch.matmul(U.T, y.reshape(-1, 1))

    # Creating d on the CPU by default here would cause a device mismatch error
    d = torch.zeros((s.shape[0], 1), dtype=X.dtype, device=X.device)
    
    # Eigenvalue shrinkage formula for Ridge Regression under SVD: s / (s^2 + alpha)
    d[idx] = s_nnz / (s_nnz**2 + alpha)
    
    d_UT_y = d * UTy
    Xi = torch.matmul(Vt.T, d_UT_y)
    
    return Xi