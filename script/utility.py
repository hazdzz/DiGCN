import numpy as np
import scipy.sparse as sp
from scipy.linalg import eig
import torch

def calc_chung_pr_gso(dir_adj, gso_type, alpha):
    if gso_type == 'rw_norm_adj' or gso_type == 'rw_norm_lap':
        gamma = 0
    elif gso_type == 'rw_renorm_adj' or gso_type == 'rw_renorm_lap':
        gamma = 1 

    n_vertex = dir_adj.shape[0]
    if sp.issparse(dir_adj):
        dir_adj = dir_adj + gamma * sp.identity(n_vertex, format='csc')
        row_out_sum = dir_adj.sum(axis=1).A1
        row_out_sum_inv = np.power(row_out_sum, -1)
        row_out_sum_inv[np.isinf(row_out_sum_inv)] = 0.
        deg_out_inv = sp.diags(row_out_sum_inv, format='csc')
    else:
        dir_adj = dir_adj + gamma * np.identity(n_vertex)
        row_out_sum = np.sum(dir_adj, axis=1)
        row_out_sum_inv = np.power(row_out_sum, -1)
        row_out_sum_inv[np.isinf(row_out_sum_inv)] = 0.
        deg_out_inv = np.diag(row_out_sum_inv)
    rw_norm_adj_out = deg_out_inv.dot(dir_adj)

    # alpha is the transportation probability
    # P_{PR} = (1 - α) * P + α / n * 1^{n × n} 
    # Inevitably, P_{PR} is a dense matrix
    if sp.issparse(rw_norm_adj_out):
        p_pr = ((1 - alpha) * rw_norm_adj_out).toarray() \
            + alpha / n_vertex * np.ones(shape=(n_vertex, n_vertex), dtype=rw_norm_adj_out.dtype)
    else:
        p_pr = ((1 - alpha) * rw_norm_adj_out) \
            + alpha / n_vertex * np.ones(shape=(n_vertex, n_vertex), dtype=rw_norm_adj_out.dtype)

    eigvals, eigvecs = eig(a=p_pr,left=True,right=False)
    eigvals, eigvecs = eigvals.real, eigvecs.real

    ind = eigvals.argsort()[-1]
    phi = eigvecs[:, ind].flatten()
    phi = phi / phi.sum()

    assert len(phi[phi < 0]) == 0
    phi_inv_sqrt = np.power(phi, -0.5)
    phi_inv_sqrt[np.isinf(phi_inv_sqrt)] = 0.
    phi_inv_sqrt[np.isnan(phi_inv_sqrt)] = 0.
    phi_inv_sqrt = np.diag(phi_inv_sqrt)
    phi_sqrt = np.power(phi, 0.5)
    phi_sqrt[np.isinf(phi_sqrt)] = 0.
    phi_sqrt[np.isnan(phi_sqrt)] = 0.
    phi_sqrt = np.diag(phi_sqrt)

    # A_{Chung} = 0.5 (Φ^{0.5} * P * Φ^{-0.5} + Φ^{-0.5} * P^{T} * Φ^{0.5})
    chung_dir_pr_norm_adj = 0.5 * (phi_sqrt.dot(p_pr).dot(phi_inv_sqrt) \
                            + phi_inv_sqrt.dot(p_pr.T).dot(phi_sqrt))
    
    # Setting a threshold, and sparse the normalized Chung's directed adjacency matrix.
    #chung_dir_pr_norm_adj = np.where(chung_dir_pr_norm_adj <= 1e-6, 0, chung_dir_pr_norm_adj)

    if gso_type == 'rw_norm_adj' or gso_type == 'rw_renorm_adj':
        gso = chung_dir_pr_norm_adj
    else:
        chung_dir_pr_norm_lap = np.eye(n_vertex) - chung_dir_pr_norm_adj
        gso = chung_dir_pr_norm_lap

    gso = gso.astype(dtype=np.float32)
    return gso

def calc_chung_ppr_gso(dir_adj, gso_type, alpha):
    if gso_type == 'rw_norm_adj' or gso_type == 'rw_norm_lap':
        gamma = 0
    elif gso_type == 'rw_renorm_adj' or gso_type == 'rw_renorm_lap':
        gamma = 1

    n_vertex = dir_adj.shape[0]
    if sp.issparse(dir_adj):
        dir_adj = dir_adj + gamma * sp.identity(n_vertex, format='csc')
        row_out_sum = dir_adj.sum(axis=1).A1
        row_out_sum_inv = np.power(row_out_sum, -1)
        row_out_sum_inv[np.isinf(row_out_sum_inv)] = 0.
        deg_out_inv = sp.diags(row_out_sum_inv, format='csc')
        rw_norm_adj_out = deg_out_inv.dot(dir_adj)
        rw_norm_adj_out = rw_norm_adj_out.toarray()
    else:
        dir_adj = dir_adj + gamma * np.identity(n_vertex)
        row_out_sum = np.sum(dir_adj, axis=1)
        row_out_sum_inv = np.power(row_out_sum, -1)
        row_out_sum_inv[np.isinf(row_out_sum_inv)] = 0.
        deg_out_inv = np.diag(row_out_sum_inv)
        rw_norm_adj_out = deg_out_inv.dot(dir_adj)

    p_ppr = np.zeros(shape=(n_vertex+1, n_vertex+1))
    p_ppr[0: -1, 0: -1] = (1 - alpha) * rw_norm_adj_out
    p_ppr[-1, 0: -1] = 1 / n_vertex
    p_ppr[0: -1, -1] = alpha
    p_ppr[-1, -1] = 0.

    eigvals, eigvecs = eig(a=p_ppr,left=True,right=False)
    eigvals, eigvecs = eigvals.real, eigvecs.real

    ind = eigvals.argsort()[-1]
    phi = eigvecs[:, ind]
    phi = phi[0: n_vertex].flatten()
    phi = phi / phi.sum()

    assert len(phi[phi < 0]) == 0
    phi_inv_sqrt = np.power(phi, -0.5)
    phi_inv_sqrt[np.isinf(phi_inv_sqrt)] = 0.
    phi_inv_sqrt[np.isnan(phi_inv_sqrt)] = 0.
    phi_inv_sqrt = np.diag(phi_inv_sqrt)
    phi_sqrt = np.power(phi, 0.5)
    phi_sqrt[np.isinf(phi_sqrt)] = 0.
    phi_sqrt[np.isnan(phi_sqrt)] = 0.
    phi_sqrt = np.diag(phi_sqrt)

    # A_{Chung} = 0.5 (Φ^{0.5} * P * Φ^{-0.5} + Φ^{-0.5} * P^{T} * Φ^{0.5})
    chung_dir_appr_norm_adj = 0.5 * (phi_sqrt.dot(rw_norm_adj_out).dot(phi_inv_sqrt) \
                            + phi_inv_sqrt.dot(rw_norm_adj_out.T).dot(phi_sqrt))
    
    # Setting a threshold, and sparse the normalized Chung's directed adjacency matrix.
    #chung_dir_pr_norm_adj = np.where(chung_dir_pr_norm_adj <= 1e-6, 0, chung_dir_pr_norm_adj)

    if gso_type == 'rw_norm_adj' or gso_type == 'rw_renorm_adj':
        gso = chung_dir_appr_norm_adj
    else:
        id = np.eye(n_vertex)
        chung_dir_pr_norm_lap = id - chung_dir_appr_norm_adj
        gso = chung_dir_pr_norm_lap

    gso = gso.astype(dtype=np.float32)
    return gso

def cnv_sparse_mat_to_coo_tensor(sp_mat, device):
    # convert a compressed sparse row (csr) or compressed sparse column (csc) matrix to a hybrid sparse coo tensor
    sp_coo_mat = sp_mat.tocoo()
    i = torch.from_numpy(np.vstack((sp_coo_mat.row, sp_coo_mat.col)))
    v = torch.from_numpy(sp_coo_mat.data)
    s = torch.Size(sp_coo_mat.shape)

    if sp_mat.dtype == np.complex64 or sp_mat.dtype == np.complex128:
        return torch.sparse_coo_tensor(indices=i, values=v, size=s, dtype=torch.complex64, device=device, requires_grad=False)
    elif sp_mat.dtype == np.float32 or sp_mat.dtype == np.float64:
        return torch.sparse_coo_tensor(indices=i, values=v, size=s, dtype=torch.float32, device=device, requires_grad=False)
    else:
        raise TypeError(f'ERROR: The dtype of {sp_mat} is {sp_mat.dtype}, not been applied in implemented models.')

def calc_accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double().sum()
    accuracy = correct / len(labels)

    return accuracy