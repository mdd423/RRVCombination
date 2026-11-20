import scipy.sparse
import numpy as np

def irls_method(rvs,err,steps,lambda_offsets=1e-5,lambda_rvs=1e-5,Q=3):
    '''
    Iteratively Reweighted Least Squares Method for RV and Offset Estimation
    Parameters
    ----------
    rvs : array-like
        Radial velocities for each observation
    err : array-like
        Errors associated with each radial velocity measurement
    time : array-like
        Time of each observation
    steps : int
        Number of iterations for the IRLS method
    lambda_offsets : float, optional
        Regularization parameter for offsets, by default 1e-5
    lambda_rvs : float, optional
        Regularization parameter for radial velocities, by default 1e-5
    Q : float, optional
        Parameter for the reweighting function, by default 3
    Returns
    -------
    opt_beta : array-like
        Optimized parameters for radial velocities and offsets
    param_info : array-like
        Information associated with each optimized parameter
    reweight : array-like
        Reweighting matrix used in the IRLS method
    '''
    m,n = rvs.shape
    info = 1/err**2
    mask = np.isnan(rvs) | np.isinf(rvs) | np.isnan(info) | np.isinf(info)
    info[mask] = 0.0
    # rvs[mask] = 0.0
    
    info_mat = scipy.sparse.spdiags(info.flatten()[None,:],diags=[0],m=m*n,n=n*m)

    ref_epoch   = np.argmax(info.sum(axis=1))
    offset_init = rvs.mean(axis=1)
    offset_init -= offset_init[ref_epoch]
    opt_beta = scipy.sparse.vstack((rvs[ref_epoch,:][:,None],offset_init[:,None]))
    design_mat = scipy.sparse.hstack((scipy.sparse.kron(np.ones((m, 1)),scipy.sparse.eye(n,n)),\
                                      scipy.sparse.kron(scipy.sparse.eye(m,m),np.ones((n, 1)))))

    row_inds = np.concatenate((np.zeros(n),np.ones(m)))
    col_inds = np.concatenate((np.arange(0,n,dtype=int),np.arange(n,m+n,dtype=int)))
    data_reg = np.concatenate((np.ones(n)/n,np.ones(m)/m))
    reg_rows = scipy.sparse.csr_array((data_reg, (row_inds, col_inds)), shape=(2, n+m))
    
    design_mat = scipy.sparse.vstack((design_mat,reg_rows)).tocsr()
    
    # check X.T @ X singular values, eigenvalues
    # Should be 2 zeros for 2 deg of freedom
    def reweighting(info_mat,y,design_mat,beta,Q):
        res = y[:,None] - (design_mat@beta)
        temp = Q**2/ (Q**2 + (res * (info_mat @ res)).T)

        out = scipy.sparse.spdiags(temp,diags=[0],m=y.shape[0],n=y.shape[0])
        # print(out.shape,'out',np.sum(out > 0))
        return out

    v = np.concatenate((rvs.flatten(),np.zeros((2))))
    reweight = scipy.sparse.eye(m=info_mat.shape[0],n=info_mat.shape[1])
    for step in range(steps):
        # print(opt_beta.shape,opt_beta)
        
        total_weight = scipy.sparse.block_diag((reweight*info_mat,\
                                                scipy.sparse.csr_matrix([[lambda_rvs]]),\
                                                scipy.sparse.csr_matrix([[lambda_offsets]])))
        # print(total_weight.shape,np.sum(total_weight > 0),np.sum(reweight > 0),np.sum(info_mat > 0))
        
        opt_beta = scipy.sparse.linalg.lsqr(design_mat.T @ total_weight @ design_mat, design_mat.T @ total_weight @ v)[0][:,None]
        reweight = reweighting(info_mat,v[:-2],design_mat[:-2,:],opt_beta,Q=Q)
        # opt_beta = opt_beta
    param_info = np.array(((reweight*info_mat) @ design_mat[:-2,:]).sum(0)).squeeze()
    return opt_beta[:n].squeeze(),opt_beta[n:].squeeze(),param_info[:n],param_info[n:],(reweight).diagonal(0).reshape(m,n) #np.array(reweight.diagonal(0)).squeeze()

def irls_method_lsocv(rvs,err,steps,lambda_offsets=1e-5,lambda_rvs=1e-5,Q=3,n_sets=5):
    '''
    Leave-Some-Out Cross Validation using Iteratively Reweighted Least Squares Method for RV and Offset Estimation
    Parameters
    ----------
    rvs : array-like
        Radial velocities for each observation
    err : array-like
        Errors associated with each radial velocity measurement
    time : array-like
        Time of each observation
    steps : int
        Number of iterations for the IRLS method
    lambda_offsets : float, optional
        Regularization parameter for offsets, by default 1e-5
    lambda_rvs : float, optional
        Regularization parameter for radial velocities, by default 1e-5
    Q : float, optional
        Parameter for the reweighting function, by default 3
    Returns
    -------
    prediction : array-like
        Prediction on all RV from each of the Leave-One-Out Cross Validation process
    '''
    
    m,n = rvs.shape

    rand_n = np.random.randint(0,n_sets,n)
    rand_m = np.random.randint(0,n_sets,m)
    
    info = 1/err**2
    ref_epoch   = np.argmax(info.sum(axis=1))
    offset_init = rvs.mean(axis=1)
    offset_init -= offset_init[ref_epoch]
    opt_beta = scipy.sparse.vstack((rvs[ref_epoch,:][:,None],offset_init[:,None]))
    

    prediction_rvs = np.zeros((m,n))
    for ii in range(n_sets):
        for jj in range(n_sets):
            info = 1/err**2
            mask = np.isnan(rvs) | np.isinf(rvs) | np.isnan(info) | np.isinf(info)
            print(ii,jj)
            m_mesh, n_mesh = np.meshgrid(np.where(rand_m == jj),np.where(rand_n == ii))
            mask[m_mesh, n_mesh] = True
        
            info[mask] = 0.0
            
            info_mat = scipy.sparse.spdiags(info.flatten()[None,:],diags=[0],m=m*n,n=n*m)
        
            design_mat = scipy.sparse.hstack((scipy.sparse.kron(np.ones((m, 1)),scipy.sparse.eye(n,n)),\
                                              scipy.sparse.kron(scipy.sparse.eye(m,m),np.ones((n, 1)))))
        
            row_inds = np.concatenate((np.zeros(n),np.ones(m)))
            col_inds = np.concatenate((np.arange(0,n,dtype=int),np.arange(n,m+n,dtype=int)))
            data_reg = np.concatenate((np.ones(n)/n,np.ones(m)/m))
            reg_rows = scipy.sparse.csr_array((data_reg, (row_inds, col_inds)), shape=(2, n+m))
            
            design_mat = scipy.sparse.vstack((design_mat,reg_rows)).tocsr()
            
            # check X.T @ X singular values, eigenvalues
            # Should be 2 zeros for 2 deg of freedom
            def reweighting(info_mat,y,design_mat,beta,Q):
                res = y[:,None] - (design_mat@beta)
                temp = Q**2/ (Q**2 + (res * (info_mat @ res)).T)
        
                out = scipy.sparse.spdiags(temp,diags=[0],m=y.shape[0],n=y.shape[0])
                # print(out.shape,'out',np.sum(out > 0))
                return out
        
            v = np.concatenate((rvs.flatten(),np.zeros((2))))
            reweight = scipy.sparse.eye(m=info_mat.shape[0],n=info_mat.shape[1])
            for step in range(steps):
                # print(opt_beta.shape,opt_beta)
                
                total_weight = scipy.sparse.block_diag((reweight*info_mat,\
                                                        scipy.sparse.csr_matrix([[lambda_rvs]]),\
                                                        scipy.sparse.csr_matrix([[lambda_offsets]])))
                # print(total_weight.shape,np.sum(total_weight > 0),np.sum(reweight > 0),np.sum(info_mat > 0))
                
                opt_beta = scipy.sparse.linalg.lsqr(design_mat.T @ total_weight @ design_mat, design_mat.T @ total_weight @ v)[0][:,None]
                reweight = reweighting(info_mat,v[:-2],design_mat[:-2,:],opt_beta,Q=Q)
                # opt_beta = opt_beta
            param_info = np.array(((reweight*info_mat) @ design_mat[:-2,:]).sum(0)).squeeze()
            def prediction():
                rvs = design_mat @ opt_beta
                rvs = rvs[:-2].reshape(m,n)
                return rvs
            prediction_rvs[m_mesh, n_mesh] = prediction()[m_mesh, n_mesh]
                
    return prediction_rvs
