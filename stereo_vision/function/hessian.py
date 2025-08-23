
def get_Hinv_jacobian(size, img_grad_x, img_grad_y):
    import numpy as np
    # Half of size length
    Len = int(0.5*(size-1))
    # Hessian matrix
    H = np.zeros((6,6), dtype=np.float64)
    # Jacobiab of reference subset warp_function (dW_dP)
    dW_dP = np.array([(1, -Len, -Len, 0, 0, 0),\
                      (0, 0, 0, 1, -Len, -Len)], dtype=np.float64)
    # Image gradient (F: computed using Sobel operator)
    dF = np.zeros((1, 2), dtype=np.float64)
    # Storage zone of Jacobiab of reference subset (F*dW_dP)
    J = np.zeros((size,size,6), dtype=np.float64) 
    # Compute jacobian of reference subset point(F*dW_dP), and then compute hessian matrix.
    for y in range(0,size,1):
        for x in range(0,size,1):
            dF[0][0] = img_grad_x[y][x] # x
            dF[0][1] = img_grad_y[y][x] # y
            dW_dP[0][1] = x-Len
            dW_dP[0][2] = y-Len
            dW_dP[1][4] = x-Len
            dW_dP[1][5] = y-Len
            # Jacobian matrix: J
            J_temp = dF @ dW_dP
            J[y][x][:] = J_temp
            H += J_temp.T @ J_temp
    # inverse of matrix H
    H_inv = np.linalg.inv(H + 1e-8 * np.eye(6))
    
    return H_inv, J
    
    
    
    
    
    