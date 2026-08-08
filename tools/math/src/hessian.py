
def get_Hinv_jacobian(subset_size_len, img_grad_x, img_grad_y):
    import numpy as np
    subset_size_len_half = int(0.5*(subset_size_len-1))
    H_mat = np.zeros((6,6), dtype=np.float64)
    dW_dP = np.array([(1, -subset_size_len_half, -subset_size_len_half, 0, 0, 0),\
                      (0, 0, 0, 1, -subset_size_len_half, -subset_size_len_half)], dtype=np.float64)
    dF = np.zeros((1, 2), dtype=np.float64)
    J_mat = np.zeros((subset_size_len,subset_size_len,6), dtype=np.float64) 
    for y in range(0,subset_size_len,1):
        for x in range(0,subset_size_len,1):
            dF[0][0] = img_grad_x[y][x] # x
            dF[0][1] = img_grad_y[y][x] # y
            dW_dP[0][1] = x - subset_size_len_half
            dW_dP[0][2] = y - subset_size_len_half
            dW_dP[1][4] = x - subset_size_len_half
            dW_dP[1][5] = y - subset_size_len_half
            J_temp = dF @ dW_dP
            J_mat[y][x][:] = J_temp
            H_mat += J_temp.T @ J_temp

    H_inv = np.linalg.inv(H_mat + 1e-8 * np.eye(6))
    return H_inv, J_mat
    
    
    
    
    
    