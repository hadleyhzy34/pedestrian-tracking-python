import numpy as np
from scipy.optimize import linear_sum_assignment


def pedestrian_tracking(rectangles, projection_matrix,matches):
    #function model to track pedestrian
    num_object = rectangles.shape[0] #number of pedestrians detected
    center_2d = np.zeros((rectangles.shape[0],2)) #array to store center point of each pedestrian
    center_2d[:,0] = rectangles[:,0] + rectangles[:,2]//2
    center_2d[:,1] = rectangles[:,1] + rectangles[:,3]
    # print('center points positions are: \n',center_2d,center_2d.shape)

    #convert to 3d scene space
    p_matrix = projection_matrix[:,[0,1,3]]; #remove third column since z axis position are all zero
    p_matrix_1 = np.linalg.inv(p_matrix) #invert matrix
    # print('inverse matrix of projection matrix is: \n',p_matrix)
    center_2d_add_1 = np.append(center_2d.T,np.ones((1,num_object)),axis=0) #add one row with ones to match projection
    # print("center_2d_add_1 is: \n",center_2d_add_1)

    center_3d = np.dot(p_matrix_1,center_2d_add_1)
    center_3d = center_3d[0:2,:].T
    # print("3d points are: \n",center_3d)

    #cost matrix
    cost = np.zeros((matches.shape[0],matches.shape[0]))
    for i,match in enumerate(matches):
        for j,points in enumerate(center_3d):
            point = match[1:];
            cost[i][j] = np.linalg.norm(point-points) #distance between point of previous frame and current frame
    
    # print('cost matrix is: \n', cost)
    
    #hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(cost)
    # print('hungarian result is: \n', row_ind, col_ind)

    #update matches
    for i in row_ind:
        matches[i][1] = center_3d[col_ind[i]][0]
        matches[i][2] = center_3d[col_ind[i]][1]
    
    # print("updated matches is: \n", matches)
    return matches

