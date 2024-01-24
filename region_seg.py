import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy.ndimage as ndimg

def region_seg(I,init_mask,max_its,alpha = 0.2,display = True):
    """
    Region Based Active Contour Segmentation
    seg = region_seg(I,init_mask,max_its,alpha,display)
    Inputs: I           2D image
             init_mask   Initialization (1 = foreground, 0 = bg)
             max_its     Number of iterations to run segmentation for
             alpha       (optional) Weight of smoothing term
                           higer = smoother.  default = 0.2
             display     (optional) displays intermediate outputs
                           default = true
    Outputs: seg        Final segmentation mask (1=fg, 0=bg)
    Description: This code implements the paper: "Active Contours Without
    Edges" By Chan Vese. This is a nice way to segment images whose
    foregrounds and backgrounds are statistically different and homogeneous.
    """
    #-- Insure image is 2D double matrix
    if len(I.shape) != 2:
        I = np.float32(cv2.cvtColor(I, cv2.COLOR_BGR2GRAY))
    #-- Create a signed distance map (SDF) from mask
    distT1 = ndimg.distance_transform_edt(init_mask)
    distT2 = ndimg.distance_transform_edt(1-init_mask)
    phi = -distT1 + distT2 - init_mask - 0.5
    #-- main loop
    for its in range(max_its):
        # Get the curve narrow
        idx = np.ravel_multi_index(np.array(np.where((phi <= 1.2) & (phi >= -1.2))),phi.shape)
        #-- find interior and exterior mean
        upts = (phi <= 0)
        vpts = (phi > 0)
        u = np.sum(I[upts])/(len(I[upts])+np.finfo(float).eps) # interior mean
        v = np.sum(I[vpts])/(len(I[vpts])+np.finfo(float).eps) # exterior mean
        F = np.square(I.flatten()[idx]-u)-np.square(I.flatten()[idx]-v) # force from image information
        curvature = get_curvature(phi,idx) # force from curvature penalty
        #-- gradient descent to minimize energy
        dphidt = np.divide(F,np.max(np.abs(F))) + alpha*curvature
        #-- maintain the CFL condition
        dt = 0.45/(np.max(dphidt)+np.finfo(float).eps)
        #-- evolve the curve
        phi.flatten()[idx] = phi.flatten()[idx] + dt*dphidt
        #-- Keep SDF smooth
        phi = sussman(phi, 0.5)
        #-- intermediate output
        if (display>0) and (its%20 == 0): 
            plt.figure()
            plt.imshow(I, vmin = 0, vmax = 255, extent = (0.5,I.shape[0]+0.5,I.shape[1]+0.5,0.5))
            plt.contour(phi,(0,0),colors = 'green',linewidths = 4)
            plt.contour(phi,(0,0),colors = 'black',linewidths = 2)
            plt.title(f"{its} Iterations")
            plt.show()
    #-- final output
    if display:
        plt.figure()
        plt.imshow(I, vmin = 0, vmax = 255, extent = (0.5,I.shape[0]+0.5,I.shape[1]+0.5,0.5))
        plt.contour(phi,(0,0),colors = 'green',linewidths = 4)
        plt.contour(phi,(0,0),colors = 'black',linewidths = 2)
        plt.title(f"{its} Iterations")
        plt.show()
    #-- make mask from SDF
    seg = phi <= 0  # Get mask from levelset
    return seg



def get_curvature(phi,idx):
    dimy, dimx = phi.shape[0],phi.shape[1]
    if idx.shape:
        [x, y] = np.unravel_index(idx, (dimx,dimy)) # get subscripts
        #-- get subscripts of neighbors
        ym1 = y-1
        xm1 = x-1
        yp1 = y+1
        xp1 = x+1
        #-- bounds checking  
        ym1[ym1 < 0] = 0
        xm1[xm1 < 0] = 0              
        yp1[yp1 > dimy-1] = dimy-1
        xp1[xp1 > dimx-1] = dimx-1
        #-- get indexes for 8 neighbors
        idup = np.ravel_multi_index((x,yp1), phi.shape)
        iddn = np.ravel_multi_index((x,ym1), phi.shape)
        idlt = np.ravel_multi_index((xm1,y), phi.shape)
        idrt = np.ravel_multi_index((xp1,y), phi.shape)
        idul = np.ravel_multi_index((xm1,yp1), phi.shape)
        idur = np.ravel_multi_index((xp1,yp1), phi.shape)
        iddl = np.ravel_multi_index((xm1,ym1), phi.shape)
        iddr = np.ravel_multi_index((xp1,ym1), phi.shape)
        #-- get central derivatives of SDF at x,y
        phi_fl = phi.flatten()
        phi_x  = -phi_fl[idlt] + phi_fl[idrt]
        phi_y  = -phi_fl[iddn] + phi_fl[idup]
        phi_xx = phi_fl[idlt] - 2*phi_fl[idx]  + phi_fl[idrt]
        phi_yy = phi_fl[iddn] - 2*phi_fl[idx] + phi_fl[idup]
        phi_xy = -0.25*phi_fl[iddl] - 0.25*phi_fl[idur] + 0.25*phi_fl[iddr] + 0.25*phi_fl[idul]
        phi_x2 = np.square(phi_x)
        phi_y2 = np.square(phi_y)
    else:
        #-- get central derivatives of SDF at x,y
        phi_x  = 0
        phi_y  = 0
        phi_xx = 0
        phi_yy = 0
        phi_xy = 0
    #-- compute curvature (Kappa)
    curvature = np.multiply(np.divide((np.multiply(phi_x2,phi_yy) + np.multiply(phi_y2,phi_xx) - 
                  2*np.multiply(phi_x,phi_y,phi_xy)), np.power(phi_x2 + phi_y2 + np.finfo(float).eps
                                                               , 3/2)),np.sqrt(phi_x2 + phi_y2))
    return curvature


def sussman(D, dt):
    """
    level set re-initialization by the sussman method
    """
    # forward/backward differences
    a = D - np.roll(D, 1, axis = 1) # backward
    b = np.roll(D, -1, axis = 1) - D # forward
    c = D - np.roll(D, 1, axis = 0) # backward
    d = np.roll(D, -1, axis = 0) - D # forward
    a_p = a
    a_n = a # a+ and a-
    b_p = b
    b_n = b
    c_p = c
    c_n = c
    d_p = d
    d_n = d
    a_p[a < 0] = 0
    a_n[a > 0] = 0
    b_p[b < 0] = 0
    b_n[b > 0] = 0
    c_p[c < 0] = 0
    c_n[c > 0] = 0
    d_p[d < 0] = 0
    d_n[d > 0] = 0
    dD = np.zeros(D.shape)
    D_neg_ind = (D < 0)
    D_pos_ind = (D > 0)
    dD[D_pos_ind] = np.sqrt(np.maximum(np.square(a_p[D_pos_ind]),np.square(b_n[D_pos_ind])) +
                         np.maximum(np.square(c_p[D_pos_ind]), np.square(d_n[D_pos_ind]))) - 1
    dD[D_neg_ind] = np.sqrt(np.maximum(np.square(a_n[D_neg_ind]), np.square(b_p[D_neg_ind])) +
                         np.maximum(np.square(c_n[D_neg_ind]), np.square(d_p[D_neg_ind]))) - 1
    sussman_sign = np.divide(D, np.sqrt(np.square(D) + 1))
    D = D - np.multiply(dt, sussman_sign, dD)
    return D