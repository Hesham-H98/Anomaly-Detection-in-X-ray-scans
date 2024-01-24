import numpy as np


def radialAverage(IMG, cx, cy, w):
	# computes the radial average of the image IMG around the cx,cy point
    # w is the vector of radii starting from zero.
    a, b = IMG.shape[0], IMG.shape[1]
    X, Y = np.meshgrid(np.arange(1,a+1,1) - cx,np.arange(1,b+1,1) - cy)
    R = np.sqrt(np.square(X)+np.square(Y))
    profile = np.array([])
    for i in w:
        mask = np.logical_and(np.greater(R,i-1),np.less(R,i+1))
        values = np.multiply((1 - abs(R[mask]-i)),IMG[mask])
        values = IMG[mask]
        profile = np.append(profile,np.mean(values.flatten()))
    return profile