import numpy as np


def funcZigZag(InMat):
	# Converts an [r,c] size matrix into a vector following a ZigZag path (1st
	# row down, 2nd row up, 3rd row down, 4th row up, ....)
	r, c = InMat.shape[0], InMat.shape[1]
	OutMat = np.zeros([r*c,1])
	for m in range(c):
		if (m+1)%2 == 0:
			OutMat[m*r:m*r+r] = np.expand_dims(np.flip(InMat[:,m]),1)
		else:
			OutMat[m*r:m*r+r] = np.expand_dims(InMat[:,m],1)
	return OutMat