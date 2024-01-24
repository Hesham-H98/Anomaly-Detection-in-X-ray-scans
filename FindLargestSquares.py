import numpy as np
def Findlargestsquares(I):
	# finds largest sqare regions with all points set to 1
	# input:  I - B/W boolean matrix
	# output: S - for each pixel I(r,c) return size of the largest 
	# 			all-white square with its upper -left corner at I(r,c)
	nr, nc = I.shape[0], I.shape[1]
	S = np.greater(I,0)*1
	for r in reversed(range(0,nr-1)):
		for c in reversed(range(0,nc-1)):
			if S[r][c]:
				a = S[r][c+1]
				b = S[r+1][c]
				d = S[r+1][c+1]
				S[r][c] = min(np.array([a,b,d])) + 1
	return S