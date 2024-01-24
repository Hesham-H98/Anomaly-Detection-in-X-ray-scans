import numpy as np
def radialavg(z,m,x0 = 0,y0 = 0):
	# Radially averaqe 2D square matrix z into m bins
	# computes the average along the radius of a unit circle 
	# inscribed in the square matrix z. The average is computed in M bins.
	# The radial average is not computed beyond the unit circle, in the
	# corners of the matrix z. The radial average is returned in Zr and the
	# mid-points of the M bins are returned in vector R. Not a Number (NaN)
	# values are excluded from the calculation. If offset values xo,yo are
	# used, the origin (0,0) of the unit circle about which the RADIALAVG is
	# computed is offset by xo and yo relative to the origin of the unit square
	# of the input z matrix.
	# INPUTs:
	#	z = square input matrix to be radially averaged
	#	m = number of bins in which to compute radial average
	#	xo = offset of x-origin relative to unit square (DEF: 0)
	#	yo = offset of y-origin relative to unit square (DEF: 0)
	# OUTPUTs:
	#	Zr = radial average of length m
	#	R  = m locations of Zr (i.e. midpoints of the m bins)
	N = z.shape[0]
	X, Y = np.meshgrid(np.arange(-1,1+2/(N-1),2/(N-1)),np.arange(-1,1+2/(N-1),2/(N-1)))
	X = X - x0
	Y = Y - y0
	r = np.sqrt(np.square(X)+np.square(Y))

	# equi-spaced points along radius which bound the bins to averaging radial values
	# bins are set so 0 (zero) is the midpoint of the first bin and 1 is the last bin
	dr = 1/(m-1)
	rbins = np.linspace(-dr/2,1+dr/2,m+1)

	# radial positions are midpoints of the bins
	R = (rbins[0:-1] + rbins[1:])/2

	Zr = np.zeros([1,m]) # vector for radial average
	nans = ~np.isnan(z)

	# loop over the bins, except the final (r=1) position
	for j in range(0,m-1):
		# find all matrix locations whose radial distance is in the jth bin
		bins = np.logical_and(np.greater_equal(r,rbins[j]),np.less(r,rbins[j+1]))
		# exclude data that is NaN
		bins = bins * nans
		# count the number of those locations
		n = np.sum(bins)
		if n != 0:
			# average the values at those binned locations
			Zr[0][j] = np.sum(z[bins])/n
		else:
			Zr[0][j] = float("Nan")

	# special case: the last bin location to not average Z values for
	# radial distances in the corners, beyond R=1
	bins = np.logical_and(np.greater_equal(r,rbins[m-1]),np.less_equal(r,1))
	bins = bins * nans

	n = np.sum(bins)
	if n != 0:
		Zr[0][m-1] = np.sum(z[bins])/n
	else:
		Zr[0][m-1] = float("Nan")

	return Zr, R