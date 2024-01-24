import numpy as np

def calc_digital_nps(I, n, px = 1, use_window = 0, average_stack = 0):
	# Calculates the digital noise-power spectrum (NPS) noise-only realizations.
	# I is a stack of symmetric n-dimensional noise realizations. The
	# realizations are stacked along the last array dimension of I. If
	# average_stack is set, the calculated NPS is averaged over the stack to
	# reduce uncertainty. px is the pixel size of the image.
	#  If use_window is set, the data is multiplied with a Hann tapering window
	# prior to NPS calculation. Windowing is useful for avoiding spectral
	# leakage in case the NPS increases rapidly towards lower spatial
	# frequencies (e.g. power-law behaviour).
	#
	# nps is the noise-power spectrum of I in units of px^n, and f is the
	# corresponding frequency vector.
	if n < len(I.shape):
		stack_size = I.shape[n]
	else:
		stack_size = 1
	size_I = I.shape
	for i in range(len(size_I)):
		for j in range(len(size_I)):
			if size_I[j] != size_I[i]:
				raise ValueError("ROI must be symmetric.")
	roi_size = size_I[0]

	# Cartesian Coordinates
	x = np.linspace(-roi_size/2,roi_size/2,roi_size)
	x = np.repeat(x[np.newaxis,...],roi_size,axis = 0).T
	# frequency vector
	f = np.linspace(-0.5,0.5,roi_size)/px
	# radial coordinates
	r2 = 0
	for p in range(1,n+1):
		r2 = r2 + np.rollaxis(np.power(x,2),p-1)
	r = np.sqrt(r2)
	# Hann window to avoid spectral leakage
	if use_window:
		h = 0.5*(1+np.cos(np.pi*r/(roi_size/2)))
		h[np.greater(r,roi_size/2)] = 0
		h = np.repeat(h,stack_size,axis = 1)
	else:
		h = 1
	# detrending by subtracting the mean of each ROI
	# more advanced schemes include subtracting a surface,
	# but that is currently not included
	S = I
	for p in range(n):
		rept = np.array(np.arange(1,n+2,1) == p+1)*(roi_size-1)+1
		S = np.squeeze(S)
		mean_ = np.mean(S,axis = p)
		for i in range(len(rept)-1):
			mean_ = np.expand_dims(mean_,i)
		SS = mean_
		for i in range(len(rept)):
			S = np.repeat(SS,rept[i],axis = p)
			SS = S
	# Detrending
	F = np.multiply(I,h)
	F = np.fft.fftshift(np.fft.fft2(F))
	# cartesian NPS
	# the normalization with h is according to Gang 2010
	nps = np.divide(np.square(np.abs(F))/(roi_size**n)*(px**n), # NPS in units of px^n
		(np.sum(np.square(h),axis=None)/np.shape(h.flatten())))
	if average_stack:
		nps = nps
	return nps, f