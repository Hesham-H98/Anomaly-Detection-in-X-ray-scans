import numpy as np


def funcCalcContrast(IMG):
	# Computes the contrast of the image/ROI
	imgMean = np.mean(IMG)
	x = (IMG - imgMean)/imgMean
	ContrastRMS = np.sqrt(np.mean(x**2)) # Root mean square of IMG
	return ContrastRMS
