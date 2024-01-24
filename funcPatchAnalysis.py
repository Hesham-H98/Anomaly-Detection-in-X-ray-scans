import numpy as np
import scipy.stats as STAT
import scipy.ndimage as ndimg
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import funcZigZag as ZZ

def funcPatchAnalysis(IMG,PatchSizeX,PatchSizeY,DEBUG_PLOT_LEVEL):
    ySize,xSize = IMG.shape[0], IMG.shape[1]
    PatchStd = np.empty([ySize-PatchSizeY+1,xSize-PatchSizeX+1])
    PatchStdGmag = np.empty([ySize-PatchSizeY+1,xSize-PatchSizeX+1])
    PatchStdGdir = np.empty([ySize-PatchSizeY+1,xSize-PatchSizeX+1])
    PatchEntropy = np.empty([ySize-PatchSizeY+1,xSize-PatchSizeX+1])
    PatchSkewness = np.empty([ySize-PatchSizeY+1,xSize-PatchSizeX+1])
    PatchKurtosis = np.empty([ySize-PatchSizeY+1,xSize-PatchSizeX+1])
    PatchZigZagDiffEnergy = np.empty([ySize-PatchSizeY+1,xSize-PatchSizeX+1])
    PatchZigZagTransDiffEnergy = np.empty([ySize-PatchSizeY+1,xSize-PatchSizeX+1])
    for n in range(ySize-PatchSizeY+1):
        for m in range(xSize-PatchSizeX+1):
            PatchTmp = IMG[n:n+PatchSizeY,m:m+PatchSizeX]
            PatchStd[n,m] = np.std(PatchTmp)
            # Gradient Analysis
            GX = ndimg.sobel(PatchTmp, axis=0)
            GY = ndimg.sobel(PatchTmp, axis=1)
            Gmag = np.sqrt((GX ** 2) + (GY ** 2))
            Gdir = np.arctan2(GY, GX) * (180 / np.pi) -90
            # Wrapping to -180, 180
            Gdir[Gdir > 180] = Gdir[Gdir > 180] - 360
            Gdir[Gdir < -180] = Gdir[Gdir < -180] + 360
            PatchStdGmag[n,m] = np.std(Gmag)
            PatchStdGdir[n,m] = np.std(Gdir)
            _, counts = np.unique(PatchTmp, return_counts=True)
            PatchEntropy[n,m] = STAT.entropy(counts,base =2)
            PatchSkewness[n,m] = STAT.skew(PatchTmp.flatten())
            PatchKurtosis[n,m] = STAT.kurtosis(PatchTmp.flatten(),fisher = False)
            PatchZigZag = ZZ.funcZigZag(PatchTmp)
            PatchZigZagTranspose = ZZ.funcZigZag(PatchTmp.T)
            PatchZigZagDiffEnergy[n,m] = np.sum(np.square(np.diff(PatchZigZag,axis =0)))
            PatchZigZagTransDiffEnergy[n,m] = np.sum(np.square(np.diff(PatchZigZagTranspose, axis =0)))
    if DEBUG_PLOT_LEVEL >= 2:
        plt.figure()
        plt.imshow(PatchStd ,cmap = 'gray')
        plt.title('Standard deviation of patches')
        plt.figure()
        plt.imshow(PatchStdGmag ,cmap = 'gray')
        plt.title('Standard deviation of gradient magnitude of patches')
        plt.figure()
        plt.imshow(PatchStdGdir ,cmap = 'gray')
        plt.title('Standard deviation of gradient directions of patches')
        plt.figure()
        plt.imshow(PatchEntropy ,cmap = 'gray')
        plt.title('Entropy of patches')
        plt.figure()
        plt.imshow(PatchSkewness ,cmap = 'gray')
        plt.title('Skewness of patches')
        plt.figure()
        plt.imshow(PatchKurtosis ,cmap = 'gray')
        plt.title('kurtosis of patches')
        plt.show()
    PatchData = np.array([PatchStd.T.flatten(),PatchStdGmag.T.flatten(),
		PatchStdGdir.T.flatten(),PatchEntropy.T.flatten(),
		PatchSkewness.T.flatten(),PatchKurtosis.T.flatten()]).T
    xSizePatchData = xSize - PatchSizeX + 1
    ySizePatchData = ySize - PatchSizeY + 1
    # k-means clustering
    kmeans = KMeans(n_clusters = 2, n_init = 5).fit(PatchData[0:ySizePatchData*
                                                           xSizePatchData,:])
    C = kmeans.cluster_centers_
    idx = kmeans.labels_
    b = np.reshape(idx, (xSizePatchData,ySizePatchData)).T
    b = -1*(b-1)
    if DEBUG_PLOT_LEVEL >= 1:
        plt.figure()
        plt.imshow(b, cmap = 'gray')
        plt.show()
    # Save this figure
    if DEBUG_PLOT_LEVEL >= 1:
        fig, ax = plt.subplots(1,3)
        ax[0].imshow(IMG, cmap = 'gray')
        ax[0].set_title('ROI for structural analysis')
        plt.xticks([]),plt.yticks([])
        IMG_c1 = IMG[0:ySizePatchData,0:xSizePatchData]
        IMG_c1 = IMG_c1 / np.max(IMG_c1)
        IMG_c2 = np.copy(IMG_c1)
        IMG_c1[b == 1] = 1
        IMG_c2[-(b-1) == 1] = 1
        ax[1].imshow(IMG_c1, cmap = 'gray')
        ax[1].set_title('Class 1')
        plt.xticks([]),plt.yticks([])
        ax[2].imshow(IMG_c2, cmap = 'gray')
        ax[2].set_title('Class 2')
        plt.xticks([]),plt.yticks([])
        plt.show()
    b_bak = b
    # AUTOMATIC SELECTION: Chose background based on variance
    if np.mean(PatchStd[np.equal(b_bak, 0)]) < np.mean(PatchStd[np.equal(b_bak, 1)]):
        b_bak = -1*(b_bak-1)
    b = b_bak
    return b
			