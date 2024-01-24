import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as interp
from skimage.measure import profile_line
from matplotlib.patches import Rectangle


def funcLineProfileNew(myimg,x,y,Gdir,vecLen,LPpoints,imgxsize,imgysize,
                       PatientNumber,DEBUG_PLOT_LEVEL):
    EdgeLineProfile = np.array([[]])
    edgePixels = np.array([]) # Edge pixels which are considered
    vecStart = np.int16(np.empty(2))
    vecEnd = np.int16(np.empty(2))
    m = 0 # Loop counter variable
    for k in range(len(x)):
        if (x[k] +1 > vecLen and y[k] +1 > vecLen and x[k]+1 + vecLen < imgxsize
            and y[k]+1 + vecLen < imgysize):
            # Define vector directions according to angles
            gradvecx = np.cos(np.radians(Gdir[y[k],x[k]]))
            gradvecy = np.sin(np.radians(Gdir[y[k],x[k]]))
            # Normalize this vector (actually, with the previous values its already normalized)
            gradvecnorm = [gradvecx,gradvecy]/np.linalg.norm([gradvecx,gradvecy])
            # Define start and end point for the line profiles to be estimated
            vecStart[0] = np.round(x[k] -vecLen*gradvecnorm[0])
            # Invert direction (-1*) due to rows start counting from top, not from bottom
            vecStart[1] = np.round(y[k] -(-1*vecLen*gradvecnorm[1]))
            vecEnd[0] = np.round(x[k] +vecLen*gradvecnorm[0])
            # Invert direction (-1*) due to rows start counting from top, not from bottom
            vecEnd[1] = np.round(y[k] -vecLen*gradvecnorm[1])
            if DEBUG_PLOT_LEVEL >= 2:
                # Plot lines in previous X-ray image figure:
                plt.figure()
                plt.plot([vecStart[0],vecEnd[0]],[vecStart[1],vecEnd[1]],linewidth = 2,
                         markerfacecolor = [1,0,0])
                plt.show()
            # Extract line profile (Edge Scan Function - ESF)
            if m == 0:
                EdgeLineProfile = improfile(myimg, vecStart, vecEnd, LPpoints)
                edgePixels = np.array([[x[k],y[k],m,k]])
            else:
                prof = improfile(myimg, vecStart, vecEnd, LPpoints)
                EdgeLineProfile = np.append(EdgeLineProfile,prof,axis = 1)
                edgePixels = np.int16(np.append(edgePixels,np.array([[x[k],y[k],m,k]]),axis = 0))
            m = m + 1
    # Plot for thesis
    if DEBUG_PLOT_LEVEL >= 1:
        plt.figure()
        plt.imshow(myimg, cmap = 'gray')
        m = 0 # Loop counter variable
        vecLen = 5
        if len(x)>=4:
            for k in range(np.arange(0,len(x),4)):
                if (x[k] > vecLen and y[k] > vecLen and x[k] + vecLen < imgxsize
                    and y[k] + vecLen < imgysize):
                    # Define vector directions according to angles
                    gradvecx = np.cos(np.radians(Gdir[y[k],x[k]]))
                    gradvecy = np.sin(np.radians(Gdir[y[k],x[k]]))
                    # Normalize this vector (actually, with the previous values its already normalized)
                    gradvecnorm = [gradvecx,gradvecy]/np.linalg.norm([gradvecx,gradvecy])
                    # Define start and end point for the line profiles to be estimated
                    vecStart[0] = np.round(x[k]-vecLen*gradvecnorm[1])
                    # Invert direction (-1*) due to rows start counting from top, not from bottom
                    vecStart[1] = np.round(y[k]-(-1*vecLen*gradvecnorm[2]))
                    vecEnd[0] = np.round(x[k]+vecLen*gradvecnorm[1])
                    # Invert direction (-1*) due to rows start counting from top, not from bottom
                    vecEnd[1]   = np.round(y[k]+(-1*vecLen*gradvecnorm[2]))
                    # Plot lines in previous X-ray image figure:
                    plt.figure()
                    plt.plot([vecStart[0],vecEnd[0]],[vecStart[1],vecEnd[1]],linewidth = 4,
                             markerfacecolor = [1,1,0])
                    # Plot rectangles for the relevant edge pixels as well
                    fig, ax = plt.subplots()
                    ax.add_patch(Rectangle((x[k]-0.5,y[k]-0.5), 1, 1,lw = 3,ec = 'g',fc = 'none'))
                    m = m + 1
                    if m == 16:
                        return
        # Export image as EPS and JPG and PNG
        imgPath = "Images"
        imgName = f"Pat-{PatientNumber}-MTF-ROI-LineProfile-Overlay"
        plt.savefig(f"{imgPath}\{imgName}.png")
        plt.show()
    return EdgeLineProfile, edgePixels

def improfile(Img, StartIdx, EndIdx, n):
    pCoordinates = np.array([StartIdx,EndIdx]).T
    nRows, nCols = Img.shape[0], Img.shape[1]
    y = [0,nRows]
    x = [0,nCols]
    getn = False
    xmin = np.min(x)
    ymin = np.min(y)
    xmax = np.max(x)
    ymax = np.max(y)
    xGridPoints = np.array([])
    if nCols > 1:
        dx = int(max( (xmax-xmin-1)/(nCols-1), np.finfo(float).eps))
        xGridPoints = np.arange(xmin,xmax,dx)
    elif nCols == 1:
        dx = 1
        Img[:,1] = float('nan')
        xGridPoints = np.array([xmin,xmin+dx])
    yGridPoints = np.array([])
    if nRows > 1:
        dy = int(max( (ymax-ymin-1)/(nRows-1), np.finfo(float).eps))
        yGridPoints = np.arange(ymin,ymax,dy)
    elif nRows == 1:
        dy = 1
        Img[1,:] = float('nan')
        yGridPoints = np.array([ymin,ymin+dy])
    squaredDiff = np.square(np.diff(pCoordinates,axis=1)).T
    sumSquaredDiff = np.sum(squaredDiff,axis=1)
    cdist = np.int16(np.array([[0],np.cumsum(np.sqrt(sumSquaredDiff))]))
    killIdx = np.array(np.where(np.diff(cdist,axis=1) == 0))
    if killIdx.shape:
        cdist = np.delete(cdist,killIdx+1)
        pCoordinates = np.delete(pCoordinates,killIdx+1,1)
    if pCoordinates.shape[0] == 1:
        xg = pCoordinates[0]
        yg = pCoordinates[1]
        if not getn:
            xg = np.matrixlib.repmat(xg,1,n)
            yg = np.matrixlib.repmat(yg,1,n)
    elif not killIdx.shape:
        xg = np.array([])
        yg = np.array([])
    else:
        profi = interp.interp1d(cdist,pCoordinates)(np.arange(0,np.max(cdist)+np.max(cdist)/(n-1),
                np.max(cdist)/(n-1)))
        xg = profi[0,:]
        yg = profi[1,:]
    if xg.shape:
        zg = interp2(xGridPoints,yGridPoints,Img,xg,yg)
        xg_pix = np.round(axes2pix(nCols, np.array([xmin,xmax]), xg))
        yg_pix = np.round(axes2pix(nRows, np.array([ymin,ymax]), yg))
        if not isinstance(zg, float):
            outside_axes = np.array(np.where( np.logical_or(np.logical_or(xg_pix <1,xg_pix >nCols-1), 
                        np.logical_or(yg_pix <1 ,yg_pix >nRows-1))))
            zg = np.double(zg)
            zg[outside_axes] = float('nan')
    else:
        zg = np.array([])
    return zg

def axes2pix(n, extent, axesCoord):
    min = extent[0]
    max = extent[np.max(extent.shape)-1]
    if n == 1:
        pixelCoord = axesCoord - min + 1
        return pixelCoord
    delta = max - min
    if delta == 0:
        xslope = 1
    else:
        xslope = (n - 1)/delta
    if xslope == 1 and min == 0:
        pixelCoord = axesCoord
    else:
        pixelCoord = xslope * (axesCoord - min) + 1
    return pixelCoord

def interp2(x, y, Grid, xq, yq, method ='nearest'):
    zq = np.empty(xq.shape)
    for i in range(len(xq)):
        x_prev = max(x[x <= xq[i]])
        x_next = min(x[x > xq[i]])
        y_prev = max(y[y <= yq[i]])
        y_next = min(y[y > yq[i]])
        if method == 'nearest':
            x_opts = np.array([x_prev,x_next])
            y_opts = np.array([y_prev,y_next])
            y_grid, x_grid = np.meshgrid(y_opts,x_opts)
            dist = np.sqrt(np.square(x_grid - xq[i]) + np.square(y_grid - yq[i]))
            nearest = (dist == np.min(dist))
            xq_i = (x_grid[nearest])[-1]
            yq_i = (y_grid[nearest])[-1]
            zq[i] = Grid[yq_i,xq_i]
    return np.array([zq]).T