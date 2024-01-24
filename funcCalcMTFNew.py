from cv2 import multiply
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import cv2
import scipy.ndimage as ndimg
import scipy.interpolate as interp
import scipy.signal as sig
import scipy.fft as FFT
from skimage.filters import unsharp_mask
from skimage import exposure
import datetime
import funcLineProfileNew as LPN
import region_seg as RS
import zeroCrossing as ZC
import funcPlotGradDirAlongEdge as PGDAE
matplotlib.use("Qt5Agg")

def funcCalcMTFNew(myimg,ActContMask,PixSpc,ImagePathInfo,ROIname,PatientNumber,DEBUG_PLOT_LEVEL):
    """
    Compute the MTF of the image/ROI
    ----------
    Important steps of this script
    ----------
    0. Load image and set global parameters
    1. Preprocessing (only used for improving edge detection)
    2. Edge detection using active contours
    3. Gradient estimation along edges
    4. Line profiles estimation perpendicular to the detected edge
    5. Bump removal in ESF curves (currently not used, replaced by funcEdgeBumpRemovalExtend)
    6. Alignment and averaging of ESF and LSF curves
    7. MTF using average LSF
    """
    #
    # 0. Load image and set global parameters
    # #######################################
    #
    ImShowSizeX, ImShowSizeY = 600, 600 #Displayed image size using imshow
    # PLOT FOR THESIS - Show selected ROI - original before inerpolation
    if DEBUG_PLOT_LEVEL >= 1:
        ySize,xSize = myimg.shape[0], myimg.shape[1]
        fig = plt.figure()
        plt.imshow(np.flip(myimg,0),cmap = 'gray')
        plt.xlim([1,xSize])
        plt.ylim([1,ySize])
        plt.axis('off')
        # Export image as EPS and JPG and PNG
        imgPath = "Images"
        imgName = f"Pat-{PatientNumber}-MTF-ROI-Original"
        plt.savefig(f"{imgPath}\{imgName}.png")
        plt.show()
    # PLOT FOR WEBPAGE PAGE (ROI)
    if DEBUG_PLOT_LEVEL == -1:
        ySize,xSize = myimg.shape[0], myimg.shape[1]
        plt.figure()
        plt.imshow(np.flip(myimg,0),cmap = 'gray')
        plt.xlim([1,xSize])
        plt.ylim([1,ySize])
        plt.axis('off')
        plt.show()
    #
    # 1. Preprocessing and parameter definition (only for edge detection)
    # ###################################################################
    #
    # Make a copy of the original, unprocessed image
    myimg = np.double(myimg)
    ImageInterPFactor = 2
    # A meshgrid of pixel coordinates
    myimg = interp2d_interleave_recursive(myimg,2)
    myimg = np.int16(np.floor(myimg))
    myimgorig = myimg
    PixSpc = PixSpc / ImageInterPFactor
    # PLOT FOR THESIS - Show selected ROI after interpolation
    # Plot edge in image
    if DEBUG_PLOT_LEVEL >= 2:
        ySize,xSize = myimg.shape[0], myimg.shape[1]
        plt.figure()
        plt.imshow(np.flip(myimg,0),cmap = 'gray')
        plt.xlim([1,xSize])
        plt.ylim([1,ySize])
        plt.axis('off')
        # Export image as EPS and JPG and PNG
        imgPath = "Images"
        imgName = f"Pat-{PatientNumber}-MTF-ROI-Interp"
        plt.savefig(f"{imgPath}\{imgName}.png")
        plt.show()
    # Preprocessing to enhance the quality of the edge detection
    # No used for the lines profiles, only for the edge detection step
    p1, p99 = np.percentile(myimg, (1, 99))
    myimg = exposure.rescale_intensity(myimg, in_range=(p1, p99))
    p1, p99 = np.percentile(myimg, (1, 99))
    myimg = exposure.rescale_intensity(myimg, in_range=(p1, p99))
    myimg = ndimg.gaussian_filter(myimg,2)
    myimg = np.int16(unsharp_mask(myimg, radius = 1, amount = 0.8, preserve_range=True))
    # PLOT FOR THESIS - Show selected ROI after interpolation and image enhancement
    if DEBUG_PLOT_LEVEL >= 2:
        ySize,xSize = myimg.shape[0], myimg.shape[1]
        plt.figure()
        plt.imshow(np.flip(myimg,0),cmap = 'gray')
        plt.xlim([1,xSize])
        plt.ylim([1,ySize])
        plt.axis('off')
        # Export image as EPS and JPG and PNG
        imgPath = "Images"
        imgName = f"Pat-{PatientNumber}-MTF-ROI-Interp-ImgEnhance"
        plt.savefig(f"{imgPath}\{imgName}.png")
        plt.show()
    #
    # 2. Edge detection using active contours
    # #######################################
    #
    # 2.0 Parameter settings for the active contours method
    # Mask definition for active contours and image resize (50%)
    m = np.zeros(myimg.shape)
    maskX =  (ImageInterPFactor**2)*ActContMask['x']
    maskY = (ImageInterPFactor**2)*ActContMask['y']
    WinWidth = ImageInterPFactor*2 # ImageInterPFactor*10
    m[maskY-WinWidth-1:maskY+WinWidth,maskX-WinWidth-1:maskX+WinWidth] = 1
    imgysize,imgxsize = myimg.shape[0], myimg.shape[1]
    # 2.1 Active contours
    if  DEBUG_PLOT_LEVEL == 0 or DEBUG_PLOT_LEVEL == -1:
        #-- Run segmentation (alpha=0.2 (DEFAULT), PLOT DISABLED) 
        seg = RS.region_seg(myimg, m, 1000, 0.2, 0)
    else:
        seg = RS.region_seg(myimg, m, 1000)
    if DEBUG_PLOT_LEVEL >= 2:
        fig, ax = plt.subplots(2,2)
        ax[0][0].imshow(myimg)
        ax[0][0].set_title('Input Image')
        ax[0][1].imshow(m)
        ax[0][1].set_title('Initialization region')
        ax[1][1].imshow(seg)
        ax[1][1].set_title('Global Region-Based Segmentation')
        plt.show()
    myimgseg = seg
    # 2.2 Generate edge matrix (edge pixels == 1, background == 0)
    myimgedge = 1/9*cv2.Sobel(np.float32(myimgseg), ddepth=cv2.CV_32F, dx=1, dy=1, ksize=5)
    myimgedge[myimgedge<0] = 1
    # Plot edge matrix
    # PLOT FOR THESIS - Show detected edge in ROI
    if DEBUG_PLOT_LEVEL >= 1:
        [ySize,xSize] = myimgedge.shape()
        plt.figure()
        plt.imshow(np.flip(myimgedge,0),cmap = 'gray')
        plt.xlim([1,xSize])
        plt.ylim([1,ySize])
        plt.axis('off')
        # Export image as EPS and JPG and PNG
        imgPath = "Images"
        imgName = f"Pat-{PatientNumber}-MTF-ROI-Edge"
        plt.savefig(f"{imgPath}\{imgName}.png")
        plt.show()
    [xEdge, yEdge] = np.where(myimgedge == 1)
    # Plot edge in image
    if DEBUG_PLOT_LEVEL >= 2:
        fig, ax = plt.subplots()
        ax.imshow(myimgorig, cmap = 'gray',extent = (0.5,myimgorig.shape[0]+0.5,myimgorig.shape[1]+0.5,0.5))
        for k in range(len(xEdge)):
            ax.add_patch(Rectangle((xEdge[k],yEdge[k]), 1, 1,lw = 2,ec = 'g',fc = 'none'))
        plt.show()
    # PLOT FOR THESIS - Show ROI and overlayed edge pixels as rectangle
    if DEBUG_PLOT_LEVEL >= 1:
        fig, ax = plt.subplots()
        ax.imshow(myimg, cmap = 'gray',extent = (0.5,myimg.shape[0]+0.5,myimg.shape[1]+0.5,0.5))
        for k in range(len(xEdge)):
            ax.add_patch(Rectangle((xEdge[k],yEdge[k]), 1, 1,lw = 2,ec = 'g',fc = 'none'))
        plt.xlim([1,xSize])
        plt.ylim([1,ySize])
        plt.axis('off')
        # Export image as EPS and JPG and PNG
        imgPath = "Images"
        imgName = f"Pat-{PatientNumber}-MTF-ROI-Edge-Overlay"
        plt.savefig(f"{imgPath}\{imgName}.png")
        plt.show()
    #
    # 3. Gradient computation
    # #######################
    #
    # This computes the gradient magnitues and direction (angle) for each pixel in the image
    # i.e. for the edge and for the background
    GX = cv2.Sobel(np.float32(myimg), ddepth=cv2.CV_32F, dx=1, dy=0)
    GY = cv2.Sobel(np.float32(myimg), ddepth=cv2.CV_32F, dx=0, dy=1)
    Gmag = np.sqrt((GX ** 2) + (GY ** 2))
    Gdir = np.arctan2(GY, GX) * (180 / np.pi) % 180
    # Create empty vector for edge gradient magnitudes
    edgemag = np.zeros([len(yEdge),1])
    # Create empty vector for edge gradient directions
    edgedir = np.zeros([len(yEdge),1])
    for k in range(len(yEdge)):
        edgemag[k,0] = Gmag[yEdge[k],xEdge[k]]
        edgedir[k,0] = Gdir[yEdge[k],xEdge[k]]
    # Plot gradient directions along one edge 
    # !!Use Carefully! (Always check manually if this works!)
    [xyEdgePlot,GdirPlotStd] = PGDAE.funcPlotGradDirAlongEdge(xEdge,yEdge,Gdir,myimgorig,
                                                              DEBUG_PLOT_LEVEL)
    #
    # 4. Line profile estimation based on gradient directions
    # #######################################################
    #
    # From here on, use the original, unfilterd image again
    myimg = myimgorig
    # Parameters for line profile
    #	Note: Length of vector for line profile:
    #	We achieved good results with a length of 15px @ PixSpc==0.33mm -> Line length ~ 5mm
    #	Use these 5mm as a rough reference
    #	length of each line in pixels
    vecLenMM = 5 #Length of line profile vector in [mm] %15px @ PixSpc = 0.33mm --> length=~5mm
    vecLen = np.round(vecLenMM/PixSpc) # Length of line profile vector in [px] %15px @ PixSpc = 0.33mm --> length=~5mm
    LPpoints = 15 # (100) Number of points used for (interpolated) line profile
    # This changes MTFarea results when vecLen is changed
    PixSpcLP = (2*vecLenMM)/LPpoints # factor "2" because length of the profile is 2*vecLen
    # Use the edge which was plotted to check the variance
    if DEBUG_PLOT_LEVEL >= 2:
        plt.figure()
        plt.imshow(myimg, cmap = 'gray')
        plt.title('Original (unfiltered) image used for line profile estimation')
        plt.show()
    [ESF,edgePixels] = LPN.funcLineProfileNew(myimg,xEdge,yEdge,Gdir,vecLen,LPpoints,
                                              imgxsize,imgysize,PatientNumber,DEBUG_PLOT_LEVEL)
    ESFbak = ESF
    # Upsample the ESF for the follwing bump removal
    PixSpcLPorig = PixSpcLP
    PixSpcLP = PixSpcLPorig
    InterpBumpRemoval=4 # Interpolation factor
    ESFInt = interp1(np.arange(0,len(ESFbak[:,0]),1),ESFbak,np.arange(0,len(ESFbak[:,0])-1
            +1/InterpBumpRemoval,1/InterpBumpRemoval),Kind = 'slinear')
    ESF = ESFInt # Replace original LSF my interpolated LSF
    xbak = np.arange(0,len(ESFbak[:,0]),1)
    if DEBUG_PLOT_LEVEL >= 2:
        x = np.arange(0,len(ESF[:,0]),1)/InterpBumpRemoval
        plt.figure()
        plt.plot(xbak,ESFbak[:,0])
        plt.plot(x,ESF[:,0])
        plt.title('ESF, Upsamples for interpolation')
        plt.show()
    PixSpcLP = PixSpcLP/InterpBumpRemoval
    ESF = ESF - np.mean(ESF)
    LSF = np.diff(ESF,axis=0)
    # Low pass filter the ESF  and plot the results
    ESFLP = ESF
    LSFLP = np.diff(ESFLP,axis=0)
    # PLOT FOR THESIS - Show ESF before and after Low Pass filtering
    N = 4
    if DEBUG_PLOT_LEVEL >= 1:
        x = np.arange(0,len(ESFLP[:,0]),1)*PixSpcLP
        plt.figure()
        mngr = plt.get_current_fig_manager()
        mngr.window.setGeometry(100,100,700,700)
        plt.grid()
        plt.plot(x,ESF[:,N-1],color = 'red',lw =4)
        plt.plot(x,ESFLP[:,N-1],color = 'blue',lw = 4)
        plt.xlabel('Length [mm]')
        plt.ylabel('Intensity [a.u.]')
        plt.xlim([0,2*vecLenMM-1])
        plt.box(False)
        plt.tick_params(axis='both',direction='out')
        plt.tick_params(axis='x',length=0.02)
        plt.tick_params(axis='y',length=0.08)
        plt.rcParams.update({'font.size': 12})
        plt.legend(['ESF','ESF (filtered)'],loc = 'lower right')
        # Export image as EPS and JPG and PNG
        imgPath = "Images"
        imgName = f"Pat-{PatientNumber}-ESF-LP-filt"
        plt.savefig(f"{imgPath}\{imgName}.png")
        plt.show()
    ESF = ESFLP
    LSF = LSFLP
    if DEBUG_PLOT_LEVEL >= 2:
        plt.figure()
        plt.plot(ESF)
        plt.title('Low pass filtered ESF')
        plt.figure()
        plt.plot(LSF)
        plt.title('Low pass filtered LSF')
        plt.show()
    # New bumpremoval, 21st century version
    # Define the range where the edge should start / end
    idxLeftLimit  = np.int16(np.round(0.20*len(ESF[:,0])))-1
    idxRightLimit = np.int16(np.round(0.80*len(ESF[:,0])))-1
    # Define the range where the LSF peak should be (with respect to the center of the LSF vector)
    idxPeakLeftLimit  = np.int16(np.round(0.40*len(ESF[:,0])))-1
    idxPeakRightLimit = np.int16(np.round(0.60*len(ESF[:,0])))-1
    edgeCenterDistance = np.int16(np.round(0.35*len(ESF[:,0])))
    idxResult = 0 # Index variable for writing to esfNoBump after succesful bump removal
    if DEBUG_PLOT_LEVEL >= 2:
        plt.figure()
    idxValid = np.array([])
    for idxCol in range(len(ESF[0,:])):
        idxZero = ZC.zeroCrossing(LSF[:,idxCol])[0] # Zero corssing detection in LSF
        # Test version
        # Center positon of the edge hopefully corresponds to the maximum of the LSF...)
        idxCenter = np.argmax(np.abs(LSF[idxPeakLeftLimit-1:idxPeakRightLimit,idxCol]))
        idxCenter = idxCenter + idxPeakLeftLimit -1
        # Find first zero crossing to the left of the center position
        idxLeft = np.where(idxZero < idxCenter)[0].tolist()
        if len(idxLeft) != 0:
            idxLeft = idxLeft[-1]
            idxLeft = idxZero[idxLeft]
            idxLeft = idxLeft + 1 # There is a shift of 1 in the zero corssing index
        idxRight = np.where(idxZero > idxCenter)[0].tolist()
        if len(idxRight) != 0:
            idxRight = idxRight[0]
            idxRight = idxZero[idxRight] + 1
        # test version
        if idxCenter-idxLeft < edgeCenterDistance and idxRight-idxCenter < edgeCenterDistance:
            tmp = ESF[:,idxCol]
            tmp[:idxLeft] = tmp[idxLeft]
            tmp[idxRight:] = tmp[idxRight]
            tmp = np.array([tmp]).T
            if idxResult == 0:
                esfNoBump = tmp
            else:
                esfNoBump = np.append(esfNoBump,tmp,axis = 1)
            tmp = np.array([[]])
            idxValid = np.append(idxValid,idxCol)
            idxResult = idxResult + 1
            if DEBUG_PLOT_LEVEL >= 2:
                plt.plot(ESF[:,idxCol])
                plt.plot(idxLeft,ESF[idxLeft,idxCol],marker = 'rd')
                plt.plot(idxRight,ESF[idxRight,idxCol],marker = 'gd')
                plt.title('Detected bump positions (considered as the end of the edge)')
    if DEBUG_PLOT_LEVEL >=2:
        plt.figure()
        plt.plot(esfNoBump)
        plt.figure()
        plt.plot(np.diff(esfNoBump))
        plt.show()
    ESF = esfNoBump
    LSF = np.diff(ESF, axis=0)
    #
    # 5. Bump removal in ESF curves (currently not used, replaced by funcEdgeBumpRemovalExtend)
    # #########################################################################################
    #
    # Downsample the ESF (was upsampled for better bump removal)
    ESFd = sig.resample(ESF,InterpBumpRemoval)
    PixSpcLP = PixSpcLP*InterpBumpRemoval # Reset pixel spacing to original value\
    if DEBUG_PLOT_LEVEL >= 2:
        xbak = np.arange(0,len(ESFbak[:,0]),1)
        x = np.arange(0,len(ESFd[:,0]),1)
        plt.figure()
        plt.plot(xbak,ESFbak[:,0]- np.mean(ESFbak[:,0]))
        plt.plot(x,ESFd[:,0])
        plt.show()
    ESF = ESFd
    LSF = np.diff(ESF,axis=0)
    # Extend the ESF to the left and right to generate more data points
    ESFT = ESF
    ESFTlen = len(ESFT[:,0])
    ESFTfill = np.zeros((ESFTlen,len(ESFT[0,:])))
    ESFT = np.concatenate((ESFTfill,ESFT),axis=0)
    ESFT = np.concatenate((ESFT,ESFTfill),axis=0)
    for k in range(len(ESFT[0,:])):
        ESFT[:ESFTlen,k] = ESFT[ESFTlen,k]
        ESFT[2*ESFTlen:,k] = ESFT[2*ESFTlen-1,k]
    ESF = ESFT
    LSF = np.diff(ESF,axis=0)
    # PLOT FOR THESIS - ESF before anf after bump removal without extension
    N = 3
    if DEBUG_PLOT_LEVEL >= 1:
        x = np.arange(0,len(ESFbak[:,0]),1)*PixSpcLP
        plt.figure()
        mngr = plt.get_current_fig_manager()
        mngr.window.setGeometry(100,100,700,700)
        plt.grid()
        plt.plot(x,ESFbak[:,N-1]-np.mean(ESFbak[:,N-1]),lw =4)
        plt.plot(x,ESFd[:,N-1]-np.mean(ESFd[:,N-1]),lw = 4)
        plt.xlabel('Length [mm]')
        plt.ylabel('Intensity [a.u.]')
        plt.xlim([0,x[-1]])
        plt.box(False)
        plt.tick_params(axis='both',direction='out',linewidth = 2)
        plt.tick_params(axis='x',length=0.02)
        plt.tick_params(axis='y',length=0.08)
        plt.rcParams.update({'font.size': 12})
        plt.legend(['ESF','ESF (edge only)'],loc = 'lower right')
        # Export image as EPS and JPG and PNG
        imgPath = "Images"
        imgName = f"Pat-{PatientNumber}-ESF-Bump-Removal-Comparison"
        plt.savefig(f"{imgPath}\{imgName}.png")
        plt.show()
    # PLOT FOR THESIS - ESF before anf after bump removal without extension
    N = 3
    if DEBUG_PLOT_LEVEL >= 1:
        x = np.arange(0,len(ESFbak[:,0]),1)*PixSpcLP
        plt.figure()
        mngr = plt.get_current_fig_manager()
        mngr.window.setGeometry(100,100,700,700)
        plt.grid()
        plt.plot(x,ESFbak[:,N-1]-np.mean(ESFbak[:,N-1]),lw =4)
        plt.plot(x,ESFd[:,N-1]-np.mean(ESFd[:,N-1]),lw = 4)
        plt.xlabel('Length [mm]')
        plt.ylabel('Intensity [a.u.]')
        plt.xlim([0,x[-1]])
        plt.box(False)
        plt.tick_params(axis='both',direction='out',linewidth = 2)
        plt.tick_params(axis='x',length=0.02)
        plt.tick_params(axis='y',length=0.08)
        plt.rcParams.update({'font.size': 12})
        plt.legend(['ESF','ESF (edge only)'],loc = 'lower right')
        # Export image as EPS and JPG and PNG
        imgPath = "Images"
        imgName = f"Pat-{PatientNumber}-ESF-Bump-Removal-Comparison"
        plt.savefig(f"{imgPath}\{imgName}.png"),plt.show()
    # PLOT FOR THESIS - LSF after bump removal without extensions
    if DEBUG_PLOT_LEVEL >= 1:
        x = np.arange(0,len(np.diff(ESFd[:,0])),1)*PixSpcLP
        plt.figure()
        mngr = plt.get_current_fig_manager()
        mngr.window.setGeometry(100,100,700,700)
        plt.grid()
        plt.plot(x,np.diff(ESFd[:,N-1]),lw =4)
        plt.xlabel('Length [mm]')
        plt.ylabel('Intensity [a.u.]')
        plt.xlim([0,x[-1]])
        plt.box(False)
        plt.tick_params(axis='both',direction='out',linewidth = 2)
        plt.tick_params(axis='x',length=0.02)
        plt.tick_params(axis='y',length=0.08)
        plt.rcParams.update({'font.size': 12})
        # Export image as EPS and JPG and PNG
        imgPath = "Images"
        imgName = f"Pat-{PatientNumber}-LSF-NoExtension"
        plt.savefig(f"{imgPath}\{imgName}.png"),plt.show()
    # PLOT FOR THESIS - LSF
    if DEBUG_PLOT_LEVEL >= 1:
        x = np.arange(0,len(LSF[:,0]),1)*PixSpcLP
        plt.figure()
        mngr = plt.get_current_fig_manager()
        mngr.window.setGeometry(100,100,700,700)
        plt.grid()
        plt.plot(x,LSF[:,N-1],lw =4)
        plt.xlabel('Length [mm]')
        plt.ylabel('Intensity [a.u.]')
        plt.xlim([0,x[-1]])
        plt.box(False)
        plt.tick_params(axis='both',direction='out',linewidth = 2)
        plt.tick_params(axis='x',length=0.02)
        plt.tick_params(axis='y',length=0.08)
        plt.rcParams.update({'font.size': 12})
        # Export image as EPS and JPG and PNG
        imgPath = "Images"
        imgName = f"Pat-{PatientNumber}-LSF"
        plt.savefig(f"{imgPath}\{imgName}.png"),plt.show()
    #
    # 6. Alignment and averaging of ESF and LSF curves
    # ################################################
    #
    # Interpolation if required
    InterpFactor = 1 # Interpolation factor
    LSFInt = interp1(np.arange(0,len(LSF[:,0]),1),LSF,
            np.arange(0,len(LSF[:,0]),1/InterpBumpRemoval), Kind = 'slinear')
    LSF = LSFInt # Replace original LSF my interpolated LSF
    #
    # 7. MTF using average LSF
    # ########################
    #
    MTF = np.abs(FFT.fft(LSF,axis =0))
    # Normalize MTF
    MTF = np.divide(MTF,MTF[1,:])
    # Consinder PixSpcLP instead of PixSpcfrom interpolating the line profile
    xaxis = np.arange(0,len(MTF[:,0]),1)/(len(MTF[:,0])-1)*InterpFactor/PixSpcLP
    # Show some debugging information about pixel spacing and sampling frequency (sampling of the line profile)    
    print('*****************************************************************')
    print('MTF frequency axis based on the line profile sampling frequency: ')
    print('Length of the line profile [mm] = ',vecLenMM)
    print('Number of pixels = ',LPpoints)
    print('Sampling frequency [Pixel/mm] = ',LPpoints/vecLenMM)
    print('Sampled pixel spacing [mm] = ',1/(LPpoints/vecLenMM))
    print('*****************************************************************')
    print(' ')
    # PLOT FOR THESIS - MTF - ALL CURVES
    if DEBUG_PLOT_LEVEL >= 1:
        plt.figure()
        mngr = plt.get_current_fig_manager()
        mngr.window.setGeometry(100,100,700,700)
        plt.grid()
        for k in range(len(MTF[0,:])):
            plt.plot(xaxis[1:],MTF[1:,k],lw = 4)
        plt.xlabel('Spatial frequency [1/mm]')
        plt.ylabel('MTF(f) (normalized)')
        plt.xlim([xaxis[1],1/PixSpcLP/2])
        plt.ylim([0,1])
        plt.box(False)
        plt.tick_params(axis='both',direction='out',linewidth = 2)
        plt.tick_params(axis='x',length=0.02)
        plt.tick_params(axis='y',length=0.08)
        plt.rcParams.update({'font.size': 12})
    	# Export image as EPS and JPG and PNG
        imgPath = "Images"
        imgName = f"Pat-{PatientNumber}-MTF-All"
        plt.savefig(f"{imgPath}\{imgName}.png"),plt.show()
    #  PLOT FOR THESIS - MTF - MEAN CURVE + PLOT FOR WEBPAGE PAGE (ROI) (-1
    if DEBUG_PLOT_LEVEL >= 1 or DEBUG_PLOT_LEVEL == -1:
        plt.figure()
        mngr = plt.get_current_fig_manager()
        mngr.window.setGeometry(100,100,700,700)
        plt.grid()
        plt.plot(xaxis[1:],np.mean(MTF[1:,:].T,axis = 0),lw = 4)
        plt.xlabel('Spatial frequency [1/mm]')
        plt.ylabel('MTF(f) (normalized)')
        plt.xlim([xaxis[1],1/PixSpcLP/2])
        plt.ylim([0,1])
        plt.box('off')
        plt.tick_params(axis='both',direction='out',labelsize = 10)
        plt.tick_params(axis='x',length=0.02)
        plt.tick_params(axis='y',length=0.08)
        plt.rcParams.update({'font.size': 12})
        if DEBUG_PLOT_LEVEL >= 1:
    		# Export image as EPS and JPG and PNG
            imgPath = "Images"
            imgName = f"Pat-{PatientNumber}-MTF-Mean"
            plt.savefig(f"{imgPath}\{imgName}.png")
        plt.show()
    # Compute the area under the MTF (half the frequency range, i.e. up to the Nyquist frequency)
    ElemNum = np.int16(np.floor(len(MTF[1:,0])/2) - 1)
    MTFarea = np.trapz(MTF[1:ElemNum-1,:].T,x = xaxis[1:ElemNum-1])
    print('Mean MTF area (under curve): %.3f' % np.mean(MTFarea))
    # PLOT FOR THESIS - EDGE USED FOR ANALYSIS OF GRADIENT ANGLES AND MTF (ALONG THIS EDGE)
    if DEBUG_PLOT_LEVEL >= 1:
        fig, ax = plt.subplots()
        mngr = plt.get_current_fig_manager()
        mngr.window.setGeometry(100,100,700,700)
        plt.grid()
        ax.imshow(myimgorig,cmap = 'gray',extent = (0.5,myimgorig.shape[0]+0.5,myimgorig.shape[1]+0.5,0.5))
        for k in range(len(edgePixels[:,0])):
            ax.add_patch(Rectangle((edgePixels[k,0],edgePixels[k,1]), 1, 1,lw = 2,ec = 'g',fc = 'none'))
        plt.xlim([1,xSize])
        plt.Aylim([1,ySize])
        plt.axis('off')
    	# Export image as EPS and JPG and PNG
        imgPath = "Images"
        imgName = f"Pat-{PatientNumber}-EdgeAnalysis-EdgePixel"
        plt.savefig(f"{imgPath}\{imgName}.png"),plt.show()
    # PLOT FOR THESIS - MTF AREA ALONG THE EDGE
    if DEBUG_PLOT_LEVEL >= 1:
        plt.figure()
        mngr = plt.get_current_fig_manager()
        mngr.window.setGeometry(100,100,700,700)
        plt.grid()
        plt.plot(MTFarea,marker='d',lw = 4)
        plt.xlabel('Edge pixel')
        plt.ylabel('Area under MTF curve')
        plt.ylim([0,np.max(MTFarea)])
        plt.box(False)
        plt.tick_params(axis='both',direction='out',linewidth = 2)
        plt.tick_params(axis='x',length=0.02)
        plt.tick_params(axis='y',length=0.08)
        plt.rcParams.update({'font.size': 12})
        # Export image as EPS and JPG and PNG
        imgPath = "Images"
        imgName = f"Pat-{PatientNumber}-EdgeAnalysis-MTFArea"
        plt.savefig(f"{imgPath}\{imgName}.png"),plt.show()
    # Save results in a txt-file
    txtPath = "Results"
    txtName = f"Pat-{PatientNumber}-Results-MTF"
    fileID = open(f"{txtPath}\{txtName}.txt","w+")
    fileID.write("-------------------------------------------------------\n")
    now = datetime.datetime.now()
    fileID.writelines(["Results timestamp: ", now.strftime("%m/%d/%Y, %H:%M:%S"),"\n"])
    fileID.writelines(["Results for Dataset: ",ImagePathInfo['SubFolderName'],"## image ",
                 ImagePathInfo['FName'],"(Patient Number used in Python:",str(PatientNumber),") \n"])
    fileID.write("Mean area under MTF curve  = %6.3f \n" %np.mean(MTFarea))
    fileID.close()
    return xaxis,MTF,MTFarea


def interp2d_interleave(z,n):
    '''performs linear interpolation on a grid

    all points are interpolated in one step not recursively

    Parameters
    ----------
    z : 2d array (M,N)
    n : int
        number of points interpolated

    Returns
    -------
    zi : 2d array ((M-1)*n+M, (N-1)*n+N)
        original and linear interpolated values

    '''
    frac = np.atleast_2d(np.arange(0,n+1)/(1.0+n)).T
    zi1 = np.kron(z[:,:-1],np.ones(len(frac))) + np.kron(np.diff(z),frac.T)
    zi1 = np.hstack((zi1,z[:,-1:]))
    zi2 = np.kron(zi1.T[:,:-1],np.ones(len(frac))) + np.kron(np.diff(zi1.T),frac.T)
    zi2 = np.hstack((zi2,zi1.T[:,-1:]))
    return zi2.T

def interp2d_interleave_recursive(z,n):
    '''interpolates by recursively interleaving n times
    '''
    zi = z.copy()
    for ii in range(1,n+1):
        zi = interp2d_interleave(zi,1)
    return zi

def interp1(x, yGrid, xq, Kind = 'linear'):
    """
    interpolate 1D grid function (yGrid) over x for interpolation data xq
    """
    for i in range(yGrid.shape[1]):
        if i == 0:
            yq = np.array([interp.interp1d(x,yGrid[:,i],kind=Kind,fill_value="extrapolate")(xq)]).T
        else:
            inter = np.array([interp.interp1d(x,yGrid[:,i],kind=Kind,fill_value="extrapolate")(xq)]).T
            yq = np.append(yq,inter,axis = 1)
    return yq