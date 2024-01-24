import pydicom as dicom
import numpy as np
import matplotlib.pyplot as plt
import os
from funcCalcMTFNew import funcCalcMTFNew
from funcCalcNPS import funcCalcNPS
from funcCalcNPScompleteROI import funcCalcNPScompleteROI
from funcRatioNpsMtf import funcRatioNpsMtf
from funcCalcContrast import funcCalcContrast
import scipy.interpolate as interp

def funcImageQuality(FName,ROI_SIZE,ROI1t,ROI2t):
    """
    Image Quality Analysis in Python
    """
    PatientNumber = 12
    DEBUG_PLOT_LEVEL = -1
    ImShowSizeX = 600
    ImShowSizeY = 600

    ds = dicom.dcmread(FName)
    IMG = np.double(ds.pixel_array)

    if DEBUG_PLOT_LEVEL == -1:
        ySize, xSize = IMG.shape[0], IMG.shape[1]
        plt.figure()
        plt.imshow(IMG ,cmap = 'gray')
        mngr = plt.get_current_fig_manager()
        mngr.window.setGeometry(0,0,ImShowSizeX,ImShowSizeY)
        plt.xlim((1,xSize))
        plt.ylim((1,ySize))
        plt.xticks([]),plt.yticks([])
        plt.show()
    
    px = (ds.PixelSpacing)[0]
    WW = ds['WindowWidth']
    WL = ds['WindowCenter']
    # vector containg all the required DICOM information
    infoDICOM = [px,WW,WL]
    # DICOM Infos as return values
    dicomManufacturer = ds['Manufacturer']
    dicomModel = ds['ManufacturerModelName']
    dicomExposureTime = ds['ExposureTime']
    dicomTubeCurrent = [] # XRayTubeCurrent
    dicomWindowWidth = ds['WindowWidth']
    dicomWindowCenter = ds['WindowCenter']

    # Correct the HU values in the image
    IMG = IMG + ds['RescaleIntercept'].value
    IMGreturn = IMG - np.min(IMG[:])
    IMGreturn = IMGreturn/np.max(IMGreturn[:])

    # ROI and active contour mask definition 
    ROI1 = [ROI1t[0],ROI1t[1],ROI_SIZE]
    ROI2 = [ROI2t[0],ROI2t[1],ROI_SIZE]
    ActContMask = {'x':10,'y':10}

    # Crop the ROI selected above from the chosen image
    # ROI1
    ROI_X, ROI_Y, ROI_SIZE = ROI1[0], ROI1[1], ROI1[2]
    # ROI1 Image
    imgROI1 = IMG[ROI_Y-int(ROI_SIZE/2)-1:ROI_Y+int(ROI_SIZE/2),
                    ROI_X-int(ROI_SIZE/2)-1:ROI_X+int(ROI_SIZE/2)]
    # ROI2
    ROI_X, ROI_Y, ROI_SIZE = ROI2[0], ROI2[1], ROI2[2]
    # ROI2 Image
    imgROI2 = IMG[ROI_Y-int(ROI_SIZE/2)-1:ROI_Y+int(ROI_SIZE/2),
                    ROI_X-int(ROI_SIZE/2)-1:ROI_X+int(ROI_SIZE/2)]

    ImShowSizeX, ImShowSizeY = 400, 400
    # Generate a new subfolder Results under the current patient folder
    PathResults = '.\\Results\\'
    SubFolderName = ''
    ImagePathInfo = {'FName':FName,
                    'PathResults': PathResults,
                    'SubFolderName': SubFolderName
                    }
    if not os.path.exists(PathResults):
        os.mkdir(PathResults)

    # MTF
    # Only for ROI1 (major fissure)
    fMTF,MTF,MTFarea = funcCalcMTFNew(imgROI1,ActContMask,infoDICOM[0],ImagePathInfo,
                                           'ROI1',PatientNumber,DEBUG_PLOT_LEVEL)
    MTFareaBak = MTFarea
    MTFarea = np.mean(MTFarea)

    # Perform the NPS Analysis
    # Parameter defintion for patch analysis (finding homogneous region):
    #  Size of the patches used for differentiating between homogenous and structured areas
    PatchSizeX = 3
    PatchSizeY = 3
    #  Size of the ROI required for NPS analysis
    NPS_roi_size = 14  # Decided with Christoph to just use the 16*16 ROI size october 2019
    NPS_pix_total = NPS_roi_size*NPS_roi_size
    # NPS fpr homogenous region
    imgROI2_NPS = funcCalcNPS(imgROI2,PatchSizeX,PatchSizeY,NPS_roi_size,NPS_pix_total,infoDICOM[0],
                ImShowSizeX, ImShowSizeY,ImagePathInfo,'ROI2',PatientNumber,DEBUG_PLOT_LEVEL)
    
    # call this function to compute the NPS for the complete ROI
    # without any preprocessing
    # NPS for ROI containing the structure (ROI1)
    imgROI1_NPS = funcCalcNPScompleteROI(imgROI1,infoDICOM[0],ImShowSizeX, ImShowSizeY,
                 ImagePathInfo,'ROI1',PatientNumber,DEBUG_PLOT_LEVEL)
    fNPS1 = imgROI1_NPS['f_avg']
    NPS1  = imgROI1_NPS['nps_measured_radavg']

    # Compare MTF NPS ratio
    funcRatioNpsMtf(fNPS1,NPS1,fMTF,MTF,PatientNumber,imgROI1,DEBUG_PLOT_LEVEL)
    # ROI mean value
    ROI1_mean = np.mean(imgROI1)

    #  Contrast estimation for ROI1 and ROI2
    # RMS-based contrast estimation
    ContrastROI1 = funcCalcContrast(imgROI1)
    ContrastROI2 = funcCalcContrast(imgROI2)
    
    # Prepare results to return
    fNPS1 = imgROI1_NPS['f_avg']
    NPS1  = imgROI1_NPS['nps_measured_radavg']
    NPS1area = imgROI1_NPS['var_NPS']
    fNPS2 = imgROI2_NPS['f_avg']
    NPS2  = imgROI2_NPS['nps_measured_radavg']
    NPS2area = imgROI2_NPS['var_NPS']
    
    fMTF = fMTF[1:] # Don't use the DC component in MTF
    MTF = np.mean(MTF[1:,:].T,axis = 0)
    fMin = fMTF[0] # min value for the freq axis
    if fNPS1[-1] < fMTF[-1]:
        fMax = fNPS1[-1] # max value for the freq axis
    else:
        fMax = fMTF[-1] # max value for the freq axis
    
    # Interpolate / resample PS (<==NPS1) and MTF curves to have the same frequency
    # points and the same number of samples
    f_int = np.arange(fMin,fMax,0.01)
    MTF_int = interp.interp1d(fMTF,MTF)(f_int)
    NPS1_int = interp.interp1d(fNPS1,NPS1)(f_int)

    return [f_int, MTF_int, NPS1_int, fNPS2, NPS2, ContrastROI1, ContrastROI2,
         MTFarea, NPS1area, NPS2area, IMGreturn, dicomManufacturer, dicomModel,
         dicomExposureTime, dicomTubeCurrent, dicomWindowWidth, dicomWindowCenter]
