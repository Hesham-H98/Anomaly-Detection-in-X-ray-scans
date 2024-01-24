import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.interpolate as interp
matplotlib.use("Qt5Agg")

def funcRatioNpsMtf(fNPS,NPS,fMTF,MTF,PatientNumber,imgROI1,DEBUG_PLOT_LEVEL):
    fMTF = fMTF[1:]
    MTF = np.mean(MTF[1:,:].T,axis = 0)
    fMin = fMTF[0]
    fMax = fNPS[-1]
    # Interpolate / resample NPS and MTF curves to have the same frequency
    # points and the same number of samples
    f_int = np.arange(fMin,fMax,0.01)
    MTF_int = interp.interp1d(fMTF,MTF)(f_int)
    NPS_int = interp.interp1d(fNPS,NPS)(f_int)
    NPS_int_scale = NPS_int/NPS_int[0]
    if DEBUG_PLOT_LEVEL >= 1:
        plt.figure()
        mngr = plt.get_current_fig_manager()
        mngr.window.setGeometry(10,10,610,610)
        plt.grid()
        plt.plot(f_int,MTF_int,marker='o',lw=2)
        plt.plot(f_int,NPS_int_scale,marker='+',lw=2)
        plt.plot(f_int,np.divide(NPS_int_scale,MTF_int),marker='d',lw=2)
        plt.plot(f_int,np.divide(np.sqrt(NPS_int_scale),MTF_int),marker='^',lw=2)
        plt.xlabel('Spatial frequency [1/mm]')
        plt.ylabel('Intensity [a.u.]')
        plt.xlim((fMin,fMax))
        plt.box(False)
        plt.tick_params(axis='both',direction='out')
        plt.tick_params(axis='x',length=0.02)
        plt.tick_params(axis='y',length=0.08)
        plt.rcParams.update({'font.size': 12})
        plt.legend(['MTF','PS','PS/MTF','sqrt(PS)/MTF'],loc = 'upper left')
        # Export image as EPS and JPG and PNG
        imgPath = "Images"
        imgName = f"Pat-{PatientNumber}-Ratio-NPS-MTF"
        plt.savefig(f"{imgPath}\{imgName}.png")
        plt.show()
    # Suggestion CH: Only NPS/MTF ratio plot
    if DEBUG_PLOT_LEVEL >= 1 or DEBUG_PLOT_LEVEL == -1:
        plt.figure()
        mngr = plt.get_current_fig_manager()
        mngr.window.setGeometry(10,10,610,610)
        plt.grid()
        plt.scatter(f_int,np.divide(NPS_int,MTF_int),marker='o',lw=2,
                    facecolors = 'none',edgecolors = 'black')
        plt.xlabel('Spatial frequency [1/mm]')
        plt.ylabel('Intensity [a.u.]')
        plt.xlim((fMin,fMax))
        plt.box(False)
        plt.tick_params(axis='both',direction='out')
        plt.tick_params(axis='x',length=0.02)
        plt.tick_params(axis='y',length=0.08)
        plt.rcParams.update({'font.size': 12})
        plt.legend(['PS/MTF'],loc = 'upper left')
        # Export image as EPS and JPG and PNG
        imgPath = "Images"
        imgName = f"Pat-{PatientNumber}-Ratio-PS-MTF-NoNorm"
        plt.savefig(f"{imgPath}\{imgName}.png")
        plt.show()
    if DEBUG_PLOT_LEVEL >= 1:
        plt.figure()
        mngr = plt.get_current_fig_manager()
        mngr.window.setGeometry(10,10,610,610)
        plt.grid()
        plt.scatter(f_int,np.divide(np.sqrt(NPS_int),MTF_int),marker='o',lw=2)
        plt.xlabel('Spatial frequency [1/mm]')
        plt.ylabel('Intensity [a.u.]')
        plt.xlim((fMin,fMax))
        plt.box(False)
        plt.tick_params(axis='both',direction='out')
        plt.tick_params(axis='x',length=0.02)
        plt.tick_params(axis='y',length=0.08)
        plt.rcParams.update({'font.size': 12})
        plt.legend(['sqrt(PS)/MTF'],loc = 'upper left')
        # Export image as EPS and JPG and PNG
        imgPath = "Images"
        imgName = f"Pat-{PatientNumber}-Ratio-sqrtPS-MTF-NoNorm"
        plt.savefig(f"{imgPath}\{imgName}.png")
        plt.show()
    return
