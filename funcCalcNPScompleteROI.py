import numpy as np
import matplotlib.pyplot as plt
import calc_digital_nps as CDNPS
import radialAverage as RA1
import radialavg as RA2
import datetime


def funcCalcNPScompleteROI(IMG,px,ImShowSizeX, ImShowSizeY, ImagePathInfo,
                                       ROIname,PatientNumber,DEBUG_PLOT_LEVEL):
    # Compute the NPS for the provided ROI without selecting a homogneous area
    # Function Inputs:
    # ...
    # ImagePathInfo - Contains the complete image name and path info 
    #               (used later for automatically save the results)
    # ROIname - Used as file name extension (eg '...-ROI1.jpg' or '...-ROI2.jpg'
    # Updates:
    NPSpix =  IMG
    # NPS computation
    # Deterend the pixel matrix
    NPSpix = NPSpix - np.mean(NPSpix)
    NPS_roi_size = len(NPSpix[:,0])
    # Measured variance
    var_measured = np.var(NPSpix)
    print('Pixel variance measured with var(): ',var_measured)
    # Compute the NPS
    [nps_measured, f] = CDNPS.calc_digital_nps(NPSpix, 2, px, 1, 0)
    # The variance is the integral of the NPS:
    var_NPS = np.trapz(np.trapz(nps_measured.T, x =f), x = f)
    print('Pixel variance from measured NPS: ', var_NPS)
    # Radial averaging of the 2D-NPS, Version 1 (i.e. of the 2D-FFT of the noise image)
    nps_measured_radagv = RA1.radialAverage(nps_measured, (NPS_roi_size+0)/2, (NPS_roi_size+0)/2,
                                        np.arange(0,NPS_roi_size/2-1,1))
    f_avg = np.linspace(0,1/px/2,num = len(nps_measured_radagv))
    [Zr, R] = RA2.radialavg(nps_measured,8)
    if DEBUG_PLOT_LEVEL >= 2:
        plt.figure()
        plt.plot(R,Zr)
        plt.show()
    # Summarize results as a structure (Dictionary)
    NPSresult = {
        "nps_measured": nps_measured,
        "f": f,
        "nps_measured_radavg": nps_measured_radagv,
        "f_avg": f_avg,
        "var_measured": var_measured,
        "var_NPS": var_NPS
        }
    # Save results in a txt-file
    txtPath = "Results"
    txtName = f"Pat-{PatientNumber}-Results-NPS-CompleteROI"
    fileID = open(f"{txtPath}\{txtName}.txt","w+")
    fileID.write("-------------------------------------------------------\n")
    now = datetime.datetime.now()
    fileID.writelines(["Results timestamp: ", now.strftime("%m/%d/%Y, %H:%M:%S"),"\n"])
    fileID.writelines(["Results for Dataset: ",ImagePathInfo['SubFolderName'],"## image ",
                 ImagePathInfo['FName'],"(Patient Number used in Python:",str(PatientNumber),") \n"])
    fileID.writelines(["!ATTENTION: the complete ROi was used to compute the NPS",
                 " (same ROI as used for MTF)! \n",str(var_measured),"\n"])
    fileID.writelines(["Pixel variance measured with var() = ",str(var_measured),"\n"])
    fileID.writelines(["Pixel variance measured with NPS() = ",str(var_NPS),"\n"])
    fileID.close()
    # Plot Results
    # PLOT FOR THESIS - 1D NPS obtained from radial avaerging
    if DEBUG_PLOT_LEVEL >= 1 or DEBUG_PLOT_LEVEL == -1:
        plt.figure()
        mngr = plt.get_current_fig_manager()
        mngr.window.setGeometry(100,100,700,700)
        plt.plot(f_avg,nps_measured_radagv, linewidth = 3)
        plt.grid()
        plt.xlabel('Frequency [mm^{-1}]')
        plt.ylabel('NPS [mm^2]')
        plt.xlim(0, 1/px/2)
        plt.ylim(0, np.max(nps_measured_radagv.flatten()))
        plt.box(False)
        plt.tick_params(axis='both',direction='out')
        plt.tick_params(axis='x',length=0.02)
        plt.tick_params(axis='y',length=0.08)
        plt.rcParams.update({'font.size': 12})
        if DEBUG_PLOT_LEVEL >= 1:
            # Export image as EPS and JPG and PNG
            imgPath = "Images"
            imgName = f"Pat-{PatientNumber}-NPS1D-CompleteROI"
            plt.savefig(f"{imgPath}\{imgName}.png")
        plt.show()
    # Plot for thesis - 2D NPS
    if DEBUG_PLOT_LEVEL >= 1:
        plt.figure()
        mngr.window.setGeometry(100,100,700,700)
        if len(nps_measured.shape()) == 2:
            # Make sure it's 3d
            nps_measured = np.expand_dims(nps_measured,1)
        plt.imshow(np.abs(nps_measured[:][:][1]).T,origin = "lower",extent = [f[0],f[1],f[0],f[1]])
        plt.xlabel('Frequency [mm^{-1}]') 
        plt.ylabel('Frequency [mm^{-1}]')
        plt.xticks([]),plt.yticks([])
        plt.xlim(-1/px/2, 1/px/2)
        plt.ylim(-1/px/2, 1/px/2)
        plt.box(False)
        plt.tick_params(axis='both',direction='out')
        plt.tick_params(axis='x',length=0.02)
        plt.tick_params(axis='y',length=0.08)
        plt.rcParams.update({'font.size': 12})
        # Export image as EPS and JPG and PNG
        imgPath = "Images"
        imgName = f"Pat-{PatientNumber}-NPS2D-CompleteROI"
        plt.savefig(f"{imgPath}\{imgName}.png")
        plt.show()
    return NPSresult