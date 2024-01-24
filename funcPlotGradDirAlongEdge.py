import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def funcPlotGradDirAlongEdge(xEdge,yEdge,Gdir,myimgorig,DEBUG_PLOT_LEVEL):
    # Plot gradient direction along the edge
    xyEdge = np.array([xEdge,yEdge]).T
    StartPos = np.argmin(xyEdge[:,0])
    xyEdgePlot = np.array([np.concatenate((xyEdge[StartPos,:],np.array([StartPos])),axis = 0)])
    CurrPos = StartPos
    for k in range(StartPos,len(xyEdge[:,0])-1):
        if xyEdge[k,0] >= xyEdge[CurrPos,0]:
            A = np.array([[xyEdge[CurrPos,0],xyEdge[CurrPos,1]]])
            B = xyEdge
            # compute Euclidean distances:
            dist2 = np.sum(np.square(B-A),axis = 1)
            dist2[dist2==0] = 1000
            # find the smallest distance and use that as an index into B:
            closest = xyEdge[dist2 == np.min(dist2),:]
            NextPos = np.argmin(dist2)
            conc = np.concatenate((xyEdge[NextPos,:], [NextPos]))
            xyEdgePlot = np.concatenate((xyEdgePlot, [conc]))
            xyEdge[CurrPos,:] = [1000, 1000]
            CurrPos = NextPos
    end = xyEdgePlot.shape[1]
    xyEdgePlot = np.delete(xyEdgePlot,np.arange(end-2,end+1),0)
    linearIndices = np.ravel_multi_index((xyEdgePlot[:,1],xyEdgePlot[:,0]), Gdir.shape)
    GdirPlot = Gdir.flatten()[linearIndices]
    GdirPlotTmp = GdirPlot
    # Angle Correction
    GdirPlot = -1*(np.abs(GdirPlot)-180)
    if DEBUG_PLOT_LEVEL == 2:
        fig, ax = plt.subplots(1)
        ax.imshow(myimgorig, cmap = 'gray',extent = (0.5,myimgorig.shape[0]+0.5,myimgorig.shape[1]+0.5,0.5))
        if len(GdirPlot) >= 2:
            for k in range(len(GdirPlot)):
                ax.add_patch(Rectangle((xyEdgePlot[k,0],xyEdgePlot[k,1]), 1, 1,
                                       lw = 2,ec = 'g',fc = 'none'))
        ax.add_patch(Rectangle((xyEdgePlot[0,0],xyEdgePlot[0,1]), 1, 1,lw = 2,ec = 'r',fc = 'none'))
        ax.add_patch(Rectangle((xyEdgePlot[-1,0],xyEdgePlot[-1,1]), 1, 1,lw = 2,ec = 'b',fc = 'none'))
        plt.title('Edge pixels used for angle analysis plot')
        plt.figure()
        plt.plot(np.arange(2,len(GdirPlotTmp),1),GdirPlotTmp[1:-2],'gd',ms = 4)
        plt.plot(1,GdirPlotTmp[0],'rd',ms = 4)
        plt.plot(len(GdirPlotTmp),GdirPlotTmp[-1],'bd',ms = 4)
        plt.xlabel('Pixel number')
        plt.ylabel('Angle with respect to x-axis')
        plt.title('Unmodified angles')
        plt.figure()
        plt.plot(np.arange(2,len(GdirPlot),1),GdirPlot[1:-2],'gd',ms = 4)
        plt.plot(1,GdirPlot[0],'rd',ms = 4)
        plt.plot(len(GdirPlot),GdirPlot[-1],'bd',ms = 4)
        plt.xlabel('Pixel number')
        plt.ylabel('Angle with respect to x-axis')
        plt.title('Modified angles')
        plt.show()
    # Compute the standard deviation of the angles along the edge in a moving window of width W
    W = 4
    if len(GdirPlot) > W:
        GdirPlotStd = np.empty([len(GdirPlot)-W,1])
        for idx in range(len(GdirPlot)-W):
            GdirPlotStd[idx] = np.std(GdirPlot[idx:idx+W])
    else:
        GdirPlotStd = []
    if DEBUG_PLOT_LEVEL == 2:
        plt.figure()
        plt.plot(GdirPlotStd,'d')
        plt.title('Standard deviation of angles. Window length = 9')
        plt.xlabel('Pixel number');
        plt.ylabel('Standard deviation')
        plt.show()
    return xyEdgePlot, GdirPlotStd

