import numpy as np
from funcCalcMTFNew import*
from region_seg import*
from FindLargestSquares import*
from funcCalcContrast import*
from funcPatchAnalysis import*
from funcLineProfileNew import*
from funcPlotGradDirAlongEdge import*
from funcZigZag import*
from funcImageQuality import*
from zeroCrossing import*

a = funcImageQuality('OVGU29NO-I0000148',18,[319,296],[389,286])
print(a[0])
# S = np.array([1,2,-2,-3,-1,0,1,-1,5,-5,0,5,6])
# a, b, c = zeroCrossing(S)
# print(a, b, c)
# I = np.array([[1,2,3,4,5],[2,4,6,8,10],[2,4,6,8,10],[2,4,6,8,10],[1,2,3,4,5]])
# E = Findlargestsquares(I)
# #print(E)
# a = funcZigZag(I)
# #print(a)
# b = funcCalcContrast(I, 2,1)
# #print(b)
# c = funcPatchAnalysis(I,2,2,2)
# print(c)
# x = [1,2]
# y = [2,4]
# Gdir = I
# xEdge = np.array([0,1,1,5])
# yEdge = np.array([0,0,2,1])
# a, b = funcLineProfileNew(I,xEdge,yEdge,Gdir,1,4,5,5,1,-1)
# print(a.shape)
# print(a,b)
# a, b = funcPlotGradDirAlongEdge(xEdge,yEdge,I,I,2)
# print(a,b)
# X = 3
# Y = 3
# xy = {'x':X,
#       'y':Y}
# px = 4
# [a, b, c] = funcCalcMTFNew(I,xy,px,1,1,1,2)