import numpy as np


def zeroCrossing(S,t = None,level = 0,imeth = 'linear'):
    """
     CROSSING find the crossings of a given level of a signal
    ind = CROSSING(S) returns an index vector ind, the signal
    S crosses zero at ind or at between ind and ind+1
    [ind,t0] = CROSSING(S,t) additionally returns a time
    vector t0 of the zero crossings of the signal S. The crossing
    times are linearly interpolated between the given times t
    [ind,t0] = CROSSING(S,t,level) returns the crossings of the
    given level instead of the zero crossings
    ind = CROSSING(S,[],level) as above but without time interpolation
    [ind,t0] = CROSSING(S,t,level,par) allows additional parameters
    par = {'none'|'linear'}.
 	With interpolation turned off (par = 'none') this function always
 	returns the value left of the zero (the data point thats nearest
    to the zero AND smaller than the zero crossing).
 
 	[ind,t0,s0] = ... also returns the data vector corresponding to 
 	the t0 values.
 
 	[ind,t0,s0,t0close,s0close] additionally returns the data points
 	closest to a zero crossing in the arrays t0close and s0close.
 
	This version has been revised incorporating the good and valuable
	bugfixes given by users on Matlabcentral. Special thanks to
 	Howard Fishman, Christian Rothleitner, Jonathan Kellogg, and
 	Zach Lewis for their input. 
    """
    if t == None:
        t = np.arange(0,S.shape[0])
    if len(t) != len(S):
        raise ValueError("t and S must be of identical length!")
    # make row vectors
    t = np.array(t.flatten()).T
    S = np.array(S.flatten()).T
    # always search for zeros. So if we want the crossing of any other threshold value "level",
    # we subtract it from the values and search for zeros.
    S = S - level
    # first look for exact zeros
    ind0 = np.where(S == 0)
    # then look for zero crossings between data points
    S1 = np.multiply(S[0:-1],S[1:])
    ind1 = np.where(S1 < 0)
    # bring exact zeros and "in-between" zeros together 
    ind = np.sort(np.append(ind0,ind1))
    #and pick the associated time values
    t0 = np.double(t[ind]) 
    s0 = S[ind]
    if imeth == 'linear':
        # linear interpolation of crossing
        for ii in range(len(t0)):
            if np.abs(S[ind[ii]]) > np.finfo(float).eps*np.abs(S[ind[ii]]):
                # interpolate only when data point is not already zero
                NUM = t[ind[ii]+1] - t[ind[ii]]
                DEN = S[ind[ii]+1] - S[ind[ii]]
                slope =  NUM / DEN
                terme = S[ind[ii]] * slope
                t0[ii] = t0[ii] - terme
                s0[ii] = 0
    return ind, t0, s0