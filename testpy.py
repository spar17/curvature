import curvature as c
import manifold as m
import numpy as np
import math
import pickle

import numpy as np

# edit: added radius as a parameter
def run_estimator(N, manifold, distType, kfunc, nn, fitVersion, nballs, spacing, radius, rmax = None, bounded = False):
    '''
    returns the mean square error for one experiment, given the parameters as inputs
    manifold: type of surface from which points are sampled
        0 = Sphere
        1 = Euclidean
        2 = Poincare Disk
        3 = Torus
        4 = Hyperboloid
    distType: type of distance metric used to get Rdist matrix
        0 = Estimated Riemannian distances (Isomap)
        1 = True Riemannian distances
        2 = Euclidean distances
    kfunc: type of kernel function used in density calculations
        0 = gauss
        1 = biweight
        2 = epanechnikov
        3 = triweight
    nn: number of nearest neighbors used in Isomap
    fitVersion: version of analytic solution for C used in quadratic fit function
    nballs: number of geodesic balls used in getting ball ratio sequence
    spacing: spacing between radii of adjacent balls if using uniform separation
    '''
    # for actual results, change sample size to 10000
    Tscs = None
    if manifold == 0:
        surf = m.Sphere()
        X = surf.sample(N, 2, R = radius)
    elif manifold == 1:
        surf = m.Euclidean()
        X = surf.sample(N, 2, radius)
    elif manifold == 2:
        surf = m.PoincareDisk()
        X = surf.sample(N, -1, radius)

    # Temporarily ommitting other manifolds because they don't have Rdist_array and distance_array functions
    '''
    elif manifold == 3:
        surf = m.Torus()
        X, thetas = surf.sample(N, 0.5, 1.5)
        Tscs = surf.exact_curvatures(thetas, 0.5, 1.5)
    else:
        surf = m.Hyperboloid()
        X = m.Hyperboloid.sample(N, 2, 1, 2)
    '''
    
    # compute true scalar curvature values at all sample points
    if Tscs == None:
        Tscs = surf.exact_curvatures(X)

    # calculates distance arrays in different ways
    if distType == 0:
        Rdist = None
    elif distType == 1:
        Rdist = surf.Rdist_array(X)
    else:
        Rdist = surf.distance_array(X)

    # BOUNDS = radii for quadratic fit, while BOUND = radii for which points to include in mse calculation
    bounds = None
    bound = None

    # sphere is unbounded so doesn't matter
    if manifold != 0:
        if bounded:
            dists = surf.Rdist0(X)
            # gets distance from boundary if less than rmax by subtracting distance to origin
            bounds = [min(radius - r, rmax) for r in dists]
        # computes subset for mse calculation -- SHOULD BE INDEPENDENT OF BOUNDS i think
        else:
            if (nballs != None) and (spacing != None):
                bound = nballs*spacing
            else:
                bound = rmax
    # creates estimator with nn and kfunc parameters, and estimates curvature at all points
    sce = c.scalar_curvature_est(X, 2, nn, kfunc, Rdist = Rdist, verbose = False, bounds = bounds)
    Ss = sce.estimate_all(rmax = rmax, nballs = nballs, spacing = spacing, version = fitVersion)

    # calculates and returns mse
    mse = surf.mse(X, Ss, radius, bound)
    print(mse)

    return mse

def make_sce(N, manifold, radius, nn, kfunc, distType):
    '''
    returns an estimator object for fixed values of N, manifold, radius, nn, kfunc, distType
    samples points once
    '''
    if manifold == 0:
        surf = m.Sphere()
        X = surf.sample(N, 2, R = radius)
    elif manifold == 1:
        surf = m.Euclidean()
        X = surf.sample(N, 2, radius)
    elif manifold == 2:
        surf = m.PoincareDisk()
        X = surf.sample(N, -1, radius)
    
    # computes appropriate Rdist array based on distType
    if distType == 0:
        Rdist = None
    elif distType == 1:
        Rdist = surf.Rdist_array(X)
    else:
        Rdist = surf.distance_array(X)
        
    sce = c.scalar_curvature_est(X, 2, nn, kfunc, Rdist = Rdist, verbose = False)
    
    return sce

def est_mse(sce, radius, manifold, fitVersion, nballs, spacing, rmax = None):
    '''
    given an sce object, computes mses for a specific radius sequence condition
    '''
    if manifold == 0:
        surf = m.Sphere()
    elif manifold == 1:
        surf = m.Euclidean()
    elif manifold == 2:
        surf = m.PoincareDisk()
    
    X = sce.X
    # uses estimator
    Ss = sce.estimate_all(rmax = rmax, nballs = nballs, spacing = spacing, version = fitVersion)
    # uses mse function specific to manifold, currently uses all points -- ADD PARAMETER FOR BOUND
    mse = surf.mse(X, Ss, radius, None)
    return mse

### FAMILY OF POPULATE FUNCTIONS ###
def populate_mses(N, manifolds, distTypes, kfuncs, nnmax, fits, nballsmax, radius):
    '''
    runs through all combos of conditions and uses run_estimator (which resamples points and generates an sce) to get error
    returns a dictionary of errors with conditions stored as keys

    spacing and nballs values are being manually set at the moment
    '''
    mses = {}

    for manifold in range(manifolds):
        for distType in range(distTypes):
            for kfunc in range(kfuncs):
                for nn in range (20, nnmax + 1, 5):
                    for fitVersion in range(1, fits + 1, 1):
                        # currently manually changing this, pass in nballs as a param
                        for nballs in [25, 50, 75, 100]:
                            # currently manually changing this, pass in spacings as a param
                            for spacing in [None, 0.025, 0.05, 0.075, 0.1]:
                                mse = run_estimator(N, manifold, distType, kfunc, nn, fitVersion, nballs, spacing)
                                mses[(manifold, distType, kfunc, nn, fitVersion, nballs, spacing)] = mse
                        print('finished fit ', fitVersion, 'for ', nn, ' neighbors with kernel function ', kfunc, ' and distType ', distType)
    return mses

def populate_specific(N, manifolds, distTypes, kfuncs, nns, fits, nballvals, radius, rmax = None):
    '''
    populates mse dictionary for a specific set of nballs?
    not entirely sure why this function was made to replace the previous
    '''
    mses = {}
    
    for manifold in manifolds:
        for distType in distTypes:
            for kfunc in kfuncs:
                for nn in nns:
                    for fitVersion in fits:
                        if rmax == None:
                            for nballs in nballvals:
                                for spacing in [0.1, 0.2, 0.3]:
                                    mse = run_estimator(N, manifold, distType, kfunc, nn, fitVersion, nballs, spacing, radius, rmax)
                                    # print("condition: ", manifold, distType, kernels[kfunc], nn, fitVersion, nballs, spacing)
                                    # print("mse: ", mse)
                                    mses[(manifold, distType, kfunc, nn, fitVersion, nballs, spacing)] = mse
                                    print('finished spacing', spacing, 'for', nballs, 'geodesic balls')
                        else:
                            for i in range(20):
                                mse = run_estimator(N, manifold, distType, kfunc, nn, fitVersion, None, None, radius, rmax)
                                # print("condition: ", manifold, distType, kernels[kfunc], nn, fitVersion, nballs, spacing)
                                # print("mse: ", mse)
                                mses[(manifold, distType, kfunc, nn, fitVersion, None, str(i))] = mse
                            print('finished')
    return mses

def fast_populate(N, manifolds, distTypes, kfuncs, nns, fits, nballvals, radius, rmax = None):
    '''
    cuts down on time by not making a new scalar estimator for each combination of parameters
    uses a new func (est_mce) in place of run_estimator
    '''
    mses = {}
    
    for manifold in manifolds:
        for distType in distTypes:
            for kfunc in kfuncs:
                for nn in nns:
                    for fitVersion in fits:
                        sce = make_sce(N, manifold, radius, nn, kfunc, distType)
                        for nballs in nballvals:
                            for spacing in [0.02, 0.04, 0.06, 0.08, 0.1]:
                                mse = est_mse(sce, radius, manifold, fitVersion, nballs, spacing, rmax = None)
                                # print("condition: ", manifold, distType, kernels[kfunc], nn, fitVersion, nballs, spacing)
                                # print("mse: ", mse)
                                mses[(manifold, distType, kfunc, nn, fitVersion, nballs, spacing)] = mse
                                print('finished spacing', spacing, 'for', nballs, 'geodesic balls')
    return mses

def avgmse_condition(manifold = -1, distType = -1, kfunc = -1, nn = -1, fitVersion = -1, nballs = -1, spacing = -1, mses = {}):
    '''
    given a set of condition values, calculates the average mse overall all other parameters
    '''
    conditions = [manifold, distType, kfunc, nn, fitVersion, nballs, spacing]
    match_errors = []
    for key in mses.keys():
        match = True
        for j, setting in enumerate(conditions):
            if (setting != -1) and (key[j] != setting):
                match = False
        if match:
            match_errors.append(mses[key])
    return np.mean(np.array(match_errors))

def unique_conditions(mse):
    '''
    returns all conditions that are iterated through in the grid search by looking at dictionary keys from the calculated mses
    '''
    vals = {'manifold': set(), 'distType': set(), 'kfunc': set(), 'nn': set(), 'fitVersion': set(), 'nballs': set(), 'spacing': set()}
    conditions = ['manifold', 'distType', 'kfunc', 'nn', 'fitVersion', 'nballs', 'spacing']
    for key in mse.keys():
        for num, condition in enumerate(conditions):
            vals[condition].add(key[num])
    return vals
    
def single_param(mse):
    '''
    enumerates average error over all other parameters for every single parameter setting
    '''
    vals = unique_conditions(mse)
    # manifold comparison
    print("average error by manifold: 0 = sphere, 1 = euclidean, 2 = poincare disk, 3 = torus, 4 = hyperboloid")
    for i in vals['manifold']:
        print(i, avgmse_condition(i, -1, -1, -1, -1, -1, -1, mse))

    # distance function comparison
    print("average error by distance function: 0 = estimated riemannian (isomap), 1 = true riemannian, 2 = euclidean")
    for i in vals['distType']:
        print(i, avgmse_condition(-1, i, -1, -1, -1, -1, -1, mse))

    # kernel function comparison
    print("average error by kernel function: 0 = gauss, 1 = biweight, 2 = epanechnikov, 3 = triweight")
    for i in vals['kfunc']:
        print(i, avgmse_condition(-1, -1, i, -1, -1, -1, -1, mse))

    # only did 20 nn for all - not sure what happened?
    print("average error by number of nearest neighbors used in isomap")
    for i in vals['nn']:
        print(i, avgmse_condition(-1, -1, -1, i, -1, -1, -1, mse))

    # fit version comparison
    print("average error by analytic fit equation version")
    for i in vals['fitVersion']:
        print(i, avgmse_condition(-1, -1, -1, -1, i, -1, -1, mse))

    # only did 20 balls for all - also not sure what happened, actually used 25
    print("average error by number of geodesic balls used to generate the quadratic fit")
    for i in vals['nballs']:
        print(i, avgmse_condition(-1, -1, -1, -1, -1, i, -1, mse))

    # spacing comparison
    print("average error by spacing: None = spaced by nearest neighbor distances, all other values are uniform")
    for i in vals['spacing']:
        print(i, avgmse_condition(-1, -1, -1, -1, -1, -1, i, mse))

    return None

def two_params(p1, p2, mse):
    '''
    extension of the previous function
    prints average error over all pairs of parameter settings
    '''
    errors = []
    vals = unique_conditions(mse)
    stuff = [-1, -1, -1, -1, -1, -1, -1]
    conditions = ['manifold', 'distType', 'kfunc', 'nn', 'fitVersion', 'nballs', 'spacing']
    param1 = conditions[p1]
    param2 = conditions[p2]
    print("average error by", param1, "and", param2)
    for i in vals[param1]:
        for j in vals[param2]:
            stuff[p1] = i
            stuff[p2] = j
            manifold, distType, kfunc, nn, fitVersion, nballs, spacing = stuff
            error = avgmse_condition(manifold, distType, kfunc, nn, fitVersion, nballs, spacing, mse)
            errors.append((i, j, error))
            print(i, j, error)
    errors.sort(key=lambda y: y[2])
    return errors
    

def adapt_mse(manifold = -1, distType = -1, kfunc = -1, nn = -1, fitVersion = -1, nballs = -1, spacing = -1, mses = {}):
    '''
    returns trimmed mse with only key/value pairs matching desired condition filters
    '''

    conditions = [manifold, distType, kfunc, nn, fitVersion, nballs, spacing]
    adapted = {}
    for key in mses.keys():
        match = True
        for j, setting in enumerate(conditions):
            if (setting != -1) and (key[j] != setting):
                match = False
        if match:
            adapted[key] = mses[key]
    return adapted

def two_params_sorted(p1, p2, mse):
    errors = []
    vals = unique_conditions(mse)
    stuff = [-1, -1, -1, -1, -1, -1, -1]
    conditions = ['manifold', 'distType', 'kfunc', 'nn', 'fitVersion', 'nballs', 'spacing']
    param1 = conditions[p1]
    param2 = conditions[p2]
    print("average error by", param1, "and", param2)
    for i in vals[param1]:
        for j in vals[param2]:
            stuff[p1] = i
            stuff[p2] = j
            manifold, distType, kfunc, nn, fitVersion, nballs, spacing = stuff
            error = avgmse_condition(manifold, distType, kfunc, nn, fitVersion, nballs, spacing, mse)
            errors.append((i, j, error))
            # print(i, j, error)
    errors.sort(key=lambda y: y[2])
    for item in errors:
        print(item)
    return None

def comp_rs(N, manifolds, distTypes, kfuncs, nns, fits, nballvals, radius, rmax):
    '''
    computes difference in error between using rmax and equally spaced balls for a fixed value of rmax
    '''
    diffs = {}

    for manifold in manifolds:
        for distType in distTypes:
            for kfunc in kfuncs:
                for nn in nns:
                    for fitVersion in fits:
                        mser = run_estimator(N, manifold, distType, kfunc, nn, fitVersion, None, None, radius, rmax)
                        for nballs in nballvals:
                            spacing = rmax / nballs
                            mse = run_estimator(N, manifold, distType, kfunc, nn, fitVersion, nballs, spacing, radius, None)
                            diffs[(manifold, distType, kfunc, nn, fitVersion, nballs, spacing)] = mse - mser
                            print('finished', nballs, 'geodesic balls')
    return diffs

def bestrNU(N, manifold, distType, kfunc, nn, fit, radius, low, high):
    '''
    uses binary search to find the best number of balls (lowest error) with no fixed spacing
    '''
    mses = {}
    currmin = run_estimator(N, manifold, distType, kfunc, nn, fit, 1, None, radius, None)
    print("range: ", low, high)
    if high >= low:
        mid = (high + low) // 2
        mse = run_estimator(N, manifold, distType, kfunc, nn, fit, mid, None, radius, None)
        mseleft = run_estimator(N, manifold, distType, kfunc, nn, fit, mid-1, None, radius, None)
        mseright = run_estimator(N, manifold, distType, kfunc, nn, fit, mid+1, None, radius, None)
        mses[mid] = mse
        print(mid, mseleft, mse, mseright)
        if (mseleft > mse) and (mseright > mse):
            return (mses, mid, mse)
        elif mse > mseleft:
            return bestrNU(N, manifold, distType, kfunc, nn, fit, radius, low, mid-1)
        else:
            return bestrNU(N, manifold, distType, kfunc, nn, fit, radius, mid+1, high)
 
    else:
        return (mses, low, "sus")

def bestspacing(N, manifold, distType, kfunc, nn, fit, nballs, radius, low, high, interval):
    '''
    uses binary search to find the best spacing for a fixed number of balls (lowest error)
    '''
    mses = {}
    currmin = run_estimator(N, manifold, distType, kfunc, nn, fit, nballs, None, radius, None)
    print("range: ", low, high)
    if high >= low:
        mid = (high + low) // 2
        mse = run_estimator(N, manifold, distType, kfunc, nn, fit, nballs, mid*interval, radius, None)
        mseleft = run_estimator(N, manifold, distType, kfunc, nn, fit, nballs, (mid-1)*interval, radius, None)
        mseright = run_estimator(N, manifold, distType, kfunc, nn, fit, nballs, (mid+1)*interval, radius, None)
        mses[mid] = mse
        print(mid, mseleft, mse, mseright)
        if (mseleft > mse) and (mseright > mse):
            return (mses, mid*interval, mse)
        elif mse > mseleft:
            return bestspacing(N, manifold, distType, kfunc, nn, fit, nballs, radius, low, mid-1, interval)
        else:
            return bestspacing(N, manifold, distType, kfunc, nn, fit, nballs, radius, mid+1, high, interval)
 
    else:
        return (mses, low, "sus")

def bestspacing_ball(sce, nballs, low, high, interval):
    '''
    uses binary search to find the best spacing for a fixed number of balls (lowest error) using est_mce
    '''
    mses = {}
    # print("range: ", low, high)
    if high >= low:
        mid = (high + low) // 2
        mse = est_mse(sce, 1, 0, 1, nballs, mid*interval, rmax = None)
        mseleft = est_mse(sce, 1, 0, 1, nballs, (mid-1)*interval, rmax = None)
        mseright = est_mse(sce, 1, 0, 1, nballs, (mid+1)*interval, rmax = None)
        mses[mid] = mse
        # print(mid, mseleft, mse, mseright)
        if (mseleft > mse) and (mseright > mse):
            return (mses, mid*interval, mse)
        elif mse > mseleft:
            return bestspacing_ball(sce, nballs, low, mid-1, interval)
        else:
            return bestspacing_ball(sce, nballs, mid+1, high, interval)
 
    else:
        return (mses, low, "sus")

# Note: bestgb is identical to bestrNU except interval, which for rNU is None, so combine into one function maybe?
    
def bestgb(N, manifold, distType, kfunc, nn, fit, radius, low, high, interval):
    mses = {}
    currmin = run_estimator(N, manifold, distType, kfunc, nn, fit, 1, interval, radius, None)
    print("range: ", low, high)
    if high >= low:
        mid = (high + low) // 2
        mse = run_estimator(N, manifold, distType, kfunc, nn, fit, mid, interval, radius, None)
        mseleft = run_estimator(N, manifold, distType, kfunc, nn, fit, mid-1, interval, radius, None)
        mseright = run_estimator(N, manifold, distType, kfunc, nn, fit, mid+1, interval, radius, None)
        mses[mid] = mse
        print(mid, mseleft, mse, mseright)
        if (mseleft > mse) and (mseright > mse):
            return (mses, mid, mse)
        elif mse > mseleft:
            return bestgb(N, manifold, distType, kfunc, nn, fit, radius, low, mid-1, interval)
        else:
            return bestgb(N, manifold, distType, kfunc, nn, fit, radius, mid+1, high, interval)
 
    else:
        return (mses, low, "sus")

def rmax_array(Ns, radii, nballvals, manifold, distType, kfunc, nn, fit, low, high, interval):
    '''
    returns the optimal spacing for different radii and nballs
    '''
    rs = {}
    for N in Ns:
        for radius in radii:
            for nball in nballvals:
                rs[(N, radius, nball)] = bestspacing(N, manifold, distType, kfunc, nn, fit, nball, radius, low, high, interval)[1:]
    return rs

def rmax_array_specific(sce, low, high, interval):
    rs = {}
    for nball in range(20, 500, 10):
        rs[nball] = bestspacing_ball(sce, nball, low, high, interval)[1:]
    return rs

def getrmaxs(rs):
    rmaxs = {}
    for key in rs.keys():
        spacing, error = rs[key]
        rmaxs[key] = np.array([key[2]*spacing, error, 1])
    return rmaxs

def getavgrmax(N, radius, nball, rmaxs):
    vals = np.zeros(3)
    for key in rmaxs.keys():
        if (N == None or N == key[0]) and (radius == None or radius == key[1]) and (nball == None or nball == key[2]):
            vals = vals + rmaxs[key]
    avgr = vals[0]/vals[2]
    avge = vals[1]/vals[2]
    return (avgr, avge)

def getallavgrs(Ns, radii, nballvals, rmaxs):
    allrs = {}
    print('rmax by Ns')
    for N in Ns:
        things = getavgrmax(N, None, None, rmaxs)
        allrs[(N, None, None)] = things
        print(N, things)
    print('rmax by radii')
    for radius in radii:
        things = getavgrmax(None, radius, None, rmaxs)
        allrs[(None, radius, None)] = things
        print(radius, things)
    print('rmax by nballs')
    for nball in nballvals:
        things = getavgrmax(None, None, nball, rmaxs)
        allrs[(None, None, nball)] = things
        print(nball, things)
    print('rmax by N AND radii')
    for N in Ns:
        for radius in radii:
            things = getavgrmax(N, radius, None, rmaxs)
            allrs[(N, radius, None)] = things
            print(N, radius, things)
    print('rmax by N AND nballs')
    for N in Ns:
        for nball in nballvals:
            things = getavgrmax(N, None, nball, rmaxs)
            allrs[(N, None, nball)] = things
            print(N, nball, things)
    print('rmax by radii AND nballs')
    for radius in radii:
        for nball in nballvals:
            things = getavgrmax(None, radius, nball, rmaxs)
            allrs[(None, radius, nball)] = things
            print(radius, nball, things)
    return allrs

def computedensity(X, kfunc, Rdist, version):
    sce = c.scalar_curvature_est(X, 2, 20, kfunc, Rdist = Rdist, verbose = True, version = version)
    return sce.density

def computeRdist(X, kfunc, radius, version):
    sce = c.scalar_curvature_est(X, 2, 20, kfunc, Rdist = None, verbose = True, version = version)
    return sce.Rdist

if __name__ == "__main__":
    pass