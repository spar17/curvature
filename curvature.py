# -*- coding: utf-8 -*-
import math
# gamma function is generalized factorial
from scipy.special import gamma
import numpy as np
from sklearn.manifold import Isomap
import multiprocessing as mp

# probably don't have to edit this class
class KDE:
    # added a parameter for KDE version, plays around with estimates on the boundary
    def __init__(self, n, X = None, D = None, kernel = None, version = 1):
        '''
        n: dimension of manifold
        X: N x d matrix containing N observations in d-dimensional ambient space
        D: distance matrix. Must input either X or D
        kernel: optional. kernel function (e.g. gauss or biweight. Default is biweight)
        '''
        # ensures we either have X (coordinates of all N sample points to run isomap on) OR D (precomputed pairwise distances, no longer need isomap)
        assert (X is not None) or (D is not None)
        self.n = n
        self.X = X
        self.D = D
        # default kernel function is biweight
        if kernel == None:
            self.kernel = KDE.biweight
        else:
            # sets self.kernel based on numeric input
            if kernel == 0:
                self.kernel = KDE.gauss
            elif kernel == 1:
                self.kernel = KDE.biweight
            elif kernel == 2:
                self.kernel = KDE.epanechnikov
            else:
                self.kernel = KDE.triweight
            
        # count the number of points in the point cloud (N) by getting row dimension of X or D
        if X is not None:
            self.N = X.shape[0]
        else:
            self.N = D.shape[0]
        # computes bandwidth proportional to value as shown in submanifold density estimation paper (aka Scott's rule)
        self.h = KDE.bandwidth(self.N, n)
        self.version = version

    # Alternate density computation with correction for boundary error
    def __call__(self, i):
        
        def dists(i, j):
            if self.D is not None:
                return self.D[i, j]
            else:
                return np.linalg.norm(self.X[j, :] - self.X[i, :])

        oldKDE = 1/(self.N*math.pow(self.h, self.n))*(sum([self.kernel(dists(i, j)/self.h, self.n) for j in range(self.N)]))

        # v0: original KDE calculation with no adjustments for boundary
        if self.version == 0:
            return oldKDE

        # v1: attempts to incorporate boundary distance estimation on submanifolds with boundary
        # lowB: lower bound of distance estimate to boundary
        # b: alternative using true distances for the Euclidean disk of radius 1 for testing purposes
        oldBDE = 1/(self.N*math.pow(self.h, self.n + 1))*(sum([self.kernel(dists(i, j)/self.h, self.n, False)*(self.X[j, :] - self.X[i, :]) for j in range(self.N)]))
        self.c = oldKDE / (math.sqrt(math.pi)*np.linalg.norm(oldBDE))
        lowB = self.h*math.sqrt(max(0, -math.log(self.c/2)))
        b = 1-np.linalg.norm(self.X[i, :])

        if self.version == 1:
            m0 = (1 + math.erf(lowB / self.h))/2
            return 1/(self.N*math.pow(self.h, self.n)*m0)*(sum([self.kernel(dists(i, j)/self.h, self.n, False) for j in range(self.N)]))
            
        if self.version == 21:
            m0 = (1 + math.erf(b / self.h))/2
            return 1/(self.N*math.pow(self.h, self.n)*m0)*(sum([self.kernel(dists(i, j)/self.h, self.n, False) for j in range(self.N)]))
 
        # v2: "cut and normalize" method for BDE vector
        nx = -oldBDE/np.linalg.norm(oldBDE)

        if self.version == 2:
            m0 = (1 + math.erf(lowB / self.h))/2
            return 1/(self.N*math.pow(self.h, self.n)*(1 + math.erf(lowB / self.h)))*(sum([self.kernel(dists(i, j)/self.h, self.n) for j in range(self.N) if np.dot(self.X[j, :], nx) <= lowB]))
            
        if self.version == 22:
            m0 = (1 + math.erf(b / self.h))/2
            return 1/(self.N*math.pow(self.h, self.n)*(1 + math.erf(b / self.h)))*(sum([self.kernel(dists(i, j)/self.h, self.n) for j in range(self.N) if np.dot(self.X[j, :], nx) <= b]))
            
        # v3: higher-order correction (intermediate calculations use h and 2h for bandwidth) 
        if self.version == 3 or self.version == 23:
            if self.version == 3:
                bval = lowB
            else:
                bval = b
            m0 = (1 + math.erf(bval / self.h))/2
            fch = 1/(self.N*math.pow(self.h, self.n)*(1 + math.erf(bval / self.h)))*(sum([self.kernel(dists(i, j)/self.h, self.n) for j in range(self.N) if np.dot(self.X[j, :], nx) <= bval]))
            fc2h = 1/(self.N*math.pow(2*self.h, self.n)*(1 + math.erf(bval / (2*self.h))))*(sum([self.kernel(dists(i, j)/(2*self.h), self.n) for j in range(self.N) if np.dot(self.X[j, :], nx) <= bval]))
            const = (1 + math.erf(bval/(2*self.h)))*math.pow(math.e, -bval**2/(4*self.h**2))/((1 + math.erf(bval/self.h))*math.pow(math.e, -bval**2/(self.h**2)))
            return (2*const*fch - fc2h)/(2*const - 1)

        # using Newton's method on v1 to get a tighter lower bound
        # v4: if Newton's method returns no values, use v1
        # v5: v4 but use v0 when no values are returned
        if self.version == 4:
            bx = KDE.newton(self.f, self.Df, lowB, 0.05, 100)
            if bx == None:
                bx = lowB
            m0 = (1 + math.erf(bx / self.h))/2
            return 1/(self.N*math.pow(self.h, self.n)*m0)*(sum([self.kernel(dists(i, j)/self.h, self.n, False) for j in range(self.N)]))
        
        if self.version == 5:
            bx = KDE.newton(self.f, self.Df, lowB, 0.05, 100)
            if bx == None:
                return oldKDE
            else:
                m0 = (1 + math.erf(bx / self.h))/2
                return 1/(self.N*math.pow(self.h, self.n)*m0)*(sum([self.kernel(dists(i, j)/self.h, self.n, False) for j in range(self.N)])) 
        
        # goal for v6: get bounds for sce using estimated distances so rmax is not constant
        if self.version == 6:
            pass
        return None
    
    def f(self, x):
        a = x/self.h
        return (1 + math.erf(a))*math.pow(math.e, a**2) - self.c

    def Df(self, x):
        a = x/self.h
        return 2/(math.sqrt(math.pi)*self.h) + 2*(1 + math.erf(a))*math.pow(math.e, a**2)*a/(self.h)

    def newton(f, Df, x0, epsilon, max_iter):
        '''Approximate solution of f(x)=0 by Newton's method.

        Parameters
        ----------
        f : function
            Function for which we are searching for a solution f(x)=0.
        Df : function
            Derivative of f(x).
        x0 : number
            Initial guess for a solution f(x)=0.
        epsilon : number
            Stopping criteria is abs(f(x)) < epsilon.
        max_iter : integer
            Maximum number of iterations of Newton's method.

        Returns
        -------
        xn : number
            Implement Newton's method: compute the linear approximation
            of f(x) at xn and find x intercept by the formula
                x = xn - f(xn)/Df(xn)
            Continue until abs(f(xn)) < epsilon and return xn.
            If Df(xn) == 0, return None. If the number of iterations
            exceeds max_iter, then return None.

        Examples
        --------
        >>> f = lambda x: x**2 - x - 1
        >>> Df = lambda x: 2*x - 1
        >>> newton(f,Df,1,1e-8,10)
        Found solution after 5 iterations.
        1.618033988749989
        '''
        try:
            xn = x0
            for n in range(0,max_iter):
                fxn = f(xn)
                if abs(fxn) < epsilon:
                    # print('Found solution after',n,'iterations.')
                    return xn
                Dfxn = Df(xn)
                if Dfxn == 0:
                    # print('Zero derivative. No solution found.')
                    return None
                xn = xn - fxn/Dfxn
                # print(xn, fxn, Dfxn)
            # print('Exceeded maximum iterations. No solution found.')
            return None
        except OverflowError:
            return None

    def density(self):
        # QUESTION: how does this density function work? (calls __call__ from above to compute density, runs in parallel with available processors)
        with mp.Pool(mp.cpu_count()) as p:
            density = p.map(self, np.arange(self.N))
        return density

    # kernel functions, all take a point out of the N and the manifold dimension
    # QUESTION : why do we normalize with manifold dimension? (integral of weights is 1, condition for kernel function of KDE)
    def gauss(x, n, normal = True):
        '''
        Returns Gaussian kernel evaluated at point x
        '''
        return (1/math.pow(math.sqrt(2*math.pi), n))*math.exp(-x*x/2)
    
    def biweight(x, n, normal = True):
        if -1 < x < 1:
            s = 2*((math.pi)**(n/2))/gamma(n/2)
            if normal:
                normalization = s*(1/n - 2/(n+2) + 1/(n+4))
            else:
                normalization = 1
            return ((1-x**2)**2)/normalization
        else:
            return 0
    
    def epanechnikov(x, n, normal = True):
        if -1 < x < 1:
            s = 2*((math.pi)**(n/2))/gamma(n/2)
            if normal:
                normalization = s*(1/n - 1/(n+2))
            else:
                normalization = 1
            return (1-x**2)/normalization
        else:
            return 0
        
    def triweight(x, n, normal = True):
        if -1 < x < 1:
            s = 2*((math.pi)**(n/2))/gamma(n/2)
            if normal:
                normalization = s*(-1/(n+6) + 3/(n+4) - 3/(n+2) + 1/n)
            else:
                normalization = 1
            return ((1-x**2)**3)/normalization
        else:
            return 0
    
    def exponential(x, n, normal = True):
        return (1/math.pow(math.sqrt(2*math.pi), n))*math.exp(-x)
    
    # "Scott's Rule" for given point cloud size (number of samples) and ambient space dimension, lower-dimensional manifold analog replaces d with n
    def bandwidth(N, d):
        return N**(-1/(d+4))
    
class scalar_curvature_est:
    # returns a whole estimator
    def __init__(self, X, n, n_nbrs = 20, kernel = None, density = None, Rdist = None, verbose = True, bounds = None, version = 1):
        '''
        X: N x d matrix containing N observations in d-dimensional ambient space
        n: Integer. dimension of the manifold
        n_nbrs: Integer. number of neighbors to use for Isomap Riemannian distance estimation (Isomap default is 5 but this is generally way too low)
        density: (optional) density[i] is an estimate of the density at X[i, :]
        Rdist: (optional) N x N matrix of Riemannian distances (exact or precomputed approximate distances).
        bounds: maximum of precomputed distance from each point to the boundary and max radius

        Initializes an estimator object, uses helper funcs to compute Riemannian distances and densities if needed
        '''
        # verbose: prints statements for functions with no console output
        self.X = X
        self.n = n
        self.n_nbrs = n_nbrs
        self.density = density
        self.N = X.shape[0] # number of sample points
        self.d = X.shape[1] # dimension of the ambient space
        self.Vn = (math.pi**(self.n/2))/gamma(self.n/2 + 1) # formula for volume of Euclidean unit n-ball
        if Rdist is None:
            self.Rdist = scalar_curvature_est.compute_Rdist(X, n_nbrs)
            if verbose: print("computed Rdist")
        else:
            self.Rdist = Rdist

        # computes largest nearest neighbor distance for any point to avoid zero division errors in using bounds
        self.low = np.max(np.sort(self.Rdist)[:, 1])

        # computes value of distance to nearest neighbor by taking the min of all "geodesic" distances to other points; snnd[i] = value for ith point in cloud
        self.nearest_nbr_dist = [np.min([self.Rdist[i, j] for j in range(self.N) if j!= i]) for i in range(self.N)]
        if verbose: print("computed nearest neighbor distances")
        
        # uses KDE to estimate densities if not given, KDE func uses formulas from KDE paper
        self.kernel = kernel
        if density is None:
            self.density = scalar_curvature_est.compute_density(n, X, kernel, version)
            if verbose: print("computed density")
        else:
            self.density = density

        self.bounds = bounds
        
    def ball_ratios(self, i, rmax = None, rs = None):
        '''
        i: index of observation in X (a row)
        rmax: (optional) positive real number. scale at which we're computing scalar curvature (max radius in the rs sequence, if rs isn't given)
        rs: (optional) increasing sequence of radii
        Must input either rmax or rs.
        
        Returns:
        rs: sequence of radii in range [0, rmax]. 
            If not initially given: r_j is estimated Riemannian distance from x_i 
                to its jth nearest neighbor.
            If given as input: returns given rs
        ball_ratios: sequence of estimaed ratios of geodesic ball volume to 
            euclidean ball volume, for each radius r in rs.
        '''
        assert (rmax is not None) or (rs is not None)
        rs, ball_vols = self.ball_volumes(i, rmax, rs)
        # enumerate gives pairs where j is the index counter and r is the actual radius value
        ball_ratios = np.array([ball_vols[j]/(self.Vn*(r**self.n)) for j, r in enumerate(rs)])
        return rs, ball_ratios
    
    def ball_volumes(self, i, rmax = None, rs = None):
        '''
        i: integer. index of observation in X (a row)
        rmax: (optional) positive real number. scale at which we're computing scalar curvature
        rs: (optional) increasing sequence of rs
        Must input either rmax or rs.
        
        Returns:
        rs: sequence of radii in range [0, rmax]. 
            If not initially given: r_j is estimated Riemannian distance from x_i to its jth nearest neighbor.
            If given as input: returns given rs
        ball_volumes: sequence of estimated geodesic ball volumes, for each radius r in rs
        '''
        assert (rmax is not None) or (rs is not None)
        if rmax is not None:
            # gets distances to neighbors within rmax, each successive radius includes one more neighbor so rs = list of distances
            rs, nbrs = self.nbr_distances(i, rmax)
            N_rs = [(j+2) for j in range(len(rs))] # N_r = number of points in ball of radius r, for r in rs -- since the first ball includes point i and its closest neighbor, index j has j+2 points
        else:
            # set rmax as largest radius in sequence
            nbr_dists, nbrs = self.nbr_distances(i, rs[-1])
            num_nbrs = len(nbr_dists)
            N_rs = []
            # iterates through distances once and finds indices at which dist <= current r, as rs are sorted, we can go successively
            k = 0
            for r in rs:
                while k < num_nbrs and nbr_dists[k] <= r:
                    k += 1
                N_rs.append(k+1) # k neighbors within B_r, plus the center point
        
        # Calculate average ball density for every ball in the sequence
        density = self.get_density()
        center_little_ball_vol = self.Vn*(.5*self.nearest_nbr_dist[i])**self.n
        num = density[i]*center_little_ball_vol
        denom = center_little_ball_vol
        avg_ball_density = []
        if rmax is not None:
            for nbr_idx in nbrs:
                little_ball_vol = self.Vn*(.5*self.nearest_nbr_dist[nbr_idx])**self.n
                num += density[nbr_idx]*little_ball_vol
                denom += little_ball_vol
                avg_ball_density.append(num/denom)
        else:
            k = 0
            for r in rs:
                if k < num_nbrs and nbr_dists[k] <= r:
                    nbr_idx = nbrs[k]
                    little_ball_vol = self.Vn*(.5*self.nearest_nbr_dist[nbr_idx])**self.n
                    num += density[nbr_idx]*little_ball_vol
                    denom += little_ball_vol
                    k += 1
                avg_ball_density.append(num/denom)
        
        # Calculate estimated ball volumes via MLE formula
        ball_volumes = [N_rs[j]/(self.N*avg_ball_density[j]) for j in range(len(rs))]
            
        # returns radii and sequence of corresponding ball volumes
        return rs, ball_volumes
       
    def compute_ball_ratios(self, i):
        '''
        Computes ball ratio sequence centered at the ith point
        '''
        # only to be called by compute_ball_ratio_seqs
        _, ball_ratios = self.ball_ratios(i, rs = self.rs)
        return ball_ratios
    
    def compute_ball_ratio_seqs(self, rmax):
        '''
        Gets ball ratio sequences for each point in point cloud
        '''
        rs, ball_ratios = self.ball_ratios(0, rmax)
        self.rs = rs
        ball_ratio_seqs = [None for i in range(self.N)]
        ball_ratio_seqs[0] = ball_ratios
        
        with mp.Pool(mp.cpu_count()) as p:
            ball_ratio_seqs[1:] = p.map(self.compute_ball_ratios, np.arange(1, self.N))
        
        self.ball_ratio_seqs = ball_ratio_seqs
        return ball_ratio_seqs
        
    def compute_Rdist(X, n_nbrs = 20):
        '''
        Uses isomap algo to get estimates for Riemannian distances to use in estimator if not given a distance matrix, takes points and neighbor count as inputs and returns a distance matrix
        '''
        # n_nbrs: integer. Parameter to pass to isomap. (number of neighbors in nearest neighbor graph)
        # QUESTION: returns the "embedded" coordinates, is this in the dimension of the manifold? is it automatically computed? (manifold, yes it's computed, but it's more info than we need)
        iso = Isomap(n_neighbors = n_nbrs, n_jobs = -1)
        iso.fit(X)
        # QUESTION: is this based on the graph or the embedded coordinates (graph, isomap func does more work than we need currently)
        Rdist = iso.dist_matrix_
        return Rdist
    
    def compute_density(n, X, kernel = None, version = 1):
        '''
        Uses KDE to estimate density at each sof the N points (default kernel is biweight)
        '''
        kde = KDE(n, X, None, kernel, version)
        density = kde.density()
        # print(density)
        return density

    # Added parameters nballs and spacing:
    # Case I: rmax only, Case II: rs only (both previously covered)
    # Case III: nballs only = get r_min from nn and get rs from each successive neighbor until reaching nballs
    # Case IV: nballs + spacing = get r_min and rs sequence from adding spacing intervals
    # passes params estimate_all -> fit_quad_coeff -> get rs functions
    
    def estimate_all(self, rmin = None, rmax = None, rs = None, nballs = None, spacing = None, version = 1):
        '''
        Computes scalar curvatures using second degree coefficient of quadratic fit to ball ratio sequence
        '''
        # TO DO- parallelize this

        Cs = [self.fit_quad_coeff(i, rmin, rmax, rs, nballs, spacing, version) for i in range(self.N)]
        Ss = [-6*(self.n + 2)*C for C in Cs]
        return Ss
        
    def estimate_with_averaging(self, rmax, k = 0):
        # k: number of neighbors to average ball_ratios over
        # TO DO- allow user to input sequence rs
        self.k = k
        
        #self.compute_ball_ratio_seqs(rmax)
        rs, ball_ratios = self.ball_ratios(0, rmax)
        self.rs = rs
        
        self.rmax = rmax
        with mp.Pool(mp.cpu_count()) as p:
            Cs = p.map(self.fit_quad_coeff_helper, np.arange(self.N))
            
        Ss = [-6*(self.n + 2)*C for C in Cs]
        return Ss
        
    def fit_quad_coeff(self, i, rmin = None, rmax = None, rs = None, nballs = None, spacing = None, version = 1):
        '''
        Parameters
        ----------
        i : integer
            index of observation in X (a row)
        rmin : nonnegative real number, optional
            Radius at which to start the sequence of rs, if rs isn't given. The default is None.
        rmax : positive real number, optional
            Max radius to consider in the sequence of rs, if rs isn't given. The default is None.
        rs : increasing sequence of nonnegative real numbers, optional
        version : {1, 2}, optional
            version 1 corresponds to Eq (10) in the overleaf. version 2 is Eq (11). The default is 1.

        Returns
        -------
        C : real number
            The quadratic coefficient of a polynomial of form 1 + C*r^2 that we fit to the data (r_i, y_i = estimated ball ratio at radius r_i)

        '''
        # if no bounds are given, obtains radii values from other parameters
        if self.bounds == None:
            if (rmax == None) and (rs == None):
                if (spacing == None):
                    rmax = self.rmax_from_nballs(i, nballs)
                else:
                    rs = self.rs_from_spacing(i, nballs, spacing)
        # in the case that distances from the boundary are given, makes sure they are larger than the minimum NN distance (self.low)
        else:
            rmax = max(self.low, self.bounds[i])

        rs, ball_ratios = self.ball_ratios(i, rmax = rmax, rs = rs)
        if rmin is not None:
            ball_ratios = ball_ratios[rs > rmin]
            rs = rs[rs > rmin]

        if version == 1:
            numerator = sum(np.array([(ball_ratios[i] - 1)*r**2 for i, r in enumerate(rs)]))
            denom = sum(np.array([r**4 for r in rs]))
            C = numerator/denom
        else:
            rs = np.append(rs, 0) # so that r[-1] = 0. need this for the rs[i] - rs[i-1] term below.
            numerator = sum(np.array([(r**2)*(ball_ratios[i] - 1)*(r - rs[i-1]) for i, r in enumerate(rs[:-1])]))
            denom = rs[-2]**5/5
            C = numerator/denom
        return C
    
    def fit_quad_coeff_helper(self, i):
        _, nbrs = self.nbr_distances(i, self.rmax)
        k = self.k
        k_nbrs = nbrs[:k]
        _, i_ball_ratios = self.ball_ratios(i, rs = self.rs)
        ball_ratio_sums = i_ball_ratios
        for nbr in k_nbrs:
            _, nbr_ball_ratios = self.ball_ratios(nbr, rs = self.rs)
            for j in range(len(self.rs)):
                ball_ratio_sums[j] = ball_ratio_sums[j] + nbr_ball_ratios[j]
        ball_ratio_avgs = [ball_sum/(k+1) for ball_sum in ball_ratio_sums]       
        numerator = sum(np.array([(ball_ratio_avgs[j] - 1)*r**2 for j, r in enumerate(self.rs)]))
        denom = sum(np.array([r**4 for r in self.rs]))
        C = numerator/denom
        return C
    
    # collection of get functions for accessing attributes of the class
    def get_ball_ratio_seqs(self, rmax = None):
        if self.ball_ratio_seqs is None:
            self.compute_ball_ratio_seqs(rmax)
        return self.ball_ratio_seqs
    
    def get_density(self):
        if self.density is None:
            self.density = scalar_curvature_est.compute_density(self.n, self.X, self.kernel)
        return self.density
    
    def get_Rdist(self):
        if self.Rdist is None:
            self.Rdist = scalar_curvature_est.compute_Rdist(self.X, self.n_nbrs)
        return self.Rdist

    def rmax_from_nballs(self, i, nballs):
        # gets matrix of all Riemannian distances
        Rdist = self.get_Rdist()
        # sorts distances to neighbors for point of interest
        distances = np.sort(Rdist[i, :])
        # returns radius value for neighbor nn
        rmax = distances[nballs]
        return rmax
    
    def dist_to_nn(self, i):
        # gets matrix of all Riemannian distances
        Rdist = self.get_Rdist()
        # sorts distances to neighbors for point of interest
        distances = np.sort(Rdist[i, :])
        # returns radius value for neighbor nn
        rmin = distances[1]
        return rmin
    
    def rs_from_spacing(self, i, nballs, spacing):
        dists = np.ones(nballs)*self.dist_to_nn(i)
        rs = np.array([dists[j] + j*spacing for j in range(nballs)])
        return rs

    def nbr_distances(self, i, rmax):
        '''
        i: index of observation in X (i.e., index of a row of X) at which we're estimating scalar curvature
        rmax: positive real number. scale at which we're computing scalar curvature
        
        Returns
            nbr_indices: sorted (ascending order by distance to X[i, :]) list of indices of neighbors that are within rmax of X[i, :]
            distances: sorted (ascending order) list of Riemannian distances from X[i, :] to its neighbors that are within rmax of X[i, :]
        '''
        # gets matrix of all Riemannian distances
        Rdist = self.get_Rdist()
        # looks at adjacencies for point of interest
        distances = Rdist[i, :]
        
        nbr_indices = np.argsort(distances)[1:] # get neighbor indices in order (sorted by distance from i), and remove index i (which is first because dist = 0)
        distances = distances[nbr_indices] # gets sorted distances? run a sample to check
        close_enough = (distances <= rmax) # counts how many are within the bound
        distances = distances[close_enough] # gets only the distances less than rmax
        nbr_indices = nbr_indices[close_enough] # corresponding indices
        return distances, nbr_indices
    
    def quad_coeff_errs(self, i, rmax, version = 1, l2norm = True):
        '''
        i: index of observation in X (a row) at which we're estimating scalar curvature
        rmax: positive real number.
        version : {1, 2}, optional
            version 1 corresponds to Eq (10) in the overleaf. version 2 is Eq (11). The default is 1.  
        '''
        rs, ball_ratios = self.ball_ratios(i, rmax)
        if l2norm:
            numerator_terms = np.array([(ball_ratios[i] - 1)*r**2 for i, r in enumerate(rs)])
            denom_terms = np.array([r**4 for r in rs])
            num_sum = 0
            denom_sum = 0
            Cs = []
            for i in range(len(rs)):
                num_sum += numerator_terms[i]
                denom_sum += denom_terms[i]
                C = num_sum/denom_sum
                Cs.append(C)
            errs = []
            err1 = 0
            err2 = 0 
            err3 = 0
            for i, r in enumerate(rs):
                err1 += r**4
                err2 += (r**2)*(1 - ball_ratios[i])
                err3 += (1 - ball_ratios[i])**2
                sq_err = (Cs[i]**2)*err1 + 2*Cs[i]*err2 + err3
                avg_err = math.sqrt(max(0, sq_err))/r # theoretically, sq_err is always positive. because of floating point errors, it can be negative (but very very small, on the order of 10^(-17))
                errs.append(avg_err)
        else:
            rs = np.append(rs, 0)
            numerator_terms = np.array([(r**2)*(ball_ratios[i] - 1)*(r - rs[i-1]) for i, r in enumerate(rs[:-1])])
            denom = (rs[-2]**5)/5
            num_sum = 0
            Cs = []
            for i, num in enumerate(numerator_terms):
                num_sum += num
                C = num_sum/denom
                Cs.append(C)
            err1 = 0
            err2 = 0
            err3 = 0
            errs = []
            for i, r in enumerate(rs[:-1]):
                err1 += (r**4)*(r - rs[i-1])
                err2 += (r**2)*(1 - ball_ratios[i])*(r - rs[i-1])
                err3 += ((1 - ball_ratios[i])**2)*(r - rs[i-1])
                sq_err = (Cs[i]**2)*err1 + 2*Cs[i]*err2 + err3
                avg_err = math.sqrt(max(0, sq_err))/r # theoretically, sq_err is always positive. because of floating point errors, it can be negative (but very very small, on the order of 10^(-17))
                errs.append(avg_err)
        return rs, Cs, errs

