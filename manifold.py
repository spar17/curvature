# -*- coding: utf-8 -*-
import numpy as np
from scipy.special import gamma
import math
import matplotlib.pyplot as plt
#from mpl_toolkits import mplot3d

##############################################################################
# Sphere sampling
##############################################################################

class Sphere:
    
    # generalized to sphere of not unit radius
    def exact_curvatures(self, X):
        N = X.shape[0]
        R2 = np.sum(X[0, :] * X[0, :])
        # formula = n(n-1)/(r**2), n = 2 for S2 so numerator is 2
        c = 2 / R2
        curvatures = [c]*N
        return curvatures

    # rewritten to not need n

    def Rdist(self, x1, x2):
        Rdist = np.arccos(np.dot(x1, x2))
        return Rdist
        
    def Rdist_array(self, X):
        N = X.shape[0]
        Rdist = np.zeros((N, N))
        for i in range(N):
            for j in range(i):
                x1 = X[i, :]
                x2 = X[j, :]
                Rdist[i, j] = self.Rdist(x1, x2)
                Rdist[j, i] = Rdist[i, j]
        return Rdist

    # gets distance array given only point coordinates

    def distance_array(self, X):
        N = X.shape[0]
        D = np.zeros((N, N))
        for i in range(N):
            x = X[i, :]
            for j in range(i):
                y = X[j, :]
                D[i, j] = np.linalg.norm(x - y)
                D[j, i] = D[i, j]
        return D
    
    def sample(self, N, n, noise = 0, R = 1):
         # To sample a point x, let x_i ~ N(0, 1) and then rescale to have norm R. then add isotropic Gaussian noise to x with variance noise^2
        X = []
        noise_mean = np.zeros(n+1)
        noise_cov = (noise**2)*np.identity(n+1)
        for i in range(N):
            x = np.random.normal(size = n+1)
            x /= np.linalg.norm(x)
            x *= R
            x += np.random.multivariate_normal(noise_mean, noise_cov)
            X.append(x)
        return np.array(X)
    
    def S2_ball_volume(self, r):
        # volume of geodesic ball of radius r in unit 2-sphere
        return 4*math.pi*(math.sin(r/2)**2)
    
    def unit_volume(self, n):
        # returns volume of Euclidean unit n-sphere
        m = n+1
        Sn = (2*(math.pi)**(m/2))/gamma(m/2)
        return Sn
    
    # can also pass radius into exact curvature function to get accurate values
    # taken in radius and bound just to have the same params as mse function for other classes
    def mse(self, X, Ss, radius, bound):
        tscs = np.array(self.exact_curvatures(X))
        ss = np.array(Ss)
        mse = (np.square(ss - tscs)).mean()
        return mse


##############################################################################
# Euclidean sampling
##############################################################################

class Euclidean:
    
    def exact_curvatures(self, X):
        N = X.shape[0]
        curvatures = [0]*N
        return curvatures

    def sample(self, N, n, R):
        # sample N points in a ball of radius R in R^n
        X = []
        for i in range(N):
            x = np.random.normal(size = n)
            u = (R**n)*np.random.random()
            r = u**(1/n)
            x *= r/np.linalg.norm(x)
            X.append(x)
        return np.array(X)

    def density(self, n, R):
        # density in a ball of radius R in R^n
        vn = (math.pi**(n/2))/gamma(n/2 + 1) # volume of Euclidean unit n-ball
        vol = vn*R**(n)
        return 1/vol

    def distance_array(self, X):
        N = X.shape[0]
        D = np.zeros((N, N))
        for i in range(N):
            x = X[i, :]
            for j in range(i):
                y = X[j, :]
                D[i, j] = np.linalg.norm(x - y)
                D[j, i] = D[i, j]
        return D
    
    def Rdist_array(self, X):
        return self.distance_array(X)
    
    def mse(self, X, Ss, radius, bound):
        # compute exact curvatures from points
        tscs = np.array(self.exact_curvatures(X))
        # get array of estimates
        ss = np.array(Ss)
        # if we only want mses of interior points (make sure bound (distance from origin) is less than radius)
        if bound != None and (radius - bound >= 0):
            N = X.shape[0]
            Rdist0 = self.Rdist0(X)
            diffs = []
            for i in range(N):
                # checks that point is within bounded region, can modify this statement to select for points with different properties
                if Rdist0[i] < radius - bound:
                    # new array of diffs
                    diffs.append(ss[i] - tscs[i])
            diffs = np.array(diffs)
            mse = (np.square(diffs)).mean()
        # if there is no bound on which points to include in mse, compute mse for all
        else:
            mse = (np.square(ss - tscs)).mean()
        return mse

    def Rdist0(self, X):
        N, d = X.shape
        Rdist = np.zeros(N)
        for i in range(N):
            x = X[i, :]
            Rdist[i] = np.linalg.norm(x)
        return Rdist
    
##############################################################################
# Torus sampling
##############################################################################

class Torus:
    
    def exact_curvatures(self, thetas, r, R):
        curvatures = [self.S_exact(theta, r, R) for theta in thetas]
        return curvatures

    def sample(self, N, r, R):
        psis = [np.random.random()*2*math.pi for i in range(N)]
        j = 0
        thetas = []
        while j < N:
            theta = np.random.random()*2*math.pi
            #eta = np.random.random()*2*(r/R) + 1 - (r/R)
            #if eta < 1 + (r/R)*math.cos(theta):
            eta = np.random.random()/math.pi
            if eta < (1 + (r/R)*math.cos(theta))/(2*math.pi):
                thetas.append(theta)
                j += 1
    
        def embed_torus(theta, psi):
            x = (R + r*math.cos(theta))*math.cos(psi)
            y = (R + r*math.cos(theta))*math.sin(psi)
            z = r*math.sin(theta)
            return [x, y, z]
    
        X = np.array([embed_torus(thetas[i], psis[i]) for i in range(N)])
        return X, np.array(thetas)
    
    def distance_array(self, X):
        N = X.shape[0]
        D = np.zeros((N, N))
        for i in range(N):
            x = X[i, :]
            for j in range(i):
                y = X[j, :]
                D[i, j] = np.linalg.norm(x - y)
                D[j, i] = D[i, j]
        return D
    
    def S_exact(self, theta, r, R):
        # Analytic scalar curvature
        S = (2*math.cos(theta))/(r*(R + r*math.cos(theta)))
        return S
    
    def theta_index(self, theta, thetas):
        # Returns index in thetas of the angle closest to theta
        err = [abs(theta_ - theta) for theta_ in thetas]
        return np.argmin(err)
    
    def mse(self, thetas, r, R, Ss):
        tscs = np.array(self.exact_curvatures(thetas, r, R))
        ss = np.array(Ss)
        mse = (np.square(ss - tscs)).mean()
        return mse
    
##############################################################################
# Poincare disk sampling
##############################################################################

class PoincareDisk:

    def exact_curvatures(self, X):
        N = X.shape[0]
        curvatures = [-1]*N
        return curvatures
    
    def sample(self, N, K = -1, Rh = 1):
        # N: number of points, K: Gaussian curvature
        # Rh: hyperbolic radius of the disk
        assert K < 0, "K must be negative"
        R = 1/math.sqrt(-K)
        thetas = 2*math.pi*np.random.random(N)
        us = np.random.random(N)
        C1 = 2/math.sqrt(-K)
        C2 = np.sinh(Rh*math.sqrt(-K)/2)
        rs = [C1*np.arcsinh(C2*math.sqrt(u)) for u in us]
        ts = [R*np.tanh(r/(2*R)) for r in rs]
        X = np.array([[ts[i]*math.cos(thetas[i]), ts[i]*math.sin(thetas[i])] for i in range(N)])
        return X
    
    def sample_polar(self, N, K = -1):
        # N: number of points
        # Gaussian curvature is K = -1
        thetas = 2*math.pi*np.random.random(N)
        us = np.random.random(N)
        C1 = 2/math.sqrt(-K)
        C2 = np.sinh(math.sqrt(-K)/2)
        rs = [C1*np.arcsinh(C2*math.sqrt(u)) for u in us]
        X = np.array([[rs[i], thetas[i]] for i in range(N)])
        return X
    
    def cartesian_to_polar(self, X, K = -1):
        R = 1/math.sqrt(-K)
        N = X.shape[0]
        X = []
        for i in range(N):
            x = X[i, 0]
            y = X[i, 1]
            r = 2*R*np.arctanh(math.sqrt(x**2 + y**2)/R)
            theta = np.arccos(x/(R*np.tanh(r/(2*R))))
            X.append([r, theta])
        return X
        
    def polar_to_cartesian(self, X, K = -1):
        R = 1/math.sqrt(-K)
        N = X.shape[0]
        rs = X[:, 0]
        thetas = X[:, 1]
        ts = [R*np.tanh(r/(2*R)) for r in rs]
        X = np.array([[ts[i]*math.cos(thetas[i]), ts[i]*math.sin(thetas[i])] for i in range(N)])
        return X
    
    def Rdist(self, u, v, K = -1):
        assert K < 0, "K must be negative"
        R = 1/math.sqrt(-K)
        z = u/R
        w = v/R
        #wconj = np.array([w[0], -w[1]]) # conjugate of w, thought of as a complex number
        z_wconj = np.array([z[0]*w[0] + z[1]*w[1], w[0]*z[1] - z[0]*w[1]]) # product of z and w_conj, thought of as complex numbers
        dist = 2*R*np.arctanh(np.linalg.norm(z - w)/np.linalg.norm(np.array([1, 0]) - z_wconj))
        return dist
    
    def Rdist_polar(self, u, v):
        # u, v: tuples. polar coordinates (r, theta)
        # Gaussian curvature is K = -1
        r1 = u[0]
        theta1 = u[1]
        r2 = v[0]
        theta2 = v[1]
        return np.arccosh(np.cosh(r1)*np.cosh(r2) - np.sinh(r1)*np.sinh(r2)*np.cos(theta2 - theta1))
    
    def Rdist_array(self, X, K = -1, polar = False):
        # K is the Gaussian curvature of the hyperbolic plane that X is sampled from
        if polar: assert K == -1
        N = X.shape[0]
        Rdist = np.zeros((N, N))
        for i in range(N):
            # print(i)
            for j in range(i):
                x1 = X[i, :]
                x2 = X[j, :]
                if polar:
                    Rdist[i, j] = self.Rdist_polar(x1, x2)
                else:
                    Rdist[i, j] = self.Rdist(x1, x2, K)
                Rdist[j, i] = Rdist[i, j]
        return Rdist
    
    def distance_array(self, X):
        N = X.shape[0]
        D = np.zeros((N, N))
        for i in range(N):
            x = X[i, :]
            for j in range(i):
                y = X[j, :]
                D[i, j] = np.linalg.norm(x - y)
                D[j, i] = D[i, j]
        return D
    
    def area(self, Rh, K = -1):
        # Rh: hyperbolic radius, K: curvature
        assert K < 0
        return (-4*math.pi/K)*(np.sinh(Rh*math.sqrt(-K)/2)**2)
    
    def mse(self, X, Ss, radius, bound):
        # see comments from mse function under Euclidean manifold class
        tscs = np.array(self.exact_curvatures(X))
        ss = np.array(Ss)
        if bound != None:
            N = X.shape[0]
            Rdist0 = self.Rdist0(X)
            diffs = []
            for i in range(N):
                if Rdist0[i] < radius - bound:
                    diffs.append(ss[i] - tscs[i])
            diffs = np.array(diffs)
            mse = (np.square(diffs)).mean()
        else:
            mse = (np.square(ss - tscs)).mean()
        return mse
    
    def Rdist0(self, X):
        N, d = X.shape
        Rdist = np.zeros(N)
        zero = np.zeros(d)
        for i in range(N):
            x = X[i, :]
            Rdist[i] = self.Rdist(x, zero)
        return Rdist


##############################################################################
# Hyperboloid sampling
##############################################################################

# TO DO- organize/clean up

class Hyperboloid:
    
    def exact_curvatures(self, X):
        zs = X[:, 2]
        return self.S(zs)

    def det_g(self, a, c, u):
        return (a**4)*(u**2) + a**2*(u**2 + 1)*c**2
    
    def sample(self, N, a, c, B, within_halfB = False):
        # if within_halfB = False, then sample N points from the hyperboloid with u in [-B, B]
        # if within_halfB = True, then sample points uniformly from u in [-B, B] until there are at least N points with u in [-.5B, .5B]
        sqrt_max_det_g = math.sqrt(self.det_g(a, c, B))
        us = []
        thetas = []
        i = 0
        while i < N:
            theta = 2*math.pi*np.random.random()
            u = 2*B*np.random.random() - B
            eta = sqrt_max_det_g*np.random.random()
            sqrt_det_g = math.sqrt(self.det_g(a, c, u))
            
            if eta < sqrt_det_g:
                us.append(u)
                thetas.append(theta)
                if (within_halfB and -.5*B <= u <= .5*B) or (not within_halfB):
                    i +=1
        xs = [a*math.cos(thetas[i])*math.sqrt(us[i]**2 + 1) for i in range(N)]
        ys = [a*math.sin(thetas[i])*math.sqrt(us[i]**2 + 1) for i in range(N)]
        zs = [c*us[i] for i in range(N)]
        X = np.array([[xs[i], ys[i], zs[i]] for i in range(N)])
        return X

    def distance_array(self, X):
        N = X.shape[0]
        D = np.zeros((N, N))
        for i in range(N):
            x = X[i, :]
            for j in range(i):
                y = X[j, :]
                D[i, j] = np.linalg.norm(x - y)
                D[j, i] = D[i, j]
        return D

    def area(self, a, c, B):
        alpha = math.sqrt(c**2 + a**2)/(c**2)
        cBalpha = c*B*alpha
        return 2*math.pi*a*(math.sqrt(cBalpha**2 + 1)*cBalpha + np.arcsinh(cBalpha))/alpha
    
    def S(self, z):
        # actual scalar curvature at z when a = b = 2 and c = 1
        return -2/((5*z**2 + 1)**2)

    def mse(self, X, Ss):
        tscs = np.array(self.exact_curvatures(X))
        ss = np.array(Ss)
        mse = (np.square(ss - tscs)).mean()
        return mse

##############################################################################
# 3d plotting
##############################################################################

# set_axes_equal and _set_axes_radius are functions from @Mateen Ulhaq and @karlo to make the aspect ratio equal for 3d plots

def set_axes_equal(ax):
    """Set 3D plot axes to equal scale.

    Make axes of 3D plot have equal scale so that spheres appear as
    spheres and cubes as cubes.  Required since `ax.axis('equal')`
    and `ax.set_aspect('equal')` don't work on 3D.
    """
    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])
    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    _set_axes_radius(ax, origin, radius)

def _set_axes_radius(ax, origin, radius):
    x, y, z = origin
    ax.set_xlim3d([x - radius, x + radius])
    ax.set_ylim3d([y - radius, y + radius])
    ax.set_zlim3d([z - radius, z + radius])
    
def plot_3d(X, vals = None):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    if vals is None:
        p = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c = X[:, 2]) # color according to z-coordinate
    else:
        p = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c = vals)
    ax.set_box_aspect([1,1,1])
    set_axes_equal(ax)
    fig.colorbar(p)
    plt.show()