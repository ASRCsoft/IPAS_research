"""Class for ice cluster calculations after formation"""

import numpy.linalg as la
import math
import shapely.ops as shops
from pyquaternion import Quaternion
import copy as cp
import numpy as np
import scipy.optimize as opt
import shapely.geometry as geom
import shapely.affinity as sha
from shapely.geometry import Point
import matplotlib.pyplot as plt
import random
from ipas import IceCrystal as crys
from descartes.patch import PolygonPatch
import descartes
import mpl_toolkits.mplot3d.art3d as art3d
import time


class IceCluster:
    def __init__(self, crystal, size=1):
        # needed for bookkeeping:
        self.rotation = Quaternion()
        self.points = np.full((size, 12), np.nan,
                              dtype=[('x', float), ('y', float), ('z', float)])
        self.points[0] = crystal.points
        self.size = size
        self.ncrystals = 1
        # used for some calculations involving shapely objects
        self.tol = 10 ** -11
        # Used for the fit_ellipse function. I really do not like that
        # I have to set this so high, arrr.
        self.tol_ellipse = 10 ** -4.5
        self.major_axis = {}
        self.minor_axis = {}

    def crystals(self, i=None):
    # return a crystal with the same points and attributes as the
    # nth crystal in the cluster
        if i is None:
            crystals = []
            for n in range(self.ncrystals):
                cr = crys.IceCrystal(1, 1)
                cr.points = self.points[n]
                cr.rotation = self.rotation
                cx = cr.points['x'].mean()
                cy = cr.points['y'].mean()
                cz = cr.points['z'].mean()
                cr.center = [cx, cy, cz]
                cr.maxz = cr.points['z'].max()
                cr.minz = cr.points['z'].min()
                crystals.append(cr)
            return crystals
        else:
            cr = crys.IceCrystal(1, 1)
            cr.points = self.points[i]  #i = 0
            cr.rotation = self.rotation
            cx = cr.points['x'].mean()
            cy = cr.points['y'].mean()
            cz = cr.points['z'].mean()
            cr.center = [cx, cy, cz]
            cr.maxz = cr.points['z'].max()
            cr.minz = cr.points['z'].min()
            return cr

    def _add_crystal(self, crystal):
        n = self.ncrystals
        if n < self.size:
            self.points[n] = crystal.points
        else:
            self.points = np.append(self.points, [crystal.points], axis=0)
        self.ncrystals += 1

    def move(self, xyz):
        # move the entire cluster
        self.points['x'][:self.ncrystals] += xyz[0]
        self.points['y'][:self.ncrystals] += xyz[1]
        self.points['z'][:self.ncrystals] += xyz[2]

    def max(self, dim):
        return self.points[dim][:self.ncrystals].max()

    def min(self, dim):
        return self.points[dim][:self.ncrystals].min()

    def _euler_to_mat(self, xyz):
        # Euler's rotation theorem, any rotation may be described using three angles
        [x, y, z] = xyz
        rx = np.matrix([[1, 0, 0], [0, np.cos(x), -np.sin(x)], [0, np.sin(x), np.cos(x)]])
        ry = np.matrix([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
        rz = np.matrix([[np.cos(z), -np.sin(z), 0], [np.sin(z), np.cos(z), 0], [0, 0, 1]])
        return rx * ry * rz

    def _rotate_mat(self, mat):
        points = cp.copy(self.points)
        self.points['x'] = points['x'] * mat[0, 0] + points['y'] * mat[0, 1] + points['z'] * mat[0, 2]
        self.points['y'] = points['x'] * mat[1, 0] + points['y'] * mat[1, 1] + points['z'] * mat[1, 2]
        self.points['z'] = points['x'] * mat[2, 0] + points['y'] * mat[2, 1] + points['z'] * mat[2, 2]

    def rotate_to(self, angles):
        # rotate to the orientation given by the 3 angles

        # get the rotation from the current position to the desired
        # rotation
        current_rot = self.rotation
        rmat = self._euler_to_mat(angles)
        desired_rot = Quaternion(matrix=rmat)
        rot_mat = (desired_rot * current_rot.inverse).rotation_matrix
        self._rotate_mat(rot_mat)

        # save the new rotation
        self.rotation = desired_rot

    # def rotate_to(self, angles):
    #     # rotate the entire cluster

    #     # first get back to the original rotation
    #     if any(np.array(self.rotation) != 0):
    #         self._rev_rotate(self.rotation)

    #     # now add the new rotation
    #     self._rotate(angles)

    #     # save the new rotation
    #     self.rotation = angles

    def center_of_mass(self):  #of cluster
        x = np.mean(self.points[:self.ncrystals]['x'])
        y = np.mean(self.points[:self.ncrystals]['y'])
        z = np.mean(self.points[:self.ncrystals]['z'])
        return [x, y, z]

    def recenter(self):
        center_move = self.center_of_mass()
        self.move([ -x for x in center_move])
        return center_move

    def plot(self):
        return geom.MultiLineString([ lines for crystal in self.crystals() for lines in crystal.plot() ])

    def _crystal_projectxy(self, n):
        return geom.MultiPoint(self.points[n][['x', 'y']]).convex_hull

    def _crystal_projectxz(self, n):
        return geom.MultiPoint(self.points[n][['x', 'z']]).convex_hull

    def _crystal_projectyz(self, n):
        return geom.MultiPoint(self.points[n][['y', 'z']]).convex_hull

    def projectxy(self):
        polygons = [ self._crystal_projectxy(n).buffer(0) for n in range(self.ncrystals) ]
        return shops.cascaded_union(polygons)

    def projectxz(self):
        polygons = [ self._crystal_projectxz(n) for n in range(self.ncrystals) ]
        return shops.cascaded_union(polygons)

    def projectyz(self):
        polygons = [ self._crystal_projectyz(n) for n in range(self.ncrystals) ]
        return shops.cascaded_union(polygons)

    def calculate_S_ratio(self, plates, crystal):
        start = time.clock()
        #Calculate separation of crystals for further collection restriction

        crystals1 = [self, crystal]
        dmaxclus = []
        dmaxnew = []
        n=0
        for i in crystals1:

            x,y = list(i.projectxy().exterior.coords.xy)

            dinit = 0
            for j in range(len(x)):
                for l in range(len(x)):
                    d =(Point(x[l],y[l]).distance(Point(x[j],y[j])))
                    if d > dinit:
                        dinit = d

                        if n == 0:
                            dmaxclus.append(d)
                        if n == 1:
                            dmaxnew.append(d)
                        xstart = l
                        ystart = l
                        xend = j
                        yend = j


            if n == 0:
                dmaxclus = max(dmaxclus)

            if n== 1:
                dmaxnew = max(dmaxnew)

            n+=1

        l = (self.projectxy().centroid).distance(crystal.projectxy().centroid)
        S = 2*l/(dmaxclus+dmaxnew)

        if plates:
            lmax = 1.0*(dmaxclus+dmaxnew)/2 #S parameter can't be higher than 0.6 for plates
        else:
            lmax = 1.0*(dmaxclus+dmaxnew)/2 #S parameter can't be higher than 0.3 for columns
        #print('lmax', plates, lmax)
        
        end = time.clock()
        #print("slow time %.2f" % (end-start))
        return S, lmax

    def place_crystal(self, plates, crystal):
        #get max distance between centers of crystals for collection bounds (no tip to tip)
        #returns a random x/y value to place the new_crystal over the aggregate within the bounded circle
        S, lmax = self.calculate_S_ratio(plates, crystal)

        lmax_bound = self.projectxy().centroid.buffer(lmax)  #new crystal center can't be outside of this circle

        rand = random.uniform(0, 1.0)
        angle = rand * np.pi * 2
        rand2 = random.uniform(0, 1.0)
        radius = np.sqrt(rand2) * lmax
        originX = self.projectxy().centroid.xy[0]
        originY = self.projectxy().centroid.xy[1]

        x = originX + radius * np.cos(angle)
        y = originY + radius * np.sin(angle)

        random_loc = [x[0], y[0], 0]

        return random_loc, lmax_bound

    def add_flow_tilt(self, crystal, lmax):

        center_aggx, center_aggy = self.projectxy().centroid.xy
        center_newx, center_newy = crystal.projectxy().centroid.xy
        #tilt fracs between 0 and 1
        tilt_fracx = ((np.abs(center_newx[0] - center_aggx[0]))/lmax)*(self.projectxy().area/crystal.projectxy().area)
        tilt_fracy = ((np.abs(center_newy[0] - center_aggy[0]))/lmax)*(self.projectxy().area/crystal.projectxy().area)
        '''
        #print('center agg',self.projectxy().centroid.xy)
        #print('center crys',crystal.projectxy().centroid.xy)
        print('top', (np.abs(center_newx[0] - center_aggx[0])))
        print('lmax', lmax)
        print('first frac center distancex',(np.abs(center_newx[0] - center_aggx[0])/lmax))
        print('first frac center distancey',(np.abs(center_newy[0] - center_aggy[0])/lmax))

        print('area ratio',self.projectxy().area/crystal.projectxy().area)
        print('tilt_fracx, tiltfracy',tilt_fracx, tilt_fracy)
        print('-------------')
        '''
        return tilt_fracx, tilt_fracy

    def add_crystal_from_above(self, crystal, lodge=0):
        # drop a new crystal onto the cluster

        # use the bounding box to determine which crystals to get
        xmax = max(crystal.projectxy().exterior.coords.xy[0])
        ymax = max(crystal.projectxy().exterior.coords.xy[1])
        xmin = min(crystal.projectxy().exterior.coords.xy[0])
        ymin = min(crystal.projectxy().exterior.coords.xy[1])

        close = np.all([self.points['x'][:self.ncrystals].max(axis=1) >= xmin,
                        self.points['x'][:self.ncrystals].min(axis=1) <= xmax,
                        self.points['y'][:self.ncrystals].max(axis=1) >= ymin,
                        self.points['y'][:self.ncrystals].min(axis=1) <= ymax], axis=0)


        which_close = np.where(close)

        close_crystals = [ self.crystals(n) for n in which_close[0] ]

        # see which crystals could actually intersect with the new crystal
        close_crystals = [ x for x in close_crystals if x.projectxy().intersects(crystal.projectxy()) ]

        # close_crystals = [ x for x in self.crystals() if x.projectxy().intersects(newpoly) ]
        if len(close_crystals) == 0:
            return False # the crystal missed!

        # we know highest hit is >= max(minzs), therefore the first
        # hit can't be below (max(minzs) - height(crystal))
        minzs = [ crystal2.minz for crystal2 in close_crystals ]
        first_hit_lower_bound = max(minzs) - (crystal.maxz - crystal.minz)
        # remove the low crystals, sort from highest to lowest
        close_crystals = [ x for x in close_crystals if x.maxz > first_hit_lower_bound ]
        close_crystals.sort(key=lambda x: x.maxz, reverse=True)

        # look to see where the new crystal hits the old ones
        mindiffz = crystal.minz - first_hit_lower_bound # the largest it can possibly be

        for crystal2 in close_crystals:
            if first_hit_lower_bound > crystal2.maxz:
                break # stop looping if the rest of the crystals are too low
            diffz = crystal.min_vert_dist(crystal2)
            if diffz is None:
                break

            #return diffz
            # update if needed
            if diffz < mindiffz:
                mindiffz = diffz
                first_hit_lower_bound = crystal.minz - mindiffz
                # take the highest hit, move the crystal to that level
        crystal.move([0, 0, -mindiffz - lodge])


        # append new crystal to list of crystals
        # self.crystals.append(crystal)
        self._add_crystal(crystal)
        # fin.
        return True

    def reorient(self, method='random', rotations=50):

        if method == 'IDL':
            # based on max_agg3.pro from IPAS
            max_area = 0
            current_rot = self.rotation
            for i in range(rotations):
                [a, b, c] = [np.random.uniform(high=np.pi / 4), np.random.uniform(high=np.pi / 4),
                             np.random.uniform(high=np.pi / 4)]
                # for mysterious reasons we are going to rotate this 3 times
                rot1 = self._euler_to_mat([a, b, c])
                rot2 = self._euler_to_mat([b * np.pi, c * np.pi, a * np.pi])
                rot3 = self._euler_to_mat([c * np.pi * 2, a * np.pi * 2, b * np.pi * 2])
                desired_rot = Quaternion(matrix=rot1 * rot2 * rot3)
                rot_mat = (desired_rot * current_rot.inverse).rotation_matrix
                self._rotate_mat(rot_mat)
                new_area = self.projectxy().area
                if new_area >= max_area:
                    max_area = new_area
                    max_rot = desired_rot

                # save our spot
                current_rot = desired_rot
            # rotate new crystal to the area-maximizing rotation
            rot_mat = (max_rot * current_rot.inverse).rotation_matrix
            self._rotate_mat(rot_mat)
        elif method == 'random':
            # same as schmitt but only rotating one time, with a real
            # random rotation
            max_area = 0
            current_rot = self.rotation
            for i in range(rotations):

                desired_rot = Quaternion.random()
                rot_mat = (desired_rot * current_rot.inverse).rotation_matrix
                self._rotate_mat(rot_mat)
                new_area = self.projectxy().area
                if new_area >= max_area:
                    max_area = new_area
                    max_rot = desired_rot

                # save our spot
                current_rot = desired_rot
            # rotate new crystal to the area-maximizing rotation
            rot_mat = (max_rot * current_rot.inverse).rotation_matrix
            self._rotate_mat(rot_mat)

        #self.rotation = Quaternion()

        # elif method == 'bh':
        #     # use a basin-hopping algorithm to look for the optimal rotation
        #     def f(x):
        #         # yrot = np.arccos(x[1]) - np.pi/2
        #         # self.rotate_to([x[0], yrot, 0])
        #         self.rotate_to([x[0], x[1], 0])
        #         return -self.projectxy().area
        #     # lbfgsb_opt = {'ftol': 1, 'maxiter': 5}
        #     # min_kwargs = {'bounds': [(0, np.pi), (0, np.pi)], 'options': lbfgsb_opt}
        #     # # min_kwargs = {'bounds': [(0, np.pi), (0, np.pi)]}
        #     # opt_rot = opt.basinhopping(f, x0=[np.pi/2, np.pi/2], niter=15, stepsize=np.pi / 7,
        #     #                            interval=5, minimizer_kwargs=min_kwargs)
        #     lbfgsb_opt = {'ftol': 1, 'maxiter': 0, 'maxfun': 4}
        #     min_kwargs = {'bounds': [(0, np.pi), (-0.99, 0.99)], 'options': lbfgsb_opt}
        #     # min_kwargs = {'bounds': [(0, np.pi), 0, np.pi)]}
        #     opt_rot = opt.basinhopping(f, x0=[np.pi/2, np.pi/2], niter=30, stepsize=np.pi / 4,
        #                                interval=10, minimizer_kwargs=min_kwargs)
        #     # xrot = opt_rot.x[0]
        #     # yrot = np.arccos(opt_rot.x[1]) - np.pi / 2
        #     [xrot, yrot] = opt_rot.x
        #     # area at rotation + pi is the same, so randomly choose to
        #     # add those
        #     # if np.random.uniform() > .5:
        #     #     xrot += np.pi
        #     # if np.random.uniform() > .5:
        #     #     yrot += np.pi
        #     zrot = np.random.uniform(high=2 * np.pi) # randomly choose z rotation
        #     self.rotate_to([xrot, yrot, zrot])
        #     self.rotation = [0, 0, 0]
        #     return opt_rot

        # elif method == 'diff_ev':
        #     def f(x):
        #         # yrot = np.arccos(x[1]) - np.pi/2
        #         # self.rotate_to([x[0], yrot, 0])
        #         self.rotate_to([x[0], x[1], 0])
        #         return -self.projectxy().area
        #     opt_rot = opt.differential_evolution(f, [(0, np.pi), (-1, 1)],
        #                                          maxiter=10, popsize=15)
        #     # xrot = opt_rot.x[0]
        #     # yrot = np.arccos(opt_rot.x[1]) - np.pi / 2
        #     [xrot, yrot] = opt_rot.x
        #     zrot = np.random.uniform(high=2 * np.pi) # randomly choose z rotation
        #     self.rotate_to([xrot, yrot, zrot])
        #     self.rotation = [0, 0, 0]
        #     return opt_rot

    def _get_moments(self, poly):
        # get 'mass moments' for this cluster's 2D polygon using a
        # variation of the shoelace algorithm
        xys = poly.exterior.coords.xy
        npoints = len(xys[0])
        # values for the three points-- point[n], point[n+1], and
        # (0,0)-- making up triangular slices from the origin to the
        # edges of the polygon
        xmat = np.array([xys[0][0:-1], xys[0][1:], np.zeros(npoints - 1)]).transpose()
        ymat = np.array([xys[1][0:-1], xys[1][1:], np.zeros(npoints - 1)]).transpose()
        # arrange the points in left-center-right order
        x_order = np.argsort(xmat, axis=1)
        ordered_xmat = xmat[np.array([range(npoints - 1)]).transpose(), x_order]
        ordered_ymat = ymat[np.array([range(npoints - 1)]).transpose(), x_order]
        xl = ordered_xmat[:, 0]
        xm = ordered_xmat[:, 1]
        xr = ordered_xmat[:, 2]
        yl = ordered_ymat[:, 0]
        ym = ordered_ymat[:, 1]
        yr = ordered_ymat[:, 2]
        # which slices have areas on the left and right sides of the
        # middle point? Ignore values smaller than 'tol' so we don't
        # run into terrible problems with division.
        left = xm - xl > self.tol_ellipse
        right = xr - xm > self.tol_ellipse
        # slope and intercept of line connecting left and right points
        has_area = xr != xl
        m3 = np.zeros(npoints - 1)
        m3[has_area] = (yr[has_area] - yl[has_area]) / (xr[has_area] - xl[has_area])
        b3 = -xl * m3 + yl
        # the y coordinate of the line connecting the left and right
        # points at the x position of the middle point
        m3_mid = yl + m3 * (xm - xl)
        # is the midpoint above or below that line?
        mid_below = ym < m3_mid
        # line connecting left and middle point (where applicable)
        m1 = (ym[left] - yl[left]) / (xm[left] - xl[left])
        b1 = -xl[left] * m1 + yl[left]
        # line connecting middle and right point (where applicable)
        m2 = (yr[right] - ym[right]) / (xr[right] - xm[right])
        b2 = -xr[right] * m2 + yr[right]
        # now that we have the points in a nice format + helpful
        # information we can calculate the integrals of the slices
        xx = np.zeros(npoints - 1)
        xy = np.zeros(npoints - 1)
        yy = np.zeros(npoints - 1)
        dxl = (xm[left] - xl[left])
        dx2l = (xm[left] ** 2 - xl[left] ** 2)
        dx3l = (xm[left] ** 3 - xl[left] ** 3)
        dx4l = (xm[left] ** 4 - xl[left] ** 4)
        dxr = (xr[right] - xm[right])
        dx2r = (xr[right] ** 2 - xm[right] ** 2)
        dx3r = (xr[right] ** 3 - xm[right] ** 3)
        dx4r = (xr[right] ** 4 - xm[right] ** 4)
        # x^2
        xx[left] = dx4l * (m1 - m3[left]) / 4 +\
                   dx3l * (b1 - b3[left]) / 3
        xx[right] += dx4r * (m2 - m3[right]) / 4 +\
                     dx3r * (b2 - b3[right]) / 3
        # x*y
        xy[left] = dx4l * (m1 ** 2 - m3[left] ** 2) / 8 +\
                   dx3l * (b1 * m1 - b3[left] * m3[left]) / 3 +\
                   dx2l * (b1 ** 2 - b3[left] ** 2) / 4
        xy[right] += dx4r * (m2 ** 2 - m3[right] ** 2) / 8 +\
                     dx3r * (b2 * m2 - b3[right] * m3[right]) / 3 +\
                     dx2r * (b2 ** 2 - b3[right] ** 2) / 4
        # y^2
        yy[left] = dx4l * (m1 ** 3 - m3[left] ** 3) / 12 +\
                   dx3l * (b1 * m1 ** 2 - b3[left] * m3[left] ** 2) / 3 +\
                   dx2l * (b1 ** 2 * m1 - b3[left] ** 2 * m3[left]) / 2 +\
                   dxl * (b1 ** 3 - b3[left] ** 3) / 3
        yy[right] += dx4r * (m2 ** 3 - m3[right] ** 3) / 12 +\
                     dx3r * (b2 * m2 ** 2- b3[right] * m3[right] ** 2) / 3 +\
                     dx2r * (b2 ** 2 * m2 - b3[right] ** 2 * m3[right]) / 2 +\
                     dxr * (b2 ** 3 - b3[right] ** 3) / 3
        # if the middle point was below the other points, multiply by
        # minus 1
        xx[mid_below] *= -1
        xy[mid_below] *= -1
        yy[mid_below] *= -1
        # find out which slices were going clockwise, and make those
        # negative
        points = np.array([xys[0], xys[1]]).transpose()
        cross_prods = np.cross(points[:-1], points[1:])
        clockwise = cross_prods < 0
        xx[clockwise] *= -1
        xy[clockwise] *= -1
        yy[clockwise] *= -1
        # add up the totals across the entire polygon
        xxtotal = np.sum(xx)
        yytotal = np.sum(yy)
        xytotal = np.sum(xy)
        # and if the points were in clockwise order, flip the sign
        if np.sum(cross_prods) < 0:
            xxtotal *= -1
            yytotal *= -1
            xytotal *= -1
        # also need to account for the holes, if they exist
        for linestring in list(poly.interiors):
            hole = geom.Polygon(linestring)
            hole_moments = self._get_moments(hole)
            xxtotal -= hole_moments[0]
            yytotal -= hole_moments[1]
            xytotal -= hole_moments[2]
        return [xxtotal, yytotal, xytotal]


    def mvee(self, tol = 0.01):  #mve = minimum volume ellipse
        # Based on work by Nima Moshtagh
        #http://www.mathworks.com/matlabcentral/fileexchange/9542

        """
        Finds the ellipse equation in "center form"
        (x-c).T * A * (x-c) = 1
        """
        pi = np.pi
        sin = np.sin
        cos = np.cos
        points_arr = np.concatenate(self.points)[:self.ncrystals*12]
        #print('points_Arr', points_arr)
        points_arr = np.array([list(i) for i in points_arr])
        N, d = points_arr.shape
        Q = np.column_stack((points_arr, np.ones(N))).T

        err = tol+1.0
        u = np.ones(N)/N
        while err > tol:
            # assert u.sum() == 1 # invariant
            X = np.dot(np.dot(Q, np.diag(u)), Q.T)
            M = np.diag(np.dot(np.dot(Q.T, la.inv(X)), Q))
            jdx = np.argmax(M)
            step_size = (M[jdx]-d-1.0)/((d+1)*(M[jdx]-1.0))
            new_u = (1-step_size)*u
            new_u[jdx] += step_size
            err = la.norm(new_u-u)
            u = new_u

        c = np.dot(u,points_arr)

        A = la.inv(np.dot(np.dot(points_arr.T, np.diag(u)), points_arr)
                   - np.multiply.outer(c,c))/d

        return A, c

    def spheroid_axes(self):
        A, c = self.mvee()
        U, D, V = la.svd(A)
        rx, ry, rz = 1./np.sqrt(D)
        return rx, ry, rz

    def ellipse(self, u, v, rx, ry, rz):
        x = rx*np.cos(u)*np.cos(v)
        y = ry*np.sin(u)*np.cos(v)
        z = rz*np.sin(v)
        return x,y,z


    def plot_ellipsoid(self):

        points_arr = np.concatenate(self.points)[:self.ncrystals*12]
        points_arr = np.array([list(i) for i in points_arr])

        A, centroid = self.mvee()
        #print('centroid', centroid)
        U, D, V = la.svd(A)
        #print(U, D, V)
        rx, ry, rz = self.spheroid_axes()

        u, v = np.mgrid[0:2*np.pi:20j, -np.pi/2:np.pi/2:10j]

        Ve = 4./3.*rx*ry*rz
        #print(Ve)

        E = np.dstack(self.ellipse(u, v, rx, ry, rz))
        E = np.dot(E,V) + centroid

        xell, yell, zell = np.rollaxis(E, axis = -1)

        x = np.zeros((len(self.points['x']),27))
        y = np.zeros((len(self.points['x']),27))
        z = np.zeros((len(self.points['x']),27))

        X = self.points['x']
        Y = self.points['y']
        Z = self.points['z']

        Xlim = self.points['x'][:self.ncrystals]
        Ylim = self.points['y'][:self.ncrystals]
        Zlim = self.points['z'][:self.ncrystals]
        #for i in range(0, 360, 60):
        #    print('angle', i)

        fig = plt.figure(figsize=(10,7))
        ax = fig.add_subplot(111, projection='3d')

        #ax.view_init(elev=90, azim=270) #z-orientation
        #ax.view_init(elev=0, azim=90) #yz
        ax.view_init(elev=0, azim=0) #xz
        ax.plot_surface(xell, yell, zell, cstride = 1, rstride = 1, alpha = 0.6)

        data = []
        #print(self.ncrystals)
        for l in range(self.ncrystals):

            prismind = [0,6,7,1,2,8,9,3,4,10,11,5]  #prism lines
            i = 0
            for n in prismind:
                x[l][i] = X[l][n]
                y[l][i] = Y[l][n]
                z[l][i] = Z[l][n]
                i+=1

            if l == len(self.points['x'][:self.ncrystals])-1:
                color = 'orange'
            else:
                color = 'orange'
            ax.plot(x[l][0:12], y[l][0:12], z[l][0:12],color=color)

            i = 0
            for n in range(0,6): #basal face lines

                x[l][i+12] = X[l][n]
                y[l][i+12] = Y[l][n]
                z[l][i+12] = Z[l][n]
                i+=1

            x[l][18] = X[l][0]
            y[l][18] = Y[l][0]
            z[l][18] = Z[l][0]

            ax.plot(x[l][12:19], y[l][12:19], z[l][12:19], color=color)

            i = 0
            for n in range(6,12): #basal face lines

                x[l][i+19] = X[l][n]
                y[l][i+19] = Y[l][n]
                z[l][i+19] = Z[l][n]
                i+=1

            x[l][25] = X[l][6]
            y[l][25] = Y[l][6]
            z[l][25] = Z[l][6]

            ax.plot(x[l][19:26], y[l][19:26], z[l][19:26], color=color)


            maxX = np.max(Xlim)
            minX = np.min(Xlim)
            maxY = np.max(Ylim)
            minY = np.min(Ylim)
            maxZ = np.max(Zlim)
            minZ = np.min(Zlim)

            maxXe = np.max(xell)
            minXe = np.min(xell)
            maxYe = np.max(yell)
            minYe = np.min(yell)
            maxZe = np.max(zell)
            minZe = np.min(zell)

            maxxyz = max(maxX, maxY, maxZ)
            minxyz = min(minX,minY,minZ)

            minell = min(minXe,minYe,minZe)
            maxell = max(maxXe, maxYe, maxZe)
            #print('min',minell, maxell)
            ax.set_xlim(minxyz, maxxyz)
            ax.set_ylim(minxyz, maxxyz)
            ax.set_zlim(minxyz, maxxyz)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_zticklabels([])
            
            dims = 'yz'
            self.plot_ellipse(dims)
            
            #ax.view_init(30, i)
            #plt.pause(.001)
        plt.show()
        #path=('/Users/vprzybylo/Desktop/fit_ellipsoid.eps')
        #plt.savefig(path)

    def plot_constraints(self, plates, new_crystal, k, plot_dots = plot):
        import shapely.ops as shops

        #fig = plt.figure(1, figsize=(5,5))
        #ax = fig.add_subplot(111)
        fig, ax = plt.subplots(1, 1)
        area = []
        centroid = []
        dmax1 = []
        dmax2 = []

        for i in range(2):

            if i == 0:
                color='#29568F'
                zorder = 3
                linecolor = '#29568F'
                self.ncrystals = self.ncrystals-1
                projpoly = self.projectxy()
                self.ncrystals = self.ncrystals + 1
            else:
                zorder = 2
                color='#e65c00'
                linecolor = '#e65c00'

                projpoly = geom.MultiPoint(self.points[self.ncrystals-1][['x', 'y']]).convex_hull
                #projpoly = new_Crystal.projectxy()

                self.ncrystals = self.ncrystals-1
                agg_nonew_area = self.projectxy().buffer(0)
                self.ncrystals = self.ncrystals + 1

                rel_area = agg_nonew_area.intersection(projpoly.buffer(0))
                ovrlpptch = PolygonPatch(rel_area, fill=True, ec='k', fc='k', zorder=3)
                ax.add_patch(ovrlpptch)
                #xovrlp,yovrlp = list(rel_area.exterior.coords.xy)
                #ax.plot(xovrlp, yovrlp, 'o',color ='green', linewidth = 3, zorder =4)

            area.append(projpoly.area)
            centroid.append(projpoly.centroid)
            cryspatch = PolygonPatch(projpoly, fill=True, ec='k', fc=color, zorder=zorder,  alpha=1.0)
            ax.add_patch(cryspatch)

            x, y = list(projpoly.exterior.coords.xy)


            dinit = 0
            for j in range(len(x)):
                for l in range(len(x)):
                    d =(Point(x[l],y[l]).distance(Point(x[j],y[j])))
                    if d > dinit:
                        dinit = d

                        if i == 0:
                            dmax1.append(d)
                        if i == 1:
                            dmax2.append(d)
                        xstart = l
                        ystart = l
                        xend = j
                        yend = j


            if i == 0:
                dmax1 = max(dmax1)

            if i == 1:
                dmax2 = max(dmax2)


            ax.plot([x[xstart], x[xend]], [y[ystart], y[yend]], color ='w', linewidth = 3, zorder=8)

            ax.plot(x, y, 'o',color ='w', linewidth = 3, zorder = 11)


        l = centroid[0].distance(centroid[1])
        S = 2*l/(dmax1+dmax2)

        if plates:
            lmax = 1.0*(dmax1+dmax2)/2 #force S = .6 for plates
        else:
            lmax = 1.0*(dmax1+dmax2)/2 #force S = .3 for columns

        lmax_bound = agg_nonew_area.centroid.buffer(lmax)
        #new crystal center can't be outside of this circle
        ax.add_patch(descartes.PolygonPatch(lmax_bound, fc='gray', ec='k', alpha=0.3, zorder=8))


        ax.plot([centroid[0].x, centroid[1].x],[centroid[0].y, centroid[1].y], color= 'gold',linewidth = 4, zorder=8)

        if plot_dots:
            i = 0
            while i < 1000:
                angle = random.uniform(0, 1) * np.pi * 2
                radius = np.sqrt(random.uniform(0, 1)) * lmax
                originX = lmax_bound.centroid.xy[0]
                originY = lmax_bound.centroid.xy[1]


                x = originX + radius * np.cos(angle)
                y = originY + radius * np.sin(angle)
                ax.scatter(x[0],y[0], color ='g', s=10, zorder = 10)
                i+=1

        #nmisses = 100
        #n = 0
        #while n < nmisses:
            #if centroid[1].within(lmax_bound):

        xmin = min(lmax_bound.exterior.coords.xy[0])
        xmax = max(lmax_bound.exterior.coords.xy[0])
        ymin = min(lmax_bound.exterior.coords.xy[1])
        ymax = max(lmax_bound.exterior.coords.xy[1])
        #print(xmin, xmax, ymin, ymax)
        #print(centroid[1])
        square = geom.Polygon([(xmin,ymax),(xmax, ymax),(xmax, ymin),(xmin,ymin)])

        #break
        #else:
                #n+=1
        #ax.add_patch(descartes.PolygonPatch(square, fc='gray', ec='k', alpha=0.1))

        ax.axis('scaled')
        if plot_dots:
            path=('/Users/vprzybylo/Desktop/ovrlp_constraint'+str(k)+'_dots'+'.pdf')
        else:
            path=('/Users/vprzybylo/Desktop/ovrlp_constraint'+str(k)+'.pdf')

        #plt.savefig(path)

        plt.show()

    def fit_ellipse(self, dims):
        # Emulating this function, but for polygons in continuous
        # space rather than blobs in discrete space:
        # http://www.idlcoyote.com/ip_tips/fit_ellipse.html

        if dims == [['x','y']] or dims =='xy':
            poly = self.projectxy()
        if dims == [['x','z']] or dims =='xz':
            poly = self.projectxz()
        if dims == [['y','z']] or dims =='yz':
            poly = self.projectyz()
        
        if dims == 'xz':
            dims = [['x','z']]
        if dims == 'yz':
            dims = [['y','z']]
        if dims == 'xy':
            dims = [['x','y']]
        xy_area = poly.area

        # center the polygon around the centroid
        centroid = poly.centroid
        poly = sha.translate(poly, -centroid.x, -centroid.y)

        # occasionally we get multipolygons
        if isinstance(poly, geom.MultiPolygon):
            xx = 0
            yy = 0
            xy = 0
            for poly2 in poly:
                moments = self._get_moments(poly2)
                xx += moments[0] / xy_area
                yy += moments[1] / xy_area
                xy -= moments[2] / xy_area
        else:
            moments = self._get_moments(poly)
            xx = moments[0] / xy_area
            yy = moments[1] / xy_area
            xy = -moments[2] / xy_area

        # get fit ellipse axes lengths, orientation, center
        m = np.matrix([[yy, xy], [xy, xx]])
        evals, evecs = np.linalg.eigh(m)
        semimajor = np.sqrt(evals[0]) * 2
        semiminor = np.sqrt(evals[1]) * 2
        major = semimajor * 2
        minor = semiminor * 2

        evec = np.squeeze(np.asarray(evecs[0]))
        orientation = np.arctan2(evec[1], evec[0]) * 180 / np.pi

        ellipse = {'xy': [centroid.x, centroid.y], 'width': minor,
                   'height': major, 'angle': orientation}
        #print('crystals',self.ncrystals)
        #print('ell',ellipse['height'])
        return ellipse


    def plot_ellipse(self, dims):
        from matplotlib.patches import Ellipse
        from descartes import PolygonPatch
        import operator

        #Only (x,z), (y,z), and (x,y) needed/allowed for dimensions
        #Depth works for both side views (x,z) and (y,z)

        if dims == [['x','z']] or dims =='xz':
            #self.rotate_to([np.pi / 2, 0, 0])
            poly = self.projectxz()
        elif dims == [['y','z']] or dims =='yz':
            #self.rotate_to([np.pi / 2, np.pi / 2, 0])
            poly = self.projectyz()
        elif dims == [['x','y']] or dims =='xy':  
            #this is the only projection used in the aggregate aspect ratio calculation
            #self.rotate_to([0, 0, 0])
            poly = self.projectxy()
        else:
            print('Not a valid dimension')
            
            
        params = self.fit_ellipse(dims)
        ellipse = Ellipse(**params)

        fig = plt.figure(0)
        ax = fig.add_subplot(111)
        ax.add_artist(ellipse)
        ellipse.set_alpha(.9)  #opacity
        ellipse.set_facecolor('darkorange')
        #if isinstance(poly, geom.multipolygon.MultiPolygon):
        #    for poly2 in poly:
        #        x, y = poly2.exterior.xy
                #ax.plot(x, y, color = 'green', linewidth = 3)
        #else:
        #    x, y = poly.exterior.xy
            #ax.plot(x, y, color = 'green', linewidth = 3)

        #maxdim = max([params['width'], params['height']]) / 2
        #ax.set_xlim([-maxdim + params['xy'][0], maxdim + params['xy'][0]])
        #ax.set_ylim([-maxdim + params['xy'][1], maxdim + params['xy'][1]])

        if dims == 'xz':
            dims = [['x','z']]
        if dims == 'yz':
            dims = [['y','z']]
        if dims == 'xy':
            dims = [['x','y']]  
            
        for l in range(len(dims)):
            crysmaxz = []
            crysminz = []
            maxzinds = []
            minzinds = []
            for i in range(self.ncrystals):
                hex1pts = self.points[dims[l]][i][0:6]  #first basal face
                poly1 = geom.Polygon([[p[0], p[1]] for p in hex1pts]) #make it into a polygon to plot
                hex2pts = self.points[dims[l]][i][6:12]  #second basal face
                poly2 = geom.Polygon([[p[0], p[1]] for p in hex2pts])
                x1,y1 = poly1.exterior.xy  #array of xy points
                x2,y2 = poly2.exterior.xy

                if i == 1:
                    color = 'navy'
                    zorder = 3
                else:
                    color = 'darkgreen'
                    zorder = 4
                for n in range(7):  #plot the prism face lines
                    x = [x1[n],x2[n]]
                    y = [y1[n],y2[n]]
                    ax.plot(x,y, color = color, zorder = zorder, linewidth = '2')

                #polypatch1 = PolygonPatch(poly1, fill=True, zorder = 1)
                #polypatch2 = PolygonPatch(poly2, fill=True, zorder = 1)
                ax.plot(x1,y1, color = color, zorder = 2, linewidth = '2') #edges of polygons
                ax.plot(x2,y2, color = color, zorder = 4, linewidth = '2')
                #ax.add_patch(polypatch1)
                #ax.add_patch(polypatch2)

                #for plotting depth line segment:
                crysminz.append(self.points['z'][i].min())
                crysmaxz.append(self.points['z'][i].max())
                minzinds.append(np.argmin(self.points['z'][i]))  #index of min pt for every xtal
                maxzinds.append(np.argmax(self.points['z'][i])) #index of max pt

            maxcrysind, self.maxz = max(enumerate(crysmaxz), key=operator.itemgetter(1))  #overall max btwn xtals
            mincrysind, self.minz = min(enumerate(crysminz), key=operator.itemgetter(1))
   
            xdepthmin = self.points[dims[l]][mincrysind][minzinds[mincrysind]]
            xdepthmax = self.points[dims[l]][maxcrysind][maxzinds[maxcrysind]]
        

            #ax.plot(xdepthmin, self.minz, 'ko', linewidth = '4')
            #ax.plot(xdepthmax, self.maxz, 'ko', linewidth = '4')
            depthlinex = [0,0]
            depthliney = [self.minz, self.maxz]
            ax.plot(depthlinex,depthliney, 'k', linewidth = '4')

            ######## plot major and minor axes ############

            maxdim = max([params['width'], params['height']]) / 2  #major axis

            #ax.set_xlim([-maxdim + params['xy'][0], maxdim + params['xy'][0]])
            #ax.set_ylim([-maxdim + params['xy'][1], maxdim + params['xy'][1]])

            leftverticex = params['xy'][0]-params['width']/2
            leftverticey = params['xy'][1]
            rightverticex = params['xy'][0]+params['width']/2
            rightverticey = params['xy'][1]
            #plt.plot(leftverticex, leftverticey, 'ro', markersize = 5)  #original vertices if no angle
            #plt.plot(rightverticex, rightverticey, 'ro', markersize = 5)
            #plt.plot(params['xy'][0], params['xy'][1], 'wo', markersize = 7)


            radangle = params['angle']*np.pi/180
            #orientation angle of ellipse

            #rotate axis points and reposition if off center
            newxleft = ((leftverticex - params['xy'][0])*np.cos(radangle)-\
                        (leftverticey-params['xy'][1])*np.sin(radangle)) + params['xy'][0]

            newxright = ((rightverticex - params['xy'][0])*np.cos(radangle)-\
                         (rightverticey - params['xy'][1])*np.sin(radangle)) + params['xy'][0]

            newyleft = ((leftverticex - params['xy'][0])*np.sin(radangle)+\
                        (leftverticey-params['xy'][1])*np.cos(radangle)) + params['xy'][1]

            newyright = ((rightverticex - params['xy'][0])*np.sin(radangle)+\
                        (rightverticey-params['xy'][1])*np.cos(radangle)) + params['xy'][1]

            newx = [newxleft, newxright]
            newy = [newyleft, newyright]
            ax.plot(newx, newy, color ='white', linewidth = 3)  #major/minor axis lines
            ax.plot(newx, newy, 'wo', markersize = 7)

            radangle1 = params['angle']*np.pi/180 + np.pi/2
            radangle = radangle1
            leftverticex = params['xy'][0]-params['height']/2
            rightverticex = params['xy'][0]+params['height']/2

            newxleft = ((leftverticex - params['xy'][0])*np.cos(radangle)-\
                        (leftverticey-params['xy'][1])*np.sin(radangle)) + params['xy'][0]

            newxright = ((rightverticex - params['xy'][0])*np.cos(radangle)-\
                         (rightverticey - params['xy'][1])*np.sin(radangle)) + params['xy'][0]

            newyleft = ((leftverticex - params['xy'][0])*np.sin(radangle)+\
                        (leftverticey-params['xy'][1])*np.cos(radangle)) + params['xy'][1]

            newyright = ((rightverticex - params['xy'][0])*np.sin(radangle)+\
                        (rightverticey-params['xy'][1])*np.cos(radangle)) + params['xy'][1]

            newx = [newxleft, newxright]
            newy = [newyleft, newyright]
            ax.plot(newx, newy, color ='white', linewidth = 3)
            ax.plot(newx, newy, 'wo', markersize = 2)


            ax.set_aspect('equal', 'datalim')

        return params
    '''
    def write_obj(self, filename):
        f = open(filename, 'w')

        faces = []
        for i, crystal in enumerate(self.crystals()):
            nc = i * 12
            # write the vertices
            for n in range(12):
                f.write('v ' + ' '.join(map(str, crystal.points[n])) + '\n')
            # write the hexagons
            for n in range(2):
                coords = range(n * 6 + 1 + nc, (n + 1) * 6 + 1 + nc)
                faces.append('f ' + ' '.join(map(str, coords)))
            # write the rectangles
            for n in range(5):
                coords = [n + 1 + nc, n + 2 + nc, n + 8 + nc, n + 7 + nc]
                faces.append('f ' + ' '.join(map(str, coords)))
            # write the last rectangle I missed
            coords = [nc + 6, nc + 1, nc + 7, nc + 12]
            faces.append('f ' + ' '.join(map(str, coords)))
        f.write('\n'.join(faces))
        f.close()

    def intersect(self):
        from operator import itemgetter
        # return a multiline object representing the edges of the prism
        hex_cntmax = np.empty([self.ncrystals],dtype='object')
        hex_cntmin = np.empty([self.ncrystals],dtype='object')
        for c in range(self.ncrystals):
            # make a line connecting the two hexagons at the max x value
            dim = ['y','z']
            hex1pts = self.points[dim][c,0:6]
            hex2pts = self.points[dim][c,6:12]
            hex1max = max(self.points[c,0:6][dim],key=itemgetter(0))
            hex2max = max(self.points[c,6:12][dim],key=itemgetter(0))
            hex_cntmax[c] = geom.LineString((hex1max, hex2max))
            print(hex1pts)
            print(hex1max)
            print(hex2pts)
            print(hex2max)
            hex1min = min(self.points[c,0:6][dim],key=itemgetter(0))
            hex2min = min(self.points[c,6:12][dim],key=itemgetter(0))

            hex_cntmin[c] = geom.LineString((hex1min, hex2min))

        intersect = False

        max_intersect = hex_cntmax[0].intersects(hex_cntmax[1])
        min_intersect = hex_cntmin[0].intersects(hex_cntmin[1])
        minmax_intersect = hex_cntmin[0].intersects(hex_cntmax[1])
        maxmin_intersect = hex_cntmax[0].intersects(hex_cntmin[1])
        if max_intersect==True:
            intersect = True
            print('max')
        if min_intersect==True:
            intersect = True
            print('min')
        if minmax_intersect==True:
            intersect = True
            print('minmax')
        if maxmin_intersect == True:
            intersect = True
            print('maxmin')
        print(intersect)

        return intersect

    '''
    def aspect_ratio(self, method, minor):
        # rotation = self.rotation

        #get depth measurement in z

        self.maxz = self.points['z'][:self.ncrystals].max()
        self.minz = self.points['z'][:self.ncrystals].min()
        #print(self.maxz, self.minz, self.maxz-self.minz)
        self.depth = self.maxz-self.minz

        self.rotate_to([0, 0, 0])

        # getting ellipse axes from 3 perspectives
        ellipse = {}
        dims = [['x','y']]
        ellipse['z'] = self.fit_ellipse(dims)
        dims = [['x','z']]
        self.rotate_to([np.pi / 2, 0, 0])
        ellipse['y'] = self.fit_ellipse(dims)
        dims = [['y','z']]
        self.rotate_to([np.pi / 2, np.pi / 2, 0])
        ellipse['x'] = self.fit_ellipse(dims)

        # put the cluster back
        self.rotate_to([0, 0, 0])

        for dim in ellipse.keys():

            self.major_axis[dim] = max(ellipse[dim]['height'], ellipse[dim]['width'])
            self.minor_axis[dim] = min(ellipse[dim]['height'], ellipse[dim]['width'])

        if minor == 'minorxy':
            if method == 1:
                return max(self.major_axis.values()) / max(self.minor_axis.values())
            elif method == 'plate':
                return max(self.minor_axis['x'],self.minor_axis['y']) / self.major_axis['z']
            elif method == 'column':
                return self.major_axis['z'] / max(self.minor_axis['x'],self.minor_axis['y'])
        elif minor == 'depth': #use depth as minor dimension of aggregate
            if method == 1:
                return max(self.major_axis.values()) / max(self.minor_axis.values())
            elif method == 'plate':
                #print(self.depth, self.major_axis['z'], self.depth/self.major_axis['z'])
                return self.depth / self.major_axis['z']
            elif method == 'column':
                #print(self.major_axis['z'], self.depth, self.major_axis['z']/self.depth)
                return self.major_axis['z'] / self.depth

    def aspect_ratio_2D(self):

        # getting fit ellipse in z orientation
        ellipse = {}

        dims = [['x','y']]
        ellipse['z'] = self.fit_ellipse(dims)

        return self.minor_axis['z']/ self.major_axis['z']

    def overlap(self, new_crystal, seedcrystal):

        self.ncrystals = self.ncrystals-1
        agg_nonew = self.projectxy().buffer(0)
        self.ncrystals = self.ncrystals + 1

        #print('aggnonew',agg_nonew.area)
        #print('agg', self.projectxy().area)
        #print('new', new_crystal.projectxy().area)
        #print('seed', seedcrystal.projectxy().area)

        rel_area = agg_nonew.intersection(new_crystal.projectxy().buffer(0))
        #rel_area = self.projectxy().buffer(0).intersection(new_crystal.projectxy().buffer(0))

        #pctovrlp1 = (rel_area.area/(seedcrystal.projectxy().area+new_crystal.projectxy().area-rel_area.area))*100
        pctovrlp = (rel_area.area/self.projectxy().area)*100
        #pctovrlp = (rel_area.area/(new_crystal.projectxy().area+self.projectxy().area))*100
        #print('rel',rel_area.area)
        #print(pctovrlp)
        return(pctovrlp)


    def overlapXYZ(self, seedcrystal, new_crystal, plates):

        #horizontal overlap
        in_out = ['y','x','z','x']
        dim_up = 'z'
        percent = []
        for i in range(2):
            xmax = self.max(in_out[i])
            xmin = self.min(in_out[i])
            xmaxseed = seedcrystal.max(in_out[i])
            xminseed = seedcrystal.min(in_out[i])
            xmaxnew = new_crystal.max(in_out[i])
            xminnew = new_crystal.min(in_out[i])
            widclus = xmax-xmin
            widseed = xmaxseed - xminseed
            widnew = xmaxnew - xminnew
            #print(widclus, xmax, xmin)

            Sx = widclus - (widclus - widseed) - (widclus - widnew)

            percentage = (Sx / widclus)*100

            percent.append(percentage)

        #print('X overlap', percent[1])
        #print('Y overlap', percent[0])

        #vertical overlap
        zmax = self.max(dim_up)
        zmin = self.min(dim_up)
        height_seed = seedcrystal.max(dim_up) - seedcrystal.min(dim_up)
        height_new = new_crystal.max(dim_up) - new_crystal.min(dim_up)
        heightclus = zmax-zmin #0 index is x
        Sz = heightclus - (heightclus - height_seed) - (heightclus - height_new)

        percentage = (Sz / heightclus)*100

        percent.append(percentage)
        #print('vert_overlap', percent[2])

        return(percent)

    def make_circle(self, points):
        # Convert to float and randomize order
        shuffled = [(float(x), float(y)) for (x, y) in points]
        random.shuffle(shuffled)

        # Progressively add points to circle or recompute circle
        c = None
        for (i, p) in enumerate(shuffled):
            if c is None or not self.is_in_circle(c, p):
                c = self._make_circle_one_point(shuffled[ : i + 1], p)
        return c

    def _make_circle_one_point(self, points, p):
        # One boundary point known
        c = (p[0], p[1], 0.0)
        for (i, q) in enumerate(points):
            if not self.is_in_circle(c, q):
                if c[2] == 0.0:
                    c = self.make_diameter(p, q)
                else:
                    c = self._make_circle_two_points(points[ : i + 1], p, q)
        return c



    def _make_circle_two_points(self, points, p, q):
        # Two boundary points known
        circ = self.make_diameter(p, q)
        left = None
        right = None
        px, py = p
        qx, qy = q

        # For each point not in the two-point circle
        for r in points:
            if self.is_in_circle(circ, r):
                continue

            # Form a circumcircle and classify it on left or right side
            cross = self._cross_product(px, py, qx, qy, r[0], r[1])
            c = self.make_circumcircle(p, q, r)
            if c is None:
                continue
            elif cross > 0.0 and (left is None or self._cross_product(px, py, qx, qy, c[0], c[1])
                                  > self._cross_product(px, py, qx, qy, left[0], left[1])):
                left = c
            elif cross < 0.0 and (right is None or self._cross_product(px, py, qx, qy, c[0], c[1])
                                  < self._cross_product(px, py, qx, qy, right[0], right[1])):
                right = c

        # Select which circle to return
        if left is None and right is None:
            return circ
        elif left is None:
            return right
        elif right is None:
            return left
        else:
            return left if (left[2] <= right[2]) else right


    def make_circumcircle(self, p0, p1, p2):
        # Mathematical algorithm from Wikipedia: Circumscribed circle
        ax, ay = p0
        bx, by = p1
        cx, cy = p2
        ox = (min(ax, bx, cx) + max(ax, bx, cx)) / 2.0
        oy = (min(ay, by, cy) + max(ay, by, cy)) / 2.0
        ax -= ox; ay -= oy
        bx -= ox; by -= oy
        cx -= ox; cy -= oy
        d = (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by)) * 2.0
        if d == 0.0:
            return None
        x = ox + ((ax * ax + ay * ay) * (by - cy) + (bx * bx + by * by) * (cy - ay) + (cx * cx + cy * cy) * (ay - by)) / d
        y = oy + ((ax * ax + ay * ay) * (cx - bx) + (bx * bx + by * by) * (ax - cx) + (cx * cx + cy * cy) * (bx - ax)) / d
        ra = math.hypot(x - p0[0], y - p0[1])
        rb = math.hypot(x - p1[0], y - p1[1])
        rc = math.hypot(x - p2[0], y - p2[1])
        return (x, y, max(ra, rb, rc))


    def make_diameter(self, p0, p1):
        cx = (p0[0] + p1[0]) / 2.0
        cy = (p0[1] + p1[1]) / 2.0
        r0 = math.hypot(cx - p0[0], cy - p0[1])
        r1 = math.hypot(cx - p1[0], cy - p1[1])
        return (cx, cy, max(r0, r1))


    def is_in_circle(self, c, p):
        _MULTIPLICATIVE_EPSILON = 1 + 1e-14
        return c is not None and math.hypot(p[0] - c[0], p[1] - c[1]) <= c[2] * _MULTIPLICATIVE_EPSILON

    def _cross_product(self, x0, y0, x1, y1, x2, y2):
        # Returns twice the signed area of the triangle defined by (x0, y0), (x1, y1), (x2, y2).
        return (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)


    def complexity(self):
        poly = self.projectxy()
        Ap = poly.area
        P = poly.length  #perim
        x, y = poly.exterior.xy

        circ = self.make_circle([x[i],y[i]] for i in range(len(x)))
        circle = Point(circ[0], circ[1]).buffer(circ[2])
        x,y = circle.exterior.xy
        Ac = circle.area

        #print(Ap, Ac, 0.1-(np.sqrt(Ac*Ap))
        C= 10*(0.1-(np.sqrt(Ac*Ap)/P**2))
        return(C)

    def cont_ang(self, seedcrystal, new_crystal, plates, dim, hypot, Sz, Sx, xmaxseed, xminseed, xmaxnew, xminnew):
        from operator import itemgetter

        #CODE IN AGG_MAIN NOTEBOOK

        return cont_ang
