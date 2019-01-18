"""Utilities for running ice particle simulations."""

import random
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
import scipy.stats as stats
import matplotlib 
import seaborn
import shapely.geometry as geom
import scipy
import warnings
import pandas as pd
import statsmodels as sm
import math
from descartes import PolygonPatch
from shapely.geometry import Point
import time  

from ipas import plots_varyphi as plts
from ipas import IceCrystal as crys
from ipas import IceCluster as clus
        
def main_ar_loop(phio, numaspectratios, ch_dist, nclusters, ncrystals, minor, rand_orient, save_plots, file_ext):   
    
    chreq = []
    chphi = []
    chphi2D = []
    ovrlp = []
    S = []
    ch_ovrlp = []
    ch_S = []
    ch_majorax = []
    ch_depth = [] 
    ch_cplx = []
    dphigam = []
    poserr_phi = []
    negerr_phi = []
    poserr_phi2D = []
    negerr_phi2D = []
    poserr_mjrax = []
    negerr_mjrax = []
    poserr_req = []
    negerr_req = []
    poserr_depth = []
    negerr_depth = []
    poserr_ovrlp = []
    negerr_ovrlp = []
    poserr_S = []
    negerr_S = []
    poserr_cplx = []
    negerr_cplx = []
    min_phi = []
    max_phi = []
    min_phi2D = []
    max_phi2D = []
    min_mjrax = []
    max_mjrax = []
    min_depth = []
    max_depth = []
    min_req = []
    max_req = []
    mean_phi = []
    mean_phi2D = []
    mean_ovrlp = []
    mean_S = []
    mean_mjrax = [] 
    mean_depth = []
    mean_req = []
    mean_cplx = []
    xrot = []
    yrot = []
    seedlengtharr = np.zeros([numaspectratios, nclusters])
    seedwidtharr = np.zeros([numaspectratios, nclusters])
    newlengtharr = np.zeros([numaspectratios, nclusters])
    newwidtharr = np.zeros([numaspectratios, nclusters])
    
    
    for i in range(len(phio)):   
        start_time = time.time()
        print('phio = ',  phio[i])
        
        #sim_clusters makes all the calls to the crystal module 
        #and creates n clusters returning overlap, contact angle, etc.
        #See below function for details on attributes
        b1 = sim_clusters(phio, i, nclusters=nclusters, ncrystals=ncrystals, 
                      numaspectratios = numaspectratios, reorient='random', 
                      minor='depth',rotations=50, rand_orient = False, 
                      speedy=False, lodge=0, max_misses=100) 
        xrot.append(b1.xrot) #append rotation for max contact angle
        yrot.append(b1.yrot)

        #After clusters are made, pass each variable name to return the characteristic
        #of the given variable distribution of n clusters.
        b1.get_characteristics(var ='req', save=save_plots, minor = minor, ch_dist=ch_dist, verbose=False)

        poserr_req.append(b1.poserr)
        negerr_req.append(b1.negerr)
        chreq.append(b1.ch)
        min_req.append(b1.min_data)
        max_req.append(b1.max_data)
        mean_req.append(b1.mean)

        b1.get_characteristics(var ='phi', save=save_plots, minor = minor, ch_dist=ch_dist, verbose=False)
        poserr_phi.append(b1.poserr)
        negerr_phi.append(b1.negerr)
        chphi.append(b1.ch)
        min_phi.append(b1.min_data)
        max_phi.append(b1.max_data)
        mean_phi.append(b1.mean)

        b1.get_characteristics(var ='phi2D', save=save_plots, minor = minor, ch_dist=ch_dist, verbose=False)
        poserr_phi2D.append(b1.poserr)
        negerr_phi2D.append(b1.negerr)
        chphi2D.append(b1.ch)
        min_phi2D.append(b1.min_data)
        max_phi2D.append(b1.max_data)
        mean_phi2D.append(b1.mean)

        b1.get_characteristics(var ='overlap', save=save_plots, minor = minor, ch_dist=ch_dist, verbose=False)  
        ch_ovrlp.append(b1.ch)
        ovrlp.append(b1.ovrlps)
        poserr_ovrlp.append(b1.poserr)
        negerr_ovrlp.append(b1.negerr)
        mean_ovrlp.append(b1.mean)

        b1.get_characteristics(var ='S', save=save_plots, minor = minor, ch_dist=ch_dist, verbose=False)  
        ch_S.append(b1.ch)
        S.append(b1.Ss)
        poserr_S.append(b1.poserr)
        negerr_S.append(b1.negerr)
        mean_S.append(b1.mean)

        b1.get_characteristics(var ='complexity', save=save_plots, minor = minor, ch_dist=ch_dist, verbose=False)  
        ch_cplx.append(b1.ch)
        poserr_cplx.append(b1.poserr)
        negerr_cplx.append(b1.negerr)
        mean_cplx.append(b1.mean)

        b1.get_characteristics(var ='major_axis', save=save_plots, minor = minor, ch_dist=ch_dist, verbose=False)

        poserr_mjrax.append(b1.poserr)
        negerr_mjrax.append(b1.negerr)
        ch_majorax.append(b1.ch)
        min_mjrax.append(b1.min_data)
        max_mjrax.append(b1.max_data)
        mean_mjrax.append(b1.mean)

        b1.get_characteristics(var ='depth', save=save_plots, minor = minor, ch_dist=ch_dist, verbose=False)

        poserr_depth.append(b1.poserr)
        negerr_depth.append(b1.negerr)
        ch_depth.append(b1.ch)       
        min_depth.append(b1.min_data)
        max_depth.append(b1.max_data)
        mean_depth.append(b1.mean)

        dphigam.append(b1.dphigam)
        
        seedlengtharr[i] = b1.seedlengtharr
        seedwidtharr[i] = b1.seedwidtharr
        newlengtharr[i] = b1.newlengtharr
        newwidtharr[i] = b1.newwidtharr

        print("--- %.2f minute(s) ---" % ((time.time() - start_time)/60))

    return plts.Make_Plots(phio, seedwidtharr, seedlengtharr, newwidtharr, newlengtharr, chreq, chphi, 
                       chphi2D, ch_ovrlp, ch_S, ovrlp, S, ch_majorax, ch_depth, dphigam, poserr_phi, negerr_phi,
                       poserr_phi2D, negerr_phi2D, poserr_mjrax, negerr_mjrax, poserr_req, negerr_req, poserr_depth,
                       negerr_depth, poserr_ovrlp, negerr_ovrlp, poserr_S, negerr_S, poserr_cplx, negerr_cplx,
                       min_phi, max_phi, min_phi2D, max_phi2D, min_mjrax, max_mjrax, 
                       min_depth,max_depth, min_req, max_req, mean_phi, mean_mjrax, mean_depth, mean_req, 
                       mean_ovrlp, mean_S, mean_cplx, ch_cplx, xrot, yrot)
                        

def sim_clusters(phio, i, nclusters, ncrystals=2, numaspectratios=1,
         reorient='random', minor='depth',rotations=50, rand_orient = False, speedy=False, lodge=0, max_misses=100):

    """Simulate crystal aggregates.

    Args:
        length (float): The column length of the crystals.
        width (float): The width of the hexagonal faces of the crystals.
        nclusters (int): The number of clusters to simulate.
        ncrystals (int): The number of crystals in each cluster.
        numaspectratios (int): The number of monomer aspect ratios to loop over.
            Default is 1.
        reorient (str): The method to use for reorienting crystals and clusters.
            'random' chooses rotations at random and selects the area-maximizing
            rotation. 'IDL' exactly reproduces the IPAS IDL code. Default is
            'random'.
        minor (str): The minor axis measurement used in calculating aggregate aspect
            ratio. 'depth' for the max-min point parallel to the z axis. 
            'minorxy' for the minor axis length from the fit ellipse in either the 
            x or y orientation (whichever is longer).  Default is 'depth'
        rotations (int): The number of rotations to use to reorient crystals and
            clusters. Default is 50.
        speedy (bool): If true, choose an optimal rotation for single crystals
            instead of reorienting them. Default is false.
        lodge (float): The vertical distance that crystals lodge into each other
            when added from above. Useful for matching the output of IPAS IDL code,
            which uses 0.5. Default is zero.

    Returns:
        An IceClusterBatch object containing the simulated clusters.
    """
    #import ipas.crystals_opt_rot as crys
        
    clusters = []
    cont_angles = []
    ovrlps = []
    cplxs = []
    Ss = []
    seedlengtharr = []
    newlengtharr = []
    seedwidtharr = []
    newwidtharr = []

    shape = 4. 
    mu = phio[i]
    minx = mu/10
    maxx = mu*10
    scale = mu/shape
    
    for n in range(nclusters):
        

        samples = np.random.lognormal(mean=np.log(phio[i]), sigma=1, size=10000)
        rand = samples[0:2]

        plates = phio[i] < 1.0

        #rand = np.random.gamma(shape, scale, 2)

        #check rand falls in between the bounds we want of the distribution
        #while (rand[0] < minx or rand[0] > maxx) or (rand[1] < minx or rand[1] > maxx): 
        #    rand = np.random.gamma(shape, scale, 2)

        if plates:  #make sure both rand values are < 1.0 for plates
            check_rand = [True if x < 1.0 else False for x in rand]
        else: #make sure both rand values are > 1.0 for columns
            check_rand = [True if x > 1.0 else False for x in rand]
   
        while all(check_rand) == False:
            samples = np.random.lognormal(mean=np.log(phio[i]), sigma=1, size=10000)
            rand = samples[0:2]
            #rand = np.random.gamma(shape, scale, 2) #pick 2 new values until check_rand is all true
            if plates:
                check_rand = [True if x < 1.0 else False for x in rand]
            else:
                check_rand = [True if x > 1.0 else False for x in rand]
        


        seedwidth = ((1000)/(rand[0]))**(1./3.) #equivalent volume length and width for each aspect ratio
        seedwidtharr.append(seedwidth)
        seedlength=rand[0]*seedwidth
        seedlengtharr.append(seedlength)
        
        newwidth = ((1000)/(rand[1]))**(1./3.) #equivalent volume length and width for each aspect ratio
        newwidtharr.append(newwidth)
        newlength=rand[1]*newwidth
        newlengtharr.append(newlength)
        

        if rand_orient:
            speedy = False
            rotations = 1
            
        if speedy:
            # get optimal y rotation for single crystals
            f = lambda x: -crys.IceCrystal(length=seedlength, width=seedwidth, rotation=[x,0,0]).projectxy().area
            xrot = opt.minimize_scalar(f, bounds=(0, np.pi/2), method='Bounded').x
            f = lambda x: -crys.IceCrystal(length=seedlength, width=seedwidth, rotation=[0,x,0]).projectxy().area
            yrot = opt.minimize_scalar(f, bounds=(0, np.pi/2), method='Bounded').x
            
            yrotrand = random.uniform(-yrot, yrot)
            xrotrand = random.uniform(-xrot, xrot)

            rotation = [xrotrand, yrotrand, random.uniform(0, 2 * np.pi)]

            seedcrystal = crys.IceCrystal(length=seedlength, width=seedwidth, rotation=rotation)

            # how this rotation works:

            # x) Zero is always one of the best choices for the x
            # rotation (along with pi/3, 2pi/3, ... ). Since the
            # crystal is symmetric it won't make any difference if we
            # rotate it to the other good values. Might as well just
            # choose zero every time.

            # y) We found the optimal y rotation above.

            # z) Rotation around the z-axis has no affect on the
            # projected area. So just choose randomly, no point in
            # simulating it.

            # Since we have values for all 3 rotations, no need to
            # test multiple rotations. These values also apply to the
            # other new crystals, since they have the same dimensions
            # as the seed crystal. So we don't need to run random
            # rotations for them either.
        else:
            xrot = 0
            yrot = 0
            # make the seed crystal, orient it
            seedcrystal = crys.IceCrystal(length=seedlength, width=seedwidth)     
            seedcrystal.reorient(method=reorient, rotations=rotations)

        # create cluster
        cluster = clus.IceCluster(seedcrystal, size=ncrystals)

        # add new crystals
        nmisses = 0
        nmiss = 0
        while cluster.ncrystals < ncrystals:  
            
            if speedy:
                
                xrotrand = random.uniform(-xrot, xrot)
                yrotrand = random.uniform(-yrot, yrot)
                zrot = random.uniform(0, 2 * np.pi)
                rotation = [xrotrand, yrotrand, zrot]
                new_crystal = crys.IceCrystal(length=newlength, width=newwidth, rotation=rotation)
                
                random_loc, lmax_bound = cluster.place_crystal(plates, new_crystal)  
                new_crystal.move(random_loc)
           
                intersect = new_crystal.projectxy().centroid.within(lmax_bound)  
                while intersect is False:        
                    print('SPEEDY MISS',nmiss)
                    nmiss += 1
                    random_loc, lmax_bound = cluster.place_crystal(plates, new_crystal)  
                    new_crystal.move(random_loc)          
                    intersect = new_crystal.projectxy().centroid.within(lmax_bound)
                    if nmiss > max_misses:
                        ('%d > %d' %(nmiss, max_misses))
                        break
                
                #'----------------- add flow to new crystal -----------------')                
                #Calculate S ratio (how far apart the centers are, 0 = complete overlap, 1 = no overlap)              
                S, lmax = cluster.calculate_S_ratio(plates, new_crystal)
                #print('S1',S, lmax)                
               
               
                
            else:
           
                # make a new crystal, orient it
                new_crystal = crys.IceCrystal(length=newlength, width=newwidth)               
                new_crystal.reorient(method=reorient, rotations=rotations)
                
                random_loc, lmax_bound = cluster.place_crystal(plates, new_crystal)  
                new_crystal.move(random_loc)
           
                intersect = new_crystal.projectxy().centroid.within(lmax_bound)  
                while intersect is False:
                    
                    nmiss += 1
                    random_loc, lmax_bound = cluster.place_crystal(plates, new_crystal)  
                    new_crystal.move(random_loc)          
                    intersect = new_crystal.projectxy().centroid.within(lmax_bound)
                    if nmiss > max_misses:
                        ('%d > %d' %(nmiss, max_misses))
                        break
                
            #'----------------- cluster - form aggregate------------------')
            
            # add to the cluster with constraints on S and flow
            # returns false if the crystal misses 
            # the 'lodge' value is just useful for testing against old IPAS code
            
            crystal_hit = cluster.add_crystal_from_above(new_crystal, lodge=lodge) # returns false if the crystal misses

            if crystal_hit:

                # recenter the cluster around the center of mass and reorient it
                cluster.recenter()      

                ovrlp = overlap(seedcrystal, new_crystal, cluster)  
                
                S, lmax = cluster.calculate_S_ratio(plates, new_crystal)
                #print('S2',S, lmax)
                #print('b4 reor', cluster.points)
                
                #rotations = 1                
                cluster.reorient(method=reorient, rotations=rotations)
                                             
                cplx = complexity(cluster)                 
                    
            else: # just in case something goes terribly wrong
                nmisses += 1
                if nmisses > max_misses:
                    #('%d > %d' %(nmisses, max_misses))
                    break
        cplxs.append(cplx)
        clusters.append(cluster)
        #cont_angles.append(cont_ang_inst)
        ovrlps.append(ovrlp)
        Ss.append(S)
        
    return IceClusterBatch(clusters, phio, i, numaspectratios, reorient, ovrlps, Ss, cplxs, nclusters,
                           xrot, yrot, plates, seedlengtharr, seedwidtharr, newlengtharr, newwidtharr)


def overlap(seedcrystal, new_crystal, cluster):
    
    cluster.ncrystals = cluster.ncrystals-1
    agg_nonew = cluster.projectxy().buffer(0)        
    cluster.ncrystals = cluster.ncrystals + 1
    
    rel_area = agg_nonew.intersection(new_crystal.projectxy().buffer(0))

    pctovrlp = (rel_area.area/cluster.projectxy().area)*100

    return(pctovrlp)



def overlapXYZ(seedcrystal, new_crystal, cluster, plates):
    
    #horizontal overlap
    in_out = ['y','x','z','x']
    dim_up = 'z'
    percent = []
    for i in range(2):
        xmax = cluster.max(in_out[i])
        xmin = cluster.min(in_out[i])
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
    zmax = cluster.max(dim_up)
    zmin = cluster.min(dim_up)
    height_seed = seedcrystal.max(dim_up) - seedcrystal.min(dim_up)   
    height_new = new_crystal.max(dim_up) - new_crystal.min(dim_up)              
    heightclus = zmax-zmin #0 index is x
    Sz = heightclus - (heightclus - height_seed) - (heightclus - height_new)    

    percentage = (Sz / heightclus)*100
    
    percent.append(percentage)
    #print('vert_overlap', percent[2])
    
    return(percent)

def make_circle(points):
    # Convert to float and randomize order
    shuffled = [(float(x), float(y)) for (x, y) in points]
    random.shuffle(shuffled)

    # Progressively add points to circle or recompute circle
    c = None
    for (i, p) in enumerate(shuffled):
        if c is None or not is_in_circle(c, p):
            c = _make_circle_one_point(shuffled[ : i + 1], p)
    return c


# One boundary point known
def _make_circle_one_point(points, p):
	c = (p[0], p[1], 0.0)
	for (i, q) in enumerate(points):
		if not is_in_circle(c, q):
			if c[2] == 0.0:
				c = make_diameter(p, q)
			else:
				c = _make_circle_two_points(points[ : i + 1], p, q)
	return c


# Two boundary points known
def _make_circle_two_points(points, p, q):
	circ = make_diameter(p, q)
	left = None
	right = None
	px, py = p
	qx, qy = q
	
	# For each point not in the two-point circle
	for r in points:
		if is_in_circle(circ, r):
			continue
		
		# Form a circumcircle and classify it on left or right side
		cross = _cross_product(px, py, qx, qy, r[0], r[1])
		c = make_circumcircle(p, q, r)
		if c is None:
			continue
		elif cross > 0.0 and (left is None or _cross_product(px, py, qx, qy, c[0], c[1]) > _cross_product(px, py, qx, qy, left[0], left[1])):
			left = c
		elif cross < 0.0 and (right is None or _cross_product(px, py, qx, qy, c[0], c[1]) < _cross_product(px, py, qx, qy, right[0], right[1])):
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


def make_circumcircle(p0, p1, p2):
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


def make_diameter(p0, p1):
    cx = (p0[0] + p1[0]) / 2.0
    cy = (p0[1] + p1[1]) / 2.0
    r0 = math.hypot(cx - p0[0], cy - p0[1])
    r1 = math.hypot(cx - p1[0], cy - p1[1])
    return (cx, cy, max(r0, r1))


def is_in_circle(c, p):
    _MULTIPLICATIVE_EPSILON = 1 + 1e-14
    return c is not None and math.hypot(p[0] - c[0], p[1] - c[1]) <= c[2] * _MULTIPLICATIVE_EPSILON


# Returns twice the signed area of the triangle defined by (x0, y0), (x1, y1), (x2, y2).
def _cross_product(x0, y0, x1, y1, x2, y2):
    return (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)


def complexity(cluster):
    poly = cluster.projectxy()
    Ap = poly.area
    P = poly.length  #perim
    x, y = poly.exterior.xy

    circ = make_circle([x[i],y[i]] for i in range(len(x)))
    circle = Point(circ[0], circ[1]).buffer(circ[2])
    x,y = circle.exterior.xy
    Ac = circle.area
    
    #print(Ap, Ac, 0.1-(np.sqrt(Ac*Ap))
    C= 10*(0.1-(np.sqrt(Ac*Ap)/P**2))
    return(C)
    


def cont_ang(seedcrystal, new_crystal, plates, dim, hypot, Sz, Sx, xmaxseed, xminseed, xmaxnew, xminnew):
    from operator import itemgetter
    
    #CODE IN AGG_MAIN NOTEBOOK
    
    return cont_ang
                          

class IceClusterBatch():
    """A collection of IceCluster objects."""

    def __init__(self, clusters, phio, i, numaspectratios, reorient, 
                 ovrlps, Ss, cplxs, nclusters, xrot, yrot, plates,
                 seedlengtharr, seedwidtharr, newlengtharr, newwidtharr): 
        
        self.phio = phio[i] 
        self.plates = plates
        self.numaspectratios = numaspectratios
        self.clusters = clusters
        self.major_axis = {}
        self.minor_axis = {}
        self.reorient = reorient
        self.ovrlps = ovrlps
        self.Ss = Ss
        self.ch = 0.0   
        self.shape = 0.0
        self.poserr = 0.0
        self.negerr = 0.0
        self.min_data = 0.0
        self.max_data = 0.0
        self.mean = 0.0
        self.cplxs = cplxs
        self.dphigam = 0.0
        self.nclusters = nclusters
        self.xrot = xrot
        self.yrot = yrot    
        self.seedwidtharr = seedwidtharr
        self.seedlengtharr = seedlengtharr
        self.newwidtharr = newwidtharr
        self.newlengtharr = newlengtharr
        
        #self.ch_ovrlps = np.zeros(3)
        
    
    def calculate_error(self, data, ch):
        mean = np.mean(data)
        std = np.std(data)
        shape = (mean/std)**2
        scale = (std**2)/mean
        shapech = mean/(self.numaspectratios*ch)
            
        pos_error = mean + std
        neg_error = mean - std
        
        min_data = min(data)
        max_data = max(data)
        
        return(pos_error, neg_error, min_data, max_data, mean)
   
        
    
    def calc_aspect_ratios(self, minor):
        """Calculate the aspect ratios of the clusters using ellipses fitted
        to the 2D cluster projections from the x-, y-, and z-axis
        perspectives.

        """             
        if self.plates:
            # if the crystals are plates do the plate version
            self.ratios = [ cluster.aspect_ratio(method='plate', minor=minor) for cluster in self.clusters ]            
            self.major_axis['z'] = [ cl.major_axis['z']/2 for cl in self.clusters ] #this has to be after ratios
            self.depth = [ cl.depth/2 for cl in self.clusters ]  #this has to be after ratios -- defined in aspect ratio
            self.req = np.power((np.power(self.major_axis['z'],2)*self.depth),(1./3.))     #major axis from fit ellipse
            
        else:
            self.ratios = [ cluster.aspect_ratio(method='column', minor=minor) for cluster in self.clusters ]
            self.major_axis['z'] = [ cl.major_axis['z']/2 for cl in self.clusters ]
            self.depth = [ cl.depth/2 for cl in self.clusters ]
            self.req = np.power((np.power(self.depth,2)*self.major_axis['z']),(1./3.))     
              
        self.minor = minor
        
    def get_characteristics(self, var, minor, ch_dist, bins=70, filename=None, ext='png',save = False, verbose=False):
        """Plot a histogram of cluster aspect ratios, sending extra arguments
        to `matplotlib.pyplot.hist`.
        """
        
        fig = plt.figure(figsize=(18,9))
        ax = plt.subplot(131)
        
        ratios = self.calc_aspect_ratios(minor)
        self.ratios_2D = [ cluster.aspect_ratio_2D() for cluster in self.clusters ]            

        if var == 'phi':
            filename = '%.2f' % (self.phio)
            data = self.ratios
            xlabel = "Aggregate Aspect Ratio"
            
            self.dphigam = self.ch - self.phio
            
        if var == 'phi2D':
            filename = 'phi2D_%.2f' % (self.phio)
            data = self.ratios_2D
            xlabel = "Aggregate Aspect Ratio (2D Projection from Ellipse)"  
             
        if var == 'req':
            filename = 'req_%.3f' % (self.phio)
            data = self.req
            xlabel = "Aggregate Eq. Vol. Radius"
            
        if var == 'overlap':
            filename = 'overlap_%.3f' % (self.phio)
            data = self.ovrlps
            xlabel = "Overlap"
            
        if var == 'S':
            filename = 'S_%.3f' % (self.phio)
            data = self.Ss
            xlabel = "S Parameter"
            
        if var == 'complexity':
            filename = 'cplx_%.3f' % (self.phio)
            data = self.cplxs
            xlabel = "Complexity"
            
                
        if var == 'depth':
            filename = 'depth_%.3f' % (self.phio)
            data = self.depth           
            xlabel = "Depth"           
           
        if var == 'major_axis':
            filename = 'mjrax_%.3f' % (self.phio)
            data = self.major_axis['z']
            xlabel = "Major Axis"
            
        if ch_dist == 'gamma':
            self.ch, self.shape = self.best_fit_distribution(data, ch_dist, xlabel=xlabel, bins=bins, ax=ax)
        else:
            self.ch = self.best_fit_distribution(data, ch_dist, xlabel=xlabel, bins=bins, ax=ax)

        self.poserr, self.negerr, self.min_data, self.max_data, self.mean = self.calculate_error(data, self.ch)     
            
        '''
        #GAMMA distribution
        fig = plt.figure(figsize=(7,5))
        ax = plt.subplot(111)
        
        #plot histogram with best fit line, weights used to normalize
        weights = np.ones_like(data)/float(len(data))        
        #print('ones_like',np.ones_like(self.ratios))
        #print(weights)
        n, bins, patches = ax.hist(data, bins=bins, weights = weights, 
                                   normed = False, facecolor = 'navy', alpha=alpha, **kwargs)
        
        shape, loc, scale = stats.gamma.fit(data)  #loc is mean, scale is stnd dev      
        
        #print('from fit', shape,loc,scale)
        #eshape, eloc, escale = self.calculateGammaParams(data)
        #print('estimates', eshape, eloc, escale)        
        #x = np.linspace(stats.gamma.pdf(min(data), shape),stats.gamma.pdf(max(data),shape),100)
        x = np.linspace(min(data),max(data),1000)
        #ey = stats.gamma.pdf(x=x, a=eshape, loc=eloc, scale=escale)
        #print(ey)
        
        g = stats.gamma.pdf(x=x, a=shape, loc=loc, scale=scale)
        #print(g)
        plt.ylim((0,max(n)))
        
        plt.plot(x, g, color='darkorange', linewidth=3, linestyle='--')
        ax.grid(b=True, which='major', alpha = 0.7, axis = 'y')     
        ax.grid(b=True, which='major', alpha = 0.7, axis = 'x')  
        
        #CHARACTERISTIC AGG ASPECT RATIO OF DISTRIBUTION
        gmax = max(g)  #highest probability
        indmax = np.argmax(g)  #FIRST index where the highest prob occurs
        chgphi = x[indmax] #characteristic aspect ratio of the distribution
        chmode=stats.mode(x)
        r = np.power((np.power(self.width,2)*self.length),(1./3.)) #of crystal
        plt.title('Monomer Req=%.3f with shape=%.3f and $\phi_n$=%.3f' %(r,shape,chphi))
        plt.xlabel ("Aggregate Eq. Vol. Radius")
        '''      
        
        
        if save:
            self.save_fig(filename = filename, ext=ext, verbose = verbose)
        plt.close() 

        
    def best_fit_distribution(self, data, ch_dist, bins, xlabel, ax=None, normed = True, facecolor='navy', 
                              alpha=1.0,**kwargs):

        import scipy.stats as st
        
        """Model data by finding best fit distribution to data"""
        # Get histogram of original data
        
        data = np.array(data)
        data[np.isinf(data)] = min(data)
        data[np.isnan(data)] = min(data)
       
        y, x = np.histogram(data, density=True)  
        #print('x hist',x)
        xx = (x + np.roll(x, -1))[:-1] / 2.0
    
        # Distributions to check
        DISTRIBUTIONS = [        
            st.alpha,st.anglit,st.arcsine,st.beta,st.betaprime,st.bradford,st.burr,st.cauchy,st.chi,st.chi2,st.cosine,
            st.dgamma,st.dweibull,st.erlang,st.expon,st.exponnorm,st.exponweib,st.exponpow,st.f,st.fatiguelife,st.fisk,
            st.foldcauchy,st.foldnorm,st.frechet_r,st.frechet_l,st.genlogistic,st.genpareto,st.gennorm,st.genexpon,
            st.genextreme,st.gausshyper,st.gamma,st.gengamma,st.genhalflogistic,st.gilbrat,st.gompertz,st.gumbel_r,
            st.gumbel_l,st.halfcauchy,st.halflogistic,st.halfnorm,st.halfgennorm,st.hypsecant,st.invgamma,st.invgauss,
            st.invweibull,st.johnsonsb,st.johnsonsu,st.ksone,st.kstwobign,st.laplace,st.levy,st.levy_l,st.levy_stable,
            st.logistic,st.loggamma,st.loglaplace,st.lognorm,st.lomax,st.maxwell,st.mielke,st.nakagami,st.ncx2,st.ncf,
                    st.nct,st.norm, st.pareto,st.pearson3, st.powerlaw,st.powerlognorm,st.powernorm,
            st.rdist,st.reciprocal,st.rayleigh, st.rice,st.recipinvgauss,st.semicircular,st.t,st.triang, st.truncexpon, 
            st.truncnorm,st.tukeylambda,st.uniform,st.vonmises,st.vonmises_line,st.wald,st.weibull_min,
            st.weibull_max,st.wrapcauchy
        ]

        # Best holders
        best_distribution = st.norm
        best_params = (0.0, 1.0)
        best_sse = np.inf
        
        #gamma holders
        gamma_name = st.gamma
        gamma_params = (0.0,1.0)

        # Estimate distribution parameters from data
        for distribution in DISTRIBUTIONS:

            # Try to fit the distribution
            try:
                # Ignore warnings from data that can't be fit
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore')

                    # fit dist to data
                    params = distribution.fit(data)

                    # Separate parts of parameters
                    arg = params[:-2]
                    loc = params[-2]
                    scale = params[-1]
                    
                  # Calculate fitted PDF and error with fit in distribution
                    #pdf = self.make_pdf(distribution, params)
                    pdf = distribution.pdf(xx, loc=loc, scale=scale, *arg)
                    sse = np.sum(np.power(y - pdf, 2.0))

                    # identify if this distribution is better
                    if best_sse > sse > 0:
                        best_distribution = distribution
                        best_params = params
                        best_sse = sse

                    if distribution == gamma_name:
                        gamma_params = params
                        arg = gamma_params[:-2]   
                        ax2 = plt.subplot(133)
                        n, bins, patches = plt.hist(data, bins=bins, normed=True, 
                                                    color='navy',range=(min(data), max(data)),**kwargs)
                        #print('distribution',distribution)
                        #pdf = self.make_pdf(distribution, gamma_params)
                        pdf = distribution.pdf(xx, loc=gamma_params[-2], scale=gamma_params[-1], *arg)
                        #print('pdfgam',pdf)
                        indmax = np.argmax(pdf)  #FIRST index where the highest prob occurs
                        gammach_var = x[indmax] #characteristic of the distribution         
                        ax2 = plt.plot(xx, pdf, color = 'darkorange', lw=3)
                        plt.title('Gamma Distribution \n shape=%.2f loc=%.2f scale=%.2f'\
                                  %(gamma_params[-3],gamma_params[-2],gamma_params[-1]))
                        
                        #plt.ylabel("Frequency")                        
                        plt.ylim((0,max(n)))

                # if axis passed in add to plot
                try:                       
                    if ax:
                        ax = plt.subplot(131)
                        ax = plt.plot(xx,pdf)
                        #print('plotting...',pdf)
                except Exception:
                    pass

            except Exception:
                #print('in except',best_distribution.name)
                pass
           
        ax = plt.subplot(131)
        arg = best_params[:-2]
        #print('best_distribution',best_distribution)
        best_dist = getattr(stats, best_distribution.name)
        #print('best_dist',best_dist)
        #print('loc, scale, arg',loc,scale,arg)
        pdf = best_distribution.pdf(xx, loc=best_params[-2], scale=best_params[-1], *arg)
        indmax = np.argmax(pdf)  #FIRST index where the highest prob occurs
        char_var = x[indmax] #characteristic of the distribution  
        ax = plt.plot(xx, pdf, lw=5, color='darkorange')
        n, bins, patches = plt.hist(data, bins=bins, normed=True, color='navy',range=(min(data), max(data)),**kwargs)
        plt.ylim((0,max(n))) 
        plt.title('All Fitted Distributions')
        plt.ylabel ("Frequency")  

        
        # Display
        ax1 = plt.subplot(132)
        ax1 = plt.plot(xx, pdf, lw=3, color='darkorange')
        #n, bins, patches = plt.hist(data, bins=bins, normed=True, color='navy',range=(min(data), max(data)))
        n, bins, patches = plt.hist(data, bins=bins, normed=True, color='navy',range=(min(data), max(data)),**kwargs)       
        param_names = (best_dist.shapes + ', loc, scale').split(', ') if best_dist.shapes else ['loc', 'scale']
        param_str = ' '.join(['{}={:0.2f}'.format(k,v) for k,v in zip(param_names, best_params)])
        dist_str = '\"{}\"\n{}'.format(best_distribution.name, param_str)

        plt.title('Best Fit Distribution= %s \n $\phi$=%.3f'%(dist_str,self.phio))
        #plt.ylabel ("Frequency") 
        plt.xlabel(xlabel)            
        plt.ylim((0,max(n))) 
        
        if ch_dist == 'gamma': 
            return (gammach_var, gamma_params[-3])
        else:
            return (char_var)
    
    def make_pdf(self, dist, params, size=10):
        """Generate distributions's Probability Distribution Function """
        
        # Separate parts of parameters
        arg = params[:-2]
        loc = params[-2]
        scale = params[-1]
        
        # Get start and end points of distribution

        #print('args, loc, scale', arg, loc, scale)
        
        start = dist.ppf(0.01, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.01, loc=loc, scale=scale)
        end = dist.ppf(0.99, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.99, loc=loc, scale=scale)
        #print('start, end',start,end)
        # Build PDF and turn into pandas Series
        x = np.linspace(start, end, size)
        y = dist.pdf(x, loc=loc, scale=scale, *arg)
        #print('x,y',x,y)
        pdf = pd.Series(y, x)
        
        return pdf


    def save_fig(self, filename, ext='png', close=True, verbose=False):

        """Save a figure from pyplot.

        Parameters
        ----------
        path : string
            The path without the extension to save the
            figure to.

        filename : string 
            name of the file to save

        ext : string (default='png')
            The file extension. This must be supported by the active
            matplotlib backend (see matplotlib.backends module).  Most
            backends support 'png', 'pdf', 'ps', 'eps', and 'svg'.

        close : boolean (default=True)
            Whether to close the figure after saving.  If you want to save
            the figure multiple times (e.g., to multiple formats), you
            should NOT close it in between saves or you will have to
            re-plot it.

        verbose : boolean (default=True)
            Whether to print information about when and where the image
            has been saved.

            """
        import os.path
                                
        if self.minor == 'minorxy':                   
            path=('/Users/vprzybylo/Desktop/icefiles/agg_model/agg_notes/graphics/python/'+
                      str(len(self.clusters))+'xtals_hist/minorxy/')
        else:
            path=('/Users/vprzybylo/Desktop/icefiles/agg_model/agg_notes/graphics/python/'+
                      str(len(self.clusters))+'xtals_hist/depth/')

        filename = "%s.%s" % (filename, ext)
        if path == '':
            path = '.'

        # If the directory does not exist, create it
        if not os.path.exists(path):
            os.makedirs(path)

        # The final path to save to
        savepath = os.path.join(path, filename)

        if verbose:
            print("Saving figure to '%s'..." % savepath),

        # Actually save the figure
        plt.savefig(savepath)

        # Close it
        if close:
            plt.close()    
            
            
            """Utilities for running ice particle simulations."""
