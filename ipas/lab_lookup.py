"""Utilities for running ice particle simulations."""
"""Editted for making the characteristic aggregate a, c, 
etc. lookup tables """

import random
import time
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
import scipy.stats as stats
#from scipy.stats import rv_continuous, gamma
import matplotlib 
import shapely.geometry as geom
import scipy
import warnings
import pandas as pd
import statsmodels as sm
import math
from descartes import PolygonPatch
from shapely.geometry import Point

from ipas import plots_phiarr as plts
from ipas import IceClusterBatchArr as batch
from ipas import IceCrystal as crys
from ipas import IceCluster as clus 

        
def parallelize_clusters(phio, lenphio, reqarr, save_plots=False, minor='depth', 
                         nclusters=300, ncrystals=2, numaspectratios=20, speedy=True, rand_orient=False,
                         ch_dist='gamma'):    
    
    c_n = np.zeros(shape=(lenphio,len(reqarr)))
    a_n = np.zeros(shape=(lenphio,len(reqarr)))
    a_avg = np.zeros(shape=(lenphio,len(reqarr)))
    dd = np.zeros(shape=(lenphio,len(reqarr)))
    
    phi_ind = 0
    for i in phio:
    #time ends after each aspect ratio
        req_ind = 0
        for r in reqarr:   #array of equivalent volume radii to loop through
            start_time = time.time()
            #equivalent volume length and width for each aspect ratio
            #r**3  volume
            width = (r**3/i)**(1./3.)
            length=i*width
            r = np.power((np.power(width,2)*length),(1./3.)) #of monomer  

            print('phi, req, c, a =', i, r, length, width)

            #sim_clusters makes all the calls to the crystal module 
            #and creates n clusters returning overlap, contact angle, etc.
            #See below function for details on attributes
            b1 = sim_clusters(r, length=length, width=width, nclusters=nclusters, ncrystals=ncrystals, 
                                  numaspectratios = numaspectratios, speedy=True, rand_orient=rand_orient) 


            #After clusters are made, pass each variable name to return the characteristic
            #of the given variable distribution of n clusters.


            b1.get_characteristics(var ='major_axis', save=save_plots, minor = minor, ch_dist=ch_dist, verbose=False)
            ch_majorax=b1.ch
            mean_mjrax=b1.mean

            b1.get_characteristics(var ='depth', save=save_plots, minor = minor, ch_dist=ch_dist, verbose=False)
            ch_depth=b1.ch
            mean_depth=b1.mean

            b1.get_characteristics(var ='density_change', save=save_plots, minor = minor, ch_dist=ch_dist, verbose=False)
            ch_dd=b1.ch

            if i < 1:  #build the lookup table 2D arrays, i loops over aspect ratio
                #have to switch a and c for plates vs. columns

                c_n[phi_ind,req_ind] = ch_depth
                a_n[phi_ind,req_ind] = ch_majorax
                a_avg[phi_ind,req_ind] = mean_mjrax
                dd[phi_ind,req_ind] = ch_dd

            else:
                c_n[phi_ind,req_ind] = ch_majorax
                a_n[phi_ind,req_ind] = ch_depth
                a_avg[phi_ind,req_ind] = mean_depth
                dd[phi_ind,req_ind] = ch_dd

            req_ind += 1
            #print(a_n)
            print("--- %.2f minute(s) ---" % ((time.time() - start_time)/60))            
            #print(i, r, b1.ch_depth)
        phi_ind += 1

    
    #output the lookup table arrays into txt files 
    print(c_n)
    df = pd.DataFrame(c_n, index=phio, columns=reqarr)
    df.to_csv('c_n_lookup_dd.txt', sep='\t')
    
    df = pd.DataFrame(a_n, index=phio, columns=reqarr)
    df.to_csv('a_n_lookup_dd.txt', sep='\t')
    
    df = pd.DataFrame(a_avg, index=phio, columns=reqarr)
    df.to_csv('a_avg_lookup_dd.txt', sep='\t')    
    
    df = pd.DataFrame(dd, index=phio, columns=reqarr)
    df.to_csv('dd_lookup.txt', sep='\t')
    
    

    #return plts.Make_Plots(phio, widtharr, lengtharr, chreq, chphi, chphi2D, ch_ovrlp, ch_S, ovrlp, S, 
    #                       ch_majorax, ch_depth, dphigam, poserr_phi, negerr_phi, poserr_phi2D, negerr_phi2D, 
    #                       poserr_mjrax, negerr_mjrax, poserr_req, negerr_req, poserr_depth, negerr_depth, poserr_ovrlp, 
    #                       negerr_ovrlp, poserr_S, negerr_S, poserr_cplx, negerr_cplx, min_phi, max_phi, min_phi2D, 
    #                       max_phi2D, min_mjrax, max_mjrax, min_depth,max_depth, min_req, max_req, mean_phi, mean_phi2D, 
    #                       mean_mjrax, mean_depth, mean_req, mean_ovrlp, mean_S, mean_cplx, ch_cplx, xrot, yrot)

def sim_clusters(r, length, width, nclusters, ncrystals=2, numaspectratios=1, reorient='random', 
                 minor='depth',rotations=50, rand_orient = False, speedy=True, lodge=0, max_misses=100):
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

    clusters = []
    cont_angles = []
    ovrlps = np.zeros((nclusters,ncrystals-1))
    Ss = np.zeros((nclusters,ncrystals-1))
    cplxs = np.zeros((nclusters,ncrystals-1))
    phi = np.zeros((nclusters,ncrystals-1))
    phi_2d = np.zeros((nclusters,ncrystals-1))
    major_axis = np.zeros((nclusters,ncrystals-1))
    depth = np.zeros((nclusters,ncrystals-1))
    req = np.zeros((nclusters,ncrystals-1))
    dd = np.zeros((nclusters,ncrystals-1))

    for n in range(nclusters):

        if n % 50 == 0:
            print('nclus',n)
        #print('nclus',n)
        plates = width > length

        if rand_orient:
            speedy = False
            xrot = 0.0
            yrot = 0.0
            rotations = 1
        else:
            speedy = True
            rotations = 50
        
        if speedy:
            # get optimal y rotation for single crystals
            f = lambda x: -crys.IceCrystal(length=length, width=width, rotation=[x,0,0]).projectxy().area
            xrot = opt.minimize_scalar(f, bounds=(0, np.pi/2), method='Bounded').x
            f = lambda x: -crys.IceCrystal(length=length, width=width, rotation=[0,x,0]).projectxy().area
            yrot = opt.minimize_scalar(f, bounds=(0, np.pi/2), method='Bounded').x

            yrotrand = random.uniform(-yrot, yrot)
            xrotrand = random.uniform(-xrot, xrot)
            
            rotation = [xrotrand, yrotrand, random.uniform(0, 2 * np.pi)]
            seedcrystal = crys.IceCrystal(length=length, width=width, rotation=rotation)


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

            # make the seed crystal, orient it
            seedcrystal = crys.IceCrystal(length=length, width=width)     
            seedcrystal.reorient(method=reorient, rotations=rotations)

        # create cluster
        cluster = clus.IceCluster(seedcrystal, size=ncrystals)

        # add new crystals
        nmisses = 0
        nmiss = 0
        
        l=0
        d1 = 1.0 #initial density of monomer (time step 1)
        while cluster.ncrystals < ncrystals: 
            
            #print('l', cluster.ncrystals)
            #if rand_orient or cluster.ncrystals>=5:                
            #    speedy = False
            #    rotations = 1

            if speedy:
                xrotrand = random.uniform(-xrot, xrot)
                yrotrand = random.uniform(-yrot, yrot)
                zrot = random.uniform(0, 2 * np.pi)
                rotation = [xrotrand, yrotrand, zrot]
                new_crystal = crys.IceCrystal(length=length, width=width, rotation=rotation)
                                
                random_loc, lmax_bound = cluster.place_crystal(plates, new_crystal)  
                new_crystal.move(random_loc)
                #print('out clus', cluster.points)
                
         
                bound = new_crystal.projectxy().buffer(0).centroid.within(lmax_bound) 
                while bound is False:      
                    #print('in bound', cluster.points)
                    #print('new in bound', new_crystal.points)
                    print('SPEEDY MISS',nmiss)
                    #cluster.plot_constraints(plates, new_crystal)
                    nmiss += 1
                    cluster.recenter()
                    new_crystal.recenter()
                    random_loc, lmax_bound = cluster.place_crystal(plates, new_crystal)  
                    new_crystal.move(random_loc)          
                   
                    bound = new_crystal.projectxy().buffer(0).centroid.within(lmax_bound)
                    
                    if nmiss > max_misses:
                        print('bound in speedy %d > %d' %(nmiss, max_misses))               
                        break
                
            else:
                # make a new crystal, orient it
                new_crystal = crys.IceCrystal(length=length, width=width)               
                new_crystal.reorient(method=reorient, rotations=rotations)
                
                random_loc, lmax_bound = cluster.place_crystal(plates, new_crystal)  
                new_crystal.move(random_loc)
                
                bound = new_crystal.projectxy().buffer(0).centroid.within(lmax_bound) 
                
                while bound is False:  #only if tilting causes the new crystal center to be outside of lmax
             
                    #from the seed crystal/agg (determined from S)
                    print('random miss', nmiss)
                    #cluster.plot_constraints(plates, new_crystal)
                    #print(cluster.points)
                    #print('new', new_crystal.points)
                    
                    nmiss += 1
                    cluster.recenter()
                    new_crystal.recenter()
                    random_loc, lmax_bound = cluster.place_crystal(plates, new_crystal)  
                    new_crystal.move(random_loc)          
                    
                    bound = new_crystal.projectxy().buffer(0).centroid.within(lmax_bound)
                    
                    if nmiss > max_misses:
                        print('bound random %d > %d' %(nmiss, max_misses))               
                        break
                
                
            #'----------------- cluster - form aggregate------------------')
   
            # add to the cluster with constraints on S 
            # returns false if the crystal misses 
            # the 'lodge' value is just useful for testing against old IPAS code
            crystal_hit = cluster.add_crystal_from_above(new_crystal, lodge=lodge) 
            
            #print(crystal_hit)
            if crystal_hit:
           
                # recenter the cluster around the center of mass
                x, y, z = cluster.recenter()   
                new_crystal.move([-x,-y,-z])                                      
                
                ovrlps[n,l] = cluster.overlap(new_crystal, cluster)                
                #print(cluster.overlap(new_crystal, cluster))
                
                Ss[n,l], lmax = cluster.calculate_S_ratio(plates, new_crystal)
                #cluster.plot_ellipsoid()
                #cluster.plot_constraints(plates, new_crystal)
                
                cluster.reorient(method=reorient, rotations=rotations)

                #cplxs[n,l] = cluster.complexity()
                
                if plates: 
                    
                    phi[n,l] = cluster.aspect_ratio(method='plate', minor=minor)
                    major_axis[n,l] = cluster.major_axis['z']/2  #this has to be after ratios
                    depth[n,l] = cluster.depth/2  #this has to be after ratios -- defined in aspect ratio
                    req[n,l] = np.power((np.power(major_axis[n,l],2)*depth[n,l]),(1./3.))        
                
                else:
                    #print(n,l)
                    phi[n,l] = cluster.aspect_ratio(method='column', minor=minor) 
                    #print(cluster.aspect_ratio(method='column', minor=minor))
                    major_axis[n,l] = cluster.major_axis['z']/2 
                    depth[n,l] = cluster.depth/2 
                    req[n,l] = np.power((np.power(depth[n,l],2)*major_axis[n,l]),(1./3.)) 
  
                phi_2d[n,l] = cluster.aspect_ratio_2D()   
                
                
                #calculate density
                
                Va = 3*np.sqrt(3)/2 * np.power(width,2) * length * cluster.ncrystals  #actual agg volume of hexagonal prisms
                rx, ry, rz = cluster.spheroid_axes()  #radii lengths - 3 axes
                #print(rx, ry, rz)
                Ve = 4/3*rx*ry*rz  #equiv volume density from fit ellipsoid
                #an equivalent ratio of densities - close to 1.0 for single monomer, otherwise <1.0
                d2 = Va/Ve 
                dd[n,l] = d1-d2 #change in density
                d1=d2 
               
                #print('dd, Va, Ve, d2',  dd, Va, Ve, d2)
                #cluster.plot_ellipsoid()
            
                l+=1  #increment # cluster counter for array indices
                
            else: # just in case something goes terribly wrong
                nmisses += 1
                if nmisses > max_misses:
                    print('crystal hit miss max %d > %d' %(nmisses, max_misses))
                    #print('cluster', cluster.points)
                    #print('new crystal',new_crystal.points)
                    break
                    
        clusters.append(cluster)
        phio = length/width
        
    #return batch.IceClusterBatch(clusters, numaspectratios, phio, plates)
    return batch.IceClusterBatch(ncrystals, clusters, length, width, r, numaspectratios, reorient, ovrlps, Ss, cplxs,
                           phi, phi_2d, major_axis, depth, req, xrot, yrot, dd, plates)
                   

class IceClusterBatch():
    """A collection of IceCluster objects."""
    
    def __init__(self, clusters, numaspectratios, phio, plates=None): 

        self.plates = plates # are they plates or columns?
        self.numaspectratios = numaspectratios
        self.clusters = clusters
        self.phio = phio
        self.major_axis = {}
        self.minor_axis = {}

        #self.ch_ovrlps = np.zeros(3)
      
    
    def calculate_error(self, data, ch):  #for error bars/shading on plots in plot.py module, mean +/- 1 std dev
        #also the max and min of each variable read in (data) for each aspect ratio
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
        perspectives.  Also saves the new a and c axis, and equivalent volume
        radius of the aggregates.

        """
     
        if self.plates:
            # if the crystals are plates do the plate version
            self.ratios = [ cluster.aspect_ratio(method='plate', minor=minor) for cluster in self.clusters ] 
            #self.ratios[np.isinf(self.ratios)] = min(self.ratios)
            #self.ratios[np.isnan(self.ratios)] = min(self.ratios)
            self.major_axis['z'] = [ cl.major_axis['z']/2 for cl in self.clusters ] #this has to be after ratios
            #self.major_axis['z'][np.isinf(self.major_axis['z'])] = min(self.major_axis['z'])
            #self.major_axis['z'][np.isnan(self.major_axis['z'])] = min(self.major_axis['z'])
            self.depth = [ cl.depth/2 for cl in self.clusters ]  #this has to be after ratios -- defined in aspect ratio
            #self.depth[np.isinf(self.depth)] = min(self.depth)
            #self.depth[np.isnan(self.depth)] = min(self.depth)
            self.req = np.power((np.power(self.major_axis['z'],2)*self.depth),(1./3.))     #major axis from fit ellipse
            self.req[np.isinf(self.req)] = min(self.req)  #occasionally there were some NaN's and/or inf's that caused the 

           
        else:
            self.ratios = [ cluster.aspect_ratio(method='column', minor=minor) for cluster in self.clusters ]
            #self.ratios[np.isinf(self.ratios)] = min(self.ratios)
            #self.ratios[np.isnan(self.ratios)] = min(self.ratios)
            self.major_axis['z'] = [ cl.major_axis['z']/2 for cl in self.clusters ]
            #self.major_axis['z'][np.isinf(self.major_axis['z'])] = min(self.major_axis['z'])
            #self.major_axis['z'][np.isnan(self.major_axis['z'])] = min(self.major_axis['z'])
            self.depth = [ cl.depth/2 for cl in self.clusters ]
            #self.depth[np.isinf(self.depth)] = min(self.depth)
            #self.depth[np.isnan(self.depth)] = min(self.depth)


        self.minor = minor
        
    def get_characteristics(self, var, minor, ch_dist, bins=70, filename=None, ext='png',save = False, verbose=False):
        """Plot a histogram of cluster aspect ratios using different distributions or a gamma dist. """
        
        fig = plt.figure(figsize=(18,9))
        ax = plt.subplot(131)
        
        ratios = self.calc_aspect_ratios(minor)
           
         
        if var == 'depth':
            filename = 'depth_%.3f' % (self.phio)
            xlabel = "Depth" 
            data = self.depth  
            if ch_dist == 'gamma':
                self.ch_depth, self.shape = self.best_fit_distribution(data, ch_dist, xlabel=xlabel, bins=bins, ax=ax)
            else:
                self.ch_depth = self.best_fit_distribution(data, ch_dist, xlabel=xlabel, bins=bins, ax=ax)
            
            self.poserr, self.negerr, self.min_data, self.max_data, self.mean_depth = self.calculate_error(data, self.ch_depth)
            
                    
        if var == 'major_axis':
            filename = 'mjrax_%.3f' % (self.phio)
            xlabel = "Major Axis"
            data = self.major_axis['z']
            if ch_dist == 'gamma':
                self.ch_majorax, self.shape = self.best_fit_distribution(data, ch_dist, xlabel=xlabel, bins=bins, ax=ax)
            else:
                self.ch_majorax = self.best_fit_distribution(data, ch_dist, xlabel=xlabel, bins=bins, ax=ax)
            
            self.poserr, self.negerr, self.min_data, self.max_data, self.mean_mjrax = self.calculate_error(data, self.ch_majorax)

        '''OTHER TESTS FOR SHAPE PARAMETER using stats.gamma Python package
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
        dphigam = chphi - self.phio    #aggregate aspect ratio - monomer aspect ratio (both crystals the same to start)
        r = np.power((np.power(self.width,2)*self.length),(1./3.)) #of crystal
        plt.title('Monomer Req=%.3f with shape=%.3f and $\phi_n$=%.3f' %(r,shape,chphi))
        plt.xlabel ("Aggregate Eq. Vol. Radius")
        '''      
        
        if save:
            self.save_fig(filename = filename, nclusters = self.nclusters, ext=ext, verbose = verbose)
        plt.close() 

        
    def best_fit_distribution(self, data, ch_dist, bins, xlabel, ax=None, normed = True, facecolor='navy', 
                              alpha=1.0,**kwargs):

        import scipy.stats as st
        
        """Model data by finding best fit distribution to data.
        Loops through a list of possible built in distribution 
        types from stats library """
        #MODIFIED FROM: 
        #https://stackoverflow.com/questions/6620471/fitting-empirical-distribution-to-theoretical-ones-with-scipy-python
        # Get histogram of original data
        
        data = np.array(data)  #turn each data list into numpy array
        data[np.isinf(data)] = min(data)  #replace any possible nans or infs
        data[np.isnan(data)] = min(data)
        
        y, x = np.histogram(data, density=True)  
        #print('x hist',x)
        xx = (x + np.roll(x, -1))[:-1] / 2.0
        
        #np.roll: Elements that roll beyond the last position are re-introduced at the first.
    
        # Distributions to check
        '''
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
        '''
        distribution = st.gamma

        # Best holders
        best_distribution = st.norm
        best_params = (0.0, 1.0)
        best_sse = np.inf
        
        #gamma holders
        gamma_name = st.gamma
        gamma_params = (0.0,1.0)

        # Estimate distribution parameters from data
        #for distribution in DISTRIBUTIONS:

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

    def save_fig(self, filename, nclusters, ext='png', close=True, verbose=False):

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
        #    path=('/home/jovyan/work/plotOutPut/'+str(nclusters)+'xtals_hist/every_ar')

        else:
            path=('/Users/vprzybylo/Desktop/icefiles/agg_model/agg_notes/graphics/python/'+
                      str(len(self.clusters))+'xtals_hist/depth/')
        #path=('/home/jovyan/work/IPAS/plotOutPut/'+str(nclusters)+'xtals_hist/every_ar/')


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
            
            
            
            