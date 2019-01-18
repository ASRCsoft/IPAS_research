"""Utilities for running ice particle simulations."""

import random
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
import scipy.stats as stats
#from scipy.stats import rv_continuous, gamma
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
from ipas import plots_Sconstrain as plts
from ipas import crystals_opt_rot_Sconstrain as crys
import time  
        
def main_ar_loop(phio, numaspectratios,nclusters, ncrystals, minor, rand_orient, save_plots, file_ext):
    import time     

    for i in phio:
        width = (1000/i)**(1./3.) #equivalent volume length and width for each aspect ratio
        length=i*width
        r = np.power((np.power(width,2)*length),(1./3.)) #of monomer
        print('eq. vol rad', r, length, width, i)
        
        
        #sim_clusters makes all the calls to the crystal module 
        #and creates n clusters returning overlap, contact angle, etc.
        #See below function for details on attributes
        b1 = sim_clusters(r, length=length, width=width, nclusters=nclusters, ncrystals=ncrystals, 
                              numaspectratios = numaspectratios, speedy=True, rand_orient=rand_orient) 

      
        print("--- %.2f minute(s) ---" % ((time.time() - start_time)/60))

    print('end')
    
def sim_clusters(r, length, width, nclusters, ncrystals=2, numaspectratios=1, reorient='random', 
                 minor='depth',rotations=50, rand_orient = False, speedy=True, lodge=0, max_misses=100):
   
    """Simulate crystal aggregates.

    Args:
        length (float): The column length of the crystals.
        width (float): The width of the hexagonal faces of the crystals.
        nclusters (int): The number of clusters to simulate.
        ncrystals (int): The number of crystals in each cluster.
        rand_orient (bool): If true, randomly orient the crystals and aggregate.
            Uses the first random orientation and sets speedy to False
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
            instead of reorienting them. Default is true.
        lodge (float): The vertical distance that crystals lodge into each other
            when added from above. Useful for matching the output of IPAS IDL code,
            which uses 0.5. Default is zero.

    Returns:
        An IceClusterBatch object containing the simulated clusters.
    """
    #import ipas.crystals_opt_rot as crys

    for n in range(nclusters):

        plates = width > length
       
        if rand_orient:
            speedy = False
            rotations = 1
        
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

        else:
            xrot = 0
            yrot = 0
            # make the seed crystal, orient it
            seedcrystal = crys.IceCrystal(length=length, width=width)     
            seedcrystal.reorient(method=reorient, rotations=rotations)

        # create cluster
        cluster = crys.IceCluster(seedcrystal, size=ncrystals)

        # add new crystals
        nmisses = 0
        while cluster.ncrystals < ncrystals:  
           
            if speedy:
                
                xrotrand = random.uniform(-xrot, xrot)
                yrotrand = random.uniform(-yrot, yrot)
                rotation = [xrotrand, yrotrand, random.uniform(0, 2 * np.pi)]
                
                new_crystal = crys.IceCrystal(length=length, width=width, rotation=rotation)               
                
            else:
           
                # make a new crystal, orient it
                new_crystal = crys.IceCrystal(length=length, width=width)               
                new_crystal.reorient(method=reorient, rotations=rotations)
                   
                
            #'----------------- cluster - form aggregate------------------')
            
            #Calculate S ratio (how far apart the centers are, 0 = complete overlap, 1 = no overlap)              
            S, lmax = cluster.calculate_S_ratio(plates, new_crystal)
            #print('S1',S, lmax)

            tilt_fracx, tilt_fracy = cluster.add_flow_tilt(new_crystal, lmax)                       

            xrottilt = xrotrand + (xrotrand*tilt_fracx)
            yrottilt = yrotrand + (yrotrand*tilt_fracy)
            rotation = [xrottilt, yrottilt, random.uniform(0, 2 * np.pi)]
            new_crystal.rotate_to(rotation)  
            #new_crystal = crys.IceCrystal(length=length, width=width, rotation=rotation)
                
            #'----------------- cluster - form aggregate------------------')
            
            random_loc = cluster.place_crystal(plates, new_crystal)            
            new_crystal.move(random_loc)
    
            # add to the cluster with constraints on S and flow
            # returns false if the crystal misses -- but it shouldn't
            # the 'lodge' value is just useful for testing against old IPAS code
            
            crystal_hit = cluster.add_crystal_from_above(new_crystal, lodge=lodge) 
            #print('after crystal_hit clus', cluster.points)
            
            if crystal_hit:

                # recenter the cluster around the center of mass and reorient it
                cluster.recenter()                           
                
                                
                S, lmax = cluster.calculate_S_ratio(plates, new_crystal)
                #print('S2',S, lmax)
                #print('b4 reor', cluster.points)
                
                rotations = 1
                cluster.reorient(method=reorient, rotations=rotations)
                #print('after reor', cluster.points)
                poly = cluster.projectxy()
                #if poly.is_valid is False:
    
                #print(poly.geom_type)
         
                if poly.geom_type =='MultiPolygon':
                    print(cluster.points)
                    print('error')
                x, y = poly.exterior.xy
            
                
            else: # just in case something goes terribly wrong
                nmisses += 1
                if nmisses > max_misses:
                    #('%d > %d' %(nmisses, max_misses))
                    break
   
        
    print('end of sim_clusters')
   
       
