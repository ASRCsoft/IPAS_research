"""Utilities for running ice particle simulations."""

import random
import numpy as np
import scipy.optimize as opt
from shapely.geometry import Point
import time  

from ipas import plots_Sconstrain as plts
from ipas import IceClusterBatch as batch
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
    lengtharr = []
    widtharr = []
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
    #tiltdiffsx = []
    #tiltdiffsy = []
    
    for i in phio:
        start_time = time.time()
        width = ((1000)/(i))**(1./3.) #equivalent volume length and width for each aspect ratio - RADII
        widtharr.append(width)
        length=i*width
        lengtharr.append(length)
        r = np.power((np.power(width,2)*length),(1./3.)) #of monomer

        print('eq. vol rad', r, length, width, i)
        
        
        #sim_clusters makes all the calls to the crystal module 
        #and creates n clusters returning overlap, contact angle, etc.
        #See below function for details on attributes
        b1 = sim_clusters(r, length=length, width=width, nclusters=nclusters, ncrystals=ncrystals, 
                              numaspectratios = numaspectratios, speedy=True, rand_orient=rand_orient) 
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
        chphi.append(b1.chphi)
        min_phi.append(b1.min_data)
        max_phi.append(b1.max_data)
        mean_phi.append(b1.mean)

        b1.get_characteristics(var ='phi2D', save=save_plots, minor = minor, ch_dist=ch_dist, verbose=False)
        poserr_phi2D.append(b1.poserr)
        negerr_phi2D.append(b1.negerr)
        chphi2D.append(b1.chphi)
        min_phi2D.append(b1.min_data)
        max_phi2D.append(b1.max_data)
        mean_phi2D.append(b1.mean)
        
        b1.get_characteristics(var ='overlap', save=save_plots, minor = minor, ch_dist=ch_dist, verbose=False)  
        ch_ovrlp.append(b1.ch_ovrlp)
        print(b1.ch_ovrlp)
        ovrlp.append(b1.ovrlps)
        poserr_ovrlp.append(b1.poserr)
        negerr_ovrlp.append(b1.negerr)
        mean_ovrlp.append(b1.mean)
        
        b1.get_characteristics(var ='S', save=save_plots, minor = minor, ch_dist=ch_dist, verbose=False)  
        ch_S.append(b1.ch_S)
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
        #tiltdiffsx.append(b1.tiltdiffsx)
        #tiltdiffsy.append(b1.tiltdiffsy)

        print("--- %.2f minute(s) ---" % ((time.time() - start_time)/60))

    return plts.Make_Plots(phio, widtharr, lengtharr, chreq, chphi, chphi2D, ch_ovrlp, ch_S, ovrlp, S, 
                           ch_majorax, ch_depth, dphigam, poserr_phi, negerr_phi, poserr_phi2D, negerr_phi2D, 
                           poserr_mjrax, negerr_mjrax, poserr_req, negerr_req, poserr_depth, negerr_depth, poserr_ovrlp, 
                           negerr_ovrlp, poserr_S, negerr_S, poserr_cplx, negerr_cplx, min_phi, max_phi, min_phi2D, 
                           max_phi2D, min_mjrax, max_mjrax, min_depth,max_depth, min_req, max_req, mean_phi, mean_phi2D, 
                           mean_mjrax, mean_depth, mean_req, mean_ovrlp, mean_S, mean_cplx, ch_cplx, xrot, yrot)

def sim_clusters(r, length, width, nclusters, ncrystals=2, numaspectratios=1, reorient='random', 
                 minor='depth',rotations=50, rand_orient = False, speedy=True, lodge=0, max_misses=20):
   
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
        
    clusters = []
    cont_angles = []
    ovrlps = np.zeros((nclusters,ncrystals-1))
    Ss = np.zeros((nclusters,ncrystals-1))
    cplxs = []
    #tiltdiffsx = []
    #tiltdiffsy = []

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
            seedcrystal = crys.IceCrystal(length=length, width=width)     
            seedcrystal.reorient(method=reorient, rotations=rotations)

        # create cluster
        cluster = clus.IceCluster(seedcrystal, size=ncrystals)

        # add new crystals
        nmisses = 0
        nmiss = 0
        
        l=0
        speedy=True
        while cluster.ncrystals < ncrystals: 
            #print('l', cluster.ncrystals, ncrystals)
            if rand_orient or cluster.ncrystals>=5:
                speedy = False
                rotations = 1
              
            xmax = cluster.max('x')
            ymax = cluster.max('y')
            zmax = cluster.max('z')
            xmin = cluster.min('x')
            ymin = cluster.min('y')
            zmin = cluster.min('z')  
            random_loc = [random.uniform(xmin, xmax), random.uniform(ymin, ymax), 0]

            if speedy:
                
                xrotrand = random.uniform(-xrot, xrot)
                yrotrand = random.uniform(-yrot, yrot)
                zrot = random.uniform(0, 2 * np.pi)
                rotation = [xrotrand, yrotrand, zrot]
                new_crystal = crys.IceCrystal(length=length, width=width, rotation=rotation)
                #new_crystal.recenter()
                #random_loc, lmax_bound = cluster.place_crystal(plates, new_crystal) 
                new_crystal.move(random_loc)
               
                
        
                #'----------------- add flow to new crystal -----------------')                
                #Calculate S ratio (how far apart the centers are, 0 = complete overlap, 1 = no overlap)              
                '''
                S, lmax = cluster.calculate_S_ratio(plates, new_crystal)
                #print('S1',S, lmax)                
                tilt_fracx, tilt_fracy = cluster.add_flow_tilt(new_crystal, lmax)                       
                xrottilt = xrotrand + (xrotrand*tilt_fracx)
                yrottilt = yrotrand + (yrotrand*tilt_fracy)                
                rotation = [xrottilt, yrottilt, zrot]
                #print(xrotrand*180/np.pi, yrotrand*180/np.pi)
                #print(xrottilt*180/np.pi, yrottilt*180/np.pi)
                #new_crystal.rotate_to(rotation) 
                '''               
                
                
                
                '''
                bound = new_crystal.projectxy().buffer(0).centroid.within(lmax_bound) 
             
                while bound is False:      
                    
                    print('SPEEDY MISS',nmiss)
                    nmiss += 1
                    #print(cluster.points)
                    #print(new_crystal.points)
                    random_loc, lmax_bound = cluster.place_crystal(plates, new_crystal)  
                    new_crystal.move(random_loc)          
                    
                    #'----------------- add flow to new crystal -----------------')                
                    #Calculate S ratio (how far apart the centers are, 0 = complete overlap, 1 = no overlap)              
                    
                    S, lmax = cluster.calculate_S_ratio(plates, new_crystal)                              
                    tilt_fracx, tilt_fracy = cluster.add_flow_tilt(new_crystal, lmax)                                           
                    xrottilt = xrotrand + (xrotrand*tilt_fracx)
                    yrottilt = yrotrand + (yrotrand*tilt_fracy)
                    rotation = [xrottilt, yrottilt, zrot]
                    #new_crystal.rotate_to(rotation) 
                    
                    bound = new_crystal.projectxy().buffer(0).centroid.within(lmax_bound)
                    
                    if nmiss > max_misses:
                        print('bound in speedy %d > %d' %(nmiss, max_misses))               
                        break
                '''    
                
                #tiltdiffx = np.abs((xrotrand*180/np.pi) - (xrottilt*180/np.pi))
                #tiltdiffy = np.abs((yrotrand*180/np.pi) - (yrottilt*180/np.pi))
                
            else:
                print('shouldnt be here')
                # make a new crystal, orient it
                new_crystal = crys.IceCrystal(length=length, width=width)   
                new_crystal.reorient(method=reorient, rotations=rotations)
                new_crystal.recenter()
                random_loc, lmax_bound = cluster.place_crystal(plates, new_crystal)  
                new_crystal.move(random_loc)
                
                '''
                S, lmax = cluster.calculate_S_ratio(plates, new_crystal)                       
                tilt_fracx, tilt_fracy = cluster.add_flow_tilt(new_crystal, lmax)                       
                xrottilt = xrotrand + (xrotrand*tilt_fracx)
                yrottilt = yrotrand + (yrotrand*tilt_fracy)
                rotation = [xrottilt, yrottilt, zrot]
                #new_crystal.rotate_to(rotation) 
                '''
                
                '''
                bound = new_crystal.projectxy().buffer(0).centroid.within(lmax_bound) 
                
                while bound is False:  #only if tilting causes the new crystal center to be outside of lmax
             
                    #from the seed crystal/agg (determined from S)
                    print('random miss', nmiss)
                    #print(cluster.points)
                    #print(new_crystal.points)
                    nmiss += 1
                    random_loc, lmax_bound = cluster.place_crystal(plates, new_crystal)  
                    new_crystal.move(random_loc)          
                
                    S, lmax = cluster.calculate_S_ratio(plates, new_crystal)                             
                    tilt_fracx, tilt_fracy = cluster.add_flow_tilt(new_crystal, lmax)                                           
                    xrottilt = xrotrand + (xrotrand*tilt_fracx)
                    yrottilt = yrotrand + (yrotrand*tilt_fracy)
                    rotation = [xrottilt, yrottilt, zrot]
                    new_crystal.rotate_to(rotation) 
                
                    bound = new_crystal.projectxy().buffer(0).centroid.within(lmax_bound)
                    
                    if nmiss > max_misses:
                        print('bound random %d > %d' %(nmiss, max_misses))               
                        break
                '''
                #tiltdiffx = np.abs((xrotrand*180/np.pi) - (xrottilt*180/np.pi))
                #tiltdiffy = np.abs((yrotrand*180/np.pi) - (yrottilt*180/np.pi))
            
                
            #'----------------- cluster - form aggregate------------------')
   
            # add to the cluster with constraints on S 
            # returns false if the crystal misses 
            # the 'lodge' value is just useful for testing against old IPAS code
            
            crystal_hit = cluster.add_crystal_from_above(new_crystal, lodge=lodge) 
            #print('l', l, cluster.ncrystals)
            
            
            #print(crystal_hit)
            if crystal_hit:
                 
                # recenter the cluster around the center of mass                       
                x, y, z = cluster.recenter()  
                
                new_crystal.move([-x,-y,-z])  #cluster.recenter doesn't move the new crystal when passed into 

                Ss[n,l], lmax = cluster.calculate_S_ratio(plates, new_crystal)
          
                #cont_ang_inst = cont_ang(seedcrystal, new_crystal, plates, dim, hypot, Sz
                #                         xmaxseed, xminseed, xmaxnew, xminnew)
               
                ovrlps[n,l] = cluster.overlap(new_crystal, seedcrystal)
         
                cluster.reorient(method=reorient, rotations=rotations)
                
                #cplx = cluster.complexity()
                l+=1
                
            else: # just in case something goes terribly wrong
                nmisses += 1
                if nmisses > max_misses:
                    print('crystal hit miss max %d > %d' %(nmisses, max_misses))
                    break
                    
        #cplxs.append(cplx)
        clusters.append(cluster)
        #cont_angles.append(cont_ang_inst)
        #tiltdiffsx.append(tiltdiffx)  #angle change in degrees after tilt added
        #tiltdiffsy.append(tiltdiffy)
        
    return batch.IceClusterBatch(clusters, length, width, r, numaspectratios, reorient, ovrlps, Ss,
                           xrot, yrot, plates)




