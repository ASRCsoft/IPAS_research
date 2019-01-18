"""Utilities for running ice particle simulations."""

import random
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
import scipy.stats as stats
#from scipy.stats import rv_continuous, gamma
import matplotlib 
import seaborn
import ipas.crystals_opt_rot as crys
import shapely.geometry as geom
import scipy
import warnings
import pandas as pd
import statsmodels as sm


def sim_clusters(length, width, nclusters, ncrystals=2, numaspectratios=1,
                 reorient='random', minor='depth',rotations=50, speedy=False, lodge=0, max_misses=100):
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
    import ipas.crystals_opt_rot as crys

    if speedy:
        # get optimal y rotation for single crystals
        #f = lambda x: -crys.IceCrystal(length=length, width=width, rotation=[x,0,0]).projectxy().area
        #xrot = opt.minimize_scalar(f, bounds=(0, np.pi/2), method='Bounded').x
        f = lambda x: -crys.IceCrystal(length=length, width=width, rotation=[0,x,0]).projectxy().area
        yrot = opt.minimize_scalar(f, bounds=(0, np.pi/2), method='Bounded').x

    clusters = []
    cont_angles = []
    ovrlps = []
    for n in range(nclusters):
        plates = width > length
        if speedy:                                
            
            if plates:
                
                rotation = [0, yrot, random.uniform(0, 2 * np.pi)]
            else:               
                          
                rotation = [0, yrot, random.uniform(0, 2 * np.pi)]
                #rotation = [np.pi/2, yrot, random.uniform(0, 2 * np.pi)]


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
        cluster = crys.IceCluster(seedcrystal, size=ncrystals)
    
        # add new crystals
        nmisses = 0
        while cluster.ncrystals < ncrystals:  
            #'-----------------cluster-------------------')
            # get the cluster's boundaries
            xmax = cluster.max('x')
            ymax = cluster.max('y')
            zmax = cluster.max('z')
            xmin = cluster.min('x')
            ymin = cluster.min('y')
            zmin = cluster.min('z')  
            random_loc = [random.uniform(xmin, xmax), random.uniform(ymin, ymax), 0]
            
            if speedy:
                if plates:
                    rotation = [0, yrot, random.uniform(0, 2 * np.pi)]

                else:
                    #rotation = [np.pi/2, yrot, random.uniform(0, 2 * np.pi)]
                    rotation = [0, yrot, random.uniform(0, 2 * np.pi)]

                new_crystal = crys.IceCrystal(length=length, width=width, rotation=rotation)  
            else:
                # make a new crystal, orient it
                
                new_crystal = crys.IceCrystal(length=length, width=width)               
                new_crystal.reorient(method=reorient, rotations=rotations)   
            new_crystal.move(random_loc)
                
            # add to the cluster
            crystal_hit = cluster.add_crystal_from_above(new_crystal, lodge=lodge) # returns false if the crystal misses
            # the 'lodge' value is just useful for testing against old IPAS code
            
            if crystal_hit:
                
                # recenter the cluster around the center of mass and reorient it
                cluster.recenter()      
                
                ovrlp = area_overlap(seedcrystal, new_crystal)  
                                
                #cont_ang_inst = cont_ang(seedcrystal, new_crystal, plates, dim, hypot, Sz
                #                         xmaxseed, xminseed, xmaxnew, xminnew)
        
                
                cluster.reorient(method=reorient, rotations=rotations)
            else: # just in case something goes terribly wrong
                nmisses += 1
                if nmisses > max_misses:
                    #('%d > %d' %(nmisses, max_misses))
                    break

        clusters.append(cluster)
        #cont_angles.append(cont_ang_inst)
        ovrlps.append(ovrlp)
    
 
    return IceClusterBatch(clusters, length, width, numaspectratios, reorient, ovrlps, plates)


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
        if plates:
            percentage = (Sx / widclus)*100   
        else:
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
    if plates:
        percentage = (Sz / heightclus)*100   
    else:
        percentage = (Sz / heightclus)*100
    percent.append(percentage)
    #print('vert_overlap', percent[2])
    
    return(percent)

def area_overlap(seedcrystal, new_crystal):

    seedpoly = seedcrystal.projectxy()
    newpoly = new_crystal.projectxy()         
    rel_area = seedcrystal.projectxy().intersection(new_crystal.projectxy()).area
    area_seed = rel_area / seedpoly.area
    area_new = rel_area / newpoly.area
    
    pctovrlp = (2*rel_area/(seedpoly.area+newpoly.area))*100

    return(pctovrlp)

def cont_ang(seedcrystal, new_crystal, plates, dim, hypot, Sz, Sx, xmaxseed, xminseed, xmaxnew, xminnew):
    from operator import itemgetter
    
    #CODE IN AGG_MAIN NOTEBOOK
    
    return cont_ang
                          
#####################################BATCH CLASS#############################################

class IceClusterBatch:
    """A collection of IceCluster objects."""
    
    def __init__(self, clusters, length, width, numaspectratios, reorient, ovrlps, plates=None):        
        self.length = length
        self.width = width
        self.phio = self.length/self.width 
        if plates is None:
            self.plates = width > length
        else:
            self.plates = plates # are they plates or columns?
        self.numaspectratios = numaspectratios
        self.clusters = clusters
        self.major_axis = {}
        self.minor_axis = {}
        self.reorient = reorient
        self.ovrlps = ovrlps
        self.chreq = 1.0
        self.chphi = 1.0
        self.ch_ovrlps = 1.0
        self.ch_majorax = 1.0
        self.ch_depth = 1.0
        self.phishape = 1.0
        self.reqshape = 1.0
        self.ovrlpshape = 1.0
        self.depthshape = 1.0
        self.majoraxshape = 1.0
        self.poserr_phi = 1.0
        self.negerr_phi = 1.0
        self.shapephi = 1.0
        self.shapechphi = 1.0
        self.shapereq = 1.0
        self.shapechreq = 1.0
        self.poserr_mjrax = 1.0
        self.negerr_mjrax = 1.0
        self.shapemjr = 1.0
        self.shapechmjr = 1.0
        self.negerr_req = 1.0
        self.poserr_depth = 1.0
        self.negerr_depth = 1.0
        self.shapedpt = 1.0
        self.shapechdpt = 1.0
        #self.ch_ovrlps = np.zeros(3)
        
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

    
    def calculateGammaParams(self, data, ch):
        mean = np.mean(data)
        std = np.std(data)
        shape = (mean/std)**2
        scale = (std**2)/mean

        shapech = mean/(self.numaspectratios*ch)
            
        pos_error = ch + std
        neg_error = ch - std
        return(pos_error, neg_error, shape, shapech)
        
    def plot_data(self, var, minor, ch_dist, bins=70, filename=None, ext='png',save = False, verbose=False):
        """Plot a histogram of cluster aspect ratios, sending extra arguments
        to `matplotlib.pyplot.hist`.
        """
        
        fig = plt.figure(figsize=(18,9))
        ax = plt.subplot(131)
        
        self.calc_aspect_ratios(minor)
        
        if var == 'phi':
            filename = '%.2f_%.2f_%.2f' % (self.phio, self.length, self.width)
            data = self.ratios
            xlabel = "Aggregate Aspect Ratio"
            if ch_dist == 'gamma':
                self.chphi, self.phishape = self.best_fit_distribution(data, ch_dist, xlabel=xlabel, bins=bins, ax=ax)
            else:
                self.chphi = self.best_fit_distribution(data, ch_dist, xlabel=xlabel, bins=bins, ax=ax)
                
            self.poserr_phi, self.negerr_phi, self.shapephi, self.shapechphi = self.calculateGammaParams(data, self.chphi)
            
            
        if var == 'req':
            filename = 'req_%.3f' % (self.phio)
            data = self.req
            xlabel = "Aggregate Eq. Vol. Radius"
            if ch_dist == 'gamma':
                self.chreq, self.reqshape = self.best_fit_distribution(data, ch_dist, xlabel=xlabel, bins=bins, ax=ax)
            else:
                self.chreq = self.best_fit_distribution(data, ch_dist, xlabel=xlabel, bins=bins, ax=ax)  
                
            self.poserr_req, self.negerr_req, self.shapereq, self.shapechreq = self.calculateGammaParams(data, self.chreq)

        if var == 'overlap':
            filename = 'overlap_%.3f' % (self.phio)
            data = self.ovrlps
            xlabel = "Overlap"
            if ch_dist == 'gamma':
                self.ch_ovrlps, self.ovrlpshape = self.best_fit_distribution(data, ch_dist, xlabel=xlabel, bins=bins, ax=ax)
            else:
                self.ch_ovrlps = self.best_fit_distribution(data, ch_dist, xlabel=xlabel, bins=bins, ax=ax)
                
        if var == 'depth':
            filename = 'depth_%.3f' % (self.phio)
            data = self.depth           

            xlabel = "Depth"           
            if ch_dist == 'gamma':
                self.ch_depth, self.depthshape = self.best_fit_distribution(data, ch_dist, xlabel=xlabel, bins=bins, ax=ax)
            else:
                self.ch_depth = self.best_fit_distribution(data, ch_dist, xlabel=xlabel, bins=bins, ax=ax)
                
            self.poserr_depth, self.negerr_depth, self.shapedpt, self.shapechdpt = \
            self.calculateGammaParams(data, self.ch_depth)
            
            
        if var == 'major_axis':
            filename = 'mjrax_%.3f' % (self.phio)
            data = self.major_axis['z']
            xlabel = "Major Axis"
            if ch_dist == 'gamma':
                self.ch_majorax, self.majoraxshape = self.best_fit_distribution(data, ch_dist, xlabel=xlabel, bins=bins, ax=ax)
            else:
                self.ch_majorax = self.best_fit_distribution(data, ch_dist, xlabel=xlabel, bins=bins, ax=ax)
    
            self.poserr_mjrax, self.negerr_mjrax, self.shapemjr, self.shapechmjr = \
            self.calculateGammaParams(data, self.ch_majorax)
            
           
            
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
        dphigam = chphi - self.phio    #aggregate aspect ratio - monomer aspect ratio (both crystals the same to start)
        r = np.power((np.power(self.width,2)*self.length),(1./3.)) #of crystal
        plt.title('Monomer Req=%.3f with shape=%.3f and $\phi_n$=%.3f' %(r,shape,chphi))
        plt.xlabel ("Aggregate Eq. Vol. Radius")
        '''      

        self.dphigam = self.chphi - self.phio
        
        if save:
            self.save_fig(filename = filename, ext=ext, verbose = verbose)
        plt.close() 

        
    def best_fit_distribution(self, data, ch_dist, bins, xlabel, ax=None, normed = True, facecolor='navy', 
                              alpha=1.0,**kwargs):

        import scipy.stats as st
        
        """Model data by finding best fit distribution to data"""
        # Get histogram of original data
       
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


    def alloverlap(self, save='False', ext='png', verbose='False', bins=70, normed = True, 
                      facecolor='navy', alpha=1.0,**kwargs):   
        xyz = ['X','Y']
        xlabels = ['X Overlap', 'Y Overlap']
        #print('self.overlaps',self.ovrlps)
            
        for i in range(len(xyz)):
            if xyz[i] == 'Y':
                 bins = 50
            self.overlaps = [item[i] for item in self.ovrlps]
            #print('overlaps @ position %s = '%(xyz[i]))                
            
            if self.reorient == 'IDL':
                filename = xyz[i]+'overlap_%.3f_IDL' % (self.phio)              
            else:
                filename = xyz[i]+'overlap_%.3f' % (self.phio)
            
            #----------------------------------------------------------
            index = np.array(range(len(self.clusters))) + 1
            data = pd.Series(self.overlaps, index=index).values.ravel()
            #data['self.overlaps'] = self.overlaps
            data = data.astype(float)
            
            fig = plt.figure(figsize=(18,9))

            ax = plt.subplot(131)
            n, bins1, patches = plt.hist(data, bins, normed=True, color='navy',range=(min(data), max(data)), **kwargs)
            plt.title('All Fitted Distributions')
            plt.ylabel ("Frequency") 
            plt.ylim((0,max(n)))
            #if xyz[i] == 'Y':
            #    print('max n', max(n), n)
            xlabel = xlabels[i]
            
            # Find best fit distribution           
            #best_fit_name, best_fir_paramms= self.best_fit_distribution(data, bins, ax=ax)
            self.ch_ovrlps[i] = self.best_fit_distribution(data, bins, xlabel, ax=ax)
            #best_dist = getattr(stats, best_fit_name)
                                
            #shape, loc, scale = stats.gamma.fit(self.overlaps)  #loc is mean, scale is stnd dev
            #print('shape, loc, scale',shape,loc,scale)
            #x = np.linspace(min(self.overlaps),max(self.overlaps),1000)
            #g = stats.gamma.pdf(x=x, a=shape, loc=loc, scale=scale)
           
            #plot histogram with best fit line
            #n, bins, patches = plt.hist(self.overlaps, bins=bins, normed = True, facecolor = facecolor, alpha=alpha, **kwargs)
            
            #plt.plot(x, g, color='darkorange', linewidth=3, linestyle='--')
            #print('g = ',g)
            #gmax = max(g)  #highest probability
            #indmax = np.argmax(g)  #FIRST index where the highest prob occurs
            #ch_over = x[indmax] #characteristic aspect ratio of the distribution      
            #print('ch_over',ch_over)
            
            #plt.title('Monomer aspect ratio=%.3f, Characteristic %s overlap=%.3f' %(self.phio, xyz[i], ch_over))
            
            #self.ch_ovrlps[i] = ch_over

            
            if save:
                self.save_fig(filename = filename, ext=ext, verbose = verbose)
            plt.close()


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
            

    
    def write_instances(self, filename='outfile.dat', save='False', ext='png', verbose=False):
        #writes data for each aspect ratio being looped over, creates file if it does not exist
        
        #if self.ch_cont_ang is None:
        #    self.contact_angle(save=save, ext=ext, verbose=verbose)

        with open(filename, 'a+') as f:
            #print(self.phio, self.chphi, self.dphigam, self.ch_ovrlps, self.chreq, self.length)
            #print(self.width, self.ch_majorax, self.ch_depth, self.phishape, self.reqshape)
            #print(self.depthshape, self.majoraxshape, self.ovrlpshape)
            writing = f.write('%10.4f\t %7.3f\t %7.3f\t %7.3f\t %7.3f\t %7.3f\t %7.3f\t %7.3f\t %7.3f\t %7.3f\t %7.3f\t %7.3f\t %7.3f\n' %(self.phio, self.chphi, self.dphigam, self.ch_ovrlps, self.chreq, self.length, self.width, self.ch_majorax, self.ch_depth, self.phishape, self.reqshape, self.depthshape, self.majoraxshape))
            
            
    def write_shape_params(self, filename='shapefile.dat', save='False', ext='png', verbose=False):
                   
        with open(filename, 'a+') as f:     
            f.write('%7.3f\t %7.3f\t %7.3f\t %7.3f\t %7.3f\t %7.3f\t %7.3f\t %7.3f\n' %(self.shapephi, self.shapechphi, self.shapereq, self.shapechreq, self.shapedpt, self.shapechdpt, self.shapemjr, self.shapechmjr))
            
    def write_error_params(self, filename='errorfile.dat', save='False', ext='png', verbose=False):
                   
        with open(filename, 'a+') as f:     
            f.write('%7.3f\t %7.3f\t %7.3f\t %7.3f\t %7.3f\t %7.3f\t %7.3f\t %7.3f\n' %(self.poserr_phi, self.negerr_phi, self.poserr_req, self.negerr_req, self.poserr_depth, self.negerr_depth, self.poserr_mjrax, self.negerr_mjrax))
            
            
    def which_plot(self, plot_name, ch_dist, savefile, read_file='outfile.dat', save=False, verbose=False, ext='png'):   
        #reads distribution characteristics into arrays
        
        import seaborn as sns
        from matplotlib.ticker import MultipleLocator, FormatStrFormatter, LogLocator, LogFormatter, ScalarFormatter
        import pandas as pd
        
        #self.numaspectratios = 26

        phio = np.zeros(self.numaspectratios)
        ch = np.zeros(self.numaspectratios)
        dphigam = np.zeros(self.numaspectratios)
        overlap = np.zeros(self.numaspectratios)
        req = np.zeros(self.numaspectratios)
        phip = np.zeros(self.numaspectratios)
        phic = np.zeros(self.numaspectratios)
        chp = np.zeros(self.numaspectratios)
        chc = np.zeros(self.numaspectratios)
        lengthp = np.zeros(self.numaspectratios)
        lengthc = np.zeros(self.numaspectratios)
        widthp = np.zeros(self.numaspectratios)
        widthc = np.zeros(self.numaspectratios)
        cont_ang = np.zeros(self.numaspectratios)
        cont_ang2 = np.zeros(self.numaspectratios)
        cont_ang3 = np.zeros(self.numaspectratios)
        cont_ang4 = np.zeros(self.numaspectratios)
        cont_ang5 = np.zeros(self.numaspectratios)
        Xoverlap = np.zeros(self.numaspectratios)
        Yoverlap = np.zeros(self.numaspectratios)
        Zoverlap = np.zeros(self.numaspectratios)
        major_axis = np.zeros(self.numaspectratios)
        depth = np.zeros(self.numaspectratios)
        length = np.zeros(self.numaspectratios)
        width = np.zeros(self.numaspectratios)
        phishape = np.zeros(self.numaspectratios)
        reqshape = np.zeros(self.numaspectratios)
        depthshape = np.zeros(self.numaspectratios)
        majoraxshape = np.zeros(self.numaspectratios)
        #ovrlpshape = np.zeros(self.numaspectratios)
        
        #read_files = ['outfile_contang2.dat','outfile_contang3.dat',
        #              'outfile_contang4.dat','outfile_contang5.dat']
        #for f in read_files:
            #fh = open(read_file, 'r')
            
        fh = open(read_file, 'r')
        header1 = fh.readline()
        header = fh.readline()
        for i in range(0,self.numaspectratios):            
            data=fh.readline().strip().split('\t')
            #print(data)
            phio[i] = float(data[0])             
            ch[i] = float(data[1])           
            dphigam[i] = float(data[2])
            overlap[i] = float(data[3])
            req[i] = float(data[4])
            length[i] = float(data[5])
            width[i] = float(data[6])
            major_axis[i] = float(data[7])
            depth[i] = float(data[8])
            phishape[i] = float(data[9])
            reqshape[i] = float(data[10])
            depthshape[i] = float(data[11]) 
            majoraxshape[i] = float(data[12]) 
            
            #ovrlpshape[i] = float(data[13]) 
            #Xoverlap[i] = float(data[4])

        '''     
                if f == 'outfile_contang2.dat':
                    cont_ang2[i] = float(data[5])
                if f == 'outfile_contang3.dat':
                    cont_ang3[i] = float(data[5])
                if f == 'outfile_contang4.dat':
                    cont_ang4[i] = float(data[5])
                if f == 'outfile_contang5.dat':
                    cont_ang5[i] = float(data[5])
        '''            
        
        
        poserr_phi = np.zeros(self.numaspectratios)
        negerr_phi = np.zeros(self.numaspectratios)
        shapephi = np.zeros(self.numaspectratios) 
        shapechphi = np.zeros(self.numaspectratios)
        poserr_req = np.zeros(self.numaspectratios) 
        negerr_req = np.zeros(self.numaspectratios)
        shapereq = np.zeros(self.numaspectratios) 
        shapechreq = np.zeros(self.numaspectratios)
        poserr_depth = np.zeros(self.numaspectratios)
        negerr_depth = np.zeros(self.numaspectratios) 
        shapedpt = np.zeros(self.numaspectratios)
        shapechdpt = np.zeros(self.numaspectratios) 
        poserr_mjrax = np.zeros(self.numaspectratios)
        negerr_mjrax = np.zeros(self.numaspectratios)
        shapemjr = np.zeros(self.numaspectratios)
        shapechmjr = np.zeros(self.numaspectratios)

        shape_file = 'shapefile.dat'
        fh = open(shape_file, 'r')
        header = fh.readline()
        for i in range(0,self.numaspectratios):            
            data=fh.readline().strip().split('\t')
            #print(data)                    
            shapephi[i] = float(data[0])
            shapechphi[i] = float(data[1])         
            shapereq[i] = float(data[2])
            shapechreq[i] = float(data[3])            
            shapedpt[i] = float(data[4])
            shapechdpt[i] = float(data[5])             
            shapemjr[i] = float(data[6]) 
            shapechmjr[i] = float(data[7]) 
            
        error_file = 'errorfile.dat'
        fh = open(error_file, 'r')
        header = fh.readline()
        for i in range(0,self.numaspectratios):            
            data=fh.readline().strip().split('\t')
            #print(data) 
            poserr_phi[i] = float(data[0])
            negerr_phi[i] = float(data[1])         
            poserr_req[i] = float(data[2])
            negerr_req[i] = float(data[3])      
            poserr_depth[i] = float(data[4])
            negerr_depth[i] = float(data[5])   
            poserr_mjrax[i] = float(data[6]) 
            negerr_mjrax[i] = float(data[7])         
                
        fig = plt.figure(figsize=(8,6))
        ax = plt.subplot(111)
        ax.set_xscale("log")
        #ax.set_yscale("log")
        ax.grid(b=True, which='minor', alpha = 0.9, axis = 'x')
        ax.grid(b=True, which='minor', alpha = 0.9, axis = 'y')
        ax.grid(b=True, which='major', alpha = 1.0, axis = 'x')            
        ax.grid(b=True, which='major', alpha = 1.0, axis = 'y')    
        
        #plt.style.use('seaborn-dark')
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.size'] = 14
        plt.rcParams['axes.labelsize'] = 14
        plt.rcParams['axes.labelweight'] = 'normal'
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['xtick.labelsize'] = 12
        plt.rcParams['ytick.labelsize'] = 12
        plt.rcParams['legend.fontsize'] = 11
        
        plt.rcParams['figure.frameon']= True
        plt.rcParams['figure.titlesize'] = 14

        
        if self.minor == 'minorxy':
            label='Using minor from x or y'
        else:
            label='Using Depth'
            
        wh=np.where(phio<1.0)
        phip[wh] = phio[wh]
        phic[wh] = np.nan
        chp[wh] = ch[wh]
        chc[wh] = np.nan
        lengthp[wh] = length[wh]
        lengthc[wh] = np.nan
        widthp[wh] = width[wh]
        widthc[wh] = np.nan
        
        wh=np.where(phio>=1.0)
        phic[wh] = phio[wh]
        phip[wh] = np.nan        
        chc[wh] = ch[wh]
        chp[wh] = np.nan
        lengthc[wh] = length[wh]
        lengthp[wh] = np.nan
        widthc[wh] = width[wh]
        widthp[wh] = np.nan
        
        phip = phip[~pd.isnull(phip)]
        chp = chp[~pd.isnull(chp)]
        phic = phic[~pd.isnull(phic)]
        chc = chc[~pd.isnull(chc)]
        lengthp = lengthp[~pd.isnull(lengthp)]
        lengthc = lengthc[~pd.isnull(lengthc)]
        widthp = widthp[~pd.isnull(widthp)]
        widthc = widthc[~pd.isnull(widthc)]
        #print(phic)
        #print(chc)      
        
        if ch_dist=='gamma':
            dist_title = 'Gamma'
        else:
            dist_title = 'Best'            
               
            
        if plot_name == 'char':                        
            '''
            #REGPLOT FIT
            p = sns.regplot(phip,chp,color='navy',scatter_kws={"s": 60},line_kws={'color':'darkorange'},lowess=False,label=label)
            q = sns.regplot(phic,chc,color='navy',scatter_kws={"s": 60},line_kws={'color':'darkorange'},lowess=True)
            slopep, interceptp, r_valuep, p_valuep, std_errp = stats.linregress(x=p.get_lines()[0].get_xdata(),y=p.get_lines()
                                                                                [0].get_ydata())
            slopeq, interceptq, r_valueq, p_valueq, std_errq = stats.linregress(x=q.get_lines()[0].get_xdata(),y=q.get_lines()
                                                                                [0].get_ydata())            
            '''
            #POLYFIT
           
            #coeffs = np.polyfit(phip,chp,4)
            #x2 = np.arange(min(phip)-1, max(phip), .01) #use more points for a smoother plot
            #y2 = np.polyval(coeffs, x2) #Evaluates the polynomial for each x2 value
            #plt.plot(x2, y2,color='darkorange',linewidth=2.5)  #best fit line
            plt.plot(phip,chp,'o',color='navy',markersize=6)
            plt.plot(phip, chp, color='navy')

            #coeffs = np.polyfit(phic,chc,4)
            #x2 = np.arange(min(phic), max(phic)+1, .01) #use more points for a smoother plot
            #y2 = np.polyval(coeffs, x2) #Evaluates the polynomial for each x2 value
            #plt.plot(x2, y2,color='darkorange',linewidth=2.5)
            plt.plot(phic,chc,'o',color='navy',markersize=6)
            plt.plot(phic, chc, color='navy')

            ax.set_yscale("log")
            
            plt.xlim((.01,100))
            #plt.ylim((min(ch),max(ch)))
            
            #ax.set_yticks([10,20,30,40, 60])

            ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))
            ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.3f"))
            #ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
            #ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
            #plt.ticklabel_format(style='plain', axis='y')
            #ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())


            #plt.title('Characteristic Aggregate Aspect Ratio for '+str(self.numaspectratios)+' Aspect Ratios')
            #plt.xlabel ("Equivalent Volume Aspect Ratio of Monomers")
            #plt.ylabel ("Characteristic Aggregate Aspect Ratio")
            plt.title('Characteristic Aggregate Aspect Ratio from %s Distribution' %dist_title)
            plt.ylabel ("Aggregate Aspect Ratio")
            #plt.legend(loc='best')                
            
        if plot_name == 'shape':
                
            #REGPLOT FIT
            
            #p = sns.regplot(phio,shape,color='navy',scatter_kws={"s": 80},\
            #    line_kws={'color':'darkorange'},lowess=True, label=label)
            #slopep, interceptp, r_valuep, p_valuep, std_errp = stats.linregress(x=p.get_lines()[0].get_xdata(),\
            #                y=p.get_lines()[0].get_ydata())
            
            #POLYFIT
            '''
       
            coeffs = np.polyfit(phio,shape,2)
            x2 = np.arange(min(phio)-1, max(phio), .01) #use more points for a smoother plot
            y2 = np.polyval(coeffs, x2) #Evaluates the polynomial for each x2 value
            plt.plot(x2, y2,color='darkorange',linewidth=2.5)
            '''
            if ch_dist == 'gamma':
                
                titlearr = ['$\phi$ ', 'Equiv. Vol. ', 'Depth ', 'Major Axis ',\
                           'Std. $\phi$ ', 'Std. Equiv. Vol. ', 'Std. Depth ', 'Std. Major Axis ',\
                           'Char $\phi$  ', 'Char Equiv. Vol. ', 'Char Depth ', 'Char Major Axis ']
                filearr = ['phi', 'Req', 'Depth', 'MjrAx','phistd', 'Reqstd', 'Depthstd', 'MjrAxstd',\
                          'phich', 'Reqch', 'Depthch', 'MjrAxch']

                shapearr = [[] for i in range(len(titlearr))]
                shapearr[0]=phishape
                shapearr[1]=reqshape 
                shapearr[2]=depthshape
                shapearr[3]=majoraxshape
                shapearr[4]=shapephi
                shapearr[5]=shapereq
                shapearr[6]=shapedpt
                shapearr[7]=shapemjr
                shapearr[8]=shapechphi              
                shapearr[9]=shapechreq               
                shapearr[10]=shapechdpt                
                shapearr[11]=shapechmjr
                
                for i in range(0,len(titlearr)):            
                    
                    fig = plt.figure(figsize=(8,6))
                    ax = plt.subplot(111)
                    ax.set_xscale("log")
                    #ax.set_yscale("log")
                    ax.grid(b=True, which='minor', alpha = 0.9, axis = 'x')
                    ax.grid(b=True, which='minor', alpha = 0.9, axis = 'y')
                    ax.grid(b=True, which='major', alpha = 1.0, axis = 'x')            
                    ax.grid(b=True, which='major', alpha = 1.0, axis = 'y')    

                    plt.plot(phio,shapearr[i],'o',color='navy',markersize=6)      
                    plt.title('%s Shape Parameter from %s Distribution' %(titlearr[i], dist_title))
                    #plt.title('Shape Parameter for '+str(self.numaspectratios)+' Aspect Ratios')           
                    plt.ylabel("Shape of %s Distribution" %titlearr[i])
                    plt.xlabel ("Equivalent Volume Aspect Ratio of Monomer")
                    plt.ylim(min(shapearr[i]),max(shapearr[i])+20) 
                    #plt.ylim(min(shape),10000) 
                    plt.xlim((.009,110))
                    ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))
                    #ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.3f"))
                    
                                          
                    if save:                      
                        savefile = '%s_shape' %filearr[i]  
                        if self.reorient == 'IDL':
                            filename = savefile+'_IDL'         
                        else:                           
                            filename = savefile     

                        self.save_fig(filename=filename, ext=ext, verbose = verbose)
                   
                
                
            else:
                print('Shape parameter plot is only applicable to the gamma distribution')
            
            
        if plot_name == 'dphigamquad':
            
            #Quadrant Plot
            
            dphi_neg_plate = dphigam[(phio<1.0) & (dphigam<0.0)]           
            phio_neg_plate = phio[(phio<1.0) & (dphigam<0.0)]
            dphi_pos_plate = dphigam[(phio<1.0) & (dphigam>0.0)]
            phio_pos_plate = phio[(phio<1.0) & (dphigam>0.0)]
            dphi_pos_col = dphigam[(phio>1.0) & (dphigam>0.0)]
            phio_pos_col = phio[(phio>1.0) & (dphigam>0.0)]
            dphi_neg_col = dphigam[(phio>1.0) & (dphigam<0.0)]
            phio_neg_col = phio[(phio>1.0) & (dphigam<0.0)]

            ax1 = plt.subplot(211)       
            ax1.grid(b=True, which='minor', alpha = 0.5, axis = 'y')
            ax1.grid(b=True, which='minor', alpha = 0.5, axis = 'x')
            ax1.grid(b=True, which='major', alpha = 0.9, axis = 'x')            
            ax1.grid(b=True, which='major', alpha = 0.9, axis = 'y')            
            ax1.set_xscale("log")
            ax1.set_yscale("log")
            plt.title('Characteristic Change in $\phi$ from %s Distribution' %dist_title)
            fig.text(0.03, 0.5, 'Aggregate $\phi$ - Monomer $\phi$', 
                     va='center', rotation='vertical', fontsize=14)
            maxc = max(dphi_pos_col)
            maxp = max(dphi_pos_plate)
            plt.ylim(.011, max(maxc, maxp)+.1)
            plt.setp(ax1.get_xticklabels(), fontsize=6)
            ax1.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))
            plt.plot(phio_pos_plate,dphi_pos_plate, color='navy', marker='o', markersize=7, label=label)
            plt.plot(phio_pos_col,dphi_pos_col, color='navy', marker='o', markersize=7)
            plt.axhline(y=0.0112, color='black', linestyle='-',linewidth=4.5)
            plt.axvline(x=1.0, color='black', linestyle='-',linewidth=2.5)
            #plt.legend(loc='best')
            
            # share x only            
            ax2 = plt.subplot(212, sharex=ax1)
            ax2.grid(b=True, which='minor', alpha = 0.5, axis = 'y')
            ax2.grid(b=True, which='minor', alpha = 0.5, axis = 'x')
            ax2.grid(b=True, which='major', alpha = 0.9, axis = 'x')            
            ax2.grid(b=True, which='major', alpha = 0.9, axis = 'y')
            ax2.set_xscale("log")
            ax2.set_yscale("log")
            ax2.set_xlabel ("Monomer Aspect Ratio")
            ax2.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))
            ax2.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))

            plt.gca().invert_yaxis()
            plt.plot(phio_neg_plate,abs(dphi_neg_plate), color='navy', marker='o', markersize=7)
            plt.plot(phio_neg_col,abs(dphi_neg_col), color='navy', marker='o', markersize=7)
            plt.axvline(x=1.0, color='black', linestyle='-',linewidth=2.5)

            #plt.ylim(max(abs(dphi_neg_col)+100),min(abs(dphi_neg_plate))) 
            plt.subplots_adjust(hspace=0)
            
        if plot_name == 'dphigamW':
            #W Plot
            plt.plot(phio,abs(dphigam),color='navy',label=label)
            plt.plot(phio,abs(dphigam), color='navy',marker='o',markersize=5)
            ax.set_yscale("log")
            #plt.title('Change in $\phi$ for '+str(self.numaspectratios)+' Aspect Ratios')
            plt.title('Change in eq. vol. $\phi$ from Characteristic of Best Distribution')
            ax.set_xlabel ("Equivalent Volume Monomer Aspect Ratio")
            ax.set_ylabel('Aggregate $\phi$ - Monomer $\phi$')
            #plt.legend(loc='best')
            ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))
            ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.3f"))
            
             
        if plot_name == 'Xoverlap':
        
            plt.xlim((.01,100))
            #plt.ylim((min(Xoverlap),100))
            ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))
            ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))
            plt.plot(phio,Xoverlap,color='navy')
            plt.plot(phio,Xoverlap, color='navy',marker='o',markersize=5)

            plt.title('Horizontal X Overlap for '+str(self.numaspectratios)+' Aspect Ratios')
            plt.ylabel ("Characteristic Overlap [%]")  
            #ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
            #ax.yaxis.get_major_formatter().set_scientific(False)
            #ax.yaxis.get_major_formatter().set_useOffset(False)
            
        if plot_name == 'Yoverlap':
        
            plt.xlim((.01,100))
            #plt.ylim((min(Yoverlap),10))
            ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))
            ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%3.2f"))
            plt.plot(phio,Yoverlap,'o',color='navy',markersize=6)
            plt.plot(phio,Yoverlap,color='navy')
            plt.title('Horizontal Y Overlap for '+str(self.numaspectratios)+' Aspect Ratios')
            plt.ylabel ("Characteristic Overlap [%]")  
            ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
            ax.yaxis.get_major_formatter().set_scientific(False)
            ax.yaxis.get_major_formatter().set_useOffset(False)
            
        if plot_name == 'overlap':
        
            plt.xlim((.01,100))
            #plt.ylim((min(overlap),10))
            ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))
            ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%3.2f"))
            plt.plot(phio,overlap,'o',color='navy',markersize=6)
            plt.plot(phio,overlap,color='navy')
            plt.title('Characteristic Overlap from %s Distribution' %dist_title)
            plt.ylabel ("Overlap [%]")  
            
        if plot_name == 'req':
        
            plt.xlim((.01,100))
            plt.ylim((0,max(req)))
            ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))
            ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%3.2f"))
            plt.plot(phio,req,'o',color='navy',markersize=6)
            plt.plot(phio,req,color='navy')
            plt.axhline(y=10, color='navy', linestyle='--')
            plt.title('Characteristic Eq. Volume Radius from %s Distribution' %dist_title)
            plt.ylabel ("Eq. Volume Radius")  
            
        if plot_name == 'major_axis':
            ax.set_yscale("log")
            plt.xlim((.01,100))
            #plt.ylim((min(major_axis),10))
            ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))
            ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%3.2f"))
            ax.fill_between(phio, poserr_mjrax, negerr_mjrax, facecolor='navy', alpha=0.5)
            ax.fill_between(phio, poserr_depth, negerr_depth, facecolor='darkorange', alpha=0.5)
            plt.plot(phio,major_axis,'o',color='navy',markersize=6,label = 'major_axis from ellipse')
            plt.plot(phio,major_axis,color='navy')
            plt.plot(phio,depth,'o',color='darkorange',markersize=6, label = 'depth')
            plt.plot(phio,depth,color='darkorange')
            #plt.plot(phip,chp,'o',color='darkgreen',markersize=6, label = 'agg aspect ratio')
            #plt.plot(phip, chp, color='darkgreen')
            #plt.plot(phic,chc,'o',color='darkgreen',markersize=6)
            #plt.plot(phic, chc, color='darkgreen')           
            plt.plot(phip,lengthp,color='darkorange', linestyle = '--')    
            plt.plot(phic,lengthc,color='navy', linestyle = '--')   
            plt.plot(phip,widthp,color='navy',linestyle = '--')
            plt.plot(phic,widthc,color='darkorange',linestyle = '--')
     
            plt.title('Major Axis, Depth, and Aggregate Aspect Ratio from %s Distribution' %dist_title)
            plt.ylabel ("Characteristic a/c axis and $\phi$ of Aggregate")  
            plt.legend(loc='best')
        plt.xlabel ("Equivalent Volume Aspect Ratio of Monomer")
        
   
        if plot_name != 'shape' and save:

            if self.reorient == 'IDL':
                filename = savefile+'_IDL'         
            else:
                filename = savefile     

            self.save_fig(filename=filename, ext=ext, verbose = verbose)

        else:
            plt.close()
            
