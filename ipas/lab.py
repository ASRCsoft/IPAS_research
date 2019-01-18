"""Utilities for running ice particle simulations."""

import random
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
import scipy.stats as stats
#from scipy.stats import rv_continuous, gamma
import matplotlib 
import seaborn
import ipas.crystals as crys

def sim_clusters(length, width, nclusters, ncrystals=2, numaspectratios=1,
                 reorient='random', minor='depth',rotations=50, speedy=False, lodge=0):
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
    import ipas.crystals as crys
    if speedy:
        # get optimal y rotation for single crystals
        f = lambda x: -crys.IceCrystal(length=length, width=width, rotation=[0,x,0]).projectxy().area
        yrot = opt.minimize_scalar(f, bounds=(0, np.pi/2), method='Bounded').x
    clusters = []
    cont_angles = []
    xoverlap = []
    for n in range(nclusters):
        if speedy:
            #rotation = [0, yrot, random.uniform(0, 2 * np.pi)]
            rotation = [0, yrot, random.uniform(0, 2 * np.pi)]
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
        plates = width > length
        while cluster.ncrystals < ncrystals:
            # get the cluster's boundaries
            xmax = cluster.max('x')
            ymax = cluster.max('y')
            zmax = cluster.max('z')
            xmin = cluster.min('x')
            ymin = cluster.min('y')
            zmin = cluster.min('z')           
          
            random_loc = [random.uniform(xmin, xmax), random.uniform(ymin, ymax), 0]
            if speedy:
                rotation = [0, yrot, random.uniform(0, 2 * np.pi)]
                new_crystal = crys.IceCrystal(length=length, width=width, center=random_loc, rotation=rotation)
            else:
                # make a new crystal, orient it
                new_crystal = crys.IceCrystal(length=length, width=width)               
                new_crystal.reorient(method=reorient, rotations=rotations)
    
                new_crystal.move(random_loc)
            # add to the cluster
            crystal_hit = cluster.add_crystal_from_above(new_crystal, lodge=lodge) # returns false if the crystal misses
            # the 'lodge' value is just useful for testing against old IPAS code
            if crystal_hit:
                #cluster.plot()          
                #plt.show()
                
                # recenter the cluster around the center of mass and reorient it
                cluster.recenter()
                 
                #update cluster boundary from addition of new crystal
                xmax = cluster.max('x')
                xmin = cluster.min('x')
                
                #X OVERLAP before reorientation of aggregate?
                    #Overlap in x and y orientation?
                
                lencluster = xmax-xmin #0 index is x
                lenseed = seedcrystal.points['x'].max() - seedcrystal.points['x'].min()
                lennew = new_crystal.points['x'].max() - new_crystal.points['x'].min()
                Sx = lencluster - (lencluster - lenseed) - (lencluster - lennew)    
                if plates:
                    Sxpercent = Sx / width
                else:
                    Sxpercent = Sx / length
                
                #print(Sx)
                              
                
                #CONTACT ANGLE -- before reorientation of aggregate               
                
                diag = np.sqrt((width**2)+(length**2))
                height_seed = seedcrystal.maxz - seedcrystal.minz
                height_new = new_crystal.maxz - new_crystal.minz
                mu1 = np.arcsin(height_seed/diag)*(180/np.pi)
                mu2 = np.arcsin(height_new/diag)*(180/np.pi)

                if plates:
                    gamma = np.arccos(width/diag)*(180/np.pi)                    
                    #theta = np.arccos(length/diag)*(180/np.pi) 
                else:
                    gamma = np.arccos(length/diag)*(180/np.pi)                    
                    #theta = np.arccos(width/diag)*(180/np.pi)
                    
                seed_ang = mu1 - gamma          
                new_ang = mu2 - gamma    
                
                if seed_ang < 0:        
                    #print(seed_ang)
                    seed_ang = gamma - mu1                    
                    #print(n, mu1, gamma, seed_ang)
                if new_ang < 0:
                    new_ang = gamma - mu2
                                                                          
                #print('cont1',cont_ang)
                #print(cont_ang, mu1, mu2)
                
                #Calculate from x/y plane for both seed and new crystal and then sum angles
                if plates:
                    seed_ang = np.arcsin((height_seed - length)/width)*(180/np.pi)
                    new_ang = np.arcsin((height_new - length)/width)*(180/np.pi)
                    
                else:
                    seed_ang = np.arcsin((height_seed - width)/length)
                    seed_ang=seed_ang*(180/np.pi)
                    new_ang = np.arcsin((height_new - width)/length)*(180/np.pi)
                    
                '''    
                pointszseed = seedcrystal.points['z']
                maxzindseed = np.where(pointszseed == seedcrystal.maxz)
                minzindseed = np.where(pointszseed == seedcrystal.minz)                
                xatmaxzseed = seedcrystal.points['x'][maxzindseed]
                xatminzseed = seedcrystal.points['x'][minzindseed]
                #print(xatmaxzseed,xatminzseed)
                
                pointsznew = new_crystal.points['z']
                maxzindnew = np.where(pointsznew == new_crystal.maxz)
                minzindnew = np.where(pointsznew == new_crystal.minz)                
                xatmaxznew = new_crystal.points['x'][maxzindnew]
                xatminznew = new_crystal.points['x'][minzindnew]
                #print(xatmaxznew,xatminznew)
                '''
                cont_ang = new_ang + seed_ang
                #print('height_seed,width,height-width',height_seed, width, (height_seed - width),length,
                #      (height_seed - width)/length)
                #print(seed_ang)
                #print('contact angle',cont_ang)
                
                cluster.reorient(method=reorient, rotations=rotations)
                
        clusters.append(cluster)
        cont_angles.append(cont_ang)
        xoverlap.append(Sxpercent)
    
 
    return IceClusterBatch(clusters, length, width, numaspectratios, reorient, cont_angles, xoverlap, plates)

class IceClusterBatch:
    """A collection of IceCluster objects."""
    
    def __init__(self, clusters, length, width, numaspectratios, reorient, cont_angles, xoverlap, plates=None):        
        self.length = length
        self.width = width
        if plates is None:
            self.plates = width > length
        else:
            self.plates = plates # are they plates or columns?
        self.numaspectratios = numaspectratios
        self.clusters = clusters
        self.ratios = None
        self.major_axis = {}
        self.minor_axis = {}
        self.chgamma = None
        self.dphigam = None
        self.shape = None
        self.phio = None
        self.reorient = reorient
        self.minor = None
        self.ch_cont_ang = None
        self.cont_angles = cont_angles
        self.xoverlap = xoverlap
        self.ch_xoverlap = None
        self.ch_overlap = None
        self.req = None
        
    def calc_aspect_ratios(self, minor):
        """Calculate the aspect ratios of the clusters using ellipses fitted
        to the 2D cluster projections from the x-, y-, and z-axis
        perspectives.

        """             
        if self.plates is None:
            # inform the user that they need to specify whether these
            # clusters are made of columns or plates
            pass
        if self.plates:
            # if the crystals are plates do the plate version
            ratios = [ cluster.aspect_ratio(method='plate', minor=minor) for cluster in self.clusters ]            
            self.major_axis['z'] = [ cl.major_axis['z'] for cl in self.clusters ] #this has to be after ratios
            depth = [ cl.depth for cl in self.clusters ] 
            self.req = np.power((np.power(self.major_axis['z'],2)*depth),(1./3.))     
            
        else:
            ratios = [ cluster.aspect_ratio(method='column', minor=minor) for cluster in self.clusters ]
            self.major_axis['z'] = [ cl.major_axis['z'] for cl in self.clusters ]
            depth = [ cl.depth for cl in self.clusters ]
            self.req = np.power((np.power(depth,2)*self.major_axis['z']),(1./3.))     
              
        self.ratios = ratios
        self.minor = minor

        return ratios
    
    def calculateGammaParams(self, data):
        mean = np.mean(data)
        std = np.std(data)
        shape = (mean/std)**2
        scale = (std**2)/mean
        return (shape, 0, scale)
 
        
    def plot_aspect_ratios(self, minor, filename=None, ext='png',save = False, verbose=False, bins=70, 
                           normed = True, facecolor='navy', alpha=1.0,**kwargs):
        """Plot a histogram of cluster aspect ratios, sending extra arguments
        to `matplotlib.pyplot.hist`.
        """
        phio = self.length/self.width  
        
        if filename is None:
            if self.reorient == 'IDL':
                filename = '%.2f_%.2f_%.2f_IDL' % (phio, self.length, self.width)              
            else:
                filename = '%.2f_%.2f_%.2f' % (phio, self.length, self.width)
            
        if self.ratios is None:
            self.calc_aspect_ratios(minor)
         
        #GAMMA distribution
        fig = plt.figure(figsize=(7,5))
        ax = plt.subplot(111)
        
        #plot histogram with best fit line, weights used to normalize
        weights = np.ones_like(self.ratios)/float(len(self.ratios))        
        #print('ones_like',np.ones_like(self.ratios))
        #print(weights)
        n, bins, patches = ax.hist(self.ratios, bins=bins, weights = weights, 
                                   normed = False, facecolor = 'navy', alpha=alpha, **kwargs)

        shape, loc, scale = stats.gamma.fit(self.ratios)  #loc is mean, scale is stnd dev      
        #print('from fit', shape,loc,scale)
        #eshape, eloc, escale = self.calculateGammaParams(self.req)
        #print('estimates', eshape, eloc, escale)        
        #x = np.linspace(stats.gamma.pdf(min(self.req), shape),stats.gamma.pdf(max(self.req),shape),100)
        x = np.linspace(min(self.ratios),max(self.ratios),1000)
        #ey = stats.gamma.pdf(x=x, a=eshape, loc=eloc, scale=escale)
        #print(ey)
        
        g = stats.gamma.pdf(x=x, a=shape, loc=loc, scale=scale)
        #print(g)
        #plt.ylim((0,max(n)))
        
        plt.plot(x, g, color='darkorange', linewidth=3, linestyle='--')
        ax.grid(b=True, which='major', alpha = 0.7, axis = 'y')     
        ax.grid(b=True, which='major', alpha = 0.7, axis = 'x')  
        
        #CHARACTERISTIC AGG ASPECT RATIO OF DISTRIBUTION
        gmax = max(g)  #highest probability
        indmax = np.argmax(g)  #FIRST index where the highest prob occurs
        chgamma = x[indmax] #characteristic aspect ratio of the distribution
        chmode=stats.mode(x)
        dphigam = chgamma - phio    #aggregate aspect ratio - monomer aspect ratio (both crystals the same to start)
        r = np.power((np.power(self.width,2)*self.length),(1./3.)) #of crystal
        #plt.title('Monomer Req=%.3f with shape=%.3f and $\phi_n$=%.3f' %(self.req,shape,chgamma))
        plt.xlabel ("Aggregate Eq. Vol. Radius")
        plt.ylabel ("Frequency")        
       
        if save:
            self.save_fig(filename = filename, ext=ext, verbose = verbose)
        plt.close() 
            
        self.chgamma=chgamma    
        self.phio=phio
        self.shape=shape
        self.dphigam=dphigam
        
    def contact_angle(self, save='False', ext='png', verbose='False', bins=70, normed = True, 
                      facecolor='navy', alpha=1.0,**kwargs):   
        '''
        depth = [ cl.depth for cl in self.clusters ]
        if self.plates:
            height = depth - self.length
            cont_ang = np.arcsin(height/self.width)*180/np.pi
        else:
            height = depth - self.width
            cont_ang = np.arcsin(height/self.length)*180/np.pi
        
        '''
        #print(self.cont_angles)
        
        #CHARACTERISTIC CONTACT ANGLE OF DISTRIBUTION
        fig = plt.figure(figsize=(7,5))
        ax = plt.subplot(111)
        shape, loc, scale = stats.gamma.fit(self.cont_angles)  #loc is mean, scale is stnd dev
        x = np.linspace(min(self.cont_angles),max(self.cont_angles),1000)
        g = stats.gamma.pdf(x=x, a=shape, loc=loc, scale=scale)
        #plot histogram with best fit line
        n = plt.hist(self.cont_angles, bins=bins, normed = normed, facecolor = facecolor, alpha=alpha)
        #plt.ylim((0,max(n)))
        plt.plot(x, g, color='darkorange', linewidth=3, linestyle='--')
        ax.grid(b=True, which='major', alpha = 0.7, axis = 'y')            
        ax.grid(b=True, which='major', alpha = 0.7, axis = 'x')  

        gmax = max(g)  #highest probability
        indmax = np.argmax(g)  #FIRST index where the highest prob occurs
        ch_cont_ang = x[indmax] #characteristic aspect ratio of the distribution        
        plt.title('Monomer aspect ratio=%.3f, Characteristic contact angle=%.3f' %(self.phio,ch_cont_ang))
        plt.xlabel ("Contact Angle")
        plt.ylabel ("Frequency")      
        
        if self.reorient == 'IDL':
            filename = 'contang1_%.3f_IDL' % (self.phio)              
        else:
            filename = 'contang_%.3f' % (self.phio)
   
        if save:
            self.save_fig(filename = filename, ext=ext, verbose = verbose)
        plt.close() 
        
        self.ch_cont_ang = ch_cont_ang
        
    def overlap(self, save='False', ext='png', verbose='False', bins=70, normed = True, 
                      facecolor='navy', alpha=1.0,**kwargs):   
        
        #CHARACTERISTIC OVERLAP OF DISTRIBUTION
        fig = plt.figure(figsize=(7,5))
        ax = plt.subplot(111)
        shape, loc, scale = stats.gamma.fit(self.xoverlap)  #loc is mean, scale is stnd dev
        x = np.linspace(min(self.xoverlap),max(self.xoverlap),1000)
        g = stats.gamma.pdf(x=x, a=shape, loc=loc, scale=scale)
        #plot histogram with best fit line
        n = plt.hist(self.xoverlap, bins=bins, normed = True, facecolor = facecolor, alpha=alpha, **kwargs)
        #plt.ylim((0,max(n)))
        plt.plot(x, g, color='darkorange', linewidth=3, linestyle='--')
        ax.grid(b=True, which='major', alpha = 0.7, axis = 'y') 
        ax.grid(b=True, which='major', alpha = 0.7, axis = 'x')  

        gmax = max(g)  #highest probability
        indmax = np.argmax(g)  #FIRST index where the highest prob occurs
        ch_xoverlap = x[indmax] #characteristic aspect ratio of the distribution        
        plt.title('Monomer aspect ratio=%.3f, Characteristic x overlap=%.3f' %(self.phio,ch_xoverlap))
        plt.xlabel ("X Overlap")
        plt.ylabel ("Frequency")      
        
        if self.reorient == 'IDL':
            filename = 'xoverlap_%.3f_IDL' % (self.phio)              
        else:
            filename = 'xoverlap_%.3f' % (self.phio)
        save=False
        if save:
            self.save_fig(filename = filename, ext=ext, verbose = verbose)
        plt.close() 

        self.ch_xoverlap = ch_xoverlap

        
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
        
        if self.ch_cont_ang is None:
            self.contact_angle(save=save, ext=ext, verbose=verbose)
            
        if self.ch_overlap is None:
            self.overlap(save=save, ext=ext, verbose=verbose)
            
            
        with open(filename, 'a+') as f:
            #print(self.phio, self.shape, self.chgamma, self.dphigam, self.ch_cont_ang, self.ch_xoverlap)
            writing = f.write('%10.4f\t %7.3f\t %.3f\t\t %7.3f\t %7.3f\t %7.3f\t %7.3f\t %7.3f\t\n'
                              %(self.phio, self.shape, self.chgamma, self.dphigam, self.ch_cont_ang, self.ch_xoverlap,
                               self.length, self.width))
            #if verbose:
            #    print("Done")
    

    def which_plot(self, plot_name, savefile, read_file='outfile.dat', save=False, verbose=False, ext='png'):   
        #reads distribution characteristics into arrays
        
        import seaborn as sns
        from matplotlib.ticker import MultipleLocator, FormatStrFormatter, LogLocator, LogFormatter
        import pandas as pd
        
        #self.numaspectratios = 4
        phio = np.zeros(self.numaspectratios)
        shape = np.zeros(self.numaspectratios)
        ch = np.zeros(self.numaspectratios)
        dphigam = np.zeros(self.numaspectratios)
        phip = np.zeros(self.numaspectratios)
        phic = np.zeros(self.numaspectratios)
        chp = np.zeros(self.numaspectratios)
        chc = np.zeros(self.numaspectratios)
        overlap = np.zeros(self.numaspectratios)
        char2 = np.zeros(self.numaspectratios)
        char3 = np.zeros(self.numaspectratios)
        char4 = np.zeros(self.numaspectratios)
        char5 = np.zeros(self.numaspectratios)
        xoverlap = np.zeros(self.numaspectratios)
        char2p = np.zeros(self.numaspectratios)
        char3p = np.zeros(self.numaspectratios)
        char4p = np.zeros(self.numaspectratios)
        char5p = np.zeros(self.numaspectratios)
        char2c= np.zeros(self.numaspectratios)
        char3c = np.zeros(self.numaspectratios)
        char4c = np.zeros(self.numaspectratios)
        char5c = np.zeros(self.numaspectratios)

        
        read_files = ['outfile_char2.dat','outfile_char3.dat',
                      'outfile_char4.dat','outfile_char5.dat']
        for f in read_files:
            fh = open(f, 'r')
            
            #fh = open(read_file, 'r')
            header1 = fh.readline()
            header = fh.readline()
            for i in range(0,self.numaspectratios):            
                data=fh.readline().strip().split('\t')
                phio[i] = float(data[0])        
                shape[i] = float(data[1])
                ch[i] = float(data[2])
                dphigam[i] = float(data[4])
                #cont_ang[i] = float(data[5])
                overlap[i] = float(data[6])


                if f == 'outfile_char2.dat':
                    char2[i] = float(data[2])
                if f == 'outfile_char3.dat':
                    char3[i] = float(data[2])
                if f == 'outfile_char4.dat':
                    char4[i] = float(data[2])
                if f == 'outfile_char5.dat':
                    char5[i] = float(data[2])
                           

        fig = plt.figure(figsize=(7,5))
        ax = plt.subplot(111)
        ax.set_xscale("log")
        ax.set_yscale("log")
        plt.style.use('seaborn-dark')
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.size'] = 14
        plt.rcParams['axes.labelsize'] = 14
        plt.rcParams['axes.labelweight'] = 'normal'
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['xtick.labelsize'] = 12
        plt.rcParams['ytick.labelsize'] = 12
        plt.rcParams['legend.fontsize'] = 11
        
        #plt.rcParams['figure.frameon']= True
        #plt.rcParams['figure.titlesize'] = 14

        ax.grid(b=True, which='minor', alpha = 0.9, axis = 'x')
        ax.grid(b=True, which='minor', alpha = 0.9, axis = 'y')
        ax.grid(b=True, which='major', alpha = 1.0, axis = 'x')            
        ax.grid(b=True, which='major', alpha = 1.0, axis = 'y')    
        
        if self.minor == 'minorxy':
            label='Using minor from x or y'
        else:
            label='Using Depth'
                
        if plot_name == 'char':
            
            wh=np.where(phio<1.0)
            phip[wh] = phio[wh]
            phic[wh] = np.nan
            chp[wh] = ch[wh]
            chc[wh] = np.nan
            wh=np.where(phio>=1.0)
            phic[wh] = phio[wh]
            phip[wh] = np.nan
            chc[wh] = ch[wh]
            chp[wh] = np.nan
                       
            phip = phip[~pd.isnull(phip)]
            chp = chp[~pd.isnull(chp)]
            phic = phic[~pd.isnull(phic)]
            chc = chc[~pd.isnull(chc)]
            #print(phic)
            #print(chc)           

                        
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
           
            coeffs = np.polyfit(phip,chp,4)
            x2 = np.arange(min(phip)-1, max(phip), .01) #use more points for a smoother plot
            y2 = np.polyval(coeffs, x2) #Evaluates the polynomial for each x2 value
            #plt.plot(x2, y2,color='darkorange',linewidth=2.5)  #best fit line
            plt.plot(phip,chp,'o',color='navy',markersize=6,label=label)
            plt.plot(phip, chp, color='navy')

            coeffs = np.polyfit(phic,chc,4)
            x2 = np.arange(min(phic), max(phic)+1, .01) #use more points for a smoother plot
            y2 = np.polyval(coeffs, x2) #Evaluates the polynomial for each x2 value
            #plt.plot(x2, y2,color='darkorange',linewidth=2.5)
            plt.plot(phic,chc,'o',color='navy',markersize=6)
            plt.plot(phic, chc, color='navy')

            
            plt.xlim((.01,100))
            plt.ylim((.1,15))
            ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))
            ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%d"))
            plt.title('Characteristic Aggregate Aspect Ratio for '+str(self.numaspectratios)+' Aspect Ratios - Eq. Vol.')
            plt.xlabel ("Aspect Ratio of Monomers")
            plt.ylabel ("Characteristic Aggregate Aspect Ratio")
            plt.legend(loc='best')                
            
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
            plt.plot(phio,shape,'o',color='navy',markersize=6, label=label)
            plt.title('Shape Parameter for '+str(self.numaspectratios)+' Equiv. Volume Aspect Ratios')
            plt.xlabel("Aspect Ratio of Monomers")
            plt.ylabel("Shape of Aspect Ratio Distribution")
            plt.ylim(min(shape),max(shape)+5) 
            plt.ylim(min(shape),10000) 
            plt.xlim((.009,110))
            ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))
            ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%d"))
            plt.legend(loc='best')
            plt.tight_layout()

            
        if plot_name == 'dphigam':
            '''
            dphi_neg_plate = dphigam[(phio<1.0) & (dphigam<0.0)]           
            phio_neg_plate = phio[(phio<1.0) & (dphigam<0.0)]
            dphi_pos_plate = dphigam[(phio<1.0) & (dphigam>0.0)]
            phio_pos_plate = phio[(phio<1.0) & (dphigam>0.0)]
            dphi_pos_col = dphigam[(phio>1.0) & (dphigam>0.0)]
            phio_pos_col = phio[(phio>1.0) & (dphigam>0.0)]
            dphi_neg_col = dphigam[(phio>1.0) & (dphigam<0.0)]
            phio_neg_col = phio[(phio>1.0) & (dphigam<0.0)]

            #plt.ylim((.01,100))
            ax1 = plt.subplot(211)       
            ax1.grid(b=True, which='major', alpha = 0.9, axis = 'x')            
            ax1.grid(b=True, which='major', alpha = 0.9, axis = 'y')            
            ax1.set_xscale("log")
            ax1.set_yscale("log")
            plt.title('Change in $\phi$ for '+str(self.numaspectratios)+' Equivalent Vol. Aspect Ratios')
            fig.text(0.04, 0.5, 'Aggregate $\phi$ - Monomer $\phi$', 
                     va='center', rotation='vertical', fontsize=14)
            ax1.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.1f"))
            
            #plt.setp(ax1.get_xticklabels(), fontsize=6)

            plt.plot(phio_pos_plate,dphi_pos_plate, color='navy', marker='o', markersize=7, label=label)
            plt.plot(phio_pos_col,dphi_pos_col, color='navy', marker='o', markersize=7)
            plt.axhline(y=0.01, color='black', linestyle='-',linewidth=4.5)
            plt.axvline(x=1.0, color='black', linestyle='-',linewidth=2.5)
            plt.legend(loc='best')
            plt.setp(ax1.get_xticklabels(),visible=False)
            '''

            plt.plot(phio,abs(dphigam),color='navy',label=label)
            plt.plot(phio,abs(dphigam), color='navy',marker='o',markersize=5)

            plt.title('Change in $\phi$ for '+str(self.numaspectratios)+' Equivalent Vol. Aspect Ratios')
            ax.set_xlabel ("Monomer Aspect Ratio")
            ax.set_ylabel('Aggregate $\phi$ - Monomer $\phi$')
            plt.legend(loc='best')
            ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))
            ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.1f"))

            '''
            # share x only            
            ax2 = plt.subplot(212, sharex=ax1)
            ax2.grid(b=True, which='major', alpha = 0.9, axis = 'x')            
            ax2.grid(b=True, which='major', alpha = 0.9, axis = 'y')       
            ax2.set_xscale("log")
            ax2.set_yscale("log")
            ax2.set_xlabel ("Monomer Aspect Ratio")
            ax2.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))
            ax2.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.1f"))

            plt.gca().invert_yaxis()
            plt.plot(phio_neg_plate,abs(dphi_neg_plate), color='navy', marker='o', markersize=7)
            plt.plot(phio_neg_col,abs(dphi_neg_col), color='navy', marker='o', markersize=7)
            plt.axvline(x=1.0, color='black', linestyle='-',linewidth=2.5)

            plt.ylim(max(abs(dphi_neg_col)+100),min(abs(dphi_neg_plate))) 
            plt.subplots_adjust(hspace=0)
            '''
            
        if plot_name == 'cont_ang':

            '''
            coeffs = np.polyfit(phio,cont_ang,2)
            x2 = np.arange(min(phio), max(phio)+1, .01) #use more points for a smoother plot
            y2 = np.polyval(coeffs, x2) #Evaluates the polynomial for each x2 value
            plt.plot(x2, y2,color='darkorange',linewidth=2.5, label=label)           
            plt.plot(phio,cont_ang,'o',color='navy',markersize=6)
         
            
            p = sns.regplot(phio,cont_ang,color='navy',scatter_kws={"s": 80},\
                line_kws={'color':'darkorange'},lowess=True, label=label)
            slopep, interceptp, r_valuep, p_valuep, std_errp = stats.linregress(x=p.get_lines()[0].get_xdata(),\
                            y=p.get_lines()[0].get_ydata())
            '''
            '''
            plt.plot(phio,cont_ang2,color='red',linewidth=1, linestyle='-', marker='o', markersize=3, label='run 1')           
            plt.plot(phio,cont_ang3,color='darkorange',linewidth=1, linestyle='-', marker='o', markersize=3, label='run 2')    
            plt.plot(phio,cont_ang4,color='darkgreen',linewidth=1, linestyle='-', marker='o', markersize=3, label='run 3')      
            plt.plot(phio,cont_ang5,color='navy',linewidth=1, linestyle='-', marker='o', markersize=3, label='run 4') 
            plt.legend(loc='best')
            
            #fill between upper and lower bound of contact angle data at each phio value
            min_cont = np.zeros(self.numaspectratios)
            max_cont = np.zeros(self.numaspectratios)
            mean_cont = np.zeros(self.numaspectratios)
            #for val in np.nditer(cont_ang2,op_dtypes=['float64']):
            for val in range(len(cont_ang2)):                
                min_cont[val] = min(cont_ang2[val], cont_ang3[val], cont_ang4[val], cont_ang5[val])
                max_cont[val] = max(cont_ang2[val], cont_ang3[val], cont_ang4[val], cont_ang5[val])
                mean_cont[val] = np.mean((cont_ang2[val], cont_ang3[val], cont_ang4[val], cont_ang5[val]))
            diff_cont = np.subtract(max_cont,min_cont)
            ytop = np.subtract(max_cont,mean_cont)
            ybot = np.subtract(mean_cont,min_cont)
            #plt.fill_between(phio, min_cont, max_cont,alpha=0.6)
            plt.errorbar(phio, mean_cont, yerr=(ybot, ytop),color='black',linewidth=1, linestyle='-', marker='o', markersize=3)

            #Statistical analysis between each model run (stnd. dev, correlation coeff):
            print("Sum of squared differences (SSD):", np.sum(np.square(max_cont - min_cont)))
                #measures how much variation there is in the observed data
           
            print("Correlation:", np.corrcoef(np.array((max_cont, min_cont)))[0, 1])
            '''
            plt.xlim((.01,100))
            plt.ylim((min(cont_ang),max(cont_ang)+50))
            #ax.set_yticks([5,10,20,30,40, 50, 60])

            ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))
            ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%d"))
            plt.plot(phio,cont_ang,'o',color='navy',markersize=6)
            plt.title('Characteristic Contact Angle for '+str(self.numaspectratios)+' Aspect Ratios - Equiv. Vol.')
            plt.xlabel ("Aspect Ratio of Monomers")
            plt.ylabel ("Characteristic Contact Angle")            
            
            
        if plot_name == 'xoverlap':
        
            plt.xlim((.01,100))
            plt.ylim((min(xoverlap),max(xoverlap)))
            ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))
            ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%d"))
            plt.plot(phio,xoverlap,'o',color='navy',markersize=6)
            plt.title('Characteristic Overlap for '+str(self.numaspectratios)+' Aspect Ratios - Equiv. Vol.')
            plt.xlabel ("Aspect Ratio of Monomers")
            plt.ylabel ("Characteristic Overlap - X Direction [%]")     
            
            
        if plot_name == 'multi_char':
            
            phip = phio[(phio<1.0)]
            phic = phio[(phio>1.0)]
            
            char2c = char2[(phio>1.0)]
            char3c = char3[(phio>1.0)]
            char4c = char4[(phio>1.0)]
            char5c = char5[(phio>1.0)]
            char2p = char2[(phio<1.0)]
            char3p = char3[(phio<1.0)]
            char4p = char4[(phio<1.0)]
            char5p = char5[(phio<1.0)]
            
            plt.plot(phip,char2p,color='red',linewidth=1, linestyle='-', marker='o', markersize=3, label='run 1')           
            plt.plot(phip,char3p,color='darkorange',linewidth=1, linestyle='-', marker='o', markersize=3, label='run 2')    
            plt.plot(phip,char4p,color='darkgreen',linewidth=1, linestyle='-', marker='o', markersize=3, label='run 3')      
            plt.plot(phip,char5p,color='navy',linewidth=1, linestyle='-', marker='o', markersize=3, label='run 4') 
            plt.plot(phic,char2c,color='red',linewidth=1, linestyle='-', marker='o', markersize=3)           
            plt.plot(phic,char3c,color='darkorange',linewidth=1, linestyle='-', marker='o', markersize=3)    
            plt.plot(phic,char4c,color='darkgreen',linewidth=1, linestyle='-', marker='o', markersize=3)      
            plt.plot(phic,char5c,color='navy',linewidth=1, linestyle='-', marker='o', markersize=3) 
            
            plt.legend(loc='best')
            
            #fill between upper and lower bound of contact angle data at each phio value
            
            min_charp = np.zeros(len(phip))
            max_charp = np.zeros(len(phip))
            mean_charp = np.zeros(len(phip))
            min_charc = np.zeros(len(phic))
            max_charc = np.zeros(len(phic))
            mean_charc = np.zeros(len(phic))
            
            #for val in np.nditer(cont_ang2,op_dtypes=['float64']):
            for val in range(len(char2p)):                
                min_charp[val] = min(char2p[val], char3p[val], char4p[val], char5p[val])
                max_charp[val] = max(char2p[val], char3p[val], char4p[val], char5p[val])
                mean_charp[val] = np.mean((char2p[val], char3p[val], char4p[val], char5p[val]))
            for val in range(len(char2c)): 
                min_charc[val] = min(char2c[val], char3c[val], char4c[val], char5c[val])
                max_charc[val] = max(char2c[val], char3c[val], char4c[val], char5c[val])
                mean_charc[val] = np.mean((char2c[val], char3c[val], char4c[val], char5c[val]))
            diff_charp = np.subtract(max_charp,min_charp)
            ytopp = np.subtract(max_charp,mean_charp)
            ybotp = np.subtract(mean_charp,min_charp)
            diff_charc = np.subtract(max_charc,min_charc)
            ytopc = np.subtract(max_charc,mean_charc)
            ybotc = np.subtract(mean_charc,min_charc)
            #plt.fill_between(phio, min_char, max_char,alpha=0.6)
            print(phip)
            print('-------------------------')
            print(mean_charp)
                  
            plt.errorbar(phip, mean_charp, yerr=(ybotp, ytopp),color='black',linewidth=1, linestyle='-', marker='o', markersize=3)
            plt.errorbar(phic, mean_charc, yerr=(ybotc, ytopc),color='black',linewidth=1, linestyle='-', marker='o', markersize=3)

            #Statistical analysis between each model run (stnd. dev, correlation coeff):
            print("Sum of squared differences plates (SSD):", np.sum(np.square(max_charp - min_charp)))
                #measures how much variation there is in the observed data
           
            print("Correlation plates:", np.corrcoef(np.array((max_charp, min_charp)))[0, 1])
            print("Correlation columns:", np.corrcoef(np.array((max_charc, min_charc)))[0, 1])

            plt.xlim((.01,100))
            #plt.ylim((min(overlap),10))
            ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))
            ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))
            plt.title('Characteristic Aggregate Aspect Ratio from Gamma Distribution')               
            plt.xlabel ("Aspect Ratio of Monomers")
            plt.ylabel ("Characteristic Aggregate Aspect Ratio")
                       
            
        if save:

            if self.reorient == 'IDL':
                filename = savefile+'_IDL'         
            else:
                filename = savefile                               

            self.save_fig(filename=filename, ext=ext, verbose = verbose)

        else:
            plt.close()
            
            
'''                             
    def write_dataframe(self, filename='outfile.csv', verbose=False):
        #writes characteristic values after all clusters have been plotted in a distribution
        #each index is a monomer aspect ratio 
        #length of outfile is the number of different aspect ratios looped over
        #index = np.array(range(len(self.clusters))) + 1  #i.e. 300 clusters - within each aspect ratio
        
        import pandas as pd
        index = np.array(range(len(self.numaspectratios)) + 1
        df = pd.DataFrame(index = index, columns=['cl', 'cw', 'chgamma', 'dphigam', 'shape'])
        df['cl'] = self.length
        df['cw'] = self.width
        #df['major_axis'] = self.major_axis['z']
        #df['minor_axis'] = self.major_axis['y']
        df['chgamma'] = self.chgamma
        df['dphigam'] = self.dphigam
        df['shape'] = self.shape
        df = df.round(3)
        if verbose is True:
            print('Writing to ' +filename+ '...')
        df.to_csv(filename, header=False, mode = 'a', ignore_index=True)
        if verbose:
            print("Done")
'''