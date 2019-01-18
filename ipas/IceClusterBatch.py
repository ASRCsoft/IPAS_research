import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import warnings
import pandas as pd
from ipas import IceCluster as clus

class IceClusterBatch():
    """A collection of IceCluster objects."""
    
    def __init__(self, clusters, length, width, r, numaspectratios, reorient, ovrlps, Ss, 
                 xrot, yrot, plates=None): 
        
        self.length = length
        self.width = width
        self.phio = self.length/self.width 
        if plates is None:
            self.plates = width > length
        else:
            self.plates = plates # are they plates or columns?
        self.r = r
        self.numaspectratios = numaspectratios
        self.clusters = clusters
        self.major_axis = {}
        self.minor_axis = {}
        self.reorient = reorient
        self.ovrlps = ovrlps
        self.Ss = Ss
        self.ch = 0.0        
        self.ch_ovrlp = np.zeros(np.shape(self.ovrlps[0,:]))
        self.ch_S = np.zeros(np.shape(self.Ss[0,:]))
        self.ncrystals = np.shape(self.ovrlps[0,:])
        self.chphi = 0.0
        self.shape = 0.0
        self.dphigam = 0.0
        self.poserr = 0.0
        self.negerr = 0.0
        self.min_data = 0.0
        self.max_data = 0.0
        self.mean = 0.0
        self.xrot = xrot
        self.yrot = yrot
    
        #self.tiltdiffsx = tiltdiffsx
        #self.tiltdiffsy = tiltdiffsy 

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
        to the 2D cluster projections from the z-axis perspective.

        """             
        if self.plates:
            # if the crystals are plates do the plate version
            self.ratios = [ cl.aspect_ratio(method='plate', minor=minor) for cl in self.clusters ]            
            self.major_axis['z'] = [ cl.major_axis['z']/2 for cl in self.clusters ] #this has to be after ratios
            self.depth = [ cl.depth/2 for cl in self.clusters ]  #this has to be after ratios -- defined in aspect ratio
            self.req = np.power((np.power(self.major_axis['z'],2)*self.depth),(1./3.))     #major axis from fit ellipse
            
        else:
         
            self.ratios = [ cl.aspect_ratio(method='column', minor=minor) for cl in self.clusters ]
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
        
        self.calc_aspect_ratios(minor)
        self.ratios_2D = [ cl.aspect_ratio_2D() for cl in self.clusters ]            
        self.cplxs = [ cl.complexity() for cl in self.clusters ]  
        
        if var == 'phi':
            filename = '%.2f_%.2f_%.2f' % (self.phio, self.length, self.width)
            data = self.ratios
            xlabel = "Aggregate Aspect Ratio"
            if ch_dist == 'gamma':
                self.chphi, self.shape = self.best_fit_distribution(data, ch_dist, xlabel=xlabel, bins=bins, ax=ax)
            else:
                self.chphi = self.best_fit_distribution(data, ch_dist, xlabel=xlabel, bins=bins, ax=ax)
                
            self.poserr, self.negerr, self.min_data, self.max_data, self.mean = self.calculate_error(data, self.chphi)
       
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
            xlabel="Overlap"
            for l in range(self.ncrystals[0]):    
                data = self.ovrlps[:,l]  
                if ch_dist == 'gamma':                
                    self.ch_ovrlp[l], self.shape = self.best_fit_distribution(data, ch_dist, xlabel=xlabel, bins=bins, ax=ax)
                    print('ch', self.best_fit_distribution(data, ch_dist, xlabel=xlabel, bins=bins, ax=ax))
                else:
                    self.ch_ovrlp[l] = self.best_fit_distribution(data, ch_dist, xlabel=xlabel, bins=bins, ax=ax)               
        
        if var == 'S':
            filename = 'S_%.3f' % (self.phio)
            xlabel = "S Parameter"    
            for l in range(self.ncrystals[0]):                 
                data = self.Ss[:,l]  
                if ch_dist == 'gamma':
                    self.ch_S[l], self.shape = self.best_fit_distribution(data, ch_dist, xlabel=xlabel, bins=bins, ax=ax)
                    if self.ch_S[l] <= 0.05:
                        save=True
                        print('S catch filename = 'filename)
                else:
                    self.ch_S[l] = self.best_fit_distribution(data, ch_dist, xlabel=xlabel, bins=bins, ax=ax)               
            
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
            
        if var != 'overlap' and var != 'S':    
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
        
        shape, loc, scale = st.gamma.fit(data)  #loc is mean, scale is stnd dev      
        
        #print('from fit', shape,loc,scale)
        #eshape, eloc, escale = self.calculateGammaParams(data)
        #print('estimates', eshape, eloc, escale)        
        #x = np.linspace(st.gamma.pdf(min(data), shape),st.gamma.pdf(max(data),shape),100)
        x = np.linspace(min(data),max(data),1000)
        #ey = st.gamma.pdf(x=x, a=eshape, loc=eloc, scale=escale)
        #print(ey)
        
        g = st.gamma.pdf(x=x, a=shape, loc=loc, scale=scale)
        #print(g)
        plt.ylim((0,max(n)))
        
        plt.plot(x, g, color='darkorange', linewidth=3, linestyle='--')
        ax.grid(b=True, which='major', alpha = 0.7, axis = 'y')     
        ax.grid(b=True, which='major', alpha = 0.7, axis = 'x')  
        
        #CHARACTERISTIC AGG ASPECT RATIO OF DISTRIBUTION
        gmax = max(g)  #highest probability
        indmax = np.argmax(g)  #FIRST index where the highest prob occurs
        chgphi = x[indmax] #characteristic aspect ratio of the distribution
        chmode=st.mode(x)
        dphigam = chphi - self.phio    #aggregate aspect ratio - monomer aspect ratio (both crystals the same to start)
        r = np.power((np.power(self.width,2)*self.length),(1./3.)) #of crystal
        plt.title('Monomer Req=%.3f with shape=%.3f and $\phi_n$=%.3f' %(r,shape,chphi))
        plt.xlabel ("Aggregate Eq. Vol. Radius")
        '''      

        self.dphigam = np.log(self.chphi) - np.log(self.phio)
        
        if save:
            self.save_fig(filename = filename, ext=ext, verbose = verbose)
        plt.close() 

        
    def best_fit_distribution(self, data, ch_dist, bins, xlabel, ax=None, normed = True, facecolor='navy', 
                              alpha=1.0,**kwargs):
        
        """Model data by finding best fit distribution to data"""
        # Get histogram of original data
        
        data = np.array(data)
        data[np.isinf(data)] = min(data)
        data[np.isnan(data)] = min(data)
        if np.isinf(data).any():
            print('inf True')
        y, x = np.histogram(data, density=True)          
        xx = (x + np.roll(x, -1))[:-1] / 2.0
    
        if ch_dist == 'best':
            # Distributions to check
            DISTRIBUTIONS =[st.alpha,st.anglit,st.arcsine,st.beta,st.betaprime,st.bradford,
                            st.burr,st.cauchy,st.chi,st.chi2,st.cosine,st.dgamma,st.dweibull,
                            st.erlang,st.expon,st.exponnorm,st.exponweib,st.exponpow,st.f,st.fatiguelife,
                            st.fisk,st.foldcauchy,st.foldnorm,st.frechet_r,st.frechet_l,st.genlogistic,
                            st.genpareto,st.gennorm,st.genexpon,st.genextreme,st.gausshyper,st.gamma,
                            st.gengamma,st.genhalflogistic,st.gilbrat,st.gompertz,st.gumbel_r,st.gumbel_l,st.halfcauchy,
                            st.halflogistic,st.halfnorm,st.halfgennorm,st.hypsecant,st.invgamma,st.invgauss,
                            st.invweibull,st.johnsonsb,st.johnsonsu,st.ksone,st.kstwobign,st.laplace,st.levy,
                            st.levy_l,st.levy_stable,st.logistic,st.loggamma,st.loglaplace,st.lognorm,
                            st.lomax,st.maxwell,st.mielke,st.nakagami,st.ncx2,st.ncf,st.nct,st.norm, st.pareto,
                            st.pearson3, st.powerlaw,st.powerlognorm,st.powernorm,st.rdist,st.reciprocal,st.rayleigh,
                            st.rice,st.recipinvgauss,st.semicircular,st.t,st.triang, st.truncexpon, 
                            st.truncnorm,st.tukeylambda,st.uniform,st.vonmises,st.vonmises_line,st.wald,
                            st.weibull_min,st.weibull_max,st.wrapcauchy]
        else:
            DISTRIBUTIONS = [st.gamma]

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
        best_dist = getattr(st, best_distribution.name)
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
        plt.show()
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
                      str(len(self.clusters))+'crystals/'+str(len(self.clusters))+'xtals_hist/minorxy/')

        else:
            path=('/Users/vprzybylo/Desktop/icefiles/agg_model/agg_notes/graphics/python/'+
                      str(len(self.clusters))+'crystals/'+str(len(self.clusters))+'xtals_hist/depth/')

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