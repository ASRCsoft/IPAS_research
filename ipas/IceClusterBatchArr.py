import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import warnings
import pandas as pd
from ipas import IceCluster as clus

class IceClusterBatch():
    """A collection of IceCluster objects."""

    def __init__(self, ncrystals, clusters, length, width, r, numaspectratios, reorient, ovrlps, Ss, cplxs, 
                 phi, phi_2d, major_axis, depth, req, xrot, yrot, dd, minor, plates=None): 
        
        self.ncrystals = ncrystals
        self.clusters = clusters
        self.length = length
        self.width = width
        if plates is None:
            self.plates = width > length
        else:
            self.plates = plates # are they plates or columns?
        self.phio = self.length/self.width 
        self.r = r
        self.numaspectratios = numaspectratios                        
        self.reorient = reorient
        self.ovrlps = ovrlps
        self.Ss = Ss
        self.cplxs = cplxs
        self.phi = phi
        self.phi_2d = phi_2d
        self.major_axis = major_axis
        self.depth = depth
        self.req = req
        self.minor_axis = {}        
        self.xrot = xrot
        self.yrot = yrot 
        self.dd = dd
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
             
  
    def get_characteristics(self, var, minor, ch_dist, bins=70, filename=None, ext='png',save = False, verbose=False):
        """Plot a histogram of cluster aspect ratios, sending extra arguments
        to `matplotlib.pyplot.hist`.
        """
        save = False
        fig = plt.figure(figsize=(18,9))
        ax = plt.subplot(131)

       
        self.ch = np.zeros(np.size(self.ovrlps[:][0])) 
        #self.ch_ovrlp = np.zeros(np.size(self.ovrlps[:][0]))
        self.shape = np.zeros(np.size(self.ovrlps[:][0]))
        self.dphigam = np.zeros(np.size(self.ovrlps[:][0]))
        self.poserr = np.zeros(np.size(self.ovrlps[:][0]))
        self.negerr = np.zeros(np.size(self.ovrlps[:][0]))
        self.min_data = np.zeros(np.size(self.ovrlps[:][0]))
        self.max_data = np.zeros(np.size(self.ovrlps[:][0]))
        self.mean = np.zeros(np.size(self.ovrlps[:][0]))
        for l in range(np.size(self.ovrlps[:][0])): 
            
            if var == 'phi':
                filename = '%.2f_%.2f_%.2f' % (self.phio, self.length, self.width)
                xlabel = "Aggregate Aspect Ratio"
                data = self.phi[:,l]  

            if var == 'phi2D':
                filename = 'phi2D_%.2f' % (self.phio)
                xlabel = "Aggregate Aspect Ratio (2D Projection from Ellipse)"  
                data = self.phi_2d[:,l]  
                
            if var == 'req':
                filename = 'req_%.3f' % (self.phio)           
                xlabel = "Aggregate Eq. Vol. Radius"  
                data = self.req[:,l]  

            if var == 'overlap':
                filename = 'overlap_%.3f' % (self.phio)
                xlabel="Overlap"
                data = self.ovrlps[:,l]  

            if var == 'S':
                filename = 'S_%.3f' % (self.phio)
                xlabel = "S Parameter"    
                data = self.Ss[:,l]  
                ch_S, self.shape = self.best_fit_distribution(data, ch_dist, xlabel=xlabel, bins=bins, ax=ax)
                if ch_S <= 0.05:
                    save=True
                    print('S catch filename = ',filename)

            if var == 'complexity':
                filename = 'cplx_%.3f' % (self.phio)
                xlabel = "Complexity"
                data = self.cplxs[:,l]  

            if var == 'depth':
                filename = 'depth_%.3f' % (self.phio)
                xlabel = "Depth"
                data = self.depth[:,l]                   

            if var == 'major_axis':
                filename = 'mjrax_%.3f' % (self.phio)
                xlabel = "Major Axis"
                data = self.major_axis[:,l]  
                
            if var == 'density_change':
                filename = 'dd_%.3f' % (self.phio)
                xlabel = "Density Change"
                data = self.dd[:,l]  
            
            if ch_dist == 'gamma':
                self.ch[l], self.shape = self.best_fit_distribution(data, ch_dist, xlabel=xlabel, bins=bins, ax=ax)

            else:
                self.ch[l] = self.best_fit_distribution(data, ch_dist, xlabel=xlabel, bins=bins, ax=ax)               

            self.poserr[l], self.negerr[l], self.min_data[l], self.max_data[l], self.mean[l] = \
                    self.calculate_error(data, self.ch[l])

            if var == 'phi':
                self.dphigam[l] = np.log(self.ch[l]) - np.log(self.phio)
        


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
        if np.isinf(data).any():
            print('inf True')
        y, x = np.histogram(data, density=True)  
        #print('x hist',x)
        xx = (x + np.roll(x, -1))[:-1] / 2.0
    
        # Distributions to check
      
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
        self.minor = 'depth'
        print('saving file..')                  
        if self.minor == 'minorxy':                   
            path=('/Users/vprzybylo/Desktop/icefiles/agg_model/agg_notes/graphics/python/'+
                      str(self.ncrystals)+'crystals/'+str(len(self.clusters))+'xtals_hist/minorxy/')
        else:
            path=('/Users/vprzybylo/Desktop/icefiles/agg_model/agg_notes/graphics/python/'+
                      str(self.ncrystals)+'crystals/'+str(len(self.clusters))+'xtals_hist/depth/')
        print(path)
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