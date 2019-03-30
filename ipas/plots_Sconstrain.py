"""Plots characteristic variables that have been looped over for every aspect ratio"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib 
import seaborn as sns
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, LogLocator, LogFormatter, ScalarFormatter
import pandas as pd
import pylab as pl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.ticker as ticker
from matplotlib.ticker import MaxNLocator
#from ipas import lab_copy as lab

class Make_Plots():
    
    def __init__(self, phio, widtharr, lengtharr, chreq, chphi, chphi2D, ch_ovrlp, ch_S, ovrlp, S, 
                           ch_majorax, ch_depth, dphigam, poserr_phi, negerr_phi, poserr_phi2D, negerr_phi2D, 
                           poserr_mjrax, negerr_mjrax, poserr_req, negerr_req, poserr_depth, negerr_depth, poserr_ovrlp, 
                           negerr_ovrlp, poserr_S, negerr_S, poserr_cplx, negerr_cplx, min_phi, max_phi, min_phi2D, 
                           max_phi2D, min_mjrax, max_mjrax, min_depth,max_depth, min_req, max_req, mean_phi, mean_phi2D, 
                           mean_mjrax, mean_depth, mean_req, mean_ovrlp, mean_S, mean_cplx, ch_cplx, xrot, yrot):
        
        self.phio = np.array(phio)
        self.width = np.array(widtharr)
        self.length = np.array(lengtharr)
        self.chreq = np.array(chreq)
        self.chphi = np.array(chphi)
        self.chphi2D = np.array(chphi2D)
        self.ch_cplx = np.array(ch_cplx)
        self.ch_ovrlp = np.array(ch_ovrlp)
        self.ch_S = np.array(ch_S)
        self.ovrlp = np.array(ovrlp)
        self.S = np.array(S)
        self.ch_majorax = np.array(ch_majorax)
        self.ch_depth = np.array(ch_depth)    
        self.dphigam = np.array(dphigam)
        self.poserr_phi = np.array(poserr_phi)
        self.negerr_phi = np.array(negerr_phi)
        self.poserr_phi2D = np.array(poserr_phi2D)
        self.negerr_phi2D = np.array(negerr_phi2D)
        self.poserr_mjrax = np.array(poserr_mjrax)
        self.negerr_mjrax = np.array(negerr_mjrax)
        self.poserr_req = np.array(poserr_req)
        self.negerr_req = np.array(negerr_req)
        self.poserr_depth = np.array(poserr_depth)
        self.negerr_depth = np.array(negerr_depth)
        self.poserr_ovrlp = np.array(poserr_ovrlp)
        self.negerr_ovrlp = np.array(negerr_ovrlp)
        self.poserr_S = np.array(poserr_S)
        self.negerr_S = np.array(negerr_S)
        self.poserr_cplx = np.array(poserr_cplx)
        self.negerr_cplx = np.array(negerr_cplx)
        self.phip = np.zeros(len(self.phio))
        self.phic = np.zeros(len(self.phio))
        self.chp = np.zeros(len(self.phio))
        self.chc = np.zeros(len(self.phio))
        self.lengthp = np.zeros(len(self.phio))
        self.lengthc = np.zeros(len(self.phio))
        self.widthp = np.zeros(len(self.phio))
        self.widthc = np.zeros(len(self.phio))
        self.min_phi = np.array(min_phi)
        self.max_phi = np.array(max_phi)
        self.min_phi2D = np.array(min_phi2D)
        self.max_phi2D = np.array(max_phi2D)
        self.min_mjrax = np.array(min_mjrax)
        self.max_mjrax = np.array(max_mjrax)
        self.min_depth = np.array(min_depth)
        self.max_depth = np.array(max_depth)
        self.min_req = np.array(min_req)
        self.max_req = np.array(max_req)
        self.mean_phi = np.array(mean_phi) 
        self.mean_phi2D = np.array(mean_phi2D) 
        self.mean_mjrax = np.array(mean_mjrax) 
        self.mean_depth = np.array(mean_depth)
        self.mean_req = np.array(mean_req)
        self.mean_cplx = np.array(mean_cplx)
        self.mean_ovrlp = np.array(mean_ovrlp)
        self.mean_S = np.array(mean_S)
        self.meanphip = np.zeros(len(self.phio))
        self.meanphic = np.zeros(len(self.phio))
        self.xrot = np.array(xrot)
        self.yrot = np.array(yrot)
        #self.tiltdiffsx = np.array(tiltdiffsx)
        #self.tiltdiffsy = np.array(tiltdiffsy)
        

    def which_plot(self,nclusters,plot_name, ch_dist, savefile, read_file='outfile.dat', save=False, verbose=False, ext='eps'):   
        #makes plots based on plot_name passed in
       
        #self.numaspectratios = 26

        fig = plt.figure(figsize=(10,7),frameon=False)
        ax = plt.subplot(111)
        
        #ax.set_xscale("log")
        ax.grid(b=True, which='minor', alpha = 0.9, axis = 'x')
        ax.grid(b=True, which='minor', alpha = 0.9, axis = 'y')
        ax.grid(b=True, which='major', alpha = 1.0, axis = 'x')            
        ax.grid(b=True, which='major', alpha = 1.0, axis = 'y')    

        plt.style.use('seaborn-dark')  #gray background
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.size'] = 18
        plt.rcParams['axes.labelsize'] = 18
        plt.rcParams['axes.labelweight'] = 'normal'
        plt.rcParams['axes.titlesize'] = 18
        plt.rcParams['xtick.labelsize'] = 18
        plt.rcParams['ytick.labelsize'] = 18
        plt.rcParams['legend.fontsize'] = 18

        self.phip = self.phio[(self.phio<=1.0)]
        self.phic = self.phio[(self.phio>=1.0)]      
        self.chp = self.chphi[(self.phio<=1.0)]
        self.chc = self.chphi[(self.phio>=1.0)]
        self.chp2D = self.chphi2D[(self.phio<=1.0)]
        self.chc2D = self.chphi2D[(self.phio>=1.0)]
        self.lengthp = self.length[(self.phio<=1.0)]
        self.widthp = self.width[(self.phio<=1.0)]
        self.lengthc = self.length[(self.phio>=1.0)]
        self.widthc = self.width[(self.phio>=1.0)]
        self.meanphip = self.mean_phi[(self.phio<1.0)]
        self.meanphic = self.mean_phi[(self.phio>=1.0)]
        self.meanphip2D = self.mean_phi2D[(self.phio<1.0)]
        self.meanphic2D = self.mean_phi2D[(self.phio>=1.0)]
        poserr_phip = self.poserr_phi[(self.phio<1.0)]
        negerr_phip = self.negerr_phi[(self.phio<1.0)]
        poserr_phic = self.poserr_phi[(self.phio>=1.0)]
        negerr_phic = self.negerr_phi[(self.phio>=1.0)]
        poserr_phip2D = self.poserr_phi2D[(self.phio<1.0)]
        negerr_phip2D = self.negerr_phi2D[(self.phio<1.0)]
        poserr_phic2D = self.poserr_phi2D[(self.phio>=1.0)]
        negerr_phic2D = self.negerr_phi2D[(self.phio>=1.0)]
        
        phiolog = np.log10(self.phio)
        phiplog = np.log10(self.phip)
        phiclog = np.log10(self.phic)
        
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
            plt.plot(self.phip, self.chp,'o',color='navy',markersize=6)
            plt.plot(self.phip, self.chp, color='navy')
            
            #coeffs = np.polyfit(phic,chc,4)
            #x2 = np.arange(min(phic), max(phic)+1, .01) #use more points for a smoother plot
            #y2 = np.polyval(coeffs, x2) #Evaluates the polynomial for each x2 value
            #plt.plot(x2, y2,color='darkorange',linewidth=2.5)
            plt.plot(self.phic, self.chc,'o',color='navy',markersize=6)
            plt.plot(self.phic, self.chc, color='navy', label = 'characteristic')
            #plt.plot(self.phio, self.mean_phi,'o',color='black',markersize=6)
      
            plt.plot(self.phip, self.meanphip, color='black')
            plt.plot(self.phic, self.meanphic, color='black', label = 'mean')
            ax.fill_between(self.phip, poserr_phip, negerr_phip, facecolor='darkorange', alpha=0.3)
            ax.fill_between(self.phic, poserr_phic, negerr_phic, facecolor='darkorange', alpha=0.3)
            
            
            #Line of no change in aspect ratio
            phiox=np.logspace(-2, 2., num=20, endpoint=True, base=10.0, dtype=None)#just columns (0,2); plates (-2,0)
            plt.ylim((.01,100))
            plt.xlim((.01,100))
            #plt.plot(phiox, phiox, color='navy')       
            
            ax.set_yscale("log")
            ax.set_xscale("log")

            #plt.xlim((2.0,100))
            #plt.ylim((min(ch),max(ch)))
            ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))
            ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.3f"))
            #ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())            

            #plt.title('Characteristic Aggregate Aspect Ratio for '+str(self.numaspectratios)+' Aspect Ratios')
            #plt.xlabel ("Equivalent Volume Aspect Ratio of Monomers")
            plt.title('Aggregate Aspect Ratio from %s Distribution' %dist_title)
            plt.ylabel ("Aggregate Aspect Ratio")
            plt.legend(loc='best')        
            
        if plot_name == '2D_phi':
            
            #ax.set_yscale("log")
            ax.set_xscale("log")
            ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))
            ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.3f"))
            
            plt.plot(self.phip, self.chp2D,'o',color='navy',markersize=6)
            plt.plot(self.phip, self.chp2D, color='navy', label = 'characteristic')          
            #plt.plot(self.phic, self.chc2D,'o',color='navy',markersize=6)
            #plt.plot(self.phic, self.chc2D, color='navy')
            #plt.plot(self.phio, self.mean_phi,'o',color='black',markersize=6)
            #print(np.mean(self.meanphip2D))
            plt.plot(self.phip, self.meanphip2D, color='black', label = 'mean')
            #plt.plot(self.phic, self.meanphic2D, color='black')
            ax.fill_between(self.phip, poserr_phip2D, negerr_phip2D, facecolor='darkorange', alpha=0.3)
            #ax.fill_between(self.phic, poserr_phic2D, negerr_phic2D, facecolor='darkorange', alpha=0.3)
            plt.title('2D Projected Aggregate Aspect Ratio from %s Distribution' %dist_title)
            plt.ylabel ("Aggregate Aspect Ratio")
            plt.legend(loc='best')
            print('mean of characteristic=', np.mean(self.chp2D))
            print('mean of all means=', np.mean(self.meanphip2D))
            
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

                titlearr = ['$\phi$ ', 'Equiv. Vol. ', 'Depth ', 'Major Axis ']
                filearr = ['phi', 'Req', 'Depth', 'MjrAx']

                shapearr = [[] for i in range(len(titlearr))]
                shapearr[0]=phishape
                shapearr[1]=reqshape 
                shapearr[2]=depthshape
                shapearr[3]=majoraxshape

                for i in range(0,len(titlearr)):            

                    fig = plt.figure(figsize=(8,6))
                    ax = plt.subplot(111)
                    ax.set_xscale("log")
                    #ax.set_yscale("log")
                    ax.grid(b=True, which='minor', alpha = 0.9, axis = 'x')
                    ax.grid(b=True, which='minor', alpha = 0.9, axis = 'y')
                    ax.grid(b=True, which='major', alpha = 1.0, axis = 'x')            
                    ax.grid(b=True, which='major', alpha = 1.0, axis = 'y')    

                    plt.plot(self.phio,shapearr[i],'o',color='navy',markersize=6)      
                    plt.title('%s Shape Parameter from %s Distribution' %(titlearr[i], dist_title))
                    #plt.title('Shape Parameter for '+str(self.numaspectratios)+' Aspect Ratios')           
                    plt.ylabel("Shape of %s Distribution" %titlearr[i])
                    plt.xlabel ("Equivalent Volume Aspect Ratio of Monomer")
                    ax.set_yscale("log")
                    ax.set_xscale("log")
                    plt.ylim(min(shapearr[i]),max(shapearr[i])+20) 
                    #plt.ylim(min(shape),10000) 
                    plt.xlim((.009,110))
                    ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))
                    #ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.3f"))
                    

                    if save:                      
                        savefile = '%s_shape' %filearr[i]  
                                               
                        filename = savefile     

                        self.save_fig(nclusters, filename=filename, ext=ext, verbose = verbose)



            else:
                print('Shape parameter plot is only applicable to the gamma distribution')
        

        if plot_name == 'dphigamquad':

            #Quadrant Plot
            
            dphi_neg_plate = self.dphigam[(self.phio<1.0) & (self.dphigam<0.0)]           
            phio_neg_plate = self.phio[(self.phio<1.0) & (self.dphigam<0.0)]
            dphi_pos_plate = self.dphigam[(self.phio<1.0) & (self.dphigam>0.0)]
            phio_pos_plate = self.phio[(self.phio<1.0) & (self.dphigam>0.0)]
            dphi_pos_col = self.dphigam[(self.phio>1.0) & (self.dphigam>0.0)]
            phio_pos_col = self.phio[(self.phio>1.0) & (self.dphigam>0.0)]
            dphi_neg_col = self.dphigam[(self.phio>1.0) & (self.dphigam<0.0)]
            phio_neg_col = self.phio[(self.phio>1.0) & (self.dphigam<0.0)]
            

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
            #maxc = max(dphi_pos_col)
            #maxp = max(dphi_pos_plate)
            #plt.ylim(.011, max(maxc, maxp)+.1)
            plt.setp(ax1.get_xticklabels(), fontsize=6)
            #ax1.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))
            '''
            plt.plot(phio_pos_plate,dphi_pos_plate, color='navy', marker='o', markersize=7)
            plt.plot(phio_pos_col,dphi_pos_col, color='navy', marker='o', markersize=7)
            plt.axhline(y=0.0112, color='black', linestyle='-',linewidth=2.5)
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
            '''
            
            ax1.plot(phio_pos_plate,dphi_pos_plate, color='navy', marker='o', markersize=7)
            ax1.plot(phio_pos_col,dphi_pos_col, color='navy', marker='o', markersize=7)
            plt.axhline(y=0.001, color='black', linestyle='-',linewidth=2.5)
            ax1.axvline(x=1.0, color='black', linestyle='-',linewidth=2.5)

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
            ax2.axvline(x=1.0, color='black', linestyle='-',linewidth=2.5)

            ax2.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))
            #ax2.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))

            #ax1.invert_yaxis()
            #ax2.invert_yaxis()
            ax2.plot(phio_neg_plate,abs(dphi_neg_plate), color='navy', marker='o', markersize=7)
            ax2.plot(phio_neg_col,abs(dphi_neg_col), color='navy', marker='o', markersize=7)             
            ax1.set_ylim(.001,1000) 
            ax2.set_ylim(1000,.001)
            plt.subplots_adjust(hspace=0)
        
        if plot_name == 'dphigamW':
            #W Plot
            plt.plot(self.phio,abs(self.dphigam),color='navy')
            plt.plot(self.phio,abs(self.dphigam), color='navy',marker='o',markersize=5)
            ax.set_yscale("log")
            #plt.title('Change in $\phi$ for '+str(self.numaspectratios)+' Aspect Ratios')
            plt.title('Change in eq. vol. $\phi$ from Characteristic of %s Distribution' %dist_title)
            ax.set_xscale("log")
            ax.set_xlabel ("Equivalent Volume Monomer Aspect Ratio")
            ax.set_ylabel('Aggregate $\phi$ - Monomer $\phi$')
            #plt.legend(loc='best')
            ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))
            ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.3f"))
            
        if plot_name == 'overlap':
            
            '''           
            self.stddevpos_ovrlp = self.poserr_ovrlp - self.mean_ovrlp
            self.stddevneg_ovrlp = self.negerr_ovrlp - self.mean_ovrlp
            cv_pos = self.stddevpos_ovrlp/self.mean_ovrlp
            cv_neg = self.stddevneg_ovrlp/self.mean_ovrlp
            
            self.ch_ovrlpp = self.ch_ovrlp[(self.phio<1.0)]
            poly = pl.polyfit(phiplog, self.ch_ovrlpp, 2)  #2 is the degree of polynomial to fit                   
            x2 = np.arange(min(phiplog), max(phiplog), .01) #use more points for a smoother plot
            y2 = np.polyval(poly, x2) #Evaluates the polynomial for each x2 value
            plt.plot(x2, y2, color='darkorange', linewidth=5)  #best fit line

            self.ch_ovrlpc = self.ch_ovrlp[(self.phio>=1.0)]
            poly = pl.polyfit(phiclog, self.ch_ovrlpc, 2)  #2 is the degree of polynomial to fit                   
            x2 = np.arange(min(phiclog), max(phiclog), .01) #use more points for a smoother plot
            y2 = np.polyval(poly, x2) #Evaluates the polynomial for each x2 value
            plt.plot(x2, y2, color='darkorange', linewidth=5)  #best fit line
            '''
            
            xstar = np.arange(0,2)
            plt.plot(xstar,self.ch_ovrlp[:,0],'*',color='blue',markersize=6, label='characteristic')
            #plt.plot(phiolog,self.ch_ovrlp,color='navy')
           
            #ax.set_xscale("log")
            #plt.xlim((.01,100))
            #plt.ylim((min(overlap),10))

            ovrlps = [np.array(i) for i in self.ovrlp[:,:,0]]
            print(self.ch_ovrlp)
            phio_formatted = [ '%.2f' % elem for elem in self.phio ] 

            #for i in range(len(phio_formatted)):
            #    print(i)
            #    if i != 0 and i != 9 and i != 19 and i != 29 and i != 39 and i != 49:
            #        print('here')
            #        phio_formatted[i]=''
                    
            plt.xticks(rotation=45)            
            plt.boxplot(ovrlps)
            ax.xaxis.set_ticklabels(phio_formatted)
            
            #ax.xaxis.set_major_locator(ticker.MultipleLocator(2))

            #ax.fill_between(self.phio, self.poserr_ovrlp, self.negerr_ovrlp, facecolor='darkorange', alpha=0.5)   
            #ax.fill_between(self.phio,  self.mean_ovrlp+cv_pos,\
            #                self.mean_ovrlp+cv_neg, facecolor='darkorange', alpha=0.5)  
            plt.legend(loc='best')
            
            plt.title('Overlap from the %s Distribution' %dist_title)
            plt.ylabel ("Overlap [%]")  
            #plt.xlim((2.0,100))
            
        if plot_name == 'overlap_contour':
            x = np.arange(len(self.phio))
            y = np.arange(len(self.ch_ovrlp[0,:]))
            ylab=np.arange(1, len(self.phio)+1)
            #ax = Axes3D(fig)

            X_grid, Y_grid = np.meshgrid(x,y)
            Z_grid = self.ch_ovrlp[X_grid,Y_grid]
            xlab = self.phio[X_grid]

            im = plt.contourf(np.log10(xlab), ylab[Y_grid], Z_grid, cmap = 'hot')
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            
            plt.title('Characteristic Overlap from the %s Distribution' %dist_title)
            ax.yaxis.label.set_size(16)
            ax.xaxis.label.set_size(16)
            #ax.zaxis.label.set_size(16)
            plt.tick_params(axis='both', which='major', labelsize=14)
            #ax.set_zlabel('[%]')
            ax.xaxis.labelpad = 10
            ax.yaxis.labelpad = 10
            ax.set_xlabel('Eq. Volume Monomer Phi')
            ax.set_ylabel('# of Monomers in Aggregate') 
            
            CBI = plt.colorbar(im, orientation='horizontal', shrink=0.8)


            l, b, w, h = plt.gca().get_position().bounds
            ll, bb, ww, hh = CBI.ax.get_position().bounds
            #CBI.ax.set_position([ll, b + 0.1*h, ww, h*0.8])

            
        if plot_name == 'S_contour':
            x = np.arange(len(self.phio))
            y = np.arange(len(self.ch_S[0,:]))
            yint=np.arange(1, len(self.phio)+1)
            
            #ax = Axes3D(fig)

            X_grid, Y_grid = np.meshgrid(x,y)
            Z_grid = self.ch_S[X_grid,Y_grid]
            im = plt.contourf(np.log10(self.phio[X_grid]), yint[Y_grid], Z_grid, cmap = 'hot')
            
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))

            plt.title('Characteristic S parameter from the %s Distribution' %dist_title)
            ax.yaxis.label.set_size(16)
            ax.xaxis.label.set_size(16)
            #ax.zaxis.label.set_size(16)
            plt.tick_params(axis='both', which='major', labelsize=14)
            #ax.set_zlabel('Separation Ratio, S~0 => centers close')
            ax.xaxis.labelpad = 10
            ax.yaxis.labelpad = 10
            #ax.zaxis.labelpad = 10
            ax.set_xlabel('Eq. Volume Monomer Phi')
            ax.set_ylabel('# of Monomers in Aggregate') 
            
            CBI = plt.colorbar(im, orientation='horizontal', shrink=0.8)


            l, b, w, h = plt.gca().get_position().bounds
            ll, bb, ww, hh = CBI.ax.get_position().bounds
            #CBI.ax.set_position([ll, b + 0.1*h, ww, h*0.8])
        
        if plot_name == 'S':   
        
            xstar = np.arange(0,2)
            
            plt.plot(xstar,self.ch_S[:,0],'*',color='blue',markersize=6, label='characteristic')
            #plt.plot(phiolog,self.ch_ovrlp,color='navy')
           
            #ax.set_xscale("log")
            
            Ss = [np.array(i) for i in self.S[:,:,0]]
           
          
            #Ss = [np.squeeze(np.array(self.S[i,:])) for i in range(self.S.shape[0])]
            phio_formatted = [ '%.2f' % elem for elem in self.phio ] 
            for i in range(len(phio_formatted)):
                if i != 0 and i != 9 and i != 19 and i != 29 and i != 39 and i != 49:
                    phio_formatted[i]=''
                    
            plt.xticks(rotation=45)            
            plt.boxplot(Ss)
            ax.xaxis.set_ticklabels(phio_formatted)
            
            #ax.xaxis.set_major_locator(ticker.MultipleLocator(2))

            #ax.fill_between(self.phio, self.poserr_ovrlp, self.negerr_ovrlp, facecolor='darkorange', alpha=0.5)   
            #ax.fill_between(self.phio,  self.mean_ovrlp+cv_pos,\
            #                self.mean_ovrlp+cv_neg, facecolor='darkorange', alpha=0.5)  
            plt.legend(loc='best')
            
            plt.title('Separation Ratio from the %s Distribution' %dist_title)
            plt.ylabel ("S")  
            #plt.xlim((2.0,100))
            
        if plot_name == 'complexity':
            
            self.ch_cplxp = self.ch_cplx[(self.phio<1.0)]
            poly = pl.polyfit(phiplog, self.ch_cplxp, 3)  #2 is the degree of polynomial to fit                   
            x2 = np.arange(min(phiplog), max(phiplog), .01) #use more points for a smoother plot
            y2 = np.polyval(poly, x2) #Evaluates the polynomial for each x2 value
            plt.plot(x2, y2, color='darkorange', linewidth=5)  #best fit line

            self.ch_cplxc = self.ch_cplx[(self.phio>=1.0)]
            poly = pl.polyfit(phiclog, self.ch_cplxc, 3)  #2 is the degree of polynomial to fit                   
            x2 = np.arange(min(phiclog), max(phiclog), .01) #use more points for a smoother plot
            y2 = np.polyval(poly, x2) #Evaluates the polynomial for each x2 value
            plt.plot(x2, y2, color='darkorange', linewidth=5)  #best fit line
            
            #plt.ylim(0,1.0)
            ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))
            ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%3.2f"))
            plt.plot(phiplog,self.ch_cplxp,'o',color='navy',markersize=6)
            plt.plot(phiplog,self.ch_cplxp,color='navy')
            plt.plot(phiclog,self.ch_cplxc,'o',color='navy',markersize=6)
            plt.plot(phiclog,self.ch_cplxc,color='navy')
            #print(self.poserr_cplx, self.negerr_cplx)
            #ax.fill_between(self.phio, self.poserr_cplx, self.negerr_cplx, facecolor='darkorange', alpha=0.5)           
            plt.title('Characteristic Complexity from %s Distribution' %dist_title)
            plt.ylabel ("Complexity") 
            
        if plot_name == 'req':
            
            agg_mass = (.9*4/3*np.pi*10**3)*2
            agg_rho = []
            for i in self.chreq:
                agg_rho.append(agg_mass/(4/3*np.pi*i**3))
            
            plt.xlim((.01,100))
            #print(agg_rho)
            ax.set_xscale("log")
            #ax.set_ylim((0,max(self.chreq)))
            ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))
            ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%3.2f"))
            ax.fill_between(self.phio, self.poserr_req, self.negerr_req, facecolor='darkorange', alpha=0.5)           
            ax.plot(self.phio,self.chreq,'o',color='navy',markersize=6)   
            ax.plot(self.phio,self.chreq,color='navy')
            ax.axhline(y=10, color='navy', linestyle='--')

            plt.title('Characteristic Eq. Volume Radius from %s Distribution' %dist_title)
            ax.set_ylabel ("Aggregate Eq. Volume Radius") 
            
        if plot_name == 'major_axis':
            
            ax.set_yscale("log")
            ax.set_xscale("log")

            plt.xlim((.01,100))
            #plt.ylim((min(major_axis),10))
            ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))
            ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%3.2f"))
            ax.fill_between(self.phio, self.negerr_mjrax, self.poserr_mjrax, facecolor='#0000e6', alpha=0.3)
            ax.fill_between(self.phio, self.negerr_depth, self.poserr_depth, facecolor='#EE7600', alpha=0.3)
            plt.plot(self.phio,self.mean_mjrax, color='black')
            plt.plot(self.phio,self.ch_majorax,'o',color='lightblue',markersize=3,label = 'Major axis from ellipse')
            plt.plot(self.phio,self.ch_majorax, color='lightblue')
            plt.plot(self.phio,self.mean_depth, color='black')
            plt.plot(self.phio,self.ch_depth,'o',color='#ffc266',markersize=3, label = 'Depth')
            plt.plot(self.phio,self.ch_depth,color='#ffc266')
            plt.plot(self.phip,self.chp,'o',color='#adebad',markersize=3, label = 'Aggregate aspect ratio')
            plt.plot(self.phip,self.chp, color='#adebad')
            plt.plot(self.phic,self.chc,'o',color='#adebad',markersize=3)
            plt.plot(self.phic,self.chc, color='#adebad') 
            plt.plot(self.phip,self.meanphip, color='black')
            plt.plot(self.phic,self.meanphic, color='black')
            ax.fill_between(self.phip, poserr_phip, negerr_phip, facecolor='darkgreen', alpha=0.3)
            ax.fill_between(self.phic, poserr_phic, negerr_phic, facecolor='darkgreen', alpha=0.3)
            plt.plot(self.phip,self.lengthp,color='darkorange', linestyle = '--')    
            plt.plot(self.phic,self.lengthc,color='navy', linestyle = '--')   
            plt.plot(self.phip,self.widthp,color='navy',linestyle = '--')
            plt.plot(self.phic,self.widthc,color='darkorange',linestyle = '--')

            plt.title('Characteristic Values from %s Distribution' %dist_title)
            plt.ylabel ("Aggregate Aspect Ratio, Major Axis, and Depth")  
            plt.legend(loc='best')
            
        if plot_name == 'dc_da':
            #ax.set_yscale("log")
            ax.set_xscale("log")

            plt.xlim((.01,100))
            #plt.ylim((min(major_axis),10))
            ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))
            ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%3.2f"))
            
            ch_majoraxp = self.ch_majorax[(self.phio<1.0)]
            ch_majoraxc = self.ch_majorax[(self.phio>=1.0)]
            ch_depthp = self.ch_depth[(self.phio<1.0)]
            ch_depthc = self.ch_depth[(self.phio>=1.0)]
            da_p = ch_majoraxp - self.widthp
            dc_p = ch_depthp - self.lengthp
            da_c = ch_depthc - self.widthc
            dc_c = ch_majoraxc - self.lengthc
            dc_da_p = dc_p/da_p
            dc_da_c = dc_c/da_c
            
            plt.plot(self.phip,da_p,'o',color='navy',markersize=3,label = 'da')  
            plt.plot(self.phip,da_p,color='navy')  
            #plt.plot(phip,dc_da_p,'o',color='darkgreen',markersize=3)  
            #plt.plot(phip,dc_da_p,color='darkgreen')
            #plt.plot(phic,dc_da_c,'o',color='darkgreen',markersize=3,label = 'dc/da')  
            #plt.plot(phic,dc_da_c,color='darkgreen')  
            plt.plot(self.phip,dc_p,'o',color='darkorange',markersize=3,label = 'dc')  
            plt.plot(self.phip,dc_p,color='darkorange') 
            plt.plot(self.phic,da_c,'o',color='navy',markersize=3)  
            plt.plot(self.phic,da_c,color='navy')  
            plt.plot(self.phic,dc_c,'o',color='darkorange',markersize=3)  
            plt.plot(self.phic,dc_c,color='darkorange') 
            #plt.axhline(y=1, color='black', linestyle='-',linewidth=2.5)

            plt.ylabel ("Aggregate c (a) - Monomer c (a)")  
            plt.legend(loc='best')
            
        if plot_name == 'dc_da_normalized':
            
            ax.set_yscale("log")
            plt.xlim((.01,100))
            #plt.ylim((min(major_axis),10))
            ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))
            ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%3.2f"))
            
            ch_majoraxp = self.ch_majorax[(self.phio<1.0)]
            ch_majoraxc = self.ch_majorax[(self.phio>=1.0)]
            ch_depthp = self.ch_depth[(self.phio<1.0)]
            ch_depthc = self.ch_depth[(self.phio>=1.0)]
            da_p = (ch_majoraxp - self.widthp)/self.widthp
            dc_p = (ch_depthp - self.lengthp)/self.lengthp
            da_c = (ch_depthc - self.widthc)/self.widthc
            dc_c = (ch_majoraxc - self.lengthc)/self.lengthc
            
            plt.plot(self.phip,da_p,'o',color='navy',markersize=3,label = 'da')  
            plt.plot(self.phip,da_p,color='navy')  
            #plt.plot(phip,dc_da_p,'o',color='darkgreen',markersize=3)  
            #plt.plot(phip,dc_da_p,color='darkgreen')
            #plt.plot(phic,dc_da_c,'o',color='darkgreen',markersize=3,label = 'dc/da')  
            #plt.plot(phic,dc_da_c,color='darkgreen')  
            plt.plot(self.phip,dc_p,'o',color='darkorange',markersize=3,label = 'dc')  
            plt.plot(self.phip,dc_p,color='darkorange') 
            plt.plot(self.phic,da_c,'o',color='navy',markersize=3)  
            plt.plot(self.phic,da_c,color='navy')  
            plt.plot(self.phic,dc_c,'o',color='darkorange',markersize=3)  
            plt.plot(self.phic,dc_c,color='darkorange') 
            #plt.axhline(y=1, color='black', linestyle='-',linewidth=2.5)

            plt.title('Normalized Change in Characteristic Axes Lengths from the %s Distribution' %dist_title)
            plt.ylabel ("(Aggregate c (a) - Monomer c (a))/ Monomer c (a)")  
            plt.legend(loc='best')
            ax.set_xscale("log")
            
        if plot_name == 'max_contact_angle':
            
            import random
            xrotp = self.xrot[(self.phio<=1.0)]*180/np.pi
            yrotp = self.yrot[(self.phio<=1.0)]*180/np.pi
            yrotc = self.yrot[(self.phio>=1.0)]
            rotplate = [max(i, j) for i, j in zip(xrotp, yrotp)]
            rotcol = yrotc*180/np.pi * 2          
            rot = np.concatenate((rotplate,rotcol))
            
            poly = pl.polyfit(phiplog, rotplate, 3)  #2 is the degree of polynomial to fit                   
            x2 = np.arange(min(phiplog), max(phiplog), .01) #use more points for a smoother plot
            y2 = np.polyval(poly, x2) #Evaluates the polynomial for each x2 value
            #plt.plot(x2, y2, color='darkorange', linewidth=5)  #best fit line
            #print('poly plates = ', poly)
               
            poly = pl.polyfit(phiclog, rotcol, 3)
            x2 = np.arange(min(phiclog), max(phiclog), .01) #use more points for a smoother plot
            y2 = np.polyval(poly, x2) #Evaluates the polynomial for each x2 value
            #plt.plot(x2, y2, color='darkorange', linewidth=5)  #best fit line
            #print('poly columns = ', poly)
                
            #plt.xlim((.01,100))
            #plt.ylim((0,max(rotplate, rotcol)))
            ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))
            ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%3d"))
            
            plt.plot(self.phio,rot,'o',color='navy',markersize=6) 
            plt.plot(self.phio,rot,color='navy')
            #plt.plot(self.phic,rotcol,'o',color='navy',markersize=6) 
            #plt.plot(self.phic,rotcol,color='navy')
            plt.title('Maximum Possible Contact Angle')
            plt.ylabel ("Aggregate Contact Angle [degrees]")  
            #ax.set_xscale("log")
         
            
        plt.xlabel ("Equivalent Volume Aspect Ratio of Monomer")


        if plot_name != 'shape' and save:

            filename = savefile     

            self.save_fig(nclusters, filename=filename, ext=ext, verbose = verbose)
            #plt.show()
        else:
            plt.close()

           
    def save_fig(self, nclusters, filename, ext='png', close=True, verbose=False):

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
                                
        
        path=('/Users/vprzybylo/Desktop/icefiles/agg_model/agg_notes/graphics/python/'
              +str(nclusters)+'xtals_hist/depth/')

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
