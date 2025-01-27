"""Plots characteristic variables that have been looped over for every aspect ratio"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib 
import seaborn
import pandas as pd
from ipas import lab_copy_varyphi as lab
from mpl_toolkits.mplot3d import Axes3D


class Make_Plots():
    
    def __init__(self, phio, seedwidtharr, seedlengtharr, newwidtharr, newlengtharr, chreq, chphi, 
                           chphi2D, ch_ovrlp, ch_S, ovrlp, S, ch_majorax, ch_depth, dphigam, poserr_phi, negerr_phi,
                           poserr_phi2D, negerr_phi2D, poserr_mjrax, negerr_mjrax, poserr_req, negerr_req, poserr_depth,
                           negerr_depth, poserr_ovrlp, negerr_ovrlp, poserr_S, negerr_S, poserr_cplx, negerr_cplx,
                           min_phi, max_phi, min_phi2D, max_phi2D, min_mjrax, max_mjrax, 
                           min_depth, max_depth, min_req, max_req, mean_phi, mean_mjrax, mean_depth, mean_req, 
                           mean_ovrlp, mean_S, mean_cplx, ch_cplx, xrot, yrot):
                        
                    
        self.phio = np.array(phio)
        self.seedwidtharr = np.array(seedwidtharr)
        self.seedlengtharr = np.array(seedlengtharr)
        self.newwidtharr = np.array(newwidtharr)
        self.newlengtharr = np.array(newlengtharr)     
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
        self.poserr_phi2D = np.array(poserr_phi2D)
        self.poserr_phi2D = np.array(poserr_phi2D)
        self.poserr_S = np.array(poserr_S)
        self.negerr_S = np.array(negerr_S)
        self.poserr_phi = np.array(poserr_phi)
        self.negerr_phi = np.array(negerr_phi)
        self.poserr_mjrax = np.array(poserr_mjrax)
        self.negerr_mjrax = np.array(negerr_mjrax)
        self.poserr_req = np.array(poserr_req)
        self.negerr_req = np.array(negerr_req)
        self.poserr_depth = np.array(poserr_depth)
        self.negerr_depth = np.array(negerr_depth)
        self.poserr_ovrlp = np.array(poserr_ovrlp)
        self.negerr_ovrlp = np.array(negerr_ovrlp)
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
        self.min_mjrax = np.array(min_mjrax)
        self.max_mjrax = np.array(max_mjrax)
        self.min_depth = np.array(min_depth)
        self.max_depth = np.array(max_depth)
        self.min_req = np.array(min_req)
        self.max_req = np.array(max_req)
        self.mean_phi = np.array(mean_phi)   
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
        

    def which_plot(self, nclusters, plot_name, ch_dist, savefile, read_file='outfile.dat', save=False, verbose=False, ext='png'):   
        #makes plots based on plot_name passed in

        import seaborn as sns
        from matplotlib.ticker import MultipleLocator, FormatStrFormatter, LogLocator, LogFormatter, ScalarFormatter

        #self.numaspectratios = 26

        fig = plt.figure(figsize=(10,7),frameon=False)
        ax = plt.subplot(111)
        ax.set_xscale("log")
        #ax.set_yscale("log")
        ax.grid(b=True, which='minor', alpha = 0.9, axis = 'x')
        ax.grid(b=True, which='minor', alpha = 0.9, axis = 'y')
        ax.grid(b=True, which='major', alpha = 1.0, axis = 'x')            
        ax.grid(b=True, which='major', alpha = 1.0, axis = 'y')    

        plt.style.use('seaborn-dark')  #gray background
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.size'] = 14
        plt.rcParams['axes.labelsize'] = 14
        plt.rcParams['axes.labelweight'] = 'normal'
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['xtick.labelsize'] = 12
        plt.rcParams['ytick.labelsize'] = 12
        plt.rcParams['legend.fontsize'] = 11
        
        wh=np.where(self.phio<=1.0)
        self.phip[wh] = self.phio[wh]
        self.phic[wh] = np.nan
        self.chp[wh] = self.chphi[wh]
        self.chc[wh] = np.nan
        #self.lengthp[wh] = self.seedlengtharr[wh]
        self.lengthc[wh] = np.nan
        #self.widthp[wh] = self.seedwidtharr[wh]
        self.widthc[wh] = np.nan
        self.meanphip = self.mean_phi[(self.phio<1.0)]
        self.meanphic = self.mean_phi[(self.phio>=1.0)]
        phip = self.phio[(self.phio<1.0)]
        
        for i in range(len(self.negerr_phi)):                
            if self.negerr_phi[i] < 0.0:
                self.negerr_phi[i] = 1.0
                print('setting negerr_phi to 1.0')

        poserr_phip = self.poserr_phi[(self.phio<1.0)]
        negerr_phip = self.negerr_phi[(self.phio<1.0)]
        poserr_phic = self.poserr_phi[(self.phio>=1.0)]
        negerr_phic = self.negerr_phi[(self.phio>=1.0)]
        
        wh=np.where(self.phio>=1.0)
        self.phic[wh] = self.phio[wh]
        self.phip[wh] = np.nan        
        self.chc[wh] = self.chphi[wh]
        self.chp[wh] = np.nan
        #self.lengthc[wh] = self.seedlengtharr[wh]
        self.lengthp[wh] = np.nan
        #self.widthc[wh] = self.seedwidtharr[wh]
        self.widthp[wh] = np.nan

        phip = self.phip[~pd.isnull(self.phip)]
        chp = self.chp[~pd.isnull(self.chp)]
        phic = self.phic[~pd.isnull(self.phic)]
        chc = self.chc[~pd.isnull(self.chc)]
        lengthp = self.lengthp[~pd.isnull(self.lengthp)]
        lengthc = self.lengthc[~pd.isnull(self.lengthc)]
        widthp = self.widthp[~pd.isnull(self.widthp)]
        widthc = self.widthc[~pd.isnull(self.widthc)]

        if ch_dist=='gamma':
            dist_title = 'Gamma'
        else:
            dist_title = 'Best'

        if plot_name == 'char':  

                    
            print(self.poserr_phi)
            print(self.negerr_phi)
            print(self.min_phi)
            print(self.max_phi)
            
            
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
      
            plt.plot(phip, self.meanphip, color='black')
            plt.plot(phic, self.meanphic, color='black', label = 'mean')
            ax.fill_between(self.phio, self.min_phi, self.max_phi, facecolor='darkorange', alpha=0.3)
            #ax.fill_between(phic, poserr_phic, negerr_phic, facecolor='darkorange', alpha=0.3)
            
            
            #Line of no change in aspect ratio
            phiox=np.logspace(-2, 2., num=20, endpoint=True, base=10.0, dtype=None)#just columns (0,2); plates (-2,0)
            #plt.ylim((.01,100))
            plt.plot(phiox, phiox, color='navy')
           
            
            ax.set_yscale("log")
            #plt.xlim((0.01,100))
            #plt.ylim((min(ch),max(ch)))
            ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))
            ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))
            #ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
            #ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
            #plt.ticklabel_format(style='plain', axis='y')
            

            #plt.title('Characteristic Aggregate Aspect Ratio for '+str(self.numaspectratios)+' Aspect Ratios')
            #plt.xlabel ("Equivalent Volume Aspect Ratio of Monomers")
            plt.title('Characteristic Aggregate Aspect Ratio from %s Distribution' %dist_title)
            plt.ylabel ("Aggregate Aspect Ratio")
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

            ax.set_xlabel ("Equivalent Volume Monomer Aspect Ratio")
            ax.set_ylabel('Aggregate $\phi$ - Monomer $\phi$')
            #plt.legend(loc='best')
            ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))
            ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.3f"))

        if plot_name == 'overlap':
            '''
            cv_pos = (self.poserr_ovrlp-self.mean_ovrlp)/self.mean_ovrlp
            cv_neg = (self.negerr_ovrlp+self.mean_ovrlp)/self.mean_ovrlp
            plt.xlim((.01,100))
            #plt.ylim((min(overlap),10))
            ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))
            ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%3.2f"))
            plt.plot(self.phio,self.ch_ovrlp,'o',color='navy',markersize=6)
            plt.plot(self.phio,self.ch_ovrlp,color='navy',markersize=6)
            plt.plot(self.phio,self.mean_ovrlp,color='black')
            #ax.fill_between(self.phio, self.poserr_ovrlp, self.negerr_ovrlp, facecolor='darkorange', alpha=0.5)   
            ax.fill_between(self.phio, self.mean_ovrlp + cv_pos,\
                            self.mean_ovrlp - cv_neg, facecolor='darkorange', alpha=0.5)   

            '''
            xstar = np.arange(0,50)
            plt.plot(xstar,self.ch_ovrlp,'*',color='blue',markersize=6, label='characteristic')


            ovrlps = [np.array(i) for i in self.ovrlp]
            phio_formatted = [ '%.2f' % elem for elem in self.phio ] 
 
            #for i in range(len(phio_formatted)):
            #    print(i)
            #    if i != 0 and i != 9 and i != 19 and i != 29 and i != 39 and i != 49:
            #        print('here')
            #        phio_formatted[i]=''
                    
            plt.xticks(rotation=45)            
            plt.boxplot(ovrlps)
            ax.xaxis.set_ticklabels(phio_formatted)
            plt.legend(loc='best')
            
            plt.title('Overlap from the %s Distribution' %dist_title)
            plt.ylabel ("Overlap [%]") 
            
           
        if plot_name == 'complexity':
           
            plt.ylim(0,1.0)
            ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))
            ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%3.2f"))
            plt.plot(self.phio,self.ch_cplx,'o',color='navy',markersize=6)
            plt.plot(self.phio,self.ch_cplx,color='navy')
            #print(self.poserr_cplx, self.negerr_cplx)
            #ax.fill_between(self.phio, self.poserr_cplx, self.negerr_cplx, facecolor='darkorange', alpha=0.5)           
            plt.title('Characteristic Complexity from %s Distribution' %dist_title)
            plt.ylabel ("Complexity") 

        if plot_name == 'req':
            
            agg_mass = (.9*4/3*np.pi*10**3)*2
            agg_rho = []
            for i in self.chreq:
                agg_rho.append(agg_mass/(4/3*np.pi*i**3))
            print(agg_rho)

            #ax.set_ylim((0,max(self.chreq)))
            ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))
            ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%3.2f"))
            ax.fill_between(self.phio, self.poserr_req, self.negerr_req, facecolor='darkorange', alpha=0.5)           
            ax.plot(self.phio,self.chreq,'o',color='navy',markersize=6)   
            ax.plot(self.phio,self.chreq,color='navy')
            ax.axhline(y=10, color='navy', linestyle='--')

            ax2 = ax.twinx()
            ax2.plot(self.phio,agg_rho,color='darkgreen')
            ax2.plot(self.phio,agg_rho,'o',color='darkgreen')
            #ax2.set_ylim((0,max(agg_rho)))
            plt.xlim(0.01, 100)
            ax2.axhline(y=1.8, color='darkgreen', linestyle='--') 

            plt.title('Characteristic Eq. Volume Radius from %s Distribution' %dist_title)
            ax.set_ylabel ("Aggregate Eq. Volume Radius", color = 'navy') 
            ax2.set_ylabel ("Aggregate Eq. Volume Density", color='darkgreen')

        if plot_name == 'major_axis':
            
            
            ax.set_yscale("log")
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
            plt.plot(phip,chp,'o',color='#adebad',markersize=3, label = 'Aggregate aspect ratio')
            plt.plot(phip, chp, color='#adebad')
            plt.plot(phic,chc,'o',color='#adebad',markersize=3)
            plt.plot(phic, chc, color='#adebad') 
            plt.plot(phip,self.meanphip, color='black')
            plt.plot(phic,self.meanphic, color='black')
            ax.fill_between(phip, poserr_phip, negerr_phip, facecolor='darkgreen', alpha=0.3)
            ax.fill_between(phic, poserr_phic, negerr_phic, facecolor='darkgreen', alpha=0.3)
            plt.plot(phip,lengthp,color='darkorange', linestyle = '--')    
            plt.plot(phic,lengthc,color='navy', linestyle = '--')   
            plt.plot(phip,widthp,color='navy',linestyle = '--')
            plt.plot(phic,widthc,color='darkorange',linestyle = '--')

            plt.title('Characteristic Values from %s Distribution' %dist_title)
            plt.ylabel ("Aggregate Aspect Ratio, Major Axis, and Depth")  
            plt.legend(loc='best')
            
        if plot_name == 'dc_da':
            #ax.set_yscale("log")
            plt.xlim((.01,100))
            #plt.ylim((min(major_axis),10))
            ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))
            ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%3.2f"))
            
            ch_majoraxp = self.ch_majorax[(self.phio<1.0)]
            ch_majoraxc = self.ch_majorax[(self.phio>=1.0)]
            ch_depthp = self.ch_depth[(self.phio<1.0)]
            ch_depthc = self.ch_depth[(self.phio>=1.0)]
            da_p = ch_majoraxp - widthp
            dc_p = ch_depthp - lengthp
            da_c = ch_depthc - widthc
            dc_c = ch_majoraxc - lengthc
            dc_da_p = dc_p/da_p
            dc_da_c = dc_c/da_c
            
            plt.plot(phip,da_p,'o',color='navy',markersize=3,label = 'da')  
            plt.plot(phip,da_p,color='navy')  
            #plt.plot(phip,dc_da_p,'o',color='darkgreen',markersize=3)  
            #plt.plot(phip,dc_da_p,color='darkgreen')
            #plt.plot(phic,dc_da_c,'o',color='darkgreen',markersize=3,label = 'dc/da')  
            #plt.plot(phic,dc_da_c,color='darkgreen')  
            plt.plot(phip,dc_p,'o',color='darkorange',markersize=3,label = 'dc')  
            plt.plot(phip,dc_p,color='darkorange') 
            plt.plot(phic,da_c,'o',color='navy',markersize=3)  
            plt.plot(phic,da_c,color='navy')  
            plt.plot(phic,dc_c,'o',color='darkorange',markersize=3)  
            plt.plot(phic,dc_c,color='darkorange') 
            #plt.axhline(y=1, color='black', linestyle='-',linewidth=2.5)

            plt.title('Change in Characteristic Axes Lengths from a %s Distribution' %dist_title)
            plt.ylabel ("Aggregate c (a) - Monomer c (a)")  
            plt.legend(loc='best')
            
        if plot_name == 'dc_da_normalized':
            #ax.set_yscale("log")
            plt.xlim((.01,100))
            #plt.ylim((min(major_axis),10))
            ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))
            ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%3.2f"))
            
            ch_majoraxp = self.ch_majorax[(self.phio<1.0)]
            ch_majoraxc = self.ch_majorax[(self.phio>=1.0)]
            ch_depthp = self.ch_depth[(self.phio<1.0)]
            ch_depthc = self.ch_depth[(self.phio>=1.0)]
            da_p = (ch_majoraxp - widthp)/widthp
            dc_p = (ch_depthp - lengthp)/lengthp
            da_c = (ch_depthc - widthc)/widthc
            dc_c = (ch_majoraxc - lengthc)/lengthc
            
            plt.plot(phip,da_p,'o',color='navy',markersize=3,label = 'da')  
            plt.plot(phip,da_p,color='navy')  
            #plt.plot(phip,dc_da_p,'o',color='darkgreen',markersize=3)  
            #plt.plot(phip,dc_da_p,color='darkgreen')
            #plt.plot(phic,dc_da_c,'o',color='darkgreen',markersize=3,label = 'dc/da')  
            #plt.plot(phic,dc_da_c,color='darkgreen')  
            plt.plot(phip,dc_p,'o',color='darkorange',markersize=3,label = 'dc')  
            plt.plot(phip,dc_p,color='darkorange') 
            plt.plot(phic,da_c,'o',color='navy',markersize=3)  
            plt.plot(phic,da_c,color='navy')  
            plt.plot(phic,dc_c,'o',color='darkorange',markersize=3)  
            plt.plot(phic,dc_c,color='darkorange') 
            #plt.axhline(y=1, color='black', linestyle='-',linewidth=2.5)

            plt.title('Normalized Change in Characteristic Axes Lengths from a %s Distribution' %dist_title)
            plt.ylabel ("(Aggregate c (a) - Monomer c (a))/ Monomer c (a)")  
            plt.legend(loc='best')
            
        if plot_name == 'max_contact_angle':
            
            xrotp = self.xrot[(self.phio<=1.0)]*180/np.pi
            yrotp = self.yrot[(self.phio<=1.0)]*180/np.pi
            yrotc = self.yrot[(self.phio>=1.0)]
            rotplate = [max(i, j) for i, j in zip(xrotp, yrotp)]
            rotcol = yrotc*180/np.pi * 2
            
            import pylab as pl
            i = 0
            while i < 2:
                if i == 0:
                    poly = pl.polyfit(phip, rotplate, 2)  #2 is the degree of polynomial to fit                   
                    x2 = np.arange(min(phip), max(phip), .01) #use more points for a smoother plot
                    y2 = np.polyval(poly, x2) #Evaluates the polynomial for each x2 value
                    plt.plot(x2, y2, color='darkorange', linewidth=5)  #best fit line
            
                else:
                    poly = pl.polyfit(phic, rotcol, 2)
                    x2 = np.arange(min(phic), max(phic), 1) #use more points for a smoother plot
                    y2 = np.polyval(poly, x2) #Evaluates the polynomial for each x2 value
                    plt.plot(x2, y2, color='darkorange', linewidth=5)  #best fit line
                    print(poly)
                    print(x2)
                    print(y2)
                def poly2latex(poly, variable="x", width=2):
                    t = ["{0:0.{width}f}"]
                    t.append(t[-1] + " {variable}")
                    t.append(t[-1] + "^{1}")

                    def f():
                        for i, v in enumerate(reversed(poly)):
                            idx = i if i < 2 else 2
                            return t[idx].format(v, i, variable=variable, width=width)

                    return "${}$".format("+".join(f()))
                if i == 0:
                    pl.text(phip[1], rotplate[21], poly2latex(poly), fontsize=14)
                else:
                    pl.text(phic[6], rotcol[4], poly2latex(poly), fontsize=14)
                i +=1
                print(poly2latex(poly))
            
            plt.xlim((.01,100))
            #plt.ylim((0,max(rotplate, rotcol)))
            ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))
            ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%3d"))
            #ax.fill_between(self., self.poserr_req, self.negerr_req, facecolor='darkorange', alpha=0.5)           
            plt.plot(phip,rotplate,'o',color='navy',markersize=6) 
            plt.plot(phip,rotplate,color='navy')
            plt.plot(phic,rotcol,'o',color='navy',markersize=6) 
            plt.plot(phic,rotcol,color='navy')
            plt.title('Maximum Possible Contact Angle')
            plt.ylabel ("Aggregate Contact Angle [degrees]")  
            
         
            
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
                                
        if len(np.shape(self.ch_ovrlp)) == 1:
            path=('/Users/vprzybylo/Desktop/icefiles/agg_model/agg_notes/graphics/python/'+
                          str(np.size(self.ch_ovrlp))+'xtals_hist/depth/')
        else:
            path=('/Users/vprzybylo/Desktop/icefiles/agg_model/agg_notes/graphics/python/'+
                          str(np.size(self.ch_ovrlp)[0])+'xtals_hist/depth/')

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
