from ipas import lab_copy1 as lab
from ipas import crystals_opt_rot as crys
from ipas import plots as plts
import numpy as np
import time  #for efficiency tests
import itertools   #to create width array and join plate/col aspect ratios
from operator import itemgetter
import shapely.geometry as geom
import matplotlib.pyplot as plt

phio = [.01,.02,.03,.04,.05,.06,.07,.08,.09,.1,.2,.3,.4,\
        .5,.6,.7,.8,.9,1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100]
reqarr = [1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100,200,300,400,500,600,700,800,900,1000]
#reqarr = [1,2]
numaspectratios=len(phio)
ch_dist='gamma'         #anything other than gamma uses the characteristic from the best distribution pdf (lowest SSE)
nclusters = 300        #changes how many aggregates per aspect ratio to consider
ncrystals = 2
minor = 'depth'         #either 'minorxy' -- from fit ellipse or 'depth' to mimic IPAS in IDL
save_plots = True     #saves all histograms to path with # of aggs and minor/depth folder
file_ext = 'png'

p300 = lab.main_ar_loop(phio, reqarr, numaspectratios, ch_dist, nclusters, ncrystals, minor, save_plots, file_ext)

