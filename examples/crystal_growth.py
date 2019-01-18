import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import crystal_dda.polygons as poly
import crystal_dda.geometry as geom
from matplotlib import path
import os

# set branched planar crystal properties
amax = 3.
ac = 0.5

fb = 0.6
ft = 0.4
fg = 0.6

nsb = 5

ag = fg*amax+(1.-fg)*ac
wt = ft*amax/2.

# test points
numxp = 300
numyp = 300
x2d, y2d = np.meshgrid(np.linspace(-amax, amax, numxp),
                       np.linspace(-amax, amax, numyp), indexing='ij')
xp = x2d.flatten()
yp = y2d.flatten()
indicator = np.empty([numxp*numyp])

# create bounding polygons for two regions
xcore, ycore = poly.make_hexagon(ac)
xg, yg = poly.make_hexagon(ag)
xbound = np.array([-amax/2., -amax/2.+wt, 0., amax/2.-wt, amax/2.])
ybound = np.array([np.sqrt(3.)*amax/2., np.sqrt(3.)*amax/2., np.sqrt(3.)*ag/2.,
                   np.sqrt(3.)*amax/2., np.sqrt(3.)*amax/2.])

# rotations for bounding star shape
xboundr, yboundr = geom.rotate(xbound, ybound, -60.)
xbound = np.concatenate((xbound, xboundr[1:]))
ybound = np.concatenate((ybound, yboundr[1:]))

for i in range(4):
    xboundr, yboundr = geom.rotate(xboundr, yboundr, -60.)
    xbound = np.concatenate((xbound, xboundr[1:]))
    ybound = np.concatenate((ybound, yboundr[1:]))

# create branched planar crystal
x, y = poly.make_branched_planar(amax, ac, ft, fb, fg, nsb, 0.)
inbranched = geom.in_polygon(x, y, xp, yp)
xp_br = xp[inbranched]
yp_br = yp[inbranched]

# get crystal shapes at various sizes
mpl.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
mpl.rc('text', usetex=True)
numa = 3
avals = np.linspace(ac, amax, numa)

for i, a in enumerate(avals):
    # subset full-size crystal to a size
    print a
    xhex, yhex = poly.make_hexagon(a)
    inhex = geom.in_polygon(xhex, yhex, xp_br, yp_br)
    xp_sub = xp_br[inhex]
    yp_sub = yp_br[inhex]

    #plot
    plt.figure(i)
    plt.scatter(xp_sub, yp_sub, c='b', s=20, edgecolor='')
    plt.plot(xcore, ycore, 'r--', lw=4.)
    plt.plot(xg, yg, 'k--', lw=4.)
    plt.plot(xbound, ybound, 'g--', lw=4.)

    ax = plt.gca()
    ax.set_aspect(1.)
    ax.set_xlim([-amax, amax])
    ax.set_ylim([-amax, amax])

    ax.set_xlabel('x distance (mm)')
    ax.set_ylabel('y distance (mm)')

    ax.tick_params(axis='both', which='major', labelsize=28, pad=20)
    ax.set_xticklabels(ax.get_xticks())
    ax.set_yticklabels(ax.get_yticks())

    ax.grid(color='k', linestyle=(0.5, [2,6]), linewidth=1.)
    imgname = 'crystal{:.1f}.png'.format(a)
    plt.savefig(imgname, dpi=60)
    os.system('convert -trim {} {}'.format(imgname, imgname))
