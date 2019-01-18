'''
Code to generate DDA files and compare accuracy of analytical area fraction.

Robert Schrom @ 11/2017
'''
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from crystal_dda.crystal_dda import branched_planar_dda
from crystal_dda.polygons import make_branched_planar, make_hexagon
from crystal_dda.geometry import afrac_dda_subregion
import os

# set values to create branched planar crystal with
amax = 3.
ac = 0.5

fb = 0.6
ft = 0.4
fg = 0.6

nsb = 5
nxp = 300
nzp = 5

# loop over a axis lengths for given crystal
numa = 21
avals = np.linspace(0.1, amax, numa)
afrac_anl_vals = np.empty(numa)
afrac_dda_vals = np.empty(numa)
xdda, ydda = make_branched_planar(amax, ac, ft, fb, fg, nsb, 0.)

for i, a in enumerate(avals):
    fname, afrac = branched_planar_dda(a, amax, ac, ft, fb, fg, nsb, nxp, nzp)

    # compare area fraction of actual DDA particle
    xhex, yhex = make_hexagon(a)
    afrac_dda = afrac_dda_subregion(xhex, yhex, xdda, ydda)
    print 'a:{:.2e}\tanalytical: {:.3f}, {:.1f}\tdda: {:.3f}'.format(a, afrac, afrac*917., afrac_dda)
    afrac_anl_vals[i] = afrac
    afrac_dda_vals[i] = afrac_dda

# print out results again
print afrac_dda_vals
print avals

# plot
mpl.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
mpl.rc('text', usetex=True)

plt.plot(avals, afrac_anl_vals, 'r--', lw=3., label='analytical')
plt.plot(avals, afrac_dda_vals, 'b--', lw=3., label='dipole shape')
plt.legend()

ax = plt.gca()
ax.set_xlabel('a-axis length (mm)', fontsize=32)
ax.set_ylabel('area fraction', fontsize=32)

ax.tick_params(axis='both', which='major', labelsize=28, pad=20)
ax.set_xticklabels(ax.get_xticks())
ax.set_yticklabels(ax.get_yticks())
ax.grid(color='k', linestyle=(0.5, [2,6]), linewidth=1.)

plt.savefig('afrac_accuracy.png')
os.system('convert -trim {} {}'.format('afrac_accuracy.png', 'afrac_accuracy.png'))
