'''
Geometry functions

Robert Schrom @ 11/2017
'''
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import path

# function to take polygon and points and return indicator
def in_polygon(xpoly, ypoly, xpoints, ypoints):
    p = path.Path([(xpoly[i], ypoly[i]) for i in range(len(xpoly))])
    points = np.empty([len(xpoints), 2])
    points[:,0] = xpoints
    points[:,1] = ypoints
    indicator = p.contains_points(points)
    return indicator

# rotation
def rotate(x, y, angle):
    angle = angle*np.pi/180.
    xrot = np.cos(angle)*x-np.sin(angle)*y
    yrot = np.sin(angle)*x+np.cos(angle)*y
    return xrot, yrot

# flip over y-axis
def flipy(x, y):
    xflip = -x
    yflip = y
    return xflip, yflip

# get equation of line from p1 to p2
def points2eqn(p1x, p1y, p2x, p2y):
    m = (p2y-p1y)/(p2x-p1x)
    b = p1y-m*p1x
    return m, b

# get point of intersection of two lines
def intersection(m1, b1, m2, b2):
    xint = (b2-b1)/(m1-m2)
    yint = m1*xint+b1
    return xint, yint

# get area fraction of stellar
def afrac_stellar(fbranch):
    afrac = 1.-(1.-fbranch)**2.
    return afrac

# determine main branch fraction width (give same width as sub-branches, or 1)
def frac_main_branch(amax, ac, ft, fb, nsb):
    wt = ft/2.*amax
    wsb = 1./((nsb-1)/fb+1.)*(amax-ac)
    wmb = min(max(wsb/2., ac/2.), min(wsb/2., ac/2.))
    fmb = wmb/(ac/2.)
    return fmb

# get areas of each region of branched planar
def area_bound(a):
    area = np.sqrt(3.)/4.*a**2.
    return area

def area_gap(a, fmb, fb, ac):
    area = np.sqrt(3.)/2.*fmb*(1.-fb)*ac*(a-ac)+\
           np.sqrt(3.)/4.*fb*(a**2.-ac**2.)
    return area

def area_star(a, fmb, fb, ft, ac, ag, amax):
    b = fb/(amax-ag)*(ft*amax-ag)
    c = (fb*ag*amax*(1.-ft)/(amax-ag))+(1.-fb)*fmb*ac
    area = np.sqrt(3.)/4.*b*(a**2.-ag**2.)+np.sqrt(3.)/2.*c*(a-ag)
    return area

# get deposition area fraction of branched planar
def dep_afrac_branched(a, amax, ac, ft, fb, fg, nsb):
    # deal with fg = 1
    fg = min(fg, 0.99999999)

    # calculate ag and fmb
    fmb = frac_main_branch(amax, ac, ft, fb, nsb)
    ag = fg*amax+(1.-fg)*ac

    # calculate area fraction (deposition)
    if a <= ac:
        afrac_dep = 1.
    elif (a>ac)&(a<=ag):
        afrac_dep = fb+ac/a*fmb*(1.-fb)
    else:
        afrac_dep = fb/(amax-ag)*(ft*amax-ag+ag*amax*(1.-ft)/a)+\
                    (1.-fb)*fmb*ac/a
    return afrac_dep

# get area fraction of branched planar
def afrac_branched(a, amax, ac, ft, fb, fg, nsb):
    # deal with fg = 1 and calculate crystal values
    fg = min(fg, 0.99999999)
    fmb = frac_main_branch(amax, ac, ft, fb, nsb)
    ag = fg*amax+(1.-fg)*ac
    
    # calculate area fraction
    if a <= ac:
        afrac = 1.
    elif (a>ac)&(a<=ag):
        afrac = (area_bound(ac)+area_gap(a, fmb, fb, ac))/area_bound(a)
    else:
        afrac = (area_bound(ac)+area_gap(ag, fmb, fb, ac)
                +area_star(a, fmb, fb, ft, ac, ag, amax))/area_bound(a)
    return afrac

# calculate area fraction in hexagonal region within dda polygon using random points
def afrac_dda_subregion(xhex, yhex, xdda_poly, ydda_poly):
    # probability method for estimating area fraction
    numtrials = 20000
    a = np.max(xhex)
    rx = 2.*a*(np.random.rand(numtrials)-0.5)
    ry = 2.*a*(np.random.rand(numtrials)-0.5)

    # determine random points in hexagon
    inhex = in_polygon(xhex, yhex, rx, ry)
    rx_hex = rx[inhex]
    ry_hex = ry[inhex]
    numhex = len(rx_hex)

    # determine which hexagon points are in polygon defining full size dda polygon
    inbranched = in_polygon(xdda_poly, ydda_poly, rx_hex, ry_hex)
    rx_br = rx_hex[inbranched]
    ry_br = ry_hex[inbranched]
    numbr = len(rx_br)

    frac = float(numbr)/float(numhex)
    return frac
