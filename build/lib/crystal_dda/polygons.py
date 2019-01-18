'''
Code to generate 2D polygons of various shapes.

Robert Schrom @ 11/2017
'''
import numpy as np
import geometry as geom

# polygon shape of hexagon
def make_hexagon(a):
    xp = np.array([a, a/2., -a/2., -a, -a/2., a/2., a])
    yp = np.array([0., a*np.sqrt(3.)/2., a*np.sqrt(3.)/2., 0.,
                  -a*np.sqrt(3.)/2., -a*np.sqrt(3.)/2., 0.])
    return xp, yp

# stellar triangle given branch fraction width and segment points
def make_stellar_triangle(fbranch, p1, p2):
    # get normal and tangential directions to p12
    p12 = p2-p1
    midpoint = (p2+p1)/2.
    segmag = np.sqrt(np.sum((p12)**2.))
    n = np.array([-p12[1], p12[0]])/segmag
    t = p12/segmag

    s1 = p1+t*fbranch*segmag/2.
    s2 = midpoint+t*(1.-fbranch)*segmag/2.
    s3 = s2+segmag*(1.-fbranch)*(n*np.sqrt(3.)/2.-t/2.)
    return s1, s2, s3

# function to create polygon of stellar crystal with branch width fraction
def make_stellar(fbranch, a):
    # define hexagonal polygon
    x, y = make_hexagon(a)
    numhex = len(x)
    numstellar = 25
    xst = np.empty([numstellar])
    yst = np.empty([numstellar])
    xst[0] = x[0]
    yst[0] = y[0]
    stind = 0

    # loop through segments of hexagon and insert points for stellar triangles
    for i in range(numhex-1):
        p1 = np.array([x[i], y[i]])
        p2 = np.array([x[i+1], y[i+1]])
        s1, s2, s3 = make_stellar_triangle(fbranch, p1, p2)
        xst[stind+1] = s1[0]
        xst[stind+2] = s3[0]
        xst[stind+3] = s2[0]
        xst[stind+4] = p2[0]

        yst[stind+1] = s1[1]
        yst[stind+2] = s3[1]
        yst[stind+3] = s2[1]
        yst[stind+4] = p2[1]
        stind = stind+4
    return xst, yst

# create branched planar crystal
def make_branched_planar(amax, ac, ft, fb, fg, nsb, diplen):
    # get x and y coordinates of bounding points
    #----------------------------------------------------
    # p1 - midpoint of ice core edge
    # p2 - intersection of ice core edge and main branch
    # p3 - intersection of main branch and tip
    # p4 - upper corner at amax
    #----------------------------------------------------

    # calculate shape quantities
    fmb = geom.frac_main_branch(amax, ac, ft, fb, nsb)
    ag = fg*amax+(1.-fg)*ac
    wmb = 1./2.*fmb*ac
    wt = 1./2.*ft*amax

    p1x = 0.
    p1y = np.sqrt(3.)/2.*ac
    p2x = ac/2.-wmb
    p2y = p1y
    p3x = amax/2.-wmb
    p3y = np.sqrt(3.)/2.*amax
    p4x = amax/2.
    p4y = p3y

    # create xcoor and ycoor of half sector
    xcoor = np.array([p1x, p2x, p3x, p4x])
    ycoor = np.array([p1y, p2y, p3y, p4y])

    # figure out sub-branches
    wsb = fb/(nsb-1)*(amax-ac-wmb)
    ssb = wsb*(1.-fb)/fb

    mbound, bbound = geom.points2eqn(0., np.sqrt(3.)/2.*ag,
                                     amax/2.*(1.-ft), np.sqrt(3.)/2.*amax)

    for i in range(nsb-1):
        # points on main branch
        sb1x = p2x+(i+0.25)*(wsb+ssb)/2.
        sb1y = p2y+(i+0.25)*(wsb+ssb)*np.sqrt(3.)/2.
        sb4x = sb1x+wsb/2.
        sb4y = sb1y+wsb*np.sqrt(3.)/2.

        # get equation of line defining each sub-branch edge (60 degrees from main branch)
        m_sb = -np.sqrt(3.)
        b_sb_low = -m_sb*sb1x+sb1y
        b_sb_high = -m_sb*sb4x+sb4y

        # points off main branch
        sb2x, sb2y = geom.intersection(mbound, bbound, m_sb, b_sb_low)
        sb3x, sb3y = geom.intersection(mbound, bbound, m_sb, b_sb_high)

        # add small delx and dely so sub-branches don't touch
        delx = diplen
        dely = -diplen*np.sqrt(3.)

        # replace with point on x=0 if below bounding line from tip to ag
        if sb2y>b_sb_low:
            sb2y = b_sb_low
            sb2x = 0.
        if sb3y>b_sb_high:
            sb3y = b_sb_high
            sb3x = 0.

        sb2x = sb2x+delx
        sb2y = sb2y+dely
        sb3x = sb3x+delx
        sb3y = sb3y+dely

        # insert sub-branch into coordinate arrays
        sbxc = np.array([sb1x, sb2x, sb3x, sb4x])
        sbyc = np.array([sb1y, sb2y, sb3y, sb4y])

        xcoor = np.insert(xcoor, 2+i*4, sbxc)
        ycoor = np.insert(ycoor, 2+i*4, sbyc)

    # flip over half sector to create full sector
    xsec = np.concatenate((-xcoor[::-1], xcoor[1:]))
    ysec = np.concatenate((ycoor[::-1], ycoor[1:]))
    
    # rotate sector 60 degrees and concatenate with previous sector
    xrot, yrot = geom.rotate(xsec, ysec, -60.)
    xhex = np.concatenate((xsec, xrot[1:]))
    yhex = np.concatenate((ysec, yrot[1:]))

    # repeat until we have the entire crystal
    for i in range(4):
        xrot, yrot = geom.rotate(xsec, ysec, -(i+2)*60.)
        xhex = np.concatenate((xhex, xrot[1:]))
        yhex = np.concatenate((yhex, yrot[1:]))

    print len(xhex)
    return xhex, yhex
