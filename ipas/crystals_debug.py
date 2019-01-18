"""Classes representing ice crystals (monomers) and ice clusters
(aggregates).

"""
import copy as cp
import numpy as np
import scipy.optimize as opt
from pyquaternion import Quaternion
import shapely.geometry as geom
import shapely.ops as shops
import shapely.affinity as sha
from shapely.geometry import Point
import random


class IceCrystal:
    """A hexagonal prism representing a single ice crystal."""
    
    def __init__(self, length, width, center=[0, 0, 0], rotation=[0, 0, 0]):
        """Create an ice crystal.


        """
        # put together the hexagonal prism
        ca = length*2 # length is c axis radius from equiv. volume radius in lab module (ca = diameter)
        mf = width*2 # maximum face dimension (mf = diameter, width = radius)
        f = np.sqrt(3) / 4 # convenient number for hexagons
        x1 = ca / 2
        
        #creates 12 point arrays for hexagonal prisms
        if length < width:  #initialize plates so that the basal face is falling down
            self.points = np.array([(mf*f, -mf / 4, x1), (mf * f, mf / 4, x1),
                        (0, mf / 2, x1), (-mf * f, mf / 4, x1),
                        (-mf * f, -mf / 4, x1), (0, -mf/2, x1),
                        (mf * f, -mf / 4, -x1), (mf * f, mf / 4, -x1),
                        (0, mf / 2, -x1), (-mf * f, mf / 4, -x1),
                        (-mf * f, -mf / 4, -x1), (0, -mf/2, -x1)],
                       dtype=[('x', float), ('y', float), ('z', float)])
            
        else:  #initialize points so that columns fall prism face down
            self.points = np.array([(x1, -mf / 4, mf * f), (x1, mf / 4, mf * f),
                        (x1, mf / 2, 0), (x1, mf / 4, -mf * f),
                        (x1, -mf / 4, -mf * f), (x1, -mf/2, 0),
                        (-x1, -mf / 4, mf * f), (-x1, mf / 4, mf * f),
                        (-x1, mf / 2, 0), (-x1, mf / 4, -mf * f),
                        (-x1, -mf / 4, -mf * f), (-x1, -mf/2, 0)],
                       dtype=[('x', float), ('y', float), ('z', float)])

        # old IDL code
        # #          1       2      3      4       5      6       7       8       9       10      11      12
        # crystal=[[ca/2.  ,ca/2.  ,ca/2. ,ca/2. ,ca/2.  ,ca/2.  ,-ca/2. ,-ca/2. ,-ca/2. ,-ca/2. ,-ca/2. ,-ca/2.],$
        #          [-mf/4. ,mf/4.  ,mf/2. ,mf/4. ,-mf/4. ,-mf/2. ,-mf/4. ,mf/4.  ,mf/2.  ,mf/4.  ,-mf/4. ,-mf/2.],$
        #          [mf*f   ,mf*f   ,0.    ,-mf*f ,-mf*f  ,0.     ,mf*f   ,mf*f   ,0.     ,-mf*f  ,-mf*f  ,0.]]

        self.center = [0, 0, 0] # start the crystal at the origin
        self.rotation = Quaternion()        
        self.rotate_to(rotation) # rotate the crystal        
        self.maxz = self.points['z'].max()
        self.minz = self.points['z'].min()        
        self.move(center) # move the crystal
        self.tol = 10 ** -11 # used for some calculations
        
    def move(self, xyz):  #moves the falling crystal anywhere over the seed crystal/aggregate within the max bounds
        self.points['x'] += xyz[0]
        self.points['y'] += xyz[1]
        self.points['z'] += xyz[2]
        # update the crystal's center:
        for n in range(3):
            self.center[n] += xyz[n]
        # update max and min
        self.maxz += xyz[2]
        self.minz += xyz[2]
        
    def center_of_mass(self):  
        x = np.mean(self.points['x'])
        y = np.mean(self.points['y'])
        z = np.mean(self.points['z'])        
        return [x, y, z]

    def recenter(self):
        self.move([ -x for x in self.center_of_mass() ])


    def max(self, dim):
        return self.points[dim].max()

    def min(self, dim):
        return self.points[dim].min()
    
    def _rotate_mat(self, mat):  #when a crystal is rotated, rotate the matrix with it
        points = cp.copy(self.points)
        self.points['x'] = points['x'] * mat[0, 0] + points['y'] * mat[0, 1] + points['z'] * mat[0, 2]
        self.points['y'] = points['x'] * mat[1, 0] + points['y'] * mat[1, 1] + points['z'] * mat[1, 2]
        self.points['z'] = points['x'] * mat[2, 0] + points['y'] * mat[2, 1] + points['z'] * mat[2, 2]
        self.maxz = self.points['z'].max()
        self.minz = self.points['z'].min()
        # old IDL code:
        # pt1r1=[point(0),point(1)*cos(angle(0))-point(2)*sin(angle(0)),point(1)*sin(angle(0))+point(2)*cos(angle(0))]
        # pt1r2=[pt1r1(0)*cos(angle(1))+pt1r1(2)*sin(angle(1)),pt1r1(1),-pt1r1(0)*sin(angle(1))+pt1r1(2)*cos(angle(1))]
        # pt1r3=[pt1r2(0)*cos(angle(2))-pt1r2(1)*sin(angle(2)),pt1r2(0)*sin(angle(2))+pt1r2(1)*cos(angle(2)),pt1r2(2)]

    def _euler_to_mat(self, xyz):
        #Euler's rotation theorem, any rotation may be described using three angles.
        #takes angles and rotates coordinate system 
        [x, y, z] = xyz
        rx = np.matrix([[1, 0, 0], [0, np.cos(x), -np.sin(x)], [0, np.sin(x), np.cos(x)]])
        ry = np.matrix([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
        rz = np.matrix([[np.cos(z), -np.sin(z), 0], [np.sin(z), np.cos(z), 0], [0, 0, 1]])
        return rx * ry * rz

    def rotate_to(self, angles):
        # rotate to the orientation given by the 3 angles
        # get the rotation from the current position to the desired rotation
        current_rot = self.rotation
        rmat = self._euler_to_mat(angles)
        desired_rot = Quaternion(matrix=rmat)
        #Quarternion: a convenient mathematical notation for representing 
        #orientations and rotations of objects in three dimensions
        rot_mat = (desired_rot * current_rot.inverse).rotation_matrix
        self._rotate_mat(rot_mat)

        # update the crystal's center:
        xyz = ['x', 'y', 'z']
        for n in range(3):
            self.center[n] = self.points[xyz[n]].mean()

        # save the new rotation
        self.rotation = desired_rot

    def reorient(self, method='random', rotations=50):        
        #reorient a crystal x random rotations to mimic IPAS in IDL instead of automatically 
        #using the xrot and yrot from max area function in lap module
        #This function was only used for old runs
        #computation time is diminished using 'speedy' and bypassing this
        
        if method == 'IDL':
            # based on max_area2.pro from IPAS
            max_area = 0
            current_rot = self.rotation
            for i in range(rotations):
                [a, b, c] = [np.random.uniform(high=np.pi), np.random.uniform(high=np.pi), np.random.uniform(high=np.pi)]
                # for mysterious reasons we are going to rotate this 3 times
                rot1 = self._euler_to_mat([a, b, c])
                rot2 = self._euler_to_mat([b * np.pi, c * np.pi, a * np.pi])
                rot3 = self._euler_to_mat([c * np.pi * 2, a * np.pi * 2, b * np.pi * 2])
                desired_rot = Quaternion(matrix=rot1 * rot2 * rot3)
                rot_mat = (desired_rot * current_rot.inverse).rotation_matrix
                self._rotate_mat(rot_mat)
                new_area = self.projectxy().area
                if new_area > max_area:
                    max_area = new_area
                    max_rot = desired_rot
                # save our spot
                current_rot = desired_rot
            # rotate new crystal to the area-maximizing rotation
            rot_mat = (max_rot * current_rot.inverse).rotation_matrix
            self._rotate_mat(rot_mat)
            
        elif method == 'random':
            # same as IDL but only rotating one time, with a real
            # random rotation
            max_area = 0
            current_rot = self.rotation
            for i in range(rotations):
                desired_rot = Quaternion.random()
                rot_mat = (desired_rot * current_rot.inverse).rotation_matrix
                self._rotate_mat(rot_mat)
                new_area = self.projectxy().area
                if new_area > max_area:
                    max_area = new_area
                    max_rot = desired_rot
                # save our spot
                current_rot = desired_rot
            # rotate new crystal to the area-maximizing rotation
            rot_mat = (max_rot * current_rot.inverse).rotation_matrix
            self._rotate_mat(rot_mat)
            
        #self.rotation = Quaternion() # set this new rotation as the default

    def plot(self):
        # return a multiline object representing the edges of the prism
        lines = []
        hex1 = self.points[0:6]  #one basal face of a crystal
        hex2 = self.points[6:12]  #the other basal face
        
        # make the lines representing each hexagon
        for hex0 in [hex1, hex2]:
            lines.append(geom.LinearRing(list(hex0)))
            
        # make the lines connecting the two hexagons
        for n in range(6):
            lines.append(geom.LineString([hex1[n], hex2[n]]))

        return geom.MultiLineString(lines)  
        #shapely automatically plots in jupyter notebook, no figure initialization needed
    
    def projectxy(self):
        return geom.MultiPoint(self.points[['x', 'y']]).convex_hull

    def bottom(self):
        #return geometry of bottom side of falling crystal
        #to be used in connecting bottom of one crystal to the top of the other
        # getting the same points regardless of the orientation
        points = [ geom.Point(list(x)) for x in self.points ]
        lines = []
        faces = []
        
        p0 = self.points[0]
        p6 = self.points[6]
        if abs(p0['x'] - p6['x']) < self.tol and abs(p0['y'] - p6['y']) < self.tol:
            # if it's vertical, only return the hexagon faces
            # (for now)
            for hexagon in range(2):
                n0 = hexagon * 6
                for i in range(5):
                    n = n0 + i
                    lines.append(geom.LineString([self.points[n], self.points[n + 1]]))
                lines.append(geom.LineString([self.points[n0 + 5], self.points[n0]]))
            # get the hexagons only-- no rectangles
            for n in range(2):
                i = n * 6
                faces.append(geom.Polygon(list(self.points[i:(i + 6)])))
        elif abs(p0['z'] - p6['z']) < self.tol:
            # lying flat on its side-- not returning hexagon faces
            if len(np.unique(self.points['z'])) == 4:
                # It's rotated so that there's a ridge on the top, and
                # the sides are vertical. Don't return any vertical
                # rectangular sides
                for n in range(5):
                    p1 = self.points[n]
                    p2 = self.points[n + 1]
                    # is it a non-vertical rectangle?
                    if abs(p1['x'] - p2['x']) >= self.tol and abs(p1['y'] - p2['y']) >= self.tol:
                        faces.append(geom.Polygon([self.points[n], self.points[n + 1],
                                                   self.points[n + 7], self.points[n + 6]]))
                # get that last rectangle missed
                p1 = self.points[5]
                p2 = self.points[0]
                if abs(p1['x'] - p2['x']) >= self.tol and abs(p1['y'] - p2['y']) >= self.tol:
                    faces.append(geom.Polygon([self.points[5], self.points[0],
                                               self.points[6], self.points[11]]))
                # get the lines around the hexagons
                for hexagon in range(2):
                    n0 = hexagon * 6
                    for i in range(5):
                        n = n0 + i
                        p1 = self.points[n]
                        p2 = self.points[n + 1]
                        if abs(p1['x'] - p2['x']) >= self.tol and abs(p1['y'] - p2['y']) >= self.tol:
                            lines.append(geom.LineString([self.points[n], self.points[n + 1]]))
                    p1 = self.points[n0 + 5]
                    p2 = self.points[n0]
                    if abs(p1['x'] - p2['x']) >= self.tol and abs(p1['y'] - p2['y']) >= self.tol:
                        lines.append(geom.LineString([self.points[n0 + 5], self.points[n0]]))
                # get the between-hexagon lines
                for n in range(6):
                    lines.append(geom.LineString([self.points[n], self.points[n + 6]]))
                
                
            # returning only rectangles
            pass
        else:
            # return all the faces

            # get the lines around the hexagons
            for hexagon in range(2):
                n0 = hexagon * 6
                for i in range(5):
                    n = n0 + i
                    lines.append(geom.LineString([self.points[n], self.points[n + 1]]))
                lines.append(geom.LineString([self.points[n0 + 5], self.points[n0]]))
            # get the between-hexagon lines
            for n in range(6):
                lines.append(geom.LineString([self.points[n], self.points[n + 6]]))
            # get the hexagons
            for n in range(2):
                i = n * 6
                faces.append(geom.Polygon(list(self.points[i:(i + 6)])))
            # get the rectangles
            for n in range(5):
                faces.append(geom.Polygon([self.points[n], self.points[n + 1],
                                           self.points[n + 7], self.points[n + 6]]))
            # get that last rectangle I missed
            faces.append(geom.Polygon([self.points[5], self.points[0],
                                       self.points[6], self.points[11]]))
        
        # return the geometry representing the bottom side of the prism

        # # similar to projectxy
        # if self.rotation[1] == math.pi / 2:
        #     # it's vertical, so just return one of the hexagons
        #     points = self.points[0:6]

        # first find top and bottom hexagon

        # remove the top two points

        # make the lines

        # make the faces

        return {'lines': lines, 'points': points, 'faces': faces}

    def top(self):
        # return the geometry representing the top side of the prism

        # first find top and bottom hexagon

        # remove the bottom two points

        # make the lines

        # make the faces

        #return {'lines': lines, 'points': points, 'faces': faces}

        # temporary, until I fix these functions
        top = self.bottom()
        # # unless it's vertical
        # if self.rotation[1] / (np.pi / 2) % 4 == 1:
        #     top['points'] = [ geom.Point(list(x)) for x in self.points[0:6] ]
        #     top['lines'] = []
        #     for i in range(5): # get the points around each hexagon
        #         top['lines'].append(geom.LineString([self.points[i], self.points[i + 1]]))
        #     top['lines'].append(geom.LineString([self.points[5], self.points[0]]))
        # elif self.rotation[1] / (np.pi / 2) % 4 == 3:
        #     top['points'] = [ geom.Point(list(x)) for x in self.points[6:12] ]
        #     top['lines'] = []
        #     for i in range(5): # get the points around each hexagon
        #         top['lines'].append(geom.LineString([self.points[i + 6], self.points[i + 7]]))
        #         top['lines'].append(geom.LineString([self.points[11], self.points[6]]))
        
        return top

    def min_vert_dist(self, crystal2):
        # find the minimum directed distance to crystal2 traveling straight downward

        rel_area = self.projectxy().buffer(0).intersection(crystal2.projectxy().buffer(0))
        if not isinstance(rel_area, geom.Polygon):
            return None
        c1_bottom = self.bottom()
        c2_top = crystal2.top()
        mindiffz = self.maxz - crystal2.minz

        # 1) lines and lines
        # all the intersections are calculated in 2d so no need to
        # convert these 3d objects!
        c1_lines = [ l for l in c1_bottom['lines'] if l.intersects(rel_area) ]
        c2_lines = [ l for l in c2_top['lines'] if l.intersects(rel_area) ]
        for line1 in c1_lines:
            for line2 in c2_lines:
                if line1.intersects(line2):
                    # get (2D) point of intersection
                    xy = line1.intersection(line2)
                    if not isinstance(xy, geom.point.Point):
                        # parallel lines don't count
                        continue
                    # get z difference
                    # make sure the damn lines aren't vertical
                    xrange1 = line1.xy[0][1] - line1.xy[0][0]
                    xrange2 = line2.xy[0][1] - line2.xy[0][0]
                    if xrange1 != 0:
                        # interpolate using x value
                        z1 = line1.interpolate((xy.x - line1.xy[0][0]) / (xrange1), normalized=True).z
                    else:
                        # interpolate using y value
                        z1 = line1.interpolate((xy.y - line1.xy[1][0]) / (line1.xy[1][1] - line1.xy[1][0]), normalized=True).z
                    if xrange2 != 0:
                        z2 = line2.interpolate((xy.x - line2.xy[0][0]) / (xrange2), normalized=True).z
                    else:
                        z2 = line2.interpolate((xy.y - line2.xy[1][0]) / (line2.xy[1][1] - line2.xy[1][0]), normalized=True).z
                    diffz = z1 - z2
                    if diffz < mindiffz:
                        mindiffz = diffz
        
        # 2) points and surfaces
        c1_points = [ p for p in c1_bottom['points'] if p.intersects(rel_area) ]
        c2_faces = [ f for f in c2_top['faces'] if f.intersects(rel_area) ]
        for point in c1_points:
            for face in c2_faces:
                if point.intersects(face):
                    # get z difference
                    z1 = point.z
                    # find the equation of the polygon's plane, plug in xy
                    a = np.array(face.exterior.coords[0])
                    AB = np.array(face.exterior.coords[1]) - a
                    AC = np.array(face.exterior.coords[2]) - a
                    normal_vec = np.cross(AB, AC)
                    # find constant value
                    d = -np.dot(normal_vec, a)
                    z2 = -(point.x * normal_vec[0] + point.y * normal_vec[1] + d) / normal_vec[2]
                    diffz = z1 - z2
                    if diffz < mindiffz:
                        mindiffz = diffz
                    # the point can only intersect one face, so we're
                    # done with this one
                    #break
                    # ^ I should be able to do that but I have to fix my 'bottom' function first!
        
        # 3) surfaces and points
        c1_faces = [ f for f in c1_bottom['faces'] if f.intersects(rel_area) ]
        c2_points = [ p for p in c2_top['points'] if p.intersects(rel_area) ]
        for point in c2_points:
            for face in c1_faces:
                if point.intersects(face):
                    # get z difference
                    z2 = point.z # z2 this time!!!
                    # find the equation of the polygon's plane, plug in xy
                    a = np.array(face.exterior.coords[0])
                    AB = np.array(face.exterior.coords[1]) - a
                    AC = np.array(face.exterior.coords[2]) - a
                    normal_vec = np.cross(AB, AC)
                    # find constant value
                    d = -np.dot(normal_vec, a)
                    z1 = -(point.x * normal_vec[0] + point.y * normal_vec[1] + d) / normal_vec[2]
                    diffz = z1 - z2
                    if diffz < mindiffz:
                        mindiffz = diffz
                        # the point can only intersect one face, so we're
                        # done with this one
                    #break

        return mindiffz

    def write_obj(self, filename):
        f = open(filename, 'w')
        # write the vertices
        for n in range(12):
            f.write('v ' + ' '.join(map(str, self.points[n])) + '\n')
        # write the hexagons
        for n in range(2):
            f.write('f ' + ' '.join(map(str, range(n * 6 + 1, (n + 1) * 6 + 1))) + '\n')
        for n in range(5):
            f.write('f ' + ' '.join(map(str, [n + 1, n + 2, n + 8, n + 7])) + '\n')
        f.write('f ' + ' '.join(map(str, [6, 1, 7, 12])) + '\n')
        f.close()

        
class IceCluster:
    def __init__(self, crystal, size=1):
        # needed for bookkeeping:
        self.rotation = Quaternion()
        self.points = np.full((size, 12), np.nan,
                              dtype=[('x', float), ('y', float), ('z', float)])
        self.points[0] = crystal.points
        self.size = size
        self.ncrystals = 1
        # used for some calculations involving shapely objects
        self.tol = 10 ** -11
        # Used for the fit_ellipse function. I really do not like that
        # I have to set this so high, arrr.
        self.tol_ellipse = 10 ** -4.5
        # to store axis lengths
        self.major_axis = {}
        self.minor_axis = {}

    def crystals(self, i=None):
    # return a crystal with the same points and attributes as the
    # nth crystal in the cluster
        if i is None:
            crystals = []
            for n in range(self.ncrystals):
                cr = IceCrystal(1, 1)
                cr.points = self.points[n]
                cr.rotation = self.rotation
                cx = cr.points['x'].mean()
                cy = cr.points['y'].mean()
                cz = cr.points['z'].mean()
                cr.center = [cx, cy, cz]
                cr.maxz = cr.points['z'].max()
                cr.minz = cr.points['z'].min()
                crystals.append(cr)
            return crystals
        else:
            cr = IceCrystal(1, 1)
            cr.points = self.points[i]  #i = 0
            cr.rotation = self.rotation
            cx = cr.points['x'].mean()
            cy = cr.points['y'].mean()
            cz = cr.points['z'].mean()
            cr.center = [cx, cy, cz]
            cr.maxz = cr.points['z'].max()
            cr.minz = cr.points['z'].min()
            return cr

    def _add_crystal(self, crystal):
        n = self.ncrystals
        if n < self.size:
            self.points[n] = crystal.points
        else:
            self.points = np.append(self.points, [crystal.points], axis=0)
        self.ncrystals += 1
    
    def move(self, xyz):
        # move the entire cluster
        self.points['x'][:self.ncrystals] += xyz[0]
        self.points['y'][:self.ncrystals] += xyz[1]
        self.points['z'][:self.ncrystals] += xyz[2]

    def max(self, dim):
        return self.points[dim][:self.ncrystals].max()

    def min(self, dim):
        return self.points[dim][:self.ncrystals].min()

    # def _rotate(self, angles):
    #     [x, y, z] = [self.points[:self.ncrystals]['x'], self.points[:self.ncrystals]['y'], self.points[:self.ncrystals]['z']]
    #     [y, z] = [y * np.cos(angles[0]) - z * np.sin(angles[0]), y * np.sin(angles[0]) + z * np.cos(angles[0])]
    #     [x, z] = [x * np.cos(angles[1]) + z * np.sin(angles[1]), -x * np.sin(angles[1]) + z * np.cos(angles[1])]
    #     [x, y] = [x * np.cos(angles[2]) - y * np.sin(angles[2]), x * np.sin(angles[2]) + y * np.cos(angles[2])]
    #     # update the crystal's points:
    #     self.points['x'][:self.ncrystals] = x
    #     self.points['y'][:self.ncrystals] = y
    #     self.points['z'][:self.ncrystals] = z

    # def _rev_rotate(self, angles):
    #     angles = [-x for x in angles ]
    #     [x, y, z] = [self.points[:self.ncrystals]['x'], self.points[:self.ncrystals]['y'], self.points[:self.ncrystals]['z']]
    #     [x, y] = [x * np.cos(angles[2]) - y * np.sin(angles[2]), x * np.sin(angles[2]) + y * np.cos(angles[2])]
    #     [x, z] = [x * np.cos(angles[1]) + z * np.sin(angles[1]), -x * np.sin(angles[1]) + z * np.cos(angles[1])]
    #     [y, z] = [y * np.cos(angles[0]) - z * np.sin(angles[0]), y * np.sin(angles[0]) + z * np.cos(angles[0])]
    #     # update the crystal's points:
    #     self.points['x'][:self.ncrystals] = x
    #     self.points['y'][:self.ncrystals] = y
    #     self.points['z'][:self.ncrystals] = z

    def _euler_to_mat(self, xyz):
        # Euler's rotation theorem, any rotation may be described using three angles
        [x, y, z] = xyz
        rx = np.matrix([[1, 0, 0], [0, np.cos(x), -np.sin(x)], [0, np.sin(x), np.cos(x)]])
        ry = np.matrix([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
        rz = np.matrix([[np.cos(z), -np.sin(z), 0], [np.sin(z), np.cos(z), 0], [0, 0, 1]])
        return rx * ry * rz

    def _rotate_mat(self, mat):
        points = cp.copy(self.points)
        self.points['x'] = points['x'] * mat[0, 0] + points['y'] * mat[0, 1] + points['z'] * mat[0, 2]
        self.points['y'] = points['x'] * mat[1, 0] + points['y'] * mat[1, 1] + points['z'] * mat[1, 2]
        self.points['z'] = points['x'] * mat[2, 0] + points['y'] * mat[2, 1] + points['z'] * mat[2, 2]

    def rotate_to(self, angles):
        # rotate to the orientation given by the 3 angles
        
        # get the rotation from the current position to the desired
        # rotation
        current_rot = self.rotation
        rmat = self._euler_to_mat(angles)
        desired_rot = Quaternion(matrix=rmat)
        rot_mat = (desired_rot * current_rot.inverse).rotation_matrix
        self._rotate_mat(rot_mat)
        
        # save the new rotation
        self.rotation = desired_rot

    # def rotate_to(self, angles):
    #     # rotate the entire cluster

    #     # first get back to the original rotation
    #     if any(np.array(self.rotation) != 0):
    #         self._rev_rotate(self.rotation)

    #     # now add the new rotation
    #     self._rotate(angles)

    #     # save the new rotation
    #     self.rotation = angles

    def center_of_mass(self):  #of cluster
        x = np.mean(self.points[:self.ncrystals]['x'])
        y = np.mean(self.points[:self.ncrystals]['y'])
        z = np.mean(self.points[:self.ncrystals]['z'])        
        return [x, y, z]

    def recenter(self):
        self.move([ -x for x in self.center_of_mass() ])

    def plot(self):
        return geom.MultiLineString([ lines for crystal in self.crystals() for lines in crystal.plot() ])

    def _crystal_projectxy(self, n):
        return geom.MultiPoint(self.points[n][['x', 'y']]).convex_hull

    def _crystal_projectxz(self, n):
        return geom.MultiPoint(self.points[n][['x', 'z']]).convex_hull

    def _crystal_projectyz(self, n):
        return geom.MultiPoint(self.points[n][['y', 'z']]).convex_hull

    def projectxy(self):
        polygons = [ self._crystal_projectxy(n).buffer(0) for n in range(self.ncrystals) ]
        return shops.cascaded_union(polygons)

    def projectxz(self):
        polygons = [ self._crystal_projectxz(n) for n in range(self.ncrystals) ]
        return shops.cascaded_union(polygons)
    
    def projectyz(self):
        polygons = [ self._crystal_projectyz(n) for n in range(self.ncrystals) ]
        return shops.cascaded_union(polygons)
    
    def calculate_S_ratio(self, plates, crystal):
    #Calculate separation of crystals for further collection restriction        
        
        crystals1 = [self, crystal]
        dmaxclus = []
        dmaxnew = []
        n=0
        for i in crystals1:    
     
            x,y = list(self.projectxy().exterior.coords.xy)

            dinit = 0
            for j in range(len(x)):
                for l in range(len(x)):
                    d =(Point(x[l],y[l]).distance(Point(x[j],y[j])))  
                    if d > dinit:
                        dinit = d

                        if n == 0:
                            dmaxclus.append(d)
                        if n == 1:
                            dmaxnew.append(d)
                        xstart = l
                        ystart = l
                        xend = j
                        yend = j
    

            if n == 0:
                dmaxclus = max(dmaxclus)

            if n== 1:
                dmaxnew = max(dmaxnew)

            n+=1

        l = (self.projectxy().centroid).distance(crystal.projectxy().centroid) 
        S = 2*l/(dmaxclus+dmaxnew)

        if plates:
            lmax = 0.6*(dmaxclus+dmaxnew)/2 #S parameter can't be higher than 0.6 for plates
        else:
            lmax = 0.3*(dmaxclus+dmaxnew)/2 #S parameter can't be higher than 0.3 for columns

        return S, lmax
    
    def place_crystal(self, plates, crystal):
        #get max distance between centers of crystals for collection bounds (no tip to tip)
        #returns a random x/y value to place the new_crystal over the aggregate within the bounded circle
        S, lmax = self.calculate_S_ratio(plates, crystal)

        lmax_bound = self.projectxy().centroid.buffer(lmax)  #new crystal center can't be outside of this circle
        
        rand = random.uniform(0, 1.0)
        angle = rand * np.pi * 2
        rand2 = random.uniform(0, 1.0)
        radius = np.sqrt(rand2) * lmax
        originX = self.projectxy().centroid.xy[0]
        originY = self.projectxy().centroid.xy[1]

        x = originX + radius * np.cos(angle)
        y = originY + radius * np.sin(angle)
        
        random_loc = [x[0], y[0], 0]        
        
        return random_loc, lmax_bound

    def add_flow_tilt(self, crystal, lmax):
        
        center_aggx, center_aggy = self.projectxy().centroid.xy
        center_newx, center_newy = crystal.projectxy().centroid.xy
        #tilt fracs between 0 and 1
        tilt_fracx = ((np.abs(center_newx[0] - center_aggx[0]))/lmax)*(self.projectxy().area/crystal.projectxy().area)
        tilt_fracy = ((np.abs(center_newy[0] - center_aggy[0]))/lmax)*(self.projectxy().area/crystal.projectxy().area)
        '''
        #print('center agg',self.projectxy().centroid.xy)
        #print('center crys',crystal.projectxy().centroid.xy)
        print('top', (np.abs(center_newx[0] - center_aggx[0])))
        print('lmax', lmax)
        print('first frac center distancex',(np.abs(center_newx[0] - center_aggx[0])/lmax))
        print('first frac center distancey',(np.abs(center_newy[0] - center_aggy[0])/lmax))
        
        print('area ratio',self.projectxy().area/crystal.projectxy().area)
        print('tilt_fracx, tiltfracy',tilt_fracx, tilt_fracy)
        print('-------------')
        '''
        return tilt_fracx, tilt_fracy

    def add_crystal_from_above(self, crystal, lodge=0):
        # drop a new crystal onto the cluster
       
        # use the bounding box to determine which crystals to get
        xmax = max(crystal.projectxy().exterior.coords.xy[0])
        ymax = max(crystal.projectxy().exterior.coords.xy[1])
        xmin = min(crystal.projectxy().exterior.coords.xy[0])
        ymin = min(crystal.projectxy().exterior.coords.xy[1])
    
        close = np.all([self.points['x'][:self.ncrystals].max(axis=1) >= xmin,
                        self.points['x'][:self.ncrystals].min(axis=1) <= xmax,
                        self.points['y'][:self.ncrystals].max(axis=1) >= ymin,
                        self.points['y'][:self.ncrystals].min(axis=1) <= ymax], axis=0)   
 
        
        which_close = np.where(close)
        
        close_crystals = [ self.crystals(n) for n in which_close[0] ]            
        
        # see which crystals could actually intersect with the new crystal
        close_crystals = [ x for x in close_crystals if x.projectxy().intersects(crystal.projectxy()) ]
        
        # close_crystals = [ x for x in self.crystals() if x.projectxy().intersects(newpoly) ]
        if len(close_crystals) == 0:
            return False # the crystal missed!
        
        # we know highest hit is >= max(minzs), therefore the first
        # hit can't be below (max(minzs) - height(crystal))
        minzs = [ crystal2.minz for crystal2 in close_crystals ]
        first_hit_lower_bound = max(minzs) - (crystal.maxz - crystal.minz)
        # remove the low crystals, sort from highest to lowest
        close_crystals = [ x for x in close_crystals if x.maxz > first_hit_lower_bound ]
        close_crystals.sort(key=lambda x: x.maxz, reverse=True)

        # look to see where the new crystal hits the old ones
        mindiffz = crystal.minz - first_hit_lower_bound # the largest it can possibly be
        
        for crystal2 in close_crystals:
            if first_hit_lower_bound > crystal2.maxz:
                break # stop looping if the rest of the crystals are too low
            diffz = crystal.min_vert_dist(crystal2)
            if diffz is None:
                break
            
            #return diffz
            # update if needed
            if diffz < mindiffz:
                mindiffz = diffz
                first_hit_lower_bound = crystal.minz - mindiffz
                # take the highest hit, move the crystal to that level
        crystal.move([0, 0, -mindiffz - lodge])
        

        # append new crystal to list of crystals
        # self.crystals.append(crystal)
        self._add_crystal(crystal)
        # fin.
        return True

    def reorient(self, method='random', rotations=50):

        if method == 'IDL':
            # based on max_agg3.pro from IPAS
            max_area = 0
            current_rot = self.rotation
            for i in range(rotations):
                [a, b, c] = [np.random.uniform(high=np.pi / 4), np.random.uniform(high=np.pi / 4), np.random.uniform(high=np.pi / 4)]
                # for mysterious reasons we are going to rotate this 3 times
                rot1 = self._euler_to_mat([a, b, c])
                rot2 = self._euler_to_mat([b * np.pi, c * np.pi, a * np.pi])
                rot3 = self._euler_to_mat([c * np.pi * 2, a * np.pi * 2, b * np.pi * 2])
                desired_rot = Quaternion(matrix=rot1 * rot2 * rot3)
                rot_mat = (desired_rot * current_rot.inverse).rotation_matrix
                self._rotate_mat(rot_mat)
                new_area = self.projectxy().area
                if new_area >= max_area:
                    max_area = new_area
                    max_rot = desired_rot
                    
                # save our spot
                current_rot = desired_rot
            # rotate new crystal to the area-maximizing rotation
            rot_mat = (max_rot * current_rot.inverse).rotation_matrix
            self._rotate_mat(rot_mat)
        elif method == 'random':
            # same as schmitt but only rotating one time, with a real
            # random rotation
            max_area = 0
            current_rot = self.rotation
            for i in range(rotations):
               
                desired_rot = Quaternion.random()
                rot_mat = (desired_rot * current_rot.inverse).rotation_matrix
                self._rotate_mat(rot_mat)
                new_area = self.projectxy().area
                if new_area >= max_area:
                    max_area = new_area
                    max_rot = desired_rot
                    
                # save our spot
                current_rot = desired_rot
            # rotate new crystal to the area-maximizing rotation
            rot_mat = (max_rot * current_rot.inverse).rotation_matrix
            self._rotate_mat(rot_mat)

        #self.rotation = Quaternion()
        
        # elif method == 'bh':
        #     # use a basin-hopping algorithm to look for the optimal rotation
        #     def f(x):
        #         # yrot = np.arccos(x[1]) - np.pi/2
        #         # self.rotate_to([x[0], yrot, 0])
        #         self.rotate_to([x[0], x[1], 0])
        #         return -self.projectxy().area
        #     # lbfgsb_opt = {'ftol': 1, 'maxiter': 5}
        #     # min_kwargs = {'bounds': [(0, np.pi), (0, np.pi)], 'options': lbfgsb_opt}
        #     # # min_kwargs = {'bounds': [(0, np.pi), (0, np.pi)]}
        #     # opt_rot = opt.basinhopping(f, x0=[np.pi/2, np.pi/2], niter=15, stepsize=np.pi / 7,
        #     #                            interval=5, minimizer_kwargs=min_kwargs)
        #     lbfgsb_opt = {'ftol': 1, 'maxiter': 0, 'maxfun': 4}
        #     min_kwargs = {'bounds': [(0, np.pi), (-0.99, 0.99)], 'options': lbfgsb_opt}
        #     # min_kwargs = {'bounds': [(0, np.pi), 0, np.pi)]}
        #     opt_rot = opt.basinhopping(f, x0=[np.pi/2, np.pi/2], niter=30, stepsize=np.pi / 4,
        #                                interval=10, minimizer_kwargs=min_kwargs)
        #     # xrot = opt_rot.x[0]
        #     # yrot = np.arccos(opt_rot.x[1]) - np.pi / 2
        #     [xrot, yrot] = opt_rot.x
        #     # area at rotation + pi is the same, so randomly choose to
        #     # add those
        #     # if np.random.uniform() > .5:
        #     #     xrot += np.pi
        #     # if np.random.uniform() > .5:
        #     #     yrot += np.pi
        #     zrot = np.random.uniform(high=2 * np.pi) # randomly choose z rotation
        #     self.rotate_to([xrot, yrot, zrot])
        #     self.rotation = [0, 0, 0]
        #     return opt_rot

        # elif method == 'diff_ev':
        #     def f(x):
        #         # yrot = np.arccos(x[1]) - np.pi/2
        #         # self.rotate_to([x[0], yrot, 0])
        #         self.rotate_to([x[0], x[1], 0])
        #         return -self.projectxy().area
        #     opt_rot = opt.differential_evolution(f, [(0, np.pi), (-1, 1)],
        #                                          maxiter=10, popsize=15)
        #     # xrot = opt_rot.x[0]
        #     # yrot = np.arccos(opt_rot.x[1]) - np.pi / 2
        #     [xrot, yrot] = opt_rot.x
        #     zrot = np.random.uniform(high=2 * np.pi) # randomly choose z rotation
        #     self.rotate_to([xrot, yrot, zrot])
        #     self.rotation = [0, 0, 0]
        #     return opt_rot

    def _get_moments(self, poly):
        # get 'mass moments' for this cluster's 2D polygon using a
        # variation of the shoelace algorithm
        xys = poly.exterior.coords.xy
        npoints = len(xys[0])
        # values for the three points-- point[n], point[n+1], and
        # (0,0)-- making up triangular slices from the origin to the
        # edges of the polygon
        xmat = np.array([xys[0][0:-1], xys[0][1:], np.zeros(npoints - 1)]).transpose()
        ymat = np.array([xys[1][0:-1], xys[1][1:], np.zeros(npoints - 1)]).transpose()
        # arrange the points in left-center-right order
        x_order = np.argsort(xmat, axis=1)
        ordered_xmat = xmat[np.array([range(npoints - 1)]).transpose(), x_order]
        ordered_ymat = ymat[np.array([range(npoints - 1)]).transpose(), x_order]
        xl = ordered_xmat[:, 0]
        xm = ordered_xmat[:, 1]
        xr = ordered_xmat[:, 2]
        yl = ordered_ymat[:, 0]
        ym = ordered_ymat[:, 1]
        yr = ordered_ymat[:, 2]
        # which slices have areas on the left and right sides of the
        # middle point? Ignore values smaller than 'tol' so we don't
        # run into terrible problems with division.
        left = xm - xl > self.tol_ellipse
        right = xr - xm > self.tol_ellipse
        # slope and intercept of line connecting left and right points
        has_area = xr != xl
        m3 = np.zeros(npoints - 1)
        m3[has_area] = (yr[has_area] - yl[has_area]) / (xr[has_area] - xl[has_area])
        b3 = -xl * m3 + yl
        # the y coordinate of the line connecting the left and right
        # points at the x position of the middle point
        m3_mid = yl + m3 * (xm - xl)
        # is the midpoint above or below that line?
        mid_below = ym < m3_mid
        # line connecting left and middle point (where applicable)
        m1 = (ym[left] - yl[left]) / (xm[left] - xl[left])
        b1 = -xl[left] * m1 + yl[left]
        # line connecting middle and right point (where applicable)
        m2 = (yr[right] - ym[right]) / (xr[right] - xm[right])
        b2 = -xr[right] * m2 + yr[right]
        # now that we have the points in a nice format + helpful
        # information we can calculate the integrals of the slices
        xx = np.zeros(npoints - 1)
        xy = np.zeros(npoints - 1)
        yy = np.zeros(npoints - 1)
        dxl = (xm[left] - xl[left])
        dx2l = (xm[left] ** 2 - xl[left] ** 2)
        dx3l = (xm[left] ** 3 - xl[left] ** 3)
        dx4l = (xm[left] ** 4 - xl[left] ** 4)
        dxr = (xr[right] - xm[right])
        dx2r = (xr[right] ** 2 - xm[right] ** 2)
        dx3r = (xr[right] ** 3 - xm[right] ** 3)
        dx4r = (xr[right] ** 4 - xm[right] ** 4)
        # x^2
        xx[left] = dx4l * (m1 - m3[left]) / 4 +\
                   dx3l * (b1 - b3[left]) / 3
        xx[right] += dx4r * (m2 - m3[right]) / 4 +\
                     dx3r * (b2 - b3[right]) / 3
        # x*y
        xy[left] = dx4l * (m1 ** 2 - m3[left] ** 2) / 8 +\
                   dx3l * (b1 * m1 - b3[left] * m3[left]) / 3 +\
                   dx2l * (b1 ** 2 - b3[left] ** 2) / 4
        xy[right] += dx4r * (m2 ** 2 - m3[right] ** 2) / 8 +\
                     dx3r * (b2 * m2 - b3[right] * m3[right]) / 3 +\
                     dx2r * (b2 ** 2 - b3[right] ** 2) / 4
        # y^2
        yy[left] = dx4l * (m1 ** 3 - m3[left] ** 3) / 12 +\
                   dx3l * (b1 * m1 ** 2 - b3[left] * m3[left] ** 2) / 3 +\
                   dx2l * (b1 ** 2 * m1 - b3[left] ** 2 * m3[left]) / 2 +\
                   dxl * (b1 ** 3 - b3[left] ** 3) / 3
        yy[right] += dx4r * (m2 ** 3 - m3[right] ** 3) / 12 +\
                     dx3r * (b2 * m2 ** 2- b3[right] * m3[right] ** 2) / 3 +\
                     dx2r * (b2 ** 2 * m2 - b3[right] ** 2 * m3[right]) / 2 +\
                     dxr * (b2 ** 3 - b3[right] ** 3) / 3
        # if the middle point was below the other points, multiply by
        # minus 1
        xx[mid_below] *= -1
        xy[mid_below] *= -1
        yy[mid_below] *= -1
        # find out which slices were going clockwise, and make those
        # negative        
        points = np.array([xys[0], xys[1]]).transpose()
        cross_prods = np.cross(points[:-1], points[1:])
        clockwise = cross_prods < 0
        xx[clockwise] *= -1
        xy[clockwise] *= -1
        yy[clockwise] *= -1
        # add up the totals across the entire polygon
        xxtotal = np.sum(xx)
        yytotal = np.sum(yy)
        xytotal = np.sum(xy)
        # and if the points were in clockwise order, flip the sign
        if np.sum(cross_prods) < 0:
            xxtotal *= -1
            yytotal *= -1
            xytotal *= -1
        # also need to account for the holes, if they exist
        for linestring in list(poly.interiors):
            hole = geom.Polygon(linestring)
            hole_moments = self._get_moments(hole)
            xxtotal -= hole_moments[0]
            yytotal -= hole_moments[1]
            xytotal -= hole_moments[2]
        return [xxtotal, yytotal, xytotal]

    def fit_ellipse(self, dims):
        # Emulating this function, but for polygons in continuous
        # space rather than blobs in discrete space:
        # http://www.idlcoyote.com/ip_tips/fit_ellipse.html
     
        if dims == [['x','y']]:
            poly = self.projectxy()
        if dims == [['x','z']]:
            poly = self.projectxz()
        if dims == [['y','z']]:
            poly = self.projectyz()
        
        xy_area = poly.area
        
        # center the polygon around the centroid
        centroid = poly.centroid
        poly = sha.translate(poly, -centroid.x, -centroid.y)

        # occasionally we get multipolygons
        if isinstance(poly, geom.MultiPolygon):
            xx = 0
            yy = 0
            xy = 0
            for poly2 in poly:
                moments = self._get_moments(poly2)
                xx += moments[0] / xy_area
                yy += moments[1] / xy_area
                xy -= moments[2] / xy_area
        else:
            moments = self._get_moments(poly)
            xx = moments[0] / xy_area
            yy = moments[1] / xy_area
            xy = -moments[2] / xy_area

        # get fit ellipse axes lengths, orientation, center
        m = np.matrix([[yy, xy], [xy, xx]])
        evals, evecs = np.linalg.eigh(m)
        semimajor = np.sqrt(evals[0]) * 2
        semiminor = np.sqrt(evals[1]) * 2
        major = semimajor * 2
        minor = semiminor * 2
       
        evec = np.squeeze(np.asarray(evecs[0]))
        orientation = np.arctan2(evec[1], evec[0]) * 180 / np.pi

        ellipse = {'xy': [centroid.x, centroid.y], 'width': minor,
                   'height': major, 'angle': orientation}
        #print('crystals',self.ncrystals)    
        #print('ell',ellipse['height'])    
        return ellipse
    
    
    def mvee(self, tol = 0.001):        
        pi = np.pi
        sin = np.sin
        cos = np.cos


            """
            Finds the ellipse equation in "center form"
            (x-c).T * A * (x-c) = 1
            """
            N, d = points.shape
            Q = np.column_stack((points, np.ones(N))).T
            err = tol+1.0
            u = np.ones(N)/N
            while err > tol:
                # assert u.sum() == 1 # invariant
                X = np.dot(np.dot(Q, np.diag(u)), Q.T)
                M = np.diag(np.dot(np.dot(Q.T, la.inv(X)), Q))
                jdx = np.argmax(M)
                step_size = (M[jdx]-d-1.0)/((d+1)*(M[jdx]-1.0))
                new_u = (1-step_size)*u
                new_u[jdx] += step_size
                err = la.norm(new_u-u)
                u = new_u
            c = np.dot(u,points)

            A = la.inv(np.dot(np.dot(points.T, np.diag(u)), points)
                       - np.multiply.outer(c,c))/d
            return A, c
    
    def spheroid_axes(self): 
        A, c = self.mvee()
        U, D, V = la.svd(A)
        rx, ry, rz = 1./np.sqrt(D)
        return rx, ry, rz
        
    def plot_ellipse(self, dims):
        import matplotlib.pyplot as plt
        import random
        from matplotlib.patches import Ellipse
        from descartes import PolygonPatch
        import operator
        
        #Only (x,z), (y,z), and (x,y) needed/allowed for dimensions
        #Depth works for both side views (x,z) and (y,z)
     
        if dims == [['x','z']]:
            #self.rotate_to([np.pi / 2, 0, 0])
            poly = self.projectxz()
        elif dims == [['y','z']]:
            #self.rotate_to([np.pi / 2, np.pi / 2, 0])
            poly = self.projectyz()
        elif dims == [['x','y']]:  #this is the only projection used in the aggregate aspect ratio calculation
            #self.rotate_to([0, 0, 0])
            poly = self.projectxy()
        else:
            print('Not a valid dimension')
       
        
        params = self.fit_ellipse(dims)
        ellipse = Ellipse(**params)
        
        fig = plt.figure(0)
        ax = fig.add_subplot(111)
        ax.add_artist(ellipse)
        ellipse.set_alpha(.9)  #opacity
        ellipse.set_facecolor('darkorange')
        #if isinstance(poly, geom.multipolygon.MultiPolygon):
        #    for poly2 in poly:
        #        x, y = poly2.exterior.xy
                #ax.plot(x, y, color = 'green', linewidth = 3)
        #else:
        #    x, y = poly.exterior.xy
            #ax.plot(x, y, color = 'green', linewidth = 3)
        
        #maxdim = max([params['width'], params['height']]) / 2
        #ax.set_xlim([-maxdim + params['xy'][0], maxdim + params['xy'][0]])
        #ax.set_ylim([-maxdim + params['xy'][1], maxdim + params['xy'][1]])
        
        for l in range(len(dims)):
            
            crysmaxz = []
            crysminz = []
            maxzinds = []
            minzinds = []
            for i in range(self.ncrystals):    
                hex1pts = self.points[dims[l]][i][0:6]  #first basal face
                poly1 = geom.Polygon([[p[0], p[1]] for p in hex1pts]) #make it into a polygon to plot
                hex2pts = self.points[dims[l]][i][6:12]  #second basal face
                poly2 = geom.Polygon([[p[0], p[1]] for p in hex2pts])
                x1,y1 = poly1.exterior.xy  #array of xy points
                x2,y2 = poly2.exterior.xy
                
                if i == 1:
                    color = 'navy'
                    zorder = 3
                else:
                    color = 'darkgreen'
                    zorder = 4
                for n in range(7):  #plot the prism face lines
                    x = [x1[n],x2[n]]
                    y = [y1[n],y2[n]]
                    ax.plot(x,y, color = color, zorder = zorder, linewidth = '2')

                #polypatch1 = PolygonPatch(poly1, fill=True, zorder = 1)     
                #polypatch2 = PolygonPatch(poly2, fill=True, zorder = 1)
                ax.plot(x1,y1, color = color, zorder = 2, linewidth = '2') #edges of polygons
                ax.plot(x2,y2, color = color, zorder = 4, linewidth = '2') 
                #ax.add_patch(polypatch1)
                #ax.add_patch(polypatch2)
                
                #for plotting depth line segment:
                crysminz.append(self.points['z'][i].min())
                crysmaxz.append(self.points['z'][i].max())                
                minzinds.append(np.argmin(self.points['z'][i]))  #index of min pt for every xtal
                maxzinds.append(np.argmax(self.points['z'][i])) #index of max pt
                
            maxcrysind, self.maxz = max(enumerate(crysmaxz), key=operator.itemgetter(1))  #overall max btwn xtals
            mincrysind, self.minz = min(enumerate(crysminz), key=operator.itemgetter(1))
            xdepthmin = self.points[dims[l]][mincrysind][minzinds[mincrysind]]  
            xdepthmax = self.points[dims[l]][maxcrysind][maxzinds[maxcrysind]]

            ax.plot(xdepthmin, self.minz, 'ko', linewidth = '4')
            ax.plot(xdepthmax, self.maxz, 'ko', linewidth = '4')
            depthlinex = [0,0]
            depthliney = [self.minz, self.maxz]
            ax.plot(depthlinex,depthliney, 'k', linewidth = '4')
            
            ######## plot major and minor axes ############
            
            maxdim = max([params['width'], params['height']]) / 2  #major axis

            #ax.set_xlim([-maxdim + params['xy'][0], maxdim + params['xy'][0]])
            #ax.set_ylim([-maxdim + params['xy'][1], maxdim + params['xy'][1]])

            leftverticex = params['xy'][0]-params['width']/2
            leftverticey = params['xy'][1]
            rightverticex = params['xy'][0]+params['width']/2
            rightverticey = params['xy'][1]
            #plt.plot(leftverticex, leftverticey, 'ro', markersize = 5)  #original vertices if no angle
            #plt.plot(rightverticex, rightverticey, 'ro', markersize = 5)
            #plt.plot(params['xy'][0], params['xy'][1], 'wo', markersize = 7)


            radangle = params['angle']*np.pi/180  
            #orientation angle of ellipse
            
            #rotate axis points and reposition if off center
            newxleft = ((leftverticex - params['xy'][0])*np.cos(radangle)-\
                        (leftverticey-params['xy'][1])*np.sin(radangle)) + params['xy'][0]

            newxright = ((rightverticex - params['xy'][0])*np.cos(radangle)-\
                         (rightverticey - params['xy'][1])*np.sin(radangle)) + params['xy'][0]  

            newyleft = ((leftverticex - params['xy'][0])*np.sin(radangle)+\
                        (leftverticey-params['xy'][1])*np.cos(radangle)) + params['xy'][1]

            newyright = ((rightverticex - params['xy'][0])*np.sin(radangle)+\
                        (rightverticey-params['xy'][1])*np.cos(radangle)) + params['xy'][1]

            newx = [newxleft, newxright]
            newy = [newyleft, newyright]
            ax.plot(newx, newy, color ='white', linewidth = 3)  #major/minor axis lines
            ax.plot(newx, newy, 'wo', markersize = 7)

            radangle1 = params['angle']*np.pi/180 + np.pi/2
            radangle = radangle1
            leftverticex = params['xy'][0]-params['height']/2
            rightverticex = params['xy'][0]+params['height']/2

            newxleft = ((leftverticex - params['xy'][0])*np.cos(radangle)-\
                        (leftverticey-params['xy'][1])*np.sin(radangle)) + params['xy'][0]

            newxright = ((rightverticex - params['xy'][0])*np.cos(radangle)-\
                         (rightverticey - params['xy'][1])*np.sin(radangle)) + params['xy'][0]  

            newyleft = ((leftverticex - params['xy'][0])*np.sin(radangle)+\
                        (leftverticey-params['xy'][1])*np.cos(radangle)) + params['xy'][1]

            newyright = ((rightverticex - params['xy'][0])*np.sin(radangle)+\
                        (rightverticey-params['xy'][1])*np.cos(radangle)) + params['xy'][1]

            newx = [newxleft, newxright]
            newy = [newyleft, newyright]
            ax.plot(newx, newy, color ='white', linewidth = 3)
            ax.plot(newx, newy, 'wo', markersize = 2)


            ax.set_aspect('equal', 'datalim')

        return params
    '''
    def write_obj(self, filename):
        f = open(filename, 'w')

        faces = []
        for i, crystal in enumerate(self.crystals()):
            nc = i * 12
            # write the vertices
            for n in range(12):
                f.write('v ' + ' '.join(map(str, crystal.points[n])) + '\n')
            # write the hexagons
            for n in range(2):
                coords = range(n * 6 + 1 + nc, (n + 1) * 6 + 1 + nc)
                faces.append('f ' + ' '.join(map(str, coords)))
            # write the rectangles
            for n in range(5):
                coords = [n + 1 + nc, n + 2 + nc, n + 8 + nc, n + 7 + nc]
                faces.append('f ' + ' '.join(map(str, coords)))
            # write the last rectangle I missed
            coords = [nc + 6, nc + 1, nc + 7, nc + 12]
            faces.append('f ' + ' '.join(map(str, coords)))
        f.write('\n'.join(faces))
        f.close()
    
    def intersect(self):
        from operator import itemgetter
        # return a multiline object representing the edges of the prism
        hex_cntmax = np.empty([self.ncrystals],dtype='object')
        hex_cntmin = np.empty([self.ncrystals],dtype='object')
        for c in range(self.ncrystals):
            # make a line connecting the two hexagons at the max x value         
            dim = ['y','z']
            hex1pts = self.points[dim][c,0:6]   
            hex2pts = self.points[dim][c,6:12]   
            hex1max = max(self.points[c,0:6][dim],key=itemgetter(0))
            hex2max = max(self.points[c,6:12][dim],key=itemgetter(0))
            hex_cntmax[c] = geom.LineString((hex1max, hex2max))           
            print(hex1pts)
            print(hex1max)
            print(hex2pts)
            print(hex2max)
            hex1min = min(self.points[c,0:6][dim],key=itemgetter(0))
            hex2min = min(self.points[c,6:12][dim],key=itemgetter(0))

            hex_cntmin[c] = geom.LineString((hex1min, hex2min))           
            
        intersect = False
        
        max_intersect = hex_cntmax[0].intersects(hex_cntmax[1])
        min_intersect = hex_cntmin[0].intersects(hex_cntmin[1])
        minmax_intersect = hex_cntmin[0].intersects(hex_cntmax[1])
        maxmin_intersect = hex_cntmax[0].intersects(hex_cntmin[1])
        if max_intersect==True:
            intersect = True
            print('max')
        if min_intersect==True:
            intersect = True
            print('min')
        if minmax_intersect==True:
            intersect = True
            print('minmax')
        if maxmin_intersect == True:
            intersect = True
            print('maxmin')
        print(intersect)
        
        return intersect
    
    '''
    def aspect_ratio(self, method, minor):
        # rotation = self.rotation
        
        #get depth measurement in z
       
        self.maxz = self.points['z'][:self.ncrystals].max()        
        self.minz = self.points['z'][:self.ncrystals].min()
        #print(self.maxz, self.minz, self.maxz-self.minz)
        self.depth = self.maxz-self.minz
        
        self.rotate_to([0, 0, 0])

        # getting ellipse axes from 3 perspectives
        ellipse = {}        
        dims = [['x','y']]
        ellipse['z'] = self.fit_ellipse(dims)
        dims = [['x','z']]
        self.rotate_to([np.pi / 2, 0, 0])
        ellipse['y'] = self.fit_ellipse(dims)
        dims = [['y','z']]
        self.rotate_to([np.pi / 2, np.pi / 2, 0])
        ellipse['x'] = self.fit_ellipse(dims)
        
        # put the cluster back
        self.rotate_to([0, 0, 0])
        
        for dim in ellipse.keys():  
            
            self.major_axis[dim] = max(ellipse[dim]['height'], ellipse[dim]['width'])
            self.minor_axis[dim] = min(ellipse[dim]['height'], ellipse[dim]['width'])
        
        if minor == 'minorxy':
            if method == 1:
                return max(self.major_axis.values()) / max(self.minor_axis.values())
            elif method == 'plate':
                return max(self.minor_axis['x'],self.minor_axis['y']) / self.major_axis['z']
            elif method == 'column':
                return self.major_axis['z'] / max(self.minor_axis['x'],self.minor_axis['y'])
        elif minor == 'depth': #use depth as minor dimension of aggregate
            if method == 1:
                return max(self.major_axis.values()) / max(self.minor_axis.values())
            elif method == 'plate':                
                #print(self.depth, self.major_axis['z'], self.depth/self.major_axis['z'])         
                return self.depth / self.major_axis['z']
            elif method == 'column':
                #print(self.major_axis['z'], self.depth, self.major_axis['z']/self.depth)
                return self.major_axis['z'] / self.depth
        
    def aspect_ratio_2D(self):
        
        # getting fit ellipse in z orientation
        ellipse = {}
        
        dims = [['x','y']]
        ellipse['z'] = self.fit_ellipse(dims)
      
        return self.minor_axis['z']/2 / self.major_axis['z']/2
        