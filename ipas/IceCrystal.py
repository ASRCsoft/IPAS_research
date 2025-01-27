"""Class representing ice crystals (monomers)"""

import copy as cp
import numpy as np
from pyquaternion import Quaternion
import shapely.geometry as geom
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
        #using the xrot and yrot from max area function in lab module
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

       