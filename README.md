# The Ice Particle and Aggregate Simulator (IPAS)
A Python implementation of the [Ice Particle Aggregate Simulator](http://www.carlgschmitt.com/Microphysics.html)

### Prerequisites
It is recomended to clone this repository and put into a virtual environment due to the packages that will need to be installed.

### Dependent on:
*numpy
*pandas
*itertools
*itemgetter
*shapely
*descartes
*scipy
*random 
*cloudpickle
*multiprocessing
*functools
*dask
..and others..

Recommended to run in a python virtual environment

## Crystals (monomers)
```python
import numpy as np
from ipas import IceCrystal as crys

# create a hexagonal crystal centered at (1,0,0)
crystal = crys.IceCrystal(length=4, width=6, center=[1, 0, 0])
crystal.points # get a numpy array containing the crystal vertices
# rotate the crystal 45 degrees around the y-axis and 90 degrees
# around the z-axis
crystal.rotate_to([0,np.pi/4,np.pi/2])
# return a shapely MultiLineString representing the crystal edges,
# which plots automatically in a jupyter notebook
crystal.plot()
```
![crystal](https://user-images.githubusercontent.com/4205859/27136311-01852f9a-50e9-11e7-8f10-db348cdddd3a.png)
```python
# project the crystal onto the xy plane, returning a shapely Polygon
crystal.projectxy()
```
![crystal_projection](https://user-images.githubusercontent.com/4205859/27136458-5f9d07ba-50e9-11e7-8665-f230dc932c6a.png)

## Cluster (aggregate)

```python
from ipas import IceCluster as clus
# use the crystal to start a cluster
cluster = clus.IceCluster(crystal)
# add a new crystal to the cluster
crystal2 = crys.IceCrystal(length=4, width=6)
cluster.add_crystal_from_above(crystal2)
cluster._add_crystal(crystal2)
cluster.rotate_to([np.pi/30,np.pi/30,0])
cluster.plot()
```
![clus](https://github.com/vprzybylo/IPAS_parallel/blob/editing_branch/project.png)
```python
cluster.plot_ellipse([['x','y']])
```

![fit_ellipse](https://github.com/vprzybylo/IPAS_parallel/blob/editing_branch/fit_ellipse.png)

## Deployment

This code can be run on a supercomputer and scaled up by relying on the dask delayed option making use of multiple workers and cores.

## Authors

* Carl Schmitt, Vanessa Przybylo, William May, Kara Sulia 

## Acknowledgments
* Based on work by:
Carl Schmitt: 
* Schmitt,  C. G. and A. J. Heymsfield,  2010:  The dimensional characteristics of ice crystal aggregates from fractal geometry. J. Atmos. Sci., 1605–1616, doi:10.1175/2009JAS3187.1
 
* Schmitt, C. G. and A. J. Heymsfield, 2014:  Observational quantification of the separation of  simple  and  complex  atmospheric ice  particles.Geophys.  Res.  Lett.,  1301–1307,  doi:80210.1002/2013GL058781
* Nima Moshtagh for the fit-ellipse function

* http://www.mathworks.com/matlabcentral/fileexchange/9542
