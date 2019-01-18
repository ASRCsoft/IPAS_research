# crystal_dda
A package for generating branched planar crystal polygons and input text files for DDA code

# Installation
Clone package to local machine. Then cd into directory and install with:

```
python setup.py install --user
```

Numpy and Matplotlib are required dependencies.

# Usage

The main function to create a branched planar dda file is ```branched_planar_dda``` which we import with:

```python
from crystal_dda.crystal_dda import branched_planar_dda
```

Next, we set the values defining the branched planar crystal shape appropriately. Let's start with the a-axis information. Each crystal has core of area fraction 1 at its center of size ```ac```. We also need a maximum a-axis length ```amax``` to define the shape at full size. And finally, we need an a-axis length ```a``` that subsets the full-size crystal within a hexagon of a-axis length ```a```. So we set the variables: 

```python
a = 1.8
amax = 3.
ac = 0.2
```

The other variables we need to set are the fractional (and thus all range between 0-1) quantities that further determine the shape. ```fb``` is the area coverage fraction of the sub-branches. ```fg``` is the fractional distance between ```ac``` and ```amax``` that gives the inscribing hexagon of length ```ag```, where ```ag = fg*amax+(1.-fg)*ac```. The region within this inscribing hexagon contains the core at its center and sub-branched beyond ```ac``` so that the area fraction between ```ac``` and ```ag``` is approximately equal to ```fb``` (the main branches and the discrete number of branches cause the true area fraction of the crystal to deviate from the analytical value; see the ```test_afrac_accuracy``` in the examples directory for more information). Finally, we have the linear fraction of the crystal tips at ```amax``` along the edge of the crystal ```ft```. We set these variables below with:

```python
fb = 0.6
ft = 0.4
fg = 0.5
```

We also set the number of sub-branches along one of the 6 main branches of the particle with:
```python
nsb = 5
```
Note that the width of the sub-branches and main branches are determined from the above constants, with the main-branch width equal to the sub-branch width unless it is greater than ```0.25*ac```; in this case the main branch width is ```0.25*ac```.

The last thing we need as input to the function to create the DDA file are the maximum number of dipoles along the x axis (```numxdip```; i.e.,  the a axis) and the z axis (```numzdip```; i.e., the c axis). These two variables define the aspect ratio of the crystal and the resolution in DDA. We set them, for example, with:

```python
numxdip = 300
numzdip = 7
```
Finally, we create the DDA input file with:

```python
fname, afrac = branched_planar_dda(a, amax, ac, ft, fb, fg, nsb, nxp, nzp)
```

where ```fname``` is the name of the input file that we just created and ```afrac``` is the analytical area fraction of the crystal. See the examples directory for more information. We can now run the DDA code with this file as an input to calculate the scattering.

![alt text](https://github.com/rskschrom/crystal_dda/blob/master/examples/crystal.gif)

The gif animation below shows DDA files produced with the ```branched_planar_dda()``` function above for ```a``` sizes of a crystal ranging from ```ac``` to ```amax``` for that crystal. The red outline indicates the core hexagon of a-axis length ```ac```, the black outline indicates the inscribing hexagon with a-axis length ```ag```, and the green outline indicates the bounding star shape of the crystal defined by points connecting the center of the main branches to the edge of the crystal tips to the center of each of the edges of the inscribing hexagon.
