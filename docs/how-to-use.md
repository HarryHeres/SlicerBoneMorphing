# Guide on how to use the module
### Visualization
Here, you can check whether you want to visualize results of each pipeline steps using Open3D.

### Input section
- Source = The mean model, i.e. a full humerus
- Target = Partial model to be reconstructed

### Preprocessing section
Before the generation process, it is possible to preprocess the models.
First is the option to downsample the models in order to relieve some of the computation.
You can configure the amount of downsampling by the following parameter:
- **Downsampling voxel size**
    - If set to 0.0, no downsampling is performed

After the downsampling, we compute the normals of the point cloud. 
The computation needs a radius for which the normals are calculated and maximum number of neighbours. 
These can be adjusted with the following parameters: 
- **Normals estimation radius** - maximum radius in which points are considered neighbouring
- **Normals estimation max neighbours** - maximum number of neighbours taken into account 

Also, we need to calculate a *(Fast) point feature histogram* in order to encode the local geometric properties of the models. 
This method uses the following parameters:
- **FPFH search radius** - maximum radius in which points are considered neighbouring
- **FPFH max neighbours** - maximum number of neighbours taken into account 

#### Registration
At this moment we have our models preprocessed and ready for the next step, which is the registration.
Here we calculate the rigid alignment of the models.
We use the **RANSAC** (Random Sample Consensus) for the computation.
The behaviour of this algorithm can be adjusted by the following parameters: 
- **Max iterations** - maximum number of iterations of the algorithm
- **Distance threshold** - maximum distance in which points are considered neighbouring
- **Fitness threshold** - the lowest fitness between the models to be accepted. 

The computed fit by the RANSAC algorithm is a bit "raw". 
To improve it further, we perform the **ICP** (Iterative closest points) algorithm.
This algorithm can be tuned by the following parameter:
- **ICP Distance threshold** - maximum corresponding points-pair distance

### Reconstruction section
Since we now have a preprocessed meshes and with defined transformations from the *source* to the *target*, we can proceed to the **reconstruction section**.
For the reconstruction we use the **BCPD** (Bayesian coherent point drift) algorithm.
Now, the BCPD allows for very fine adjustments of its behaviour using lots of different parameters. 
For the exact description of their effects, please refer to the official documentation [here](https://github.com/ohirose/bcpd/blob/master/README.md).

> **Note: You do NOT have to perform any kind of installation process, the BCPD and its geodesic variant are already pre-built and preconfigured for immediate use in this module.**

**Not implemented options:**
- Terminal output 
- File output

### Postprocessing section
After the model is reconstructed, we a postprocessing section is included for you to be able to slightly modify the result, if necessary.
For these, we let you modify the following parameters:
- **Clustering scaling**
    - Scaled size of voxel for within vertices that are clustered together (additionally refer to [here](http://www.open3d.org/docs/0.7.0/python_api/open3d.geometry.simplify_vertex_clustering.html))
    - If set to 1.0, no scaling is performed
- **Smoothing iterations** - Number of iterations of mesh smoothing
    - If set to 0, no smoothing is applied 

After the whole process is done, the generated mesh is imported back into the current Slicer scene.
