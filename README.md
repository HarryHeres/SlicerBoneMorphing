# SlicerBoneMorphing
Extension for 3D Slicer for bone mesh morphing.

At the moment, this module specializes for the *humerus* bone. 
TODO: Explain in more depth

## Installation
- Download the latest ZIP package from Releases
- Extract the ZIP contents to your desired folder
- Open up 3D Slicer, go to Edit -> Application Settings
- In the modules section, add the extracted contents' path to "Additional Module Paths" 
- Restart 3D Slicer

    **DISCLAIMER! After restarting, the installation process will begin. If there are any Python modules not available in Slicer, they will be installed, so the startup will take SIGNIFICANTLY MORE amount of time. Do not be scared, this is intended behaviour. DISCLAIMER!**

## Usage
After a successful install, the module will be available in the **Morphing** section. 
When switching to the module, you should be greeted with the following UI: 

<p align="center"> 
<img src="docs/assets/ui.png" width="400px" height="900px">
</p>

The UI consists of **4** main modules
- Input
- Preprocessing
- BCPD
- Postprocessing

## Architecture
TODO: Create a workflow diagram

## Module sections

### Input
This section is self-explanatory. Here, you choose two input models:
- Source = Source for the generation; This is the model that represents the 
- Target = Model which is non-complete => Needs its missing portions generated

### Preprocessing parameters
Before the generation process, we usually want to preprocess the model.
Reasons could be to remove unwanted **outliers** or to smooth out the models.

#### Point cloud preprocessing
In this seciton, the following parameters can be adjusted:
- Downsampling distance threshold: When downsampling, we need to create a threshold for distance in which two points are considered neighbours. This will represent this threshold.
- Normals estimation radius: After downsampling, there are 






