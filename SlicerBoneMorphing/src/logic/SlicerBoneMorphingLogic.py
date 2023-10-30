from slicer.ScriptedLoadableModule import ScriptedLoadableModuleLogic
import slicer
from slicer import vtkMRMLModelNode

try:
    import open3d as o3d
except ModuleNotFoundError:
    print("Module Open3D is not installed. Installing...")
    slicer.util.pip_install('open3d')

import numpy as np
import os
import glob
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import subprocess
import tempfile

from .Constants import *

BCPD_EXEC = os.path.dirname(os.path.abspath(__file__)) + "/../../Resources/BCPD/exec/"

from sys import platform

#NOTE: Needs relative path to the main module script
if platform == "linux" or platform == "linux2":
    BCPD_EXEC += "bcpd_linux_x86_64"
elif platform == "darwin":
    BCPD_EXEC += "bcpd_macos_x86_64" # Slicer is running through Rosetta, so x86 version needs to be used for now
elif platform == "win32":
    BCPD_EXEC += "bcpd_win32.exe"

RADIUS_NORMAL_SCALING = 4
RADIUS_FEATURE_SCALING = 10 
MAX_NN_NORMALS = 30
MAX_NN_FPFH = 100
VOXEL_SIZE_SCALING = 55 
CLUSTERING_VOXEL_SCALING = 3
SMOOTHING_ITERATIONS = 2
FILTERING_ITEARTIONS = 100


deformed = None


class SlicerBoneMorphingLogic(ScriptedLoadableModuleLogic):
    """This class should implement all the actual
    computation done by your module.  The interface
    should be such that other python code can import
    this class and make use of the functionality without
    requiring an instance of the Widget.
    Uses ScriptedLoadableModuleLogic base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModuleLogic.__init__(self, parent)

    def convert_mesh_to_vtk_polydata(self, mesh: o3d.geometry.TriangleMesh):
        """
            Convert o3d.geometry.TriangleMesh to vtkPolyData

            Parameters
            ----------
            o3d.geometry.TriangleMesh mesh - mesh to be converted

            Returns
            -------
            vtk.vtkPolyData
        """
        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)

        points = vtk.vtkPoints()

        for v in vertices:
            points.InsertNextPoint(v)

        polys = vtk.vtkCellArray()

        for t in triangles:
            polys.InsertNextCell(3, t)

        vtkPolyData = vtk.vtkPolyData()
        vtkPolyData.SetPoints(points)
        vtkPolyData.SetPolys(polys)

        return vtkPolyData
        
    def convert_model_to_mesh(self, model: vtkMRMLModelNode): 
        """
            Convert vtkMRMLModelNode to open3d.geometry.TriangleMesh

            Parameters
            ----------
            slicer.vtkMRMLNode model - model to be converted

            Returns
            -------
            open3d.geometry.TriangleMesh 
        """
        vtkPolyData = model.GetPolyData()

        # Get vertices from vtk_polydata
        numpyVertices = vtk_to_numpy(vtkPolyData.GetPoints().GetData())

        # Get normals if present
        numpyNormals = None
        modelNormals = vtkPolyData.GetPointData().GetNormals()
        if(modelNormals is not None):
            numpyNormals = vtk_to_numpy(vtkPolyData.GetPointData().GetNormals())

        # Get indices (triangles), this would be a (n, 3) shape numpy array where n is the number of triangles.
        # If the vtkPolyData does not represent a valid triangle mesh, the indices may not form valid triangles.
        numpyIndices = vtk_to_numpy(vtkPolyData.GetPolys().GetData()).reshape(-1, 4)[:, 1:]

        # convert numpy array to open3d TriangleMesh
        open3dMesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(numpyVertices),
        o3d.utility.Vector3iVector(numpyIndices))

        if numpyNormals is not None:
            open3dMesh.vertex_normals = o3d.utility.Vector3dVector(numpyNormals)

        return open3dMesh
    

    def generate_model(self, sourceModel: vtkMRMLModelNode, targetModel: vtkMRMLModelNode): 
        """
            Generates new model based on the BCPD algorithm fit between source and target models. 

            Parameters
            ----------
            vtkMRMLModelNode source_model - source (partial) model to be fit-generated
            vtkMRMLModelNode target_model - model to fit the partial source by. 

            Returns
            -------
            Tuple[int status, vtkPolyData model data]:
                - status: EXIT_OK or EXIT_FAILURE
                - model: representing the new generated model or None, if EXIT_FAILURE

        """

        NODE_NOT_SELECTED = 0 # Via Slicer documentation

        if(sourceModel == NODE_NOT_SELECTED or targetModel == NODE_NOT_SELECTED):
            print("Input or foundation model(s) were not selected")
            return EXIT_FAILURE

        sourceMesh = self.convert_model_to_mesh(sourceModel)        
        targetMesh = self.convert_model_to_mesh(targetModel)

        sourcePcd = self.convert_to_point_cloud(sourceMesh)
        targetPcd = self.convert_to_point_cloud(targetMesh)

        print(sourcePcd)
        
        #Calculate object size
        size = np.linalg.norm(np.asarray(targetPcd.get_max_bound()) - np.asarray(targetPcd.get_min_bound()))
        voxelSize = float(size / VOXEL_SIZE_SCALING) # the 55 value seems to be too much downscaling
        
        sourcePcdDownsampled, sourcePcdFpfh = self.preprocess_point_cloud(sourcePcd, voxelSize)
        targetPcdDownsampled, targetPcdFpfh = self.preprocess_point_cloud(targetPcd, voxelSize)

        print(sourcePcdDownsampled)
        
        # Global registration
        try:
            maxAttempts = 20
            resultRansac = self.ransac_pcd_registration(sourcePcdDownsampled, targetPcdDownsampled,sourcePcdFpfh,targetPcdFpfh,voxelSize, maxAttempts)
        except RuntimeError:
            print("No ideal fit was found using the RANSAC algorithm. Please, try to adjust the parameters")
            return EXIT_FAILURE 

        resultIcp = self.refine_registration(sourcePcd, targetPcd, resultRansac, voxelSize)
        
        # Deformable registration
        deformed = self.deformable_registration(sourceMesh.transform(resultIcp.transformation), targetMesh) 
        if(deformed == None):
            return EXIT_FAILURE
        deformed.compute_vertex_normals()
        targetMesh.compute_vertex_normals()

        # Combine meshes (alternative - to crop the first before merging)
        combined = deformed + targetMesh 
        combined.compute_vertex_normals()

        # Simplify mesh (smoothing and filtering)
        meshSmp = combined.simplify_vertex_clustering(voxelSize/CLUSTERING_VOXEL_SCALING, contraction=o3d.geometry.SimplificationContraction.Average)
        meshSmp = meshSmp.filter_smooth_simple(number_of_iterations=SMOOTHING_ITERATIONS)
        meshSmp = meshSmp.filter_smooth_taubin(number_of_iterations=FILTERING_ITEARTIONS)
        # mesh_smp.compute_vertex_normals() # TODO: Is this needed? 

        vtkPolyData = self.convert_mesh_to_vtk_polydata(meshSmp)
        
        return [EXIT_OK, vtkPolyData]

        
    def convert_to_point_cloud(self, mesh: o3d.geometry.TriangleMesh):
        """
            Convert o3d.geometry.TriangleMesh to o3d.geometry.PointCloud

            Parameters
            ----------
            o3d.geometry.TriangleMesh mesh - mesh to be converted

            Returns
            -------
            o3d.geometry.PointCloud
        """

        # mesh_center = mesh.get_center()
        # mesh.translate(-mesh_center, relative=False) # Not needed for Slicer
        pcd = o3d.geometry.PointCloud()
        pcd.points = mesh.vertices
        pcd.colors = mesh.vertex_colors
        pcd.normals = mesh.vertex_normals

        return pcd
 


    def preprocess_point_cloud(self, pcd: o3d.geometry.PointCloud, voxelSize: float): 
      ''' 
          Perform downsampling of a mesh, normal estimation and computing FPFH feature of the point cloud.

          Returns
          -------
          Tuple: [Downsampled PCD: open3d.geometry.PointCloud,
                  FPFH: open3d.pipelines.registration.Feature]
      '''

      radiusNormal = voxelSize * RADIUS_NORMAL_SCALING
      radiusFeature = voxelSize * RADIUS_FEATURE_SCALING

      print("Downsampling mesh with a voxel size %.3f." % voxelSize)
      pcdDown: o3d.geometry.PointCloud = pcd.voxel_down_sample(voxelSize)

      print("Estimating normal with search radius %.3f." % radiusNormal)
      pcdDown.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radiusNormal, max_nn=MAX_NN_NORMALS))

      print("Compute FPFH feature with search radius %.3f." % radiusFeature)
      pcdFpfh = o3d.pipelines.registration.compute_fpfh_feature(pcdDown, o3d.geometry.KDTreeSearchParamHybrid(radius=radiusFeature, max_nn=MAX_NN_FPFH))

      return pcdDown, pcdFpfh

    def ransac_pcd_registration(self,
                                sourcePcdDown: o3d.geometry.PointCloud, 
                              targetPcdDown: o3d.geometry.PointCloud, 
                              sourceFpfh: o3d.pipelines.registration.Feature, 
                              targetFpfh: o3d.pipelines.registration.Feature,
                              voxelSize: float, 
                              ransacMaxAttempts: int):
      ''' Perform a registration of nearest neighbourgs using the RANSAC algorithm.
          Distance threshold will be set as 1.5 times the voxel size.
          
          Parameters
          ----------
          source_pcd_down: Downsampled SOURCE point cloud
          target_pcd_down: Downsampled TARGET point cloud
          source_fpfh: Source PCD Fast-Point-Feature-Histogram
          target_fpfh: Target PCD Fast-Point-Feature-Histogram
          voxel_size: Difference between source and target PCD max voxel_size
          ransac_max_attempts: Maximum number of iterations of the RANSAC algorithm

          Returns
          -------
          Best PCD fit: open3d.pipelines.registration.RegistrationResult

      '''
      distanceThreshold = voxelSize * 1.5
      fitness = 0
      count = 0
      bestResult = None
      fitnessMin, fitnessMax = 0.999, 1

      while (fitness < fitnessMin and fitness < fitnessMax and count < ransacMaxAttempts):
          result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
              sourcePcdDown,
              targetPcdDown,
              sourceFpfh,
              targetFpfh,
              True,
              distanceThreshold,
              o3d.pipelines.registration.TransformationEstimationPointToPoint(True),
              3,
              [   
                  o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                  o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distanceThreshold)
              ],
              o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999) # NOTE: JH - Would this not make a good fit? Why is the outer loop neccessary? 
          )
          if result.fitness > fitness and result.fitness < 1:
            fitness = result.fitness
            bestResult = result
          count += 1
      return bestResult

    def refine_registration(self, sourcePcdDown: o3d.geometry.PointCloud, 
                          targetPcdDown: o3d.geometry.PointCloud, 
                          ransacResult: o3d.pipelines.registration.RegistrationResult,
                          voxelSize: float):
      ''' Perform a point-clouds' registration refinement.
          Distance threshold for refinement is 0.4 times voxel_size
          
          Parameters
          ----------
          source_down: Downsampled SOURCE point cloud
          target_down: Downsampled TARGET point cloud
          source_fpfh: Source PCD Fast-Point-Feature-Histogram
          target_fpfh: Target PCD Fast-Point-Feature-Histogram
          voxel_size: Difference between source and target PCD max voxel_size

      '''
      distance_threshold = voxelSize * 0.4

      sourcePcdDown.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxelSize * RADIUS_NORMAL_SCALING, max_nn=MAX_NN_NORMALS))
      targetPcdDown.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxelSize * RADIUS_NORMAL_SCALING, max_nn=MAX_NN_NORMALS))

      print(":: Point-to-plane ICP registration is applied on original point")
      print("   clouds to refine the alignment. This time we use a strict")
      print("   distance threshold %.3f." % distance_threshold)
      result = o3d.pipelines.registration.registration_icp(
          sourcePcdDown, 
          targetPcdDown, 
          distance_threshold, 
          ransacResult.transformation,
          o3d.pipelines.registration.TransformationEstimationPointToPlane())
      return result

    def deformable_registration(self, sourcePcd, targetPcd):
        sourceArray = np.asarray(sourcePcd.vertices,dtype=np.float32)
        targetArray = np.asarray(targetPcd.vertices,dtype=np.float32)

        targetPath = tempfile.gettempdir() + '/slicer_bone_morphing_target.txt'
        sourcePath = tempfile.gettempdir() + '/slicer_bone_morphing_source.txt'
        outputPath = tempfile.gettempdir() + '/output_'

        np.savetxt (targetPath, targetArray, delimiter=',')
        np.savetxt (sourcePath, sourceArray, delimiter=',')

        cmd = f'{BCPD_EXEC} -h -x {targetPath} -y {sourcePath} -l10 -b10 -g0.1 -K140 -J500 -c1e-6 -p -d7 -e0.3 -f0.3 -ux -DB,5000,0.08 -ux -N1 -o {outputPath}'

        subprocess.run(cmd, shell=True, check=True, text=True, capture_output=True)

        try:
            bcpdResult = np.loadtxt(outputPath + "y.interpolated.txt")
        except FileNotFoundError:
            print("No results generated by BCPD. Refer to the output in the console.")
            return None

        for fl in glob.glob(outputPath + "*.txt"):
            os.remove(fl)
        os.remove(targetPath)
        os.remove(sourcePath)

        deformed = o3d.geometry.TriangleMesh()
        deformed.vertices = o3d.utility.Vector3dVector(np.asarray(bcpdResult))
        deformed.triangles = sourcePcd.triangles

        return deformed
