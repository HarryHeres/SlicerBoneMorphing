from slicer.ScriptedLoadableModule import ScriptedLoadableModuleLogic
import slicer

try:
    import open3d as o3d
except ModuleNotFoundError:
    print("Module Open3D not installed. Trying to install...")
    slicer.util.pip_install('open3d')
    
import numpy as np
import ctypes
import os
import glob
from vtk.util.numpy_support import vtk_to_numpy

import tempfile

from .Constants import *

bcpd_lib = os.path.dirname(os.path.abspath(__file__)) + "/../../Resources/BCPD/"

from sys import platform

#NOTE: Needs relative path to the main module script
if platform == "linux" or platform == "linux2":
    bcpd_lib += "libbcpd_linux_x86_64.so"
elif platform == "darwin":
    bcpd_lib += "libbcpd_macos_x86_64.dylib" # Slicer is running through Rosetta, so x86 version needs to be used for now
elif platform == "win32":
    bcpd_lib += "libbcpd.dll"

BCPD = ctypes.CDLL(bcpd_lib)

RADIUS_NORMAL_SCALING = 4
RADIUS_FEATURE_SCALING = 10 
MAX_NN_NORMALS = 30
MAX_NN_FPFH = 100
VOXEL_SIZE_SCALING = 55 

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

    def convert_model_to_mesh(self, model): 
        vtk_polydata = model.GetPolyData()

        # Get vertices from vtk_polydata
        numpy_vertices = vtk_to_numpy(vtk_polydata.GetPoints().GetData())

        # Get normals if present
        numpy_normals = None
        model_normals = vtk_polydata.GetPointData().GetNormals()
        if(model_normals is not None):
            numpy_normals = vtk_to_numpy(vtk_polydata.GetPointData().GetNormals())

        # Get indices (triangles), this would be a (n, 3) shape numpy array where n is the number of triangles.
        # If the vtkPolyData does not represent a valid triangle mesh, the indices may not form valid triangles.
        numpy_indices = vtk_to_numpy(vtk_polydata.GetPolys().GetData()).reshape(-1, 4)[:, 1:]

        # convert numpy array to open3d TriangleMesh
        open3d_mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(numpy_vertices),
        o3d.utility.Vector3iVector(numpy_indices))

        if numpy_normals is not None:
            open3d_mesh.vertex_normals = o3d.utility.Vector3dVector(numpy_normals)

        return open3d_mesh
    

    def generate_model(self, source_model, target_model):
        NODE_NOT_SELECTED = 0 # Via Slicer documentation

        if(source_model == NODE_NOT_SELECTED or target_model == NODE_NOT_SELECTED):
            print("Input or foundation model(s) were not selected")
            return

        source_mesh = self.convert_model_to_mesh(source_model)        
        target_mesh = self.convert_model_to_mesh(target_model)

        source_pcd = self.convert_to_point_cloud(source_mesh)
        target_pcd = self.convert_to_point_cloud(target_mesh)

        print(source_pcd)
        
        #Calculate object size
        size = np.linalg.norm(np.asarray(target_pcd.get_max_bound()) - np.asarray(target_pcd.get_min_bound()))
        voxel_size = float(size / VOXEL_SIZE_SCALING) # the 55 value seems to be too much downscaling
        
        source_pcd_downsampled, source_pcd_fpfh = self.preprocess_point_cloud(source_pcd, voxel_size)
        target_pcd_downsampled, target_pcd_fpfh = self.preprocess_point_cloud(target_pcd, voxel_size)

        print(source_pcd_downsampled)
        
        # Global registration
        try:
            max_attempts = 20
            result_ransac = self.ransac_pcd_registration(source_pcd_downsampled, target_pcd_downsampled,source_pcd_fpfh,target_pcd_fpfh,voxel_size, max_attempts)
        except RuntimeError:
            print("No ideal fit was found using the RANSAC algorithm. Please, try to adjust the parameters")
            return EXIT_OK 

        result_icp = self.refine_registration(source_pcd, target_pcd, result_ransac, voxel_size)
        
        # Deformable registration
        deformed = self.deformable_registration(source_mesh.transform(result_icp.transformation), target_mesh) 
        deformed.compute_vertex_normals()
        target_mesh.compute_vertex_normals()

        # Combine meshes (alternative - to crop the first before merging)
        combined = deformed + target_mesh 
        combined.compute_vertex_normals()

        # Simplify mesh (smoothing and filtering)
        mesh_smp = combined.simplify_vertex_clustering(voxel_size/3, contraction=o3d.geometry.SimplificationContraction.Average)
        mesh_smp = mesh_smp.filter_smooth_simple(number_of_iterations=2)
        mesh_smp = mesh_smp.filter_smooth_taubin(number_of_iterations=100)
        mesh_smp.compute_vertex_normals()

        return mesh_smp

        
    def convert_to_point_cloud(self, mesh: o3d.geometry.TriangleMesh):
        ''' Converts a mesh to a open3d.geometry.PointCloud'''

        mesh_center = mesh.get_center()
        mesh.translate(-mesh_center, relative=False)
        pcd = o3d.geometry.PointCloud()
        pcd.points = mesh.vertices
        pcd.colors = mesh.vertex_colors
        pcd.normals = mesh.vertex_normals

        return pcd
 


    def preprocess_point_cloud(self, pcd: o3d.geometry.PointCloud, voxel_size: float): 
      ''' 
          Perform downsampling of a mesh, normal estimation and computing FPFH feature of the point cloud.

          Returns
          -------
          Tuple: [Downsampled PCD: open3d.geometry.PointCloud,
                  FPFH: open3d.pipelines.registration.Feature]
      '''

      radius_normal = voxel_size * RADIUS_NORMAL_SCALING
      radius_feature = voxel_size * RADIUS_FEATURE_SCALING

      print("Downsampling mesh with a voxel size %.3f." % voxel_size)
      pcd_down: o3d.geometry.PointCloud = pcd.voxel_down_sample(voxel_size)

      print("Estimating normal with search radius %.3f." % radius_normal)
      pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=MAX_NN_NORMALS))

      print("Compute FPFH feature with search radius %.3f." % radius_feature)
      pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(pcd_down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=MAX_NN_FPFH))

      return pcd_down, pcd_fpfh

    def ransac_pcd_registration(self,
                                source_pcd_down: o3d.geometry.PointCloud, 
                              target_pcd_down: o3d.geometry.PointCloud, 
                              source_fpfh: o3d.pipelines.registration.Feature, 
                              target_fpfh: o3d.pipelines.registration.Feature,
                              voxel_size: float, 
                              ransac_max_attempts: int):
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
      distance_threshold = voxel_size * 1.5
      fitness = 0
      count = 0
      best_result = None
      fitness_min, fitness_max = 0.999, 1

      while (fitness < fitness_min and fitness < fitness_max and count < ransac_max_attempts):
          result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
              source_pcd_down,
              target_pcd_down,
              source_fpfh,
              target_fpfh,
              True,
              distance_threshold,
              o3d.pipelines.registration.TransformationEstimationPointToPoint(True),
              3,
              [   
                  o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                  o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
              ],
              o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999) # NOTE: JH - Would this not make a good fit? Why is the outer loop neccessary? 
          )
          if result.fitness > fitness and result.fitness < 1:
            fitness = result.fitness
            best_result = result
          count += 1
      return best_result

    def refine_registration(self, source_pcd_down: o3d.geometry.PointCloud, 
                          target_pcd_down: o3d.geometry.PointCloud, 
                          ransac_result: o3d.pipelines.registration.RegistrationResult,
                          voxel_size: float):
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
      distance_threshold = voxel_size * 0.4

      # NOTE: Why does it need to be estimated again? 
      source_pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * RADIUS_NORMAL_SCALING, max_nn=MAX_NN_NORMALS))
      target_pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * RADIUS_NORMAL_SCALING, max_nn=MAX_NN_NORMALS))

      print(":: Point-to-plane ICP registration is applied on original point")
      print("   clouds to refine the alignment. This time we use a strict")
      print("   distance threshold %.3f." % distance_threshold)
      result = o3d.pipelines.registration.registration_icp(
          source_pcd_down, 
          target_pcd_down, 
          distance_threshold, 
          ransac_result.transformation,
          o3d.pipelines.registration.TransformationEstimationPointToPlane())
      return result

    def deformable_registration(self, source_pcd, target_pcd):

        source_array = np.asarray(source_pcd.vertices,dtype=np.float32)
        target_array = np.asarray(target_pcd.vertices,dtype=np.float32)
        target_path = tempfile.gettempdir() + '/slicer_bone_morphing_target.txt'
        source_path = tempfile.gettempdir() + '/slicer_bone_morphing_source.txt'
        output_path = tempfile.gettempdir() + '/output_'
        np.savetxt (target_path, target_array, delimiter=',')
        np.savetxt (source_path, source_array, delimiter=',')


        cmd = f'bcdp -x {target_path} -y {source_path} -l10 -b10 -g0.1 -K140 -J500 -c1e-6 -p -d7 -e0.3 -f0.3 -ux -DB,5000,0.08 -ux -N1 -o {output_path}'

        # Split the cmd string into individual arguments
        cmd_parts = cmd.split()

        # Convert the Python list of arguments to a C-style array of char pointers
        argv_c = (ctypes.c_char_p * len(cmd_parts))()
        argv_c[:] = [arg.encode('utf-8') for arg in cmd_parts]

        # Set argc to the number of arguments
        argc = len(cmd_parts)

        BCPD.main.argtypes = (ctypes.c_int, ctypes.POINTER(ctypes.c_char_p))

        # TODO: JH - Better usage, rather than the main method
        # Might be considered in the future, but this should be enough, for now
        BCPD.main(argc, argv_c)

        bcpd_result = np.loadtxt(output_path + "y.interpolated.txt")

        for fl in glob.glob(output_path + "*.txt"):
            os.remove(fl)

        os.remove(target_path)
        os.remove(source_path)

        deformed = o3d.geometry.TriangleMesh()
        deformed.vertices = o3d.utility.Vector3dVector(np.asarray(bcpd_result))
        deformed.triangles = source_pcd.triangles

        return deformed
