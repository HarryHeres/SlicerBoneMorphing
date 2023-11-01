from sys import platform
from typing import Tuple
from slicer.ScriptedLoadableModule import ScriptedLoadableModuleLogic
import slicer
from slicer import vtkMRMLModelNode

try:
    import open3d as o3d
except ModuleNotFoundError:
    print("Module Open3D is not installed. Installing...")
    slicer.util.pip_install('open3d')

import open3d as o3d
import numpy as np
import os
import glob
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import subprocess
import tempfile

from .Constants import *

BCPD_EXEC = os.path.dirname(
    os.path.abspath(__file__)) + "/../../Resources/BCPD/exec/"

# NOTE: Needs relative path to the main module script
if platform == "linux" or platform == "linux2":
    BCPD_EXEC += "bcpd_linux_x86_64"
elif platform == "darwin":
    # Slicer is running through Rosetta, so x86 version needs to be used for now
    BCPD_EXEC += "bcpd_macos_x86_64"
elif platform == "win32":
    BCPD_EXEC += "bcpd_win32.exe"

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

    def generate_model(
            self, source_model: vtkMRMLModelNode,
            target_model: vtkMRMLModelNode,
            parameters) -> Tuple[int, vtk.vtkPolyData, vtk.vtkPolyData]:
        """
            Generates new model based on the BCPD algorithm fit between source and target models.

            Parameters
            ----------
            vtkMRMLModelNode source_model - source (partial) model to be fit-generated
            vtkMRMLModelNode target_model - model to fit the partial source by.
            string parameters - parameters for the preprocessing, BCPD and postprocessing

            Returns
            -------
            Tuple[int status, vtk.vtkPolyData generatedPolydata, vtk.vtkPolyData mergedPolydata]:
                - status: EXIT_OK or EXIT_FAILURE
                - generatedPolydata: Generated model by the BCPD
                - mergedPolydata: generatedPolydata that had been merged with the targetModel
        """

        if (source_model == VALUE_NODE_NOT_SELECTED
                or target_model == VALUE_NODE_NOT_SELECTED):
            print("Input or foundation model(s) were not selected")
            return EXIT_FAILURE, None, None

        err, result_icp = self.preprocess_model(source_model, target_model,
                                                parameters[PREPROCESSING_KEY])
        if err == EXIT_FAILURE:
            print("Cannot continue with generating. Aborting...")
            return EXIT_FAILURE, None, None

        # BCPD stage
        deformed = self.deformable_registration(
            source_mesh.transform(result_icp.transformation), target_mesh,
            parameters[BCPD_KEY])
        if (deformed == None):
            return EXIT_FAILURE, None, None

        self.postprocess_result_mesh(deformed)

        return EXIT_OK, generated_polydata, merged_polydata

    def convert_mesh_to_vtk_polydata(
            self, mesh: o3d.geometry.TriangleMesh) -> vtk.vtkPolyData:
        """
            Convert o3d.geometry.TriangleMesh to vtkPolyData

            Parameters
            ----------
            o3d.geometry.TriangleMesh mesh - mesh to be converted

            Returns
            -------
            vtk.vtkPolyData representation of the input mesh
        """
        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)

        points = vtk.vtkPoints()

        for v in vertices:
            points.InsertNextPoint(v)

        polys = vtk.vtkCellArray()

        for t in triangles:
            polys.InsertNextCell(3, t)

        vtk_poly_data = vtk.vtkPolyData()
        vtk_poly_data.SetPoints(points)
        vtk_poly_data.SetPolys(polys)

        return vtk_poly_data

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
        vtk_poly_data = model.GetPolyData()

        # Get vertices from vtk_polydata
        numpy_vertices = vtk_to_numpy(vtk_poly_data.GetPoints().GetData())

        # Get normals if present
        numpy_normals = None
        model_normals = vtk_poly_data.GetPointData().GetNormals()
        if (model_normals is not None):
            numpy_normals = vtk_to_numpy(
                vtk_poly_data.GetPointData().GetNormals())

        # Get indices (triangles), this would be a (n, 3) shape numpy array where n is the number of triangles.
        # If the vtkPolyData does not represent a valid triangle mesh, the indices may not form valid triangles.
        numpy_indices = vtk_to_numpy(
            vtk_poly_data.GetPolys().GetData()).reshape(-1, 4)[:, 1:]

        # convert numpy array to open3d TriangleMesh
        open3d_mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(numpy_vertices),
            o3d.utility.Vector3iVector(numpy_indices))

        if numpy_normals is not None:
            open3d_mesh.vertex_normals = o3d.utility.Vector3dVector(
                numpy_normals)

        return open3d_mesh

    def convert_to_point_cloud(
            self, mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.PointCloud:
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

    def preprocess_model(
        self, source_model: vtkMRMLModelNode, target_model: vtkMRMLModelNode,
        parameters
    ) -> Tuple[int, o3d.pipelines.registration.RegistrationResult]:
        """
            Preprocess model before advancing into the generation (BCPD) stage. This method converts the input models into respective point clouds, then performs RANSAC best fit transformation from source to target and returns a refined registration result

            Parameters
            ----------
            vtkMRMLModelNode source_model: Source MRML model
            vtkMRMLModelNode target_model: Target MRML model


        """
        source_mesh = self.convert_model_to_mesh(source_model)
        target_mesh = self.convert_model_to_mesh(target_model)

        source_pcd = self.convert_to_point_cloud(source_mesh)
        target_pcd = self.convert_to_point_cloud(target_mesh)

        # Calculate object size
        voxel_size_diff = np.linalg.norm(
            np.asarray(target_pcd.get_max_bound()) -
            np.asarray(target_pcd.get_min_bound()))

        distance_threshold = float(
            voxel_size_diff / parameters[PREPROCESSING_KEY_VOXEL_SIZE_SCALING])

        source_pcd_downsampled, source_pcd_fpfh = self.preprocess_point_cloud(
            source_pcd, distance_threshold)
        target_pcd_downsampled, target_pcd_fpfh = self.preprocess_point_cloud(
            target_pcd, distance_threshold)

        try:
            result_ransac = self.ransac_pcd_registration(
                source_pcd_downsampled, target_pcd_downsampled,
                source_pcd_fpfh, target_pcd_fpfh, distance_threshold,
                parameters[PREPROCESSING_RANSAC_MAX_ATTEMPTS])
        except RuntimeError:
            print(
                "No ideal fit was found using the RANSAC algorithm. Please, try adjusting the parameters"
            )
            return EXIT_FAILURE, None

        result_icp = self.refine_registration(source_pcd, target_pcd,
                                              result_ransac,
                                              distance_threshold)

        return EXIT_OK, result_icp

    def preprocess_point_cloud(
        self, pcd: o3d.geometry.PointCloud, distance_threshold: float
    ) -> Tuple[o3d.geometry.PointCloud, o3d.pipelines.registration.Feature]:
        ''' 
            Perform downsampling of a mesh, normal estimation and computing FPFH feature of the point cloud.

            Returns
            -------
            Tuple: [Downsampled PCD: open3d.geometry.PointCloud,
                    FPFH: open3d.pipelines.registration.Feature]
        '''

        radius_normal = distance_threshold * RADIUS_NORMAL_SCALING
        radius_feature = distance_threshold * RADIUS_FEATURE_SCALING

        print("Downsampling mesh with a %.3f." % distance_threshold)
        pcd_downsampled: o3d.geometry.PointCloud = pcd.voxel_down_sample(
            distance_threshold)

        print("Estimating normal with search radius %.3f." % radius_normal)
        pcd_downsampled.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal,
                                                 max_nn=MAX_NN_NORMALS))

        print("Compute FPFH feature with search radius %.3f." % radius_feature)
        pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd_downsampled,
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature,
                                                 max_nn=MAX_NN_FPFH))

        return pcd_downsampled, pcd_fpfh

    def ransac_pcd_registration(
        self,
        source_pcd_down: o3d.geometry.PointCloud,
        target_pcd_down: o3d.geometry.PointCloud,
        source_fpfh: o3d.pipelines.registration.Feature,
        target_fpfh: o3d.pipelines.registration.Feature,
        distance_threshold: float,
        ransac_max_attempts: int,
    ) -> o3d.pipelines.registration.RegistrationResult:
        ''' Perform a registration of nearest neighbours using the RANSAC algorithm.
            Distance threshold will be set as the voxel difference.

            Parameters
            ----------
            source_pcd_down: Downsampled SOURCE point cloud
            target_pcd_down: Downsampled TARGET point cloud
            source_fpfh: Source PCD Fast-Point-Feature-Histogram
            target_fpfh: Target PCD Fast-Point-Feature-Histogram
            distance_threshold: Threshold in which a near point is considered a neighbour
            ransac_max_attempts: Maximum number of iterations of the RANSAC algorithm

            Returns
            -------
            Best PCD fit: open3d.pipelines.registration.RegistrationResult

        '''
        fitness = 0
        count = 0
        best_result = None
        fitness_min, fitness_max = 0.999, 1

        while (fitness < fitness_min and fitness < fitness_max
               and count < ransac_max_attempts):
            result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
                source_pcd_down,
                target_pcd_down,
                source_fpfh,
                target_fpfh,
                True,
                distance_threshold,
                o3d.pipelines.registration.
                TransformationEstimationPointToPoint(True),
                3,
                [
                    o3d.pipelines.registration.
                    CorrespondenceCheckerBasedOnEdgeLength(0.9),
                    o3d.pipelines.registration.
                    CorrespondenceCheckerBasedOnDistance(distance_threshold)
                ],
                # NOTE: JH - Would this not make a good fit? Why is the outer loop neccessary?
                o3d.pipelines.registration.RANSACConvergenceCriteria(
                    100000, 0.999))
            if result.fitness > fitness and result.fitness < 1:
                fitness = result.fitness
                best_result = result
            count += 1
        return best_result

    def refine_registration(
        self, source_pcd_down: o3d.geometry.PointCloud,
        target_pcd_down: o3d.geometry.PointCloud,
        ransacResult: o3d.pipelines.registration.RegistrationResult,
        distance_threshold: float
    ) -> o3d.pipelines.registration.RegistrationResult:
        ''' Perform a point-clouds' registration refinement.

            Parameters
            ----------
            source_down: Downsampled SOURCE point cloud
            target_down: Downsampled TARGET point cloud
            source_fpfh: Source PCD Fast-Point-Feature-Histogram
            target_fpfh: Target PCD Fast-Point-Feature-Histogram
            distance_threshold: Threshold in which a point is considered a neighbour

        '''
        source_pcd_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=distance_threshold *
                                                 RADIUS_NORMAL_SCALING,
                                                 max_nn=MAX_NN_NORMALS))
        target_pcd_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=distance_threshold *
                                                 RADIUS_NORMAL_SCALING,
                                                 max_nn=MAX_NN_NORMALS))

        result = o3d.pipelines.registration.registration_icp(
            source_pcd_down, target_pcd_down, distance_threshold,
            ransacResult.transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPlane())

        return result

    def deformable_registration(self, source_pcd: o3d.geometry.PointCloud,
                                target_pcd: o3d.geometry.PointCloud,
                                bcpd_parameters) -> o3d.geometry.TriangleMesh:
        """
            Perform a BCPD deformable registration from the source point cloud to the target point cloud

            Parameters
            ----------
            o3d.geometry.PointCloud sourcePcd: source point cloud
            o3d.geometry.PointCloud targetPcd: target point cloud 
            string bcpdParameters: parameters for the BCPD algorithm

            Returns
            -------
            o3d.geometry.TriangleMesh representing the new deformed mesh 
        """
        source_array = np.asarray(source_pcd.vertices, dtype=np.float32)
        target_array = np.asarray(target_pcd.vertices, dtype=np.float32)

        target_path = tempfile.gettempdir(
        ) + '/slicer_bone_morphing_target.txt'
        source_path = tempfile.gettempdir(
        ) + '/slicer_bone_morphing_source.txt'
        output_path = tempfile.gettempdir() + '/output_'

        np.savetxt(target_path, target_array, delimiter=',')
        np.savetxt(source_path, source_array, delimiter=',')

        cmd = f'{BCPD_EXEC} -h -x {target_path} -y {source_path} {bcpd_parameters} -o {output_path}'

        subprocess.run(cmd,
                       shell=True,
                       check=True,
                       text=True,
                       capture_output=True)

        try:
            bcpdResult = np.loadtxt(output_path + "y.interpolated.txt")
        except FileNotFoundError:
            print(
                "No results generated by BCPD. Refer to the output in the console."
            )
            return None

        for fl in glob.glob(output_path + "*.txt"):
            os.remove(fl)
        os.remove(target_path)
        os.remove(source_path)

        deformed = o3d.geometry.TriangleMesh()
        deformed.vertices = o3d.utility.Vector3dVector(np.asarray(bcpdResult))
        deformed.triangles = source_pcd.triangles

        return deformed
