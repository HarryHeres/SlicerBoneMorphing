from sys import platform
from typing import Tuple
from slicer.ScriptedLoadableModule import ScriptedLoadableModuleLogic
import slicer
from slicer import vtkMRMLModelNode

try:
    import open3d as o3d
except ModuleNotFoundError:
    print("Module Open3D is not installed. Installing...")
    slicer.util.pip_install('open3d===0.16.0')  # Version fix because of silicon based Macs

import open3d as o3d
import numpy as np
import os
import glob
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import subprocess
import tempfile

from .Constants import *

# NOTE: Path is relative to the main module class
BCPD_EXEC = os.path.dirname(os.path.abspath(__file__)) + "/../../Resources/BCPD/exec/"

if platform == "linux" or platform == "linux2":
    BCPD_EXEC += "bcpd_linux_x86_64"
elif platform == "darwin":
    BCPD_EXEC += "bcpd_macos_x86_64"  # Slicer is running through Rosetta, so x86 version needs to be used for now
elif platform == "win32":
    BCPD_EXEC += "bcpd_win32.exe"

SOURCE_VISUALIZATION_COLOR = [0, 1, 0]
TARGET_VISUALIZATION_COLOR = [1, 0, 0]


class SlicerBoneMorphingLogic(ScriptedLoadableModuleLogic):
    def __init__(self, parent):
        ScriptedLoadableModuleLogic.__init__(self, parent)

    def __visualize(self, source, target):
        """
        Only for debugging purposes
        """
        models = []

        if (source != None):
            source.paint_uniform_color(SOURCE_VISUALIZATION_COLOR)
            models.append(source)

        if (target != None):
            target.paint_uniform_color(TARGET_VISUALIZATION_COLOR)
            models.append(target)

        o3d.visualization.draw_geometries(models)

    def generate_model(
            self,
            source_model: vtkMRMLModelNode,
            target_model: vtkMRMLModelNode,
            parameters: dict
    ) -> Tuple[int, vtk.vtkPolyData, vtk.vtkPolyData]:
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

        if (source_model == VALUE_NODE_NOT_SELECTED or target_model == VALUE_NODE_NOT_SELECTED):
            print("Input or foundation model(s) were not selected")
            return EXIT_FAILURE, None, None

        source_mesh = self.__convert_model_to_mesh(source_model)
        target_mesh = self.__convert_model_to_mesh(target_model)

        err, result_icp = self.__preprocess_model(source_mesh, target_mesh, parameters[PREPROCESSING_KEY])
        if err == EXIT_FAILURE:
            print("Cannot continue with generating. Aborting...")
            return EXIT_FAILURE, None, None

        source_mesh.transform(result_icp.transformation)

        # self.__visualize(source_mesh, target_mesh)

        # BCPD stage
        deformed = self.__deformable_registration(source_mesh, target_mesh, parameters[BCPD_KEY])
        if (deformed == None):
            return EXIT_FAILURE, None, None

        # self.__visualize(deformed, None)

        generated_polydata, merged_polydata = self.__postprocess_meshes(deformed, target_mesh, parameters[POSTPROCESSING_KEY])

        return EXIT_OK, generated_polydata, merged_polydata

    def __convert_mesh_to_vtk_polydata(self, mesh: o3d.geometry.TriangleMesh) -> vtk.vtkPolyData:
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

    def __convert_model_to_mesh(self, model: vtkMRMLModelNode) -> o3d.geometry.TriangleMesh:
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
            numpy_normals = vtk_to_numpy(vtk_poly_data.GetPointData().GetNormals())

        # Get indices (triangles), this would be a (n, 3) shape numpy array where n is the number of triangles.
        # If the vtkPolyData does not represent a valid triangle mesh, the indices may not form valid triangles.
        numpy_indices = vtk_to_numpy(vtk_poly_data.GetPolys().GetData()).reshape(-1, 4)[:, 1:]

        # convert numpy array to open3d TriangleMesh
        open3d_mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(numpy_vertices), o3d.utility.Vector3iVector(numpy_indices))

        if numpy_normals is not None:
            open3d_mesh.vertex_normals = o3d.utility.Vector3dVector(numpy_normals)

        return open3d_mesh

    def __convert_mesh_to_point_cloud(self, mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.PointCloud:
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

    def __preprocess_model(
            self,
            source_mesh: o3d.geometry.TriangleMesh,
            target_mesh: o3d.geometry.TriangleMesh,
            parameters: dict
    ) -> Tuple[int, o3d.pipelines.registration.RegistrationResult]:
        """
            Preprocess model before advancing into the generation (BCPD) stage. This method converts the input models into respective point clouds, then performs RANSAC best fit transformation from source to target and returns the result

            Parameters
            ----------
            o3d.geometry.TriangleMesh source_mesh: Source MRML model
            o3d.geometry.TriangleMesh target_mesh: Target MRML model

            Returns
            -------
            Tuple[int, o3d.pipelines.registration.RegistrationResult] - if Tuple[0] equals EXIT_OK, then Tuple[1] will carry the registration result
        """

        source_pcd = self.__convert_mesh_to_point_cloud(source_mesh)
        target_pcd = self.__convert_mesh_to_point_cloud(target_mesh)

        source_pcd_downsampled, source_pcd_fpfh = self.__preprocess_point_cloud(
            source_pcd,
            parameters[PREPROCESSING_KEY_DOWNSAMPLING_DISTANCE_THRESHOLD],
            parameters[PREPROCESSING_KEY_NORMALS_ESTIMATION_RADIUS],
            parameters[PREPROCESSING_KEY_FPFH_ESTIMATION_RADIUS],
            parameters[PREPROCESSING_KEY_MAX_NN_NORMALS],
            parameters[PREPROCESSING_KEY_MAX_NN_FPFH]
        )

        target_pcd_downsampled, target_pcd_fpfh = self.__preprocess_point_cloud(
            target_pcd,
            parameters[PREPROCESSING_KEY_DOWNSAMPLING_DISTANCE_THRESHOLD],
            parameters[PREPROCESSING_KEY_NORMALS_ESTIMATION_RADIUS],
            parameters[PREPROCESSING_KEY_FPFH_ESTIMATION_RADIUS],
            parameters[PREPROCESSING_KEY_MAX_NN_NORMALS],
            parameters[PREPROCESSING_KEY_MAX_NN_FPFH]
        )

        try:
            result_ransac = self.__ransac_pcd_registration(
                source_pcd_downsampled, target_pcd_downsampled,
                source_pcd_fpfh, target_pcd_fpfh,
                parameters[REGISTRATION_KEY_DISTANCE_THRESHOLD],
                parameters[REGISTRATION_KEY_FITNESS_THRESHOLD],
                parameters[REGISTRATION_KEY_MAX_ITERATIONS]
            )
            if result_ransac == None:
                raise RuntimeError
        except RuntimeError:
            print("No registration fit was found using the RANSAC algorithm. Please, try adjusting the preprocessing parameters")
            return EXIT_FAILURE, None

        result_icp = o3d.pipelines.registration.registration_icp(
            source_pcd_downsampled, target_pcd_downsampled,
            parameters[REGISTRATION_KEY_ICP_DISTANCE_THRESHOLD],
            result_ransac.transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPlane()
        )

        return EXIT_OK, result_icp

    def __preprocess_point_cloud(
            self,
            pcd: o3d.geometry.PointCloud,
            downsampling_distance_threshold: float,
            normals_estimation_radius: float,
            fpfh_estimation_radius: float,
            max_nn_normals: int,
            max_nn_fpfh: int
    ) -> Tuple[o3d.geometry.PointCloud, o3d.pipelines.registration.Feature]:
        '''
            Perform downsampling of a mesh, normal estimation and computing FPFH feature of the point cloud.

            Parameters
            ----------
            o3d.geometry.PointCloud pcd: Source point cloud
            float downsampling_distance_threshold: Distance threshold for downsampling
            float normals_estimation_radius: Radius for estimating normals
            float fpfh_estimation_radius: Radius for the FPFH computation
            int max_nn_normals: Maximum number of neighbours considered for normals estimation
            int max_nn_fpfh: Maximum number of neighbours considered for the FPFH calculation

            Returns
            -------
            Tuple: [Downsampled PCD: open3d.geometry.PointCloud,
                    FPFH: open3d.pipelines.registration.Feature]
        '''

        pcd_downsampled: o3d.geometry.PointCloud = pcd.voxel_down_sample(downsampling_distance_threshold)

        pcd_downsampled.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=normals_estimation_radius, max_nn=max_nn_normals))

        pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(pcd_downsampled, o3d.geometry.KDTreeSearchParamHybrid(radius=fpfh_estimation_radius, max_nn=max_nn_fpfh))

        return pcd_downsampled, pcd_fpfh

    def __ransac_pcd_registration(
        self,
        source_pcd_down: o3d.geometry.PointCloud,
        target_pcd_down: o3d.geometry.PointCloud,
        source_fpfh: o3d.pipelines.registration.Feature,
        target_fpfh: o3d.pipelines.registration.Feature,
        distance_threshold: float,
        fitness_threshold: float,
        max_iterations: int,
    ) -> o3d.pipelines.registration.RegistrationResult:
        ''' Perform a registration of nearest neighbours using the RANSAC algorithm.

            Parameters
            ----------
            o3d.geometry.PointCloud source_pcd_down: Downsampled SOURCE point cloud
            o3d.geometry.PointCloud target_pcd_down: Downsampled TARGET point cloud
            o3d.pipelines.registration.Feature source_fpfh: Source PCD Fast-Point-Feature-Histogram
            o3d.pipelines.registration.Feature target_fpfh: Target PCD Fast-Point-Feature-Histogram
            float distance_threshold: Threshold in which a near point is considered a neighbour
            float fitness_threshold: Minimal value for iterations until it is reached
            max_iterations: Maximum number of iterations of the RANSAC algorithm

            Returns
            -------
            Best PCD fit: open3d.pipelines.registration.RegistrationResult

        '''
        fitness = 0
        count = 0
        best_result = None
        fitness_max = 1

        while (fitness < fitness_threshold and fitness < fitness_max and count < max_iterations):
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
                # NOTE: Just for earlier termination, but still needs the outer loop for proper convergence
                o3d.pipelines.registration.RANSACConvergenceCriteria(100000, fitness_threshold)
            )

            if result.fitness > fitness and result.fitness < 1:
                fitness = result.fitness
                best_result = result

            count += 1

        return best_result

    def __deformable_registration(
        self,
        source_pcd: o3d.geometry.PointCloud,
        target_pcd: o3d.geometry.PointCloud,
        bcpd_parameters: dict
    ) -> o3d.geometry.TriangleMesh:
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

        target_path = tempfile.gettempdir() + '/slicer_bone_morphing_target.txt'
        source_path = tempfile.gettempdir() + '/slicer_bone_morphing_source.txt'
        output_path = tempfile.gettempdir() + '/output_'

        np.savetxt(target_path, target_array, delimiter=',')
        np.savetxt(source_path, source_array, delimiter=',')

        cmd = f'{BCPD_EXEC} -h -x {target_path} -y {source_path}'

        for key in bcpd_parameters.keys():
            cmd += f' {key}{bcpd_parameters[key]}'

        cmd += f' -o {output_path}'
        print("BCPD: " + cmd)

        subprocess.run(cmd,
                       shell=True,
                       check=True,
                       text=True,
                       capture_output=True)

        try:
            bcpdResult = np.loadtxt(output_path + "y.interpolated.txt")
        except FileNotFoundError:
            print("No results generated by BCPD. Refer to the output in the console.")
            return None

        for fl in glob.glob(output_path + "*.txt"):
            os.remove(fl)
        os.remove(target_path)
        os.remove(source_path)

        deformed = o3d.geometry.TriangleMesh()
        deformed.vertices = o3d.utility.Vector3dVector(np.asarray(bcpdResult))
        deformed.triangles = source_pcd.triangles

        return deformed

    def __postprocess_meshes(
        self,
        generated: o3d.geometry.TriangleMesh,
        target_mesh: o3d.geometry.TriangleMesh,
        parameters: dict
    ) -> Tuple[vtk.vtkPolyData, vtk.vtkPolyData]:
        """
            Perform a postprocessing (smoothing and filtering) of the result of the BCPD algorithm

            Parameters
            ----------
            o3d.geometry.TriangleMesh generated: BCPD result mesh
            o3d.geometry.TriangleMesh target_mesh: Target model mesh
            dict parameters: parameters dict for the postprocessing stage

            Returns
            -------
            Tuple[vtk.vtkPolyData, vtk.vtkPolyData]: Tuple[0] will represent the postprocessed BCPD mesh, Tuple[1] will represent the COMBINED mesh (i.e. BCPD result merged with the target mesh)

        """

        combined = generated + target_mesh

        # Compute normals before postprocessing
        generated.compute_vertex_normals()
        combined.compute_vertex_normals()

        # Simplify mesh (smoothing and filtering)
        if parameters[POSTPROCESSING_KEY_CLUSTERING_SCALING] > 1.0:
            generated = generated.simplify_vertex_clustering(parameters[POSTPROCESSING_KEY_CLUSTERING_SCALING], contraction=o3d.geometry.SimplificationContraction.Average)
            combined = combined.simplify_vertex_clustering(parameters[POSTPROCESSING_KEY_CLUSTERING_SCALING], contraction=o3d.geometry.SimplificationContraction.Average)

        if parameters[POSTPROCESSING_KEY_SMOOTHING_ITERATIONS] > 0:
            generated = generated.filter_smooth_simple(number_of_iterations=parameters[POSTPROCESSING_KEY_SMOOTHING_ITERATIONS])
            generated = generated.filter_smooth_taubin(number_of_iterations=parameters[POSTPROCESSING_KEY_SMOOTHING_ITERATIONS])
            combined = combined.filter_smooth_simple(number_of_iterations=parameters[POSTPROCESSING_KEY_SMOOTHING_ITERATIONS])
            combined = combined.filter_smooth_taubin(number_of_iterations=parameters[POSTPROCESSING_KEY_SMOOTHING_ITERATIONS])

        generated_polydata = self.__convert_mesh_to_vtk_polydata(generated)
        merged_polydata = self.__convert_mesh_to_vtk_polydata(combined)

        return generated_polydata, merged_polydata
