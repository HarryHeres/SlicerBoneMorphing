import src.logic.Constants as const
import tempfile
import subprocess
from subprocess import CalledProcessError
from vtk.util.numpy_support import vtk_to_numpy
import vtk
import glob
import os
import numpy as np
from sys import platform
from typing import Tuple
from slicer.ScriptedLoadableModule import ScriptedLoadableModuleLogic
from slicer import vtkMRMLModelNode
import slicer.util as su

try:
    import open3d as o3d
except ModuleNotFoundError:
    print("Module Open3D not found")
    if su.confirmOkCancelDisplay(text="This module requires the 'open3d' Python package. Click OK to install it now.") is True:
        su.pip_install('open3d===0.16.0')  # Version fix because of silicon based Macs
        import open3d as o3d
    else:
        print("Open3D is not installed, but is required")


# NOTE: Path is relative to the main module class
BCPD_EXEC = os.path.dirname(os.path.abspath(__file__)) + "/../../Resources/BCPD/exec/"

if platform == "linux" or platform == "linux2":
    BCPD_EXEC += "bcpd_linux_x86_64"
elif platform == "darwin":
    BCPD_EXEC += "bcpd_macos_x86_64"  # Slicer is running through Rosetta, so x86 version needs to be used for now
elif platform == "win32":
    BCPD_EXEC += "bcpd_win32.exe"


class SlicerBoneMorphingLogic(ScriptedLoadableModuleLogic):
    def __init__(self, parent=None):
        ScriptedLoadableModuleLogic.__init__(self, parent)

    def __visualize(self, source, target, window_name: str = "", source_color=const.VISUALIZATION_DEFAULT_VALUE_TARGET_MODEL_COLOR, target_color=const.VISUALIZATION_KEY_TARGET_MODEL_COLOR):
        models = []

        if (source is not None):
            source.paint_uniform_color(np.array([source_color.red() / 255, source_color.green() / 255, source_color.blue() / 255]))
            models.append(source)

        if (target is not None):
            target.paint_uniform_color(np.array([target_color.red() / 255, target_color.green() / 255, target_color.blue() / 255]))
            models.append(target)

        o3d.visualization.draw_geometries(models, window_name=window_name, mesh_show_wireframe=True, point_show_normal=True)

    def generate_model(
            self,
            source_model: vtkMRMLModelNode,
            target_model: vtkMRMLModelNode,
            parameters: dict
    ) -> Tuple[int, vtk.vtkPolyData]:
        """
            Generates new model based on the BCPD algorithm fit between source and target models.

            Parameters
            ----------
            source_model - source (partial) model to be fit-generated
            target_model - model to fit the partial source by.
            parameters - parameters for the preprocessing, BCPD and postprocessing

            Returns
            -------
            Tuple[status, generatedPolydata]:
                - status: EXIT_OK or EXIT_FAILURE
                - generatedPolydata: Generated model by the BCPD
        """
        source_mesh = self.__convert_model_to_mesh(source_model)
        target_mesh = self.__convert_model_to_mesh(target_model)

        err, result_icp = self.__preprocess_model(source_mesh, target_mesh, parameters[const.PREPROCESSING_KEY])
        if err == const.EXIT_FAILURE:
            print("Cannot continue to reconstruction. Aborting...")
            return const.EXIT_FAILURE, None

        source_mesh.transform(result_icp.transformation)

        visualization_params = parameters[const.VISUALIZATION_KEY]

        if (visualization_params[const.VISUALIZATION_KEY_SHOULD_VISUALIZE] is True):
            self.__visualize(source_mesh, target_mesh, "Preprocessed models", visualization_params[const.VISUALIZATION_KEY_SOURCE_MODEL_COLOR], visualization_params[const.VISUALIZATION_KEY_TARGET_MODEL_COLOR])

        # BCPD stage
        deformed = self.__deformable_registration(source_mesh, target_mesh, parameters[const.BCPD_KEY])
        if (deformed is None):
            return const.EXIT_FAILURE, None

        if (visualization_params[const.VISUALIZATION_KEY_SHOULD_VISUALIZE] is True):
            self.__visualize(deformed, None, "Reconstructed model", visualization_params[const.VISUALIZATION_KEY_SOURCE_MODEL_COLOR], visualization_params[const.VISUALIZATION_KEY_TARGET_MODEL_COLOR])

        generated_polydata = self.__postprocess_meshes(deformed, parameters[const.POSTPROCESSING_KEY])

        if (visualization_params[const.VISUALIZATION_KEY_SHOULD_VISUALIZE] is True):
            self.__visualize(deformed, None, "Postprocessed model", visualization_params[const.VISUALIZATION_KEY_SOURCE_MODEL_COLOR], visualization_params[const.VISUALIZATION_KEY_TARGET_MODEL_COLOR])

        return const.EXIT_OK, generated_polydata

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
            model - model to be converted

            Returns
            -------
            Converted TriangleMesh
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
            Convert TriangleMesh to PointCloud

            Parameters
            ----------
            mesh - mesh to be converted

            Returns
            -------
            Converted PointCloud
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
            source_mesh: Source MRML model
            target_mesh: Target MRML model

            Returns
            -------
             - if [0] equals EXIT_OK, then [1] will carry the registration result
        """

        source_pcd = self.__convert_mesh_to_point_cloud(source_mesh)
        target_pcd = self.__convert_mesh_to_point_cloud(target_mesh)

        source_pcd_downsampled, source_pcd_fpfh = self.__preprocess_point_cloud(
            source_pcd,
            parameters[const.PREPROCESSING_KEY_DOWNSAMPLING_VOXEL_SIZE],
            parameters[const.PREPROCESSING_KEY_NORMALS_ESTIMATION_RADIUS],
            parameters[const.PREPROCESSING_KEY_FPFH_ESTIMATION_RADIUS],
            parameters[const.PREPROCESSING_KEY_MAX_NN_NORMALS],
            parameters[const.PREPROCESSING_KEY_MAX_NN_FPFH]
        )

        target_pcd_downsampled, target_pcd_fpfh = self.__preprocess_point_cloud(
            target_pcd,
            parameters[const.PREPROCESSING_KEY_DOWNSAMPLING_VOXEL_SIZE],
            parameters[const.PREPROCESSING_KEY_NORMALS_ESTIMATION_RADIUS],
            parameters[const.PREPROCESSING_KEY_FPFH_ESTIMATION_RADIUS],
            parameters[const.PREPROCESSING_KEY_MAX_NN_NORMALS],
            parameters[const.PREPROCESSING_KEY_MAX_NN_FPFH]
        )

        try:
            result_ransac = self.__ransac_pcd_registration(
                source_pcd_downsampled, target_pcd_downsampled,
                source_pcd_fpfh, target_pcd_fpfh,
                parameters[const.REGISTRATION_KEY_DISTANCE_THRESHOLD],
                parameters[const.REGISTRATION_KEY_FITNESS_THRESHOLD],
                parameters[const.REGISTRATION_KEY_MAX_ITERATIONS]
            )
            if result_ransac is None:
                raise RuntimeError
        except RuntimeError:
            print("No registration fit was found using the RANSAC algorithm. Please, try adjusting the preprocessing parameters")
            return const.EXIT_FAILURE, None

        result_icp = o3d.pipelines.registration.registration_icp(
            source_pcd_downsampled, target_pcd_downsampled,
            parameters[const.REGISTRATION_KEY_ICP_DISTANCE_THRESHOLD],
            result_ransac.transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPlane()
        )

        return const.EXIT_OK, result_icp

    def __preprocess_point_cloud(
            self,
            pcd: o3d.geometry.PointCloud,
            downsampling_voxel_size: float,
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
            Tuple: [open3d.geometry.PointCloud, open3d.pipelines.registration.Feature]
                - [0] = downsampled PCD
                - [1] = FPFH
        '''

        if downsampling_voxel_size > 0.0:
            pcd = pcd.voxel_down_sample(downsampling_voxel_size)

        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=normals_estimation_radius, max_nn=max_nn_normals))

        pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=fpfh_estimation_radius, max_nn=max_nn_fpfh))

        return pcd, pcd_fpfh

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
            source_pcd_down: Downsampled SOURCE point cloud
            target_pcd_down: Downsampled TARGET point cloud
            source_fpfh: Source PCD Fast-Point-Feature-Histogram
            target_fpfh: Target PCD Fast-Point-Feature-Histogram
            distance_threshold: Threshold in which a near point is considered a neighbour
            fitness_threshold: Minimal value for iterations until it is reached
            max_iterations: Maximum number of iterations of the RANSAC algorithm

            Returns
            -------
            RANSAC registration result

        '''
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
            o3d.pipelines.registration.RANSACConvergenceCriteria(max_iteration=max_iterations, confidence=const.REGISTRATION_DEFAULT_VALUE_RANSAC_CONVERGENCE_CONFIDENCE)
        )

        return result

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
            sourcePcd: source point cloud
            targetPcd: target point cloud
            bcpdParameters: parameters for the BCPD algorithm

            Returns
            -------
            New deformed mesh
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

        try:
            subprocess.run(cmd,
                           shell=True,
                           check=True,
                           text=True,
                           capture_output=True)
        except CalledProcessError as e:
            print("BCPD subprocess returned with error (code {}): {}".format(e.returncode, e.output))
            print("Process output: {}".format(e.output))
            print("Errors: {}".format(e.stderr))
            return None

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
        parameters: dict
    ) -> vtk.vtkPolyData:
        """
            Perform a postprocessing (smoothing and filtering) of the result of the BCPD algorithm

            Parameters
            ----------
            generated: BCPD result mesh
            parameters: parameters for the postprocessing stage

            Returns
            -------
            BCPD generated mesh
        """

        # Compute normals before postprocessing
        generated.compute_vertex_normals()

        # Simplify mesh (smoothing and filtering)
        if parameters[const.POSTPROCESSING_KEY_CLUSTERING_SCALING] > 1.0:
            generated = generated.simplify_vertex_clustering(parameters[const.POSTPROCESSING_KEY_CLUSTERING_SCALING], contraction=o3d.geometry.SimplificationContraction.Average)

        if parameters[const.POSTPROCESSING_KEY_SMOOTHING_ITERATIONS] > 0:
            generated = generated.filter_smooth_simple(number_of_iterations=parameters[const.POSTPROCESSING_KEY_SMOOTHING_ITERATIONS])
            generated = generated.filter_smooth_taubin(number_of_iterations=parameters[const.POSTPROCESSING_KEY_SMOOTHING_ITERATIONS])

        generated_polydata = self.__convert_mesh_to_vtk_polydata(generated)

        return generated_polydata
