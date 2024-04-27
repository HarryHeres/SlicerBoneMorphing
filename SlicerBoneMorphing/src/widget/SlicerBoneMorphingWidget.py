from src.logic.Constants import *
import ctk
import slicer
from slicer.ScriptedLoadableModule import ScriptedLoadableModuleWidget
from src.logic.SlicerBoneMorphingLogic import SlicerBoneMorphingLogic

from src.logic.Enums import BcpdAccelerationMode, BcpdKernelType, BcpdNormalizationOptions, BcpdStandardKernel
from qt import QComboBox
from enum import Enum

import os


class SlicerBoneMorphingWidget(ScriptedLoadableModuleWidget):
    def __init__(self, parent):
        """Called when the application opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.__init__(self, parent)
        self.__bcpd_options = {}

    def setup(self) -> None:
        """Called when the application opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.setup(self)

        # Load widget from .ui file (created by Qt Designer)
        self.__uiWidget = slicer.util.loadUI(self.resourcePath("UI/SlicerBoneMorphing.ui"))
        self.layout.addWidget(self.__uiWidget)
        self.__ui = slicer.util.childWidgetVariables(self.__uiWidget)
        self.__logic = SlicerBoneMorphingLogic(self)

        self.__setup_ui()
        self.__reset_parameters_to_default()

    def __setup_ui(self) -> None:
        """
            Method that sets up all UI elements and their dependencies
        """

        self.__ui.sourceNodeSelectionBox.setMRMLScene(slicer.mrmlScene)
        self.__ui.targetNodeSelectionBox.setMRMLScene(slicer.mrmlScene)

        self.__ui.bcpdAdvancedControlsGroupBox.setVisible(False)

        self.__setup_combo_box(self.__ui.bcpdKernelTypeComboBox, BcpdKernelType, self.__show_kernel_type)
        self.__ui.bcpdGeodesicKernelGroupBox.setVisible(False)

        self.__setup_combo_box(self.__ui.bcpdStandardKernelComboBox, BcpdStandardKernel, None)

        self.__setup_combo_box(self.__ui.bcpdAccelerationModeComboBox, BcpdAccelerationMode, self.__show_acceleration_mode)
        self.__ui.bcpdAccelerationManualGroupBox.setVisible(False)

        self.__ui.bcpdGeodesicKernelInputMeshLineEdit.setVisible(False)
        self.__ui.bcpdGeodesicKernelInputMeshLabel.setVisible(False)

        self.__setup_combo_box(self.__ui.bcpdNormalizationComboBox, BcpdNormalizationOptions, None)

        self.__ui.bcpdDownsamplingCollapsibleGroupBox.visible = False

        self.__ui.bcpdResetParametersPushButton.clicked.connect(self.__reset_parameters_to_default)
        self.__ui.generateModelButton.clicked.connect(self.__generate_model)

    def __reset_parameters_to_default(self) -> None:
        ## Preprocessing parameters ##
        self.__ui.preprocessingDownsamplingDistanceThresholdDoubleSpinBox.value = PREPROCESSING_DEFAULT_VALUE_DOWNSAMPLING_DISTANCE_THRESHOLD
        self.__ui.preprocessingNormalsEstimationRadiusDoubleSpinBox.value = PREPROCESSING_DEFAULT_VALUE_RADIUS_NORMAL_SCALE
        self.__ui.preprocessingNormalsEstimationMaxNeighboursSpinBox.value = PREPROCESSING_DEFAULT_VALUE_MAX_NN_NORMALS
        self.__ui.preprocessingFpfhRadiusDoubleSpinBox.value = PREPROCESSING_DEFAULT_VALUE_RADIUS_FEATURE_SCALE
        self.__ui.preprocessingFpfhMaxNeighboursSpinBox.value = PREPROCESSING_DEFAULT_VALUE_MAX_NN_FPFH

        ## Registration parameters ##
        self.__ui.registrationMaxIterationsSpinBox.value = REGISTRATION_DEFAULT_VALUE_MAX_ITERATIONS
        self.__ui.registrationDistanceThresholdDoubleSpinBox.value = REGISTRATION_DEFAULT_VALUE_DISTANCE_THRESHOLD
        self.__ui.registrationFitnessThresholdDoubleSpinBox.value = REGISTRATION_DEFAULT_VALUE_FITNESS_THRESHOLD
        self.__ui.registrationIcpDistanceThresholdDoubleSpinBox.value = REGISTRATION_DEFAULT_VALUE_ICP_DISTANCE_THRESHOLD

        ## Tuning parameters ##
        self.__ui.bcpdOmegaDoubleSpinBox.value = BCPD_DEFAULT_VALUE_OMEGA
        self.__ui.bcpdLambdaDoubleSpinBox.value = BCPD_DEFAULT_VALUE_LAMBDA
        self.__ui.bcpdBetaDoubleSpinBox.value = BCPD_DEFAULT_VALUE_BETA
        self.__ui.bcpdGammaDoubleSpinBox.value = BCPD_DEFAULT_VALUE_GAMMA
        self.__ui.bcpdKappaDoubleSpinBox.value = BCPD_DEFAULT_VALUE_KAPPA

        ## Kernel parameters ##
        self.__ui.bcpdKernelTypeComboBox.setCurrentIndex(BCPD_DEFAULT_VALUE_KERNEL_TYPE)
        self.__ui.bcpdStandardKernelComboBox.setCurrentIndex(BCPD_DEFAULT_VALUE_STANDARD_KERNEL_TYPE)
        self.__ui.bcpdGeodesicKernelTauDoubleSpinBox.value = BCPD_DEFAULT_VALUE_TAU
        self.__ui.bcpdGeodesicKernelInputMeshCheckBox.checked = False
        self.__ui.bcpdGeodesicKernelInputMeshLineEdit.text = BCPD_DEFAULT_VALUE_INPUT_MESH_PATH
        self.__ui.bcpdGeodesicKernelNeighboursSpinBox.value = BCPD_DEFAULT_VALUE_KERNEL_NEIGBOURS
        self.__ui.bcpdGeodesicKernelRadiusDoubleSpinBox.value = BCPD_DEFAULT_VALUE_KERNEL_NEIGHBOUR_RADIUS
        self.__ui.bcpdGeodesicKernelBetaDoubleSpinBox.value = BCPD_DEFAULT_VALUE_KERNEL_BETA
        self.__ui.bcpdGeodesicKernelKTildeDoubleSpinBox.value = BCPD_DEFAULT_VALUE_KERNEL_K_TILDE
        self.__ui.bcpdGeodesicKernelEpsilonDoubleSpinBox.value = BCPD_DEFAULT_VALUE_KERNEL_EPSILON

        ## Acceleration parameters ##
        self.__ui.bcpdAccelerationModeComboBox.setCurrentIndex(BCPD_DEFAULT_VALUE_ACCELERATION_MODE)
        self.__ui.bcpdAccelerationAutomaticVbiCheckBox.checked = True
        self.__ui.bcpdAccelerationAutomaticPlusPlusCheckBox.checked = True
        self.__ui.bcpdAccelerationManualNystormGSpinBox.value = BCPD_DEFAULT_VALUE_ACCELERATION_NYSTORM_SAMPLES_G
        self.__ui.bcpdAccelerationManualNystormJSpinBox.value = BCPD_DEFAULT_VALUE_ACCELERATION_NYSTORM_SAMPLES_J
        self.__ui.bcpdAccelerationManualNystormRSpinBox.value = BCPD_DEFAULT_VALUE_ACCELERATION_NYSTORM_SAMPLES_R
        self.__ui.bcpdAccelerationManualKdTreeScaleDoubleSpinBox.value = BCPD_DEFAULT_VALUE_ACCELERATION_KD_TREE_SCALE
        self.__ui.bcpdAccelerationManualKdTreeRadiusDoubleSpinBox.value = BCPD_DEFAULT_VALUE_ACCELERATION_KD_TREE_RADIUS
        self.__ui.bcpdAccelerationManualKdTreeThresholdDoubleSpinBox.value = BCPD_DEFAULT_VALUE_ACCELERATION_KD_TREE_SIGMA_THRESHOLD

        ## Downsampling options ##
        self.__ui.bcpdDownsamplingLineEdit.text = BCPD_DEFAULT_VALUE_DOWNSAMPLING_OPTIONS

        ## Convergence options ##
        self.__ui.bcpdConvergenceToleranceDoubleSpinBox.value = BCPD_DEFAULT_VALUE_CONVERGENCE_TOLERANCE
        self.__ui.bcpdConvergenceMaxIterationsSpinBox.value = BCPD_DEFAULT_VALUE_CONVERGENCE_MAX_ITERATIONS
        self.__ui.bcpdConvergenceMinIterationsSpinBox.value = BCPD_DEFAULT_VALUE_CONVERGENCE_MIN_ITERATIONS

        ## Normalization options ##
        self.__ui.bcpdNormalizationComboBox.setCurrentIndex(BCPD_DEFAULT_VALUE_NORMALIZATION_OPTIONS)

        self.__ui.postprocessingClusteringScalingDoubleSpinBox.value = POSTPROCESSING_DEFAULT_VALUE_CLUSTERING_SCALING
        self.__ui.processingSmoothingIterationsSpinBox.value = POSTPROCESSING_DEFAULT_VALUE_SMOOTHING_ITERATIONS

    def __setup_combo_box(self, combo_box: QComboBox, enum: Enum, on_selection_changed):
        """
            Method for setting up combo box and its possible values
        """
        for mode in list(enum):
            combo_box.addItem(mode.name, mode)
        if on_selection_changed is not None:
            combo_box.currentIndexChanged.connect(on_selection_changed)
        combo_box.setCurrentIndex(0)

    def __show_kernel_type(self, current_index) -> None:
        """
            Kernel type callback
        """
        if current_index == BcpdKernelType.STANDARD.value:
            show_standard_setting = True
            show_geodesic_settings = False
        else:
            show_standard_setting = False
            show_geodesic_settings = True

        self.__ui.bcpdStandardKernelGroupBox.setVisible(show_standard_setting)
        self.__ui.bcpdGeodesicKernelGroupBox.setVisible(show_geodesic_settings)

    def __show_acceleration_mode(self, current_index: int) -> None:
        """
            Acceleration mode combo box callback
        """
        if current_index == BcpdAccelerationMode.AUTOMATIC.value:
            show_automatic = True
            show_manual = False
        else:
            show_automatic = False
            show_manual = True

        self.__ui.bcpdAccelerationAutomaticGroupBox.setVisible(show_automatic)
        self.__ui.bcpdAccelerationManualGroupBox.setVisible(show_manual)

    def __parse_parameters(self) -> dict:
        """
            Parsing parameters from the UI user option elements
        """
        params = {}
        params[PREPROCESSING_KEY] = self.__parse_parameters_preprocessing()
        params[BCPD_KEY] = self.__parse_parameters_bcpd()
        params[POSTPROCESSING_KEY] = self.__parse_parameters_postprocessing()

        return params

    def __parse_parameters_preprocessing(self) -> dict:
        params = {}

        # Preprocessing
        params[PREPROCESSING_KEY_DOWNSAMPLING_DISTANCE_THRESHOLD] = self.__ui.preprocessingDownsamplingDistanceThresholdDoubleSpinBox.value
        params[PREPROCESSING_KEY_NORMALS_ESTIMATION_RADIUS] = self.__ui.preprocessingNormalsEstimationRadiusDoubleSpinBox.value
        params[PREPROCESSING_KEY_MAX_NN_NORMALS] = self.__ui.preprocessingNormalsEstimationMaxNeighboursSpinBox.value
        params[PREPROCESSING_KEY_FPFH_ESTIMATION_RADIUS] = self.__ui.preprocessingFpfhRadiusDoubleSpinBox.value
        params[PREPROCESSING_KEY_MAX_NN_FPFH] = self.__ui.preprocessingFpfhMaxNeighboursSpinBox.value

        # Registration
        params[REGISTRATION_KEY_MAX_ITERATIONS] = self.__ui.registrationMaxIterationsSpinBox.value
        params[REGISTRATION_KEY_DISTANCE_THRESHOLD] = self.__ui.registrationDistanceThresholdDoubleSpinBox.value
        params[REGISTRATION_KEY_FITNESS_THRESHOLD] = self.__ui.registrationFitnessThresholdDoubleSpinBox.value
        params[REGISTRATION_KEY_ICP_DISTANCE_THRESHOLD] = self.__ui.registrationIcpDistanceThresholdDoubleSpinBox.value

        return params

    def __parse_parameters_bcpd(self) -> dict:
        """
            Parsing parameters from the UI for the BCPD stage
        """
        params = {}

        ## Tuning parameters ##
        params[BCPD_VALUE_KEY_OMEGA] = self.__ui.bcpdOmegaDoubleSpinBox.value
        params[BCPD_VALUE_KEY_LAMBDA] = self.__ui.bcpdLambdaDoubleSpinBox.value
        params[BCPD_VALUE_KEY_BETA] = self.__ui.bcpdBetaDoubleSpinBox.value
        params[BCPD_VALUE_KEY_GAMMA] = self.__ui.bcpdGammaDoubleSpinBox.value

        kappa = self.__ui.bcpdKappaDoubleSpinBox.value
        if kappa < BCPD_MAX_VALUE_KAPPA:  # Setting it to BCPD_MAX_VALUE_KAPPA will behave as "infinity"
            params[BCPD_VALUE_KEY_KAPPA] = kappa

        # if self.ui.bcpdAdvancedParametersCheckBox.checked == True:
        self.__parse_advanced_parameters(params)
        return params

    def __parse_advanced_parameters(self, params: dict) -> None:
        """
            Parsing parameters under the "Advanced" section
        """
        ## Kernel settings ##
        kernel_params = ""
        kernel_type = self.__ui.bcpdKernelTypeComboBox.currentIndex

        if (kernel_type == BcpdKernelType.STANDARD.value):
            selected_kernel = self.__ui.bcpdStandardKernelComboBox.currentIndex
            # Default kernel is Gauss, which does not need to be specified
            if selected_kernel != BCPD_DEFAULT_VALUE_STANDARD_KERNEL_TYPE:
                kernel_params += str(selected_kernel)
        else:  # Geodesic Kernel
            kernel_params += "geodesic" + BCPD_MULTIPLE_VALUES_SEPARATOR

            tau = self.__ui.bcpdGeodesicKernelTauDoubleSpinBox.value
            kernel_params += str(tau) + BCPD_MULTIPLE_VALUES_SEPARATOR

            if self.__ui.bcpdGeodesicKernelInputMeshCheckBox.checked == True:
                input_mesh_path = self.__ui.bcpdGeodesicKernelInputMeshLineEdit.text
                if not os.path.exists(input_mesh_path):
                    print("File '" + input_mesh_path + "' does not exist. Cancelling process...")
                    return
                kernel_params += input_mesh_path
            else:
                kernel_params += str(self.__ui.bcpdGeodesicKernelNeighboursSpinBox.value) + BCPD_MULTIPLE_VALUES_SEPARATOR
                kernel_params += str(self.__ui.bcpdGeodesicKernelRadiusDoubleSpinBox.value)

        if kernel_params != "":
            params[BCPD_VALUE_KEY_KERNEL] = kernel_params

        ## Acceleration settings ##
        if self.__ui.bcpdAccelerationModeComboBox.currentIndex == BcpdAccelerationMode.AUTOMATIC.value:
            if self.__ui.bcpdAccelerationAutomaticVbiCheckBox.checked == True:
                params[BCPD_VALUE_KEY_NYSTORM_G] = 70
                params[BCPD_VALUE_KEY_NYSTORM_P] = 300
                # Option switch without a value
                params[BCPD_VALUE_KEY_KD_TREE] = ""
                params[BCPD_VALUE_KEY_KD_TREE_SCALE] = 7
                params[BCPD_VALUE_KEY_KD_TREE_RADIUS] = 0.15

            if self.__ui.bcpdAccelerationAutomaticPlusPlusCheckBox.checked == True:
                params[BCPD_VALUE_KEY_DOWNSAMPLING] = "B,10000,0.08"
        else:  # Manual acceleration
            if self.__ui.bcpdAccelerationManualNystormGroupBox.checked == True:
                params[BCPD_VALUE_KEY_NYSTORM_G] = self.__ui.bcpdAccelerationManualNystormGSpinBox.value
                params[BCPD_VALUE_KEY_NYSTORM_P] = self.__ui.bcpdAccelerationManualNystormJSpinBox.value
                params[BCPD_VALUE_KEY_NYSTORM_R] = self.__ui.bcpdAccelerationManualNystormRSpinBox.value

            if self.__ui.bcpdAccelerationManualKdTreeGroupBox.checked == True:
                # Option switch without a value
                params[BCPD_VALUE_KEY_KD_TREE] = ""
                params[BCPD_VALUE_KEY_KD_TREE_SCALE] = self.__ui.bcpdAccelerationManualKdTreeScaleDoubleSpinBox.value
                params[BCPD_VALUE_KEY_KD_TREE_RADIUS] = self.__ui.bcpdAccelerationManualKdTreeRadiusDoubleSpinBox.value
                params[BCPD_VALUE_KEY_KD_TREE_THRESHOLD] = self.__ui.bcpdAccelerationManualKdTreeThresholdDoubleSpinBox.value

        ## Downsampling settings ##

        if params.get(BCPD_VALUE_KEY_DOWNSAMPLING) is None:
            params[BCPD_VALUE_KEY_DOWNSAMPLING] = self.__ui.bcpdDownsamplingLineEdit.text

        ## Convergence options ##
        params[BCPD_VALUE_KEY_CONVERGENCE_TOLERANCE] = self.__ui.bcpdConvergenceToleranceDoubleSpinBox.value
        params[BCPD_VALUE_KEY_CONVERGENCE_MIN_ITERATIONS] = self.__ui.bcpdConvergenceMinIterationsSpinBox.value
        params[BCPD_VALUE_KEY_CONVERGENCE_MAX_ITERATIONS] = self.__ui.bcpdConvergenceMaxIterationsSpinBox.value

        ## Normalization options ##
        params[BCPD_VALUE_KEY_NORMALIZATION] = self.__ui.bcpdNormalizationComboBox.currentText.lower()

    def __parse_parameters_postprocessing(self) -> dict:
        params = {}

        params[POSTPROCESSING_KEY_CLUSTERING_SCALING] = self.__ui.postprocessingClusteringScalingDoubleSpinBox.value
        params[POSTPROCESSING_KEY_SMOOTHING_ITERATIONS] = self.__ui.processingSmoothingIterationsSpinBox.value

        return params

    def __generate_model(self) -> None:
        """
            Generate button callback. Calls the Logic's generate_model method and adds the results into the scene
        """
        params = self.__parse_parameters()

        err, generated_polydata, merged_polydata = self.__logic.generate_model(
            self.__ui.sourceNodeSelectionBox.currentNode(),
            self.__ui.targetNodeSelectionBox.currentNode(), params)

        if (err == EXIT_OK):
            model_node = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLModelNode', 'BCPD generated')
            model_node.SetAndObservePolyData(generated_polydata)
            model_node.CreateDefaultDisplayNodes()

            model_node = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLModelNode', 'BCPD merged')
            model_node.SetAndObservePolyData(merged_polydata)
            model_node.CreateDefaultDisplayNodes()
