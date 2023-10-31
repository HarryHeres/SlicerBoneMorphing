from src.logic.Constants import *
import ctk
import slicer
from slicer.ScriptedLoadableModule import ScriptedLoadableModuleWidget
from src.logic.SlicerBoneMorphingLogic import SlicerBoneMorphingLogic

from src.logic.Enums import BCPDAccelerationMode, BCPDKernelMode, BCPDNormalizationOptions, BCPDStandardKernelMode
from enum import Enum
from qt import QComboBox

import os 

class SlicerBoneMorphingWidget(ScriptedLoadableModuleWidget):
  """Uses ScriptedLoadableModuleWidget base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self, parent):
    """Called when the application opens the module the first time and the widget is initialized."""
    ScriptedLoadableModuleWidget.__init__(self, parent)
    self.bcpd_options = {}


  def setup(self):
    """Called when the application opens the module the first time and the widget is initialized."""
    ScriptedLoadableModuleWidget.setup(self)
    
    # Load widget from .ui file (created by Qt Designer)
    self.uiWidget = slicer.util.loadUI(self.resourcePath("UI/SlicerBoneMorphing.ui"))
    self.layout.addWidget(self.uiWidget)
    self.ui = slicer.util.childWidgetVariables(self.uiWidget)
    self.logic = SlicerBoneMorphingLogic(self)

    self.setup_ui()


  def setup_ui(self):
    """
        Method that sets up all UI elements and their dependencies
    """

    self.ui.sourceNodeSelectionBox.setMRMLScene(slicer.mrmlScene)
    self.ui.targetNodeSelectionBox.setMRMLScene(slicer.mrmlScene)

    self.ui.bcpdAdvancedControlsGroupBox.setVisible(False)

    self.setup_combo_box(self.ui.bcpdKernelTypeComboBox, BCPDKernelMode, self.show_kernel_type)
    self.ui.bcpdGeodesicKernelGroupBox.setVisible(False)

    self.setup_combo_box(self.ui.bcpdStandardKernelComboBox, BCPDStandardKernelMode, None)

    self.setup_combo_box(self.ui.bcpdAccelerationModeComboBox, BCPDAccelerationMode, self.show_acceleration_mode)
    self.ui.bcpdAccelerationManualGroupBox.setVisible(False)

    self.ui.bcpdGeodesicKernelInputMeshLineEdit.setVisible(False)
    self.ui.bcpdGeodesicKernelInputMeshLabel.setVisible(False)

    self.setup_combo_box(self.ui.bcpdNormalizationComboBox, BCPDNormalizationOptions, None)
 
    self.ui.bcpdResetParametersPushButton.clicked.connect(self.reset_parameters_to_default)
    self.ui.generateModelButton.clicked.connect(self.generate_model)


  def reset_parameters_to_default(self):
      ## Tuning parameters ## 
      self.ui.bcpdOmegaDoubleSpinBox.value = BCPD_DEFAULT_VALUE_OMEGA
      self.ui.bcpdLambdaDoubleSpinBox.value = BCPD_DEFAULT_VALUE_LAMBDA
      self.ui.bcpdBetaDoubleSpinBox.value = BCPD_DEFAULT_VALUE_BETA
      self.ui.bcpdGammaDoubleSpinBox.value = BCPD_DEFAULT_VALUE_GAMMA
      self.ui.bcpdKappaDoubleSpinBox.value = BCPD_DEFAULT_VALUE_KAPPA

      # self.ui.bcpdAdvancedParametersCheckBox.checked = False
      
      ## Kernel parameters ##
      self.ui.bcpdKernelTypeComboBox.setCurrentIndex(BCPD_DEFAULT_VALUE_KERNEL_TYPE)
      self.ui.bcpdStandardKernelComboBox.setCurrentIndex(BCPD_DEFAULT_VALUE_STANDARD_KERNEL_TYPE)
      self.ui.bcpdGeodesicKernelTauDoubleSpinBox.value = BCPD_DEFAULT_VALUE_TAU
      self.ui.bcpdGeodesicKernelInputMeshCheckBox.checked = False
      self.ui.bcpdGeodesicKernelInputMeshLineEdit.text = BCPD_DEFAULT_VALUE_INPUT_MESH_PATH
      self.ui.bcpdGeodesicKernelNeighboursSpinBox.value = BCPD_DEFAULT_VALUE_KERNEL_NEIGBOURS
      self.ui.bcpdGeodesicKernelRadiusDoubleSpinBox.value = BCPD_DEFAULT_VALUE_KERNEL_NEIGHBOUR_RADIUS
      self.ui.bcpdGeodesicKernelBetaDoubleSpinBox.value = BCPD_DEFAULT_VALUE_KERNEL_BETA
      self.ui.bcpdGeodesicKernelKTildeDoubleSpinBox.value = BCPD_DEFAULT_VALUE_KERNEL_K_TILDE
      self.ui.bcpdGeodesicKernelEpsilonDoubleSpinBox.value = BCPD_DEFAULT_VALUE_KERNEL_EPSILON


      ## Acceleration parameters ##
      self.ui.bcpdAccelerationModeComboBox.setCurrentIndex(BCPD_DEFAULT_VALUE_ACCELERATION_MODE)
      self.ui.bcpdAccelerationAutomaticVbiCheckBox.checked = True
      self.ui.bcpdAccelerationAutomaticPlusPlusCheckBox.checked = True
      self.ui.bcpdAccelerationManualNystormGSpinBox.value = BCPD_DEFAULT_VALUE_ACCELERATION_NYSTORM_SAMPLES_G
      self.ui.bcpdAccelerationManualNystormJSpinBox.value = BCPD_DEFAULT_VALUE_ACCELERATION_NYSTORM_SAMPLES_J
      self.ui.bcpdAccelerationManualNystormRSpinBox.value = BCPD_DEFAULT_VALUE_ACCELERATION_NYSTORM_SAMPLES_R
      self.ui.bcpdAccelerationManualKdTreeScaleDoubleSpinBox.value = BCPD_DEFAULT_VALUE_ACCELERATION_KD_TREE_SCALE
      self.ui.bcpdAccelerationManualKdTreeRadiusDoubleSpinBox.value = BCPD_DEFAULT_VALUE_ACCELERATION_KD_TREE_RADIUS
      self.ui.bcpdAccelerationManualKdTreeThresholdDoubleSpinBox.value = BCPD_DEFAULT_VALUE_ACCELERATION_KD_TREE_SIGMA_THRESHOLD

      ## Downsampling options ##
      self.ui.bcpdDownsamplingLineEdit.text = BCPD_DEFAULT_VALUE_DOWNSAMPLING_OPTIONS

      ## Convergence options ## 
      self.ui.bcpdConvergenceToleranceDoubleSpinBox.value = BCPD_DEFAULT_VALUE_CONVERGENCE_TOLERANCE
      self.ui.bcpdConvergenceMaxIterationsSpinBox.value = BCPD_DEFAULT_VALUE_CONVERGENCE_MAX_ITERATIONS
      self.ui.bcpdConvergenceMinIterationsSpinBox.value = BCPD_DEFAULT_VALUE_CONVERGENCE_MIN_ITERATIONS

      ## Normalization options ## 
      self.ui.bcpdNormalizationComboBox.setCurrentIndex(BCPD_DEFAULT_VALUE_NORMALIZATION_OPTIONS)
      


  def setup_combo_box(self, comboBox: QComboBox, enum, onSelectionChanged): 
    """
        Method for setting up combo box and its possible values
    """
    for mode in list(enum):
        comboBox.addItem(mode.name, mode)
    if(onSelectionChanged != None):
        comboBox.currentIndexChanged.connect(onSelectionChanged)
    comboBox.setCurrentIndex(0) 

  def show_kernel_type(self, selectedIndex):
    """
        Kernel type callback
    """
    if selectedIndex == BCPDKernelMode.STANDARD.value: 
        showStandardSettings = True
        showGeodesicSettings = False
    else:
        showStandardSettings = False
        showGeodesicSettings = True

    self.ui.bcpdStandardKernelGroupBox.setVisible(showStandardSettings)
    self.ui.bcpdGeodesicKernelGroupBox.setVisible(showGeodesicSettings)

  def show_acceleration_mode(self, selectedIndex):
    """
        Acceleration mode combo box callback
    """
    if selectedIndex == BCPDAccelerationMode.AUTOMATIC.value:
        showAutomatic = True 
        showManual = False
    else:
        showAutomatic = False
        showManual = True
    
    self.ui.bcpdAccelerationAutomaticGroupBox.setVisible(showAutomatic)
    self.ui.bcpdAccelerationManualGroupBox.setVisible(showManual)

    def parse_parameters(self): 
        params = {}
        
        ## Tuning parameters ## 
        params[BCPD_VALUE_KEY_OMEGA] = self.ui.bcpdOmegaDoubleSpinBox.value
        params[BCPD_VALUE_KEY_LAMBDA] = self.ui.bcpdLambdaDoubleSpinBox.value
        params[BCPD_VALUE_KEY_BETA] = self.ui.bcpdBetaDoubleSpinBox.value
        params[BCPD_VALUE_KEY_GAMMA] = self.ui.bcpdGammaDoubleSpinBox.value
        params[BCPD_VALUE_KEY_KAPPA] = self.ui.bcpdKappaDoubleSpinBox.value

        if self.ui.bcpdAdvancedParametersCheckBox.checked == True:
            return self.parse_advanced_parameters(params)

        return params

  def parse_advanced_parameters(self, params: dict):
      if(self.ui.bcpdKernelTypeComboBox.value == BCPDKernelMode.STANDARD.value):
          ## Tuning params ## 
          params[BCPD_VALUE_KEY_OMEGA] = self.ui.bcpdOmegaDoubleSpinBox.value
          params[BCPD_VALUE_KEY_LAMBDA] = self.ui.bcpdLambdaDoubleSpinBox.value
          params[BCPD_VALUE_KEY_BETA] = self.ui.bcpdBetaDoubleSpinBox.value
          params[BCPD_VALUE_KEY_GAMMA] = self.ui.bcpdGammaDoubleSpinBox.value

          kappa = self.ui.bcpdKappaDoubleSpinBox.value
          if kappa < BCPD_MAX_VALUE_KAPPA: # Setting it to max behaves like "infinity" 
              params[BCPD_VALUE_KEY_KAPPA] = kappa 

          ## Kernel settings ##
          kernelType = self.ui.bcpdKernelTypeComboBox.getCurrentIndex()
          kernelParams = ""
          if(kernelType == BCPDKernelMode.STANDARD.value):
              selectedKernel = self.ui.bcpdStandardKernelComboBox.getCurrentIndex()
              if selectedKernel != BCPD_DEFAULT_VALUE_STANDARD_KERNEL_TYPE: # Default kernel is Gauss, which does not need to be specified 
                  kernelParams += str(selectedKernel)
          else: # Geodesic Kernel 
              kernelParams += "geodesic" + BCPD_MULTIPLE_VALUES_SEPARATOR   

              tau = self.ui.bcpdGeodesicKernelTauDoubleSpinBox.value
              kernelParams += str(tau) + BCPD_MULTIPLE_VALUES_SEPARATOR

              if self.ui.bcpdGeodesicKernelInputMeshCheckBox.checked == True:
                  input_mesh_path = self.ui.bcpdGeodesicKernelInputMeshLineEdit.text
                  if not os.path.exists(input_mesh_path):
                      print("File '" + input_mesh_path + "' does not exist. Cancelling process...")
                      return EXIT_FAILURE 
                  kernelParams += input_mesh_path 
              else:
                  kernelParams += str(self.ui.bcpdGeodesicKernelNeighboursSpinBox.value) + BCPD_MULTIPLE_VALUES_SEPARATOR
                  kernelParams += str(self.ui.bcpdGeodesicKernelRadiusDoubleSpinBox.value) 

              params[BCPD_VALUE_KEY_KERNEL] = kernelParams


  def generate_model(self):
    """
        Generate button callback
    """
    # mode = self.ui.bcpdModeComboBox.currentData
    # if(mode == None):
    #     print("Please, select a proper BCPD registration mode")
    # self.bcpd_options["mode"] = mode

    # dvl = self.ui.bcpdDeformationVectorLengthSpinBox.value
    # self.bcpd_options["b"] = dvl


      # code, vtk_polydata = self.logic.generate_model(self.ui.sourceNodeSelectionBox.currentNode(), self.ui.targetNodeSelectionBox.currentNode())

    # if(code == EXIT_OK):
    #     model_node = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLModelNode', 'BCPD result')
    #     model_node.SetAndObservePolyData(vtk_polydata)
    #     model_node.CreateDefaultDisplayNodes()  

    


            
        
