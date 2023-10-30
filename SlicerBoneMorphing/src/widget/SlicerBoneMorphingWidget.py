import src.logic.Constants 
import ctk
import slicer
from slicer.ScriptedLoadableModule import ScriptedLoadableModuleWidget
from src.logic.SlicerBoneMorphingLogic import SlicerBoneMorphingLogic
from enum import Enum

class BCPDMode(Enum):
    STANDARD = "bcpd"
    GEODESIC = "gbcpd"

class BCPDAccelerationMode(Enum):
    AUTOMATIC = "Automatic"
    MANUAL = "Manual"

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

    self.ui.sourceNodeSelectionBox.setMRMLScene(slicer.mrmlScene)
    self.ui.targetNodeSelectionBox.setMRMLScene(slicer.mrmlScene)
    self.ui.generateModelButton.clicked.connect(self.generate_model)

    self.set_default_parameters()


  def set_default_parameters(self):
    self.ui.bcpdAdvancedControlsGroupBox.setVisible(False)

    acceleration_combo_box = self.ui.bcpdAccelerationModeComboBox
    for mode in list(BCPDAccelerationMode):
        acceleration_combo_box.addItem(mode.value, mode)
    acceleration_combo_box.setCurrentIndex(0)
    acceleration_combo_box.currentIndexChanged.connect(self.show_acceleration_mode)

  def show_acceleration_mode(self, currentMode):
    if currentMode == BCPDAccelerationMode.AUTOMATIC.name:
        showAutomatic = True 
        showManual = False
    else:
        showAutomatic = False
        showManual = True

    
    self.ui.bcpdAccelerationAutomaticGroupBox.setVisible(showAutomatic)
    self.ui.bcpdAccelerationManualGroupBox.setVisible(showManual)


  def generate_model(self):
    mode = self.ui.bcpdModeComboBox.currentData
    if(mode == None):
        print("Please, select a proper BCPD registration mode")
    self.bcpd_options["mode"] = mode

    dvl = self.ui.bcpdDeformationVectorLengthSpinBox.value
    self.bcpd_options["b"] = dvl

    print("TEST: " + str(mode.value))

      # code, vtk_polydata = self.logic.generate_model(self.ui.sourceNodeSelectionBox.currentNode(), self.ui.targetNodeSelectionBox.currentNode())

      # if(code == EXIT_OK):
      #   model_node = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLModelNode', 'BCPD result')
      #   model_node.SetAndObservePolyData(vtk_polydata)
      #   model_node.CreateDefaultDisplayNodes()  

    


            
        
