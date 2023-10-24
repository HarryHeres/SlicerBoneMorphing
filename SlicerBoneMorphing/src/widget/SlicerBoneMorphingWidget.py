from src.logic.Constants import EXIT_OK
import ctk
import slicer
from slicer.ScriptedLoadableModule import ScriptedLoadableModuleWidget
from src.logic.SlicerBoneMorphingLogic import SlicerBoneMorphingLogic


class SlicerBoneMorphingWidget(ScriptedLoadableModuleWidget):
  """Uses ScriptedLoadableModuleWidget base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self, parent):
    """Called when the application opens the module the first time and the widget is initialized."""
    ScriptedLoadableModuleWidget.__init__(self, parent)


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

  def generate_model(self):
      code, vtk_polydata = self.logic.generate_model(self.ui.sourceNodeSelectionBox.currentNode(), self.ui.targetNodeSelectionBox.currentNode())

      if(code == EXIT_OK):
        model_node = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLModelNode', 'TEST')
        model_node.SetAndObservePolyData(vtk_polydata)
        model_node.CreateDefaultDisplayNodes()  # Optional


            
        
