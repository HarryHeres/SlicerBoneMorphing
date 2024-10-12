from slicer.ScriptedLoadableModule import ScriptedLoadableModule
from src.widget.SlicerBoneMorphingWidget import SlicerBoneMorphingWidget


class SlicerBoneMorphing(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "Slicer Bone Morphing"
        self.parent.categories = ["Morphing"]
        self.parent.dependencies = []
        self.parent.contributors = ["Jan Heres, Eva C. Herbst (ETH Zurich), Arthur Porto (Louisiana State University)"]

        self.parent.helpText = """
        This module gives a user ability to recreate bone mesh models based on their partial scan.
        Please, start with importing your model and checking out the options on the left side afterwards.
        Version: 0.2.0-rc.2
        """

        self.parent.acknowledgementText = """
        Credits: Jan Heres
        I would love to thank my awesome colleagues Eva C. Herbst and Arthur Porto for their priceless contributions to this project!"""
