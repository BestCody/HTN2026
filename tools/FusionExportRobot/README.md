# FusionExportRobot

This is the Fusion script bundle. In Fusion, open **Utilities -> Scripts and
Add-Ins**, choose **+ -> Script or add-in from device**, and select this entire
`FusionExportRobot` directory. Do not select only the Python file.

Run it with the imported `Sharma_Ishaan_robotAssem.SLDASM` design active. The
exporter writes raw meshes for every occurrence and uses the checked
`robot_models/four_dof_desktop_arm/fusion_mapping.json` file to group the
original SolidWorks occurrence names into five canonical links. Renaming or
re-parenting the imported parts is unnecessary. It also captures every analytic
cylindrical face in assembly coordinates so the four revolute axes can be fit
from the CAD export. You do not need to unground components or manually create
Fusion joints.

The entry point loads the maintained implementation from
`tools/fusion_export_robot.py`, so keep this directory in its current repository
location.
