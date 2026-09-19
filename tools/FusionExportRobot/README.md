# FusionExportRobot

This is the Fusion script bundle. In Fusion, open **Utilities → Scripts and
Add-Ins**, choose **+ → Script or add-in from device**, and select this entire
`FusionExportRobot` directory. Do not select only the Python file.

Run it with the imported `Sharma_Ishaan_robotAssem.SLDASM` design active. The
exporter writes raw meshes for every occurrence and marks the result complete
only after the five canonical link components and four configured joints have
been named as described in `robot_models/four_dof_desktop_arm/README.md`.

The entry point loads the maintained implementation from
`tools/fusion_export_robot.py`, so keep this directory in its current repository
location.
