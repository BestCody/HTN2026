Drop STL files exported from Fusion 360 here to replace the placeholder geometry:

  base.stl      - fixed base pedestal, oriented with its rotation axis on world Y
  shoulder.stl  - upper arm link, meshed with its shoulder pivot at origin
  elbow.stl    - forearm link, meshed with its elbow pivot at origin
  gripper.stl   - gripper assembly, meshed with the wrist mount at origin

Export procedure (Fusion 360 free/personal):
  1. In the browser tree, right-click a component -> Save as Mesh.
  2. Format: STL (Binary), Units: meters (or scale in viewer), Refinement: Medium.
  3. Save the file with the exact name above into this directory.

Any file that isn't present will fall back to the default placeholder shape.
