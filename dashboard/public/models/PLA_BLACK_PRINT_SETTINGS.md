# Black PLA print settings

These settings cover every current production STL. They assume a **0.4 mm nozzle**, dry 1.75 mm PLA, and a reasonably calibrated printer. The CSV beside this file contains the complete per-part table.

## Global slicer profile

- Nozzle: start at **205 C**; use the filament manufacturer's range if different.
- Bed: **60 C first layer, 55 C afterward**.
- Fan: **0% for layers 1-2, then 100% from layer 4**.
- Line width: **0.42-0.45 mm**. First layer: **0.24 mm high**, 20 mm/s.
- Speeds: outer wall **35 mm/s**, inner wall **55 mm/s**, infill **65 mm/s**, small perimeters and gear teeth **25 mm/s**.
- Infill pattern: gyroid or cubic. For rows marked 100%, use rectilinear or concentric solid infill.
- Wall order: inner/outer/inner or the slicer's dimensional-accuracy mode. Slow the external wall rather than over-extruding it.
- Seam: paint it away from gear teeth, bearing journals, horn pockets, bushings, and gripping faces.
- Supports: 55-degree threshold, 0.20 mm top Z gap, 0.35 mm XY gap, two interface layers. Use only where the per-part table requests them.
- Elephant-foot compensation: start at **0.15 mm**. Never scale a mechanical part to correct a fit problem.
- Dimensional target: ordinary M3 clearance holes about **3.4-3.6 mm** after cleanup; the long P12/P13/P18/P19 through-passages are modeled at **3.8 mm**. M5 clearance holes should finish about **5.3-5.5 mm**. Gauge or hand-ream every load-bearing hole; do not force screws through undersized holes.

## Strength modifiers

Where the CSV says **local 100% infill**, add a cylindrical or box modifier extending at least 8 mm around the named hole, hub, gear root, or lug. Walls carry most of the load, so do not reduce the specified wall counts merely because the infill is high.

The seated STAR horn bridges and P41 retainer are printed solid because their small horn interfaces transmit servo torque. P09 and P15 are also printed solid so they can be reamed without exposing sparse infill.

Lay `STAR_yaw` on its maximum-Y broad face, and lay the shoulder/elbow STAR bridges on their maximum-Z broad faces. Print P12/P13/P18/P19 with the long axis in the bed plane. Lay P38 and P39 on their maximum-Y broad faces and block support from every journal and bore.

## Black PLA cautions

Black PLA absorbs heat quickly in sunlight. Keep the finished arm below about 45 C, away from a hot car or sunny window, and avoid holding a stalled MG996R against a hard stop. Heat from a stalled servo can soften the nearby horn pocket even when the rest of the arm is cool.

## Before committing both arms

1. Print one mock servo, one P09 bushing, one P15 bushing, and one seated STAR bridge.
2. Confirm the real servo body slides into the mount without spreading it.
3. Confirm the real horn fully seats in the bridge without rocking.
4. Confirm the bushing rotates freely after light hand reaming.
5. Print one P36/P37 gear pair and verify smooth hand meshing through the complete 0-35 degree jaw range before printing the duplicates.
6. Use the real M3/M5 hardware as gauges. Hand-ream tight bores and polish printed journals; any part that must be forced together is a failed fit, not an assembly step.

Print **two copies of every production STL**. The mock servo is a shared fit gauge and normally needs only one copy.
