"""Read-only inspection of robot CAD and print-package evidence."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path

_CORE_3MF_NS = {"m": "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"}


@dataclass(frozen=True)
class ThreeMFReport:
    unit: str
    build_items: int
    plate_count: int
    mesh_names: tuple[str, ...]
    triangle_count: int
    repaired_meshes: tuple[str, ...]
    has_joint_metadata: bool
    layout_kind: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def inspect_3mf(path: str | Path) -> ThreeMFReport:
    """Inspect a slicer 3MF without extracting or trusting archive paths."""

    package = Path(path)
    if not package.is_file():
        raise FileNotFoundError(f"3MF package does not exist: {package}")
    with zipfile.ZipFile(package) as archive:
        names = set(archive.namelist())
        required = {"3D/3dmodel.model", "Metadata/model_settings.config"}
        missing = required - names
        if missing:
            raise ValueError("3MF package is missing: " + ", ".join(sorted(missing)))

        root = ET.fromstring(archive.read("3D/3dmodel.model"))
        unit = str(root.get("unit", "")).casefold()
        if unit not in {"micron", "millimeter", "centimeter", "inch", "foot", "meter"}:
            raise ValueError(f"3MF package has an unsupported unit: {unit or '<missing>'}")
        build_items = len(root.findall(".//m:build/m:item", _CORE_3MF_NS))

        settings = ET.fromstring(archive.read("Metadata/model_settings.config"))
        meshes: dict[str, int] = {}
        repaired: set[str] = set()
        for item in settings.findall("./object"):
            name_node = next(
                (node for node in item.findall("./metadata") if node.get("key") == "name"),
                None,
            )
            name = "unnamed" if name_node is None else str(name_node.get("value", "unnamed"))
            statistic = item.find("./part/mesh_stat")
            if statistic is None:
                continue
            triangles = int(statistic.get("face_count", "0"))
            meshes.setdefault(name, triangles)
            if any(
                int(value) != 0
                for key, value in statistic.attrib.items()
                if key != "face_count" and re.fullmatch(r"-?\d+", value)
            ):
                repaired.add(name)

        plates = len(
            [
                name
                for name in names
                if re.fullmatch(r"Metadata/plate_\d+\.png", name)
            ]
        )
        has_joint_metadata = any("joint" in name.casefold() for name in names)
        return ThreeMFReport(
            unit=unit,
            build_items=build_items,
            plate_count=plates,
            mesh_names=tuple(sorted(meshes, key=str.casefold)),
            triangle_count=sum(meshes.values()),
            repaired_meshes=tuple(sorted(repaired, key=str.casefold)),
            has_joint_metadata=has_joint_metadata,
            layout_kind="print_plate" if plates else "unknown",
        )
