import type { ReactNode } from "react";
import { useEffect, useMemo, useState } from "react";
import { Canvas } from "@react-three/fiber";
import { Grid, OrbitControls } from "@react-three/drei";
import type { BufferGeometry } from "three";
import { STLLoader } from "three/examples/jsm/loaders/STLLoader.js";
import type { ArmSnapshot } from "./types";

const DEG = Math.PI / 180;

interface LinkMeshProps {
  stlUrl?: string;
  color: string;
  children: ReactNode;
}

function LinkMesh({ stlUrl, color, children }: LinkMeshProps) {
  const [geom, setGeom] = useState<BufferGeometry | null>(null);
  useEffect(() => {
    if (!stlUrl) {
      setGeom(null);
      return;
    }
    let cancelled = false;
    const loader = new STLLoader();
    loader.load(
      stlUrl,
      (g) => {
        if (!cancelled) {
          g.computeVertexNormals();
          setGeom(g);
        }
      },
      undefined,
      () => {
        if (!cancelled) setGeom(null);
      }
    );
    return () => {
      cancelled = true;
    };
  }, [stlUrl]);

  if (geom) {
    return (
      <mesh geometry={geom} castShadow receiveShadow>
        <meshStandardMaterial color={color} metalness={0.2} roughness={0.6} />
      </mesh>
    );
  }
  return <>{children}</>;
}

interface ArmProps {
  angles: Record<string, number>;
  models: Record<string, string | undefined>;
}

function Arm({ angles, models }: ArmProps) {
  const base = (angles.base ?? 90) - 90;
  const shoulder = (angles.shoulder ?? 90) - 90;
  const elbow = (angles.elbow ?? 90) - 90;
  const gripper = (angles.gripper ?? 30) * 0.5;

  return (
    <group>
      {/* Base — rotates about Y */}
      <group rotation={[0, base * DEG, 0]}>
        <LinkMesh stlUrl={models.base} color="#455060">
          <mesh position={[0, 0.15, 0]} castShadow receiveShadow>
            <cylinderGeometry args={[0.35, 0.4, 0.3, 32]} />
            <meshStandardMaterial color="#455060" />
          </mesh>
        </LinkMesh>

        {/* Shoulder pivot sits on top of base */}
        <group position={[0, 0.3, 0]} rotation={[shoulder * DEG, 0, 0]}>
          <mesh castShadow>
            <sphereGeometry args={[0.12, 24, 24]} />
            <meshStandardMaterial color="#2a2f3a" />
          </mesh>

          <LinkMesh stlUrl={models.shoulder} color="#7089a8">
            <mesh position={[0, 0.55, 0]} castShadow receiveShadow>
              <boxGeometry args={[0.18, 1.1, 0.18]} />
              <meshStandardMaterial color="#7089a8" />
            </mesh>
          </LinkMesh>

          {/* Elbow pivot at end of upper arm */}
          <group position={[0, 1.1, 0]} rotation={[elbow * DEG, 0, 0]}>
            <mesh castShadow>
              <sphereGeometry args={[0.1, 24, 24]} />
              <meshStandardMaterial color="#2a2f3a" />
            </mesh>

            <LinkMesh stlUrl={models.elbow} color="#8fa4be">
              <mesh position={[0, 0.45, 0]} castShadow receiveShadow>
                <boxGeometry args={[0.14, 0.9, 0.14]} />
                <meshStandardMaterial color="#8fa4be" />
              </mesh>
            </LinkMesh>

            {/* Wrist / gripper mount */}
            <group position={[0, 0.9, 0]}>
              <mesh castShadow>
                <boxGeometry args={[0.22, 0.12, 0.22]} />
                <meshStandardMaterial color="#2a2f3a" />
              </mesh>

              <LinkMesh stlUrl={models.gripper} color="#d0b070">
                <group>
                  <mesh
                    position={[0.08 + gripper * 0.005, 0.16, 0]}
                    castShadow
                  >
                    <boxGeometry args={[0.04, 0.24, 0.14]} />
                    <meshStandardMaterial color="#d0b070" />
                  </mesh>
                  <mesh
                    position={[-0.08 - gripper * 0.005, 0.16, 0]}
                    castShadow
                  >
                    <boxGeometry args={[0.04, 0.24, 0.14]} />
                    <meshStandardMaterial color="#d0b070" />
                  </mesh>
                </group>
              </LinkMesh>
            </group>
          </group>
        </group>
      </group>
    </group>
  );
}

interface Props {
  snapshot: ArmSnapshot | null;
  showTargetGhost?: boolean;
}

export function ArmViewer({ snapshot, showTargetGhost = true }: Props) {
  const models = useMemo(
    () => ({
      base: "/models/base.stl",
      shoulder: "/models/shoulder.stl",
      elbow: "/models/elbow.stl",
      gripper: "/models/gripper.stl",
    }),
    []
  );

  return (
    <Canvas
      shadows
      camera={{ position: [2.2, 1.8, 2.6], fov: 45 }}
      style={{ background: "#0e1116" }}
    >
      <ambientLight intensity={0.4} />
      <directionalLight
        position={[3, 5, 3]}
        intensity={1.1}
        castShadow
        shadow-mapSize={[1024, 1024]}
      />
      <Grid
        args={[8, 8]}
        cellSize={0.25}
        cellColor="#2a2f3a"
        sectionSize={1}
        sectionColor="#3a4252"
        infiniteGrid
        fadeDistance={12}
        position={[0, 0, 0]}
      />
      {snapshot && (
        <>
          {showTargetGhost && (
            <group>
              <Arm angles={snapshot.target} models={{}} />
            </group>
          )}
          <Arm angles={snapshot.current} models={models} />
        </>
      )}
      <OrbitControls target={[0, 1, 0]} makeDefault />
    </Canvas>
  );
}
