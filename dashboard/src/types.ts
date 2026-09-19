export type JointName = string;

export interface ArmSnapshot {
  type: "state";
  enabled: boolean;
  hardware: boolean;
  current: Record<JointName, number>;
  target: Record<JointName, number>;
  limits: Record<JointName, [number, number]>;
}

export type OutgoingMessage =
  | { type: "set_target"; joint: JointName; degrees: number }
  | { type: "set_targets"; joints: Record<JointName, number> }
  | { type: "home" }
  | { type: "set_enabled"; enabled: boolean };
