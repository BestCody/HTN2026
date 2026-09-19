import type { ArmSnapshot, OutgoingMessage } from "./types";

interface Props {
  snapshot: ArmSnapshot | null;
  onSend: (msg: OutgoingMessage) => void;
}

export function JointControls({ snapshot, onSend }: Props) {
  if (!snapshot) return <div className="panel">Waiting for controller…</div>;

  const jointNames = Object.keys(snapshot.limits);
  const hasPositionJoints =
    snapshot.modes == null || Object.values(snapshot.modes).some((mode) => mode === "position");

  return (
    <div className="panel">
      <header className="panel-head">
        <div>
          <div className="pill">
            {snapshot.hardware ? "PCA9685" : "MOCK DRIVER"}
          </div>
          <div className="pill" data-enabled={snapshot.enabled}>
            {snapshot.enabled ? "ENABLED" : "DISABLED"}
          </div>
        </div>
        <div className="row">
          <button
            className="btn"
            onClick={() =>
              onSend({ type: "set_enabled", enabled: !snapshot.enabled })
            }
          >
            {snapshot.enabled ? "Disable" : "Enable"}
          </button>
          {hasPositionJoints && (
            <button className="btn" onClick={() => onSend({ type: "home" })}>
              Home
            </button>
          )}
          <button
            className="btn btn-danger"
            onClick={() => onSend({ type: "set_enabled", enabled: false })}
          >
            E-Stop
          </button>
        </div>
      </header>

      <div className="joints">
        {jointNames.map((name) => {
          const [min, max] = snapshot.limits[name];
          const current = snapshot.current[name] ?? 0;
          const target = snapshot.target[name] ?? 0;
          const continuous = snapshot.modes?.[name] === "continuous";
          const speed = snapshot.speeds?.[name] ?? 0;
          const speedLabel =
            speed === 0
              ? "stopped"
              : `${speed < 0 ? "reverse" : "forward"} ${Math.abs(speed).toFixed(0)}%`;
          return (
            <div key={name} className="joint">
              <div className="joint-head">
                <span className="joint-name">{name}</span>
                {continuous ? (
                  <span className="joint-values dim">{speedLabel}</span>
                ) : (
                  <span className="joint-values">
                    <span title="target">{target.toFixed(1)}°</span>
                    <span className="dim" title="current">
                      {" / "}
                      {current.toFixed(1)}°
                    </span>
                  </span>
                )}
              </div>
              {continuous ? (
                <>
                  <input
                    type="range"
                    min={-100}
                    max={100}
                    step={5}
                    value={speed}
                    disabled={!snapshot.enabled}
                    onChange={(e) =>
                      onSend({
                        type: "set_speed",
                        joint: name,
                        speed: Number(e.target.value),
                      })
                    }
                  />
                  <div className="joint-range">
                    <span>← Reverse</span>
                    <button
                      className="btn"
                      disabled={!snapshot.enabled || speed === 0}
                      onClick={() => onSend({ type: "set_speed", joint: name, speed: 0 })}
                    >
                      Stop
                    </button>
                    <span>Forward →</span>
                  </div>
                </>
              ) : (
                <>
                  <input
                    type="range"
                    min={min}
                    max={max}
                    step={0.5}
                    value={target}
                    onChange={(e) =>
                      onSend({
                        type: "set_target",
                        joint: name,
                        degrees: Number(e.target.value),
                      })
                    }
                  />
                  <div className="joint-range">
                    <span>{min}°</span>
                    <span>{max}°</span>
                  </div>
                </>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
