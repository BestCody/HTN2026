import { useEffect, useMemo, useState } from "react";
import { ArmViewer } from "./ArmViewer";
import { JointControls } from "./JointControls";
import { useArmSocket } from "./useArmSocket";

const STORAGE_KEY = "arm-ws-url";

function defaultWsUrl(): string {
  const saved = localStorage.getItem(STORAGE_KEY);
  if (saved) return saved;
  const host = window.location.hostname || "arm.local";
  const controllerHost = host === "localhost" || host === "127.0.0.1" ? "arm.local" : host;
  const proto = window.location.protocol === "https:" ? "wss" : "ws";
  return `${proto}://${controllerHost}:8000/ws`;
}

export default function App() {
  const [wsUrl, setWsUrl] = useState<string>(defaultWsUrl);
  const [draft, setDraft] = useState<string>(wsUrl);
  const { snapshot, status, send } = useArmSocket(wsUrl);

  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, wsUrl);
  }, [wsUrl]);

  const statusLabel = useMemo(() => {
    if (status === "open") return "connected";
    if (status === "connecting") return "connecting…";
    return "disconnected";
  }, [status]);

  return (
    <div className="app">
      <aside className="side">
        <h1>Arm 1</h1>
        <div className="ws-row">
          <input
            className="ws-input"
            value={draft}
            spellCheck={false}
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") setWsUrl(draft);
            }}
          />
          <button className="btn" onClick={() => setWsUrl(draft)}>
            Connect
          </button>
        </div>
        <div className={`status status-${status}`}>{statusLabel}</div>
        <JointControls snapshot={snapshot} onSend={send} />
      </aside>
      <main className="viewer">
        <ArmViewer snapshot={snapshot} />
      </main>
    </div>
  );
}
