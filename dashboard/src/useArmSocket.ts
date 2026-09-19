import { useEffect, useRef, useState } from "react";
import type { ArmSnapshot, OutgoingMessage } from "./types";

export type ConnectionStatus = "connecting" | "open" | "closed";

export function useArmSocket(url: string) {
  const [snapshot, setSnapshot] = useState<ArmSnapshot | null>(null);
  const [status, setStatus] = useState<ConnectionStatus>("connecting");
  const socketRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    let cancelled = false;
    let ws: WebSocket | null = null;
    let reconnectTimer: number | null = null;

    const connect = () => {
      if (cancelled) return;
      setStatus("connecting");
      try {
        ws = new WebSocket(url);
      } catch {
        setStatus("closed");
        reconnectTimer = window.setTimeout(connect, 1500);
        return;
      }
      socketRef.current = ws;

      ws.addEventListener("open", () => setStatus("open"));
      ws.addEventListener("message", (event) => {
        try {
          const parsed = JSON.parse(event.data) as ArmSnapshot;
          if (parsed.type === "state") setSnapshot(parsed);
        } catch {
          /* ignore malformed */
        }
      });
      const scheduleReconnect = () => {
        setStatus("closed");
        socketRef.current = null;
        if (!cancelled) reconnectTimer = window.setTimeout(connect, 1500);
      };
      ws.addEventListener("close", scheduleReconnect);
      ws.addEventListener("error", () => ws?.close());
    };

    connect();

    return () => {
      cancelled = true;
      if (reconnectTimer !== null) window.clearTimeout(reconnectTimer);
      ws?.close();
      socketRef.current = null;
    };
  }, [url]);

  const send = (msg: OutgoingMessage) => {
    const ws = socketRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify(msg));
    }
  };

  return { snapshot, status, send };
}
