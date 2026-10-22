import { useCallback, useEffect, useMemo, useRef, useState } from "react";

type Roi = { x: number; y: number; width: number; height: number };
type CaptureTab = "jewellery" | "tare";
type TareMode = "pledge" | "release";
type CaptureState = Record<string, any>;

const FLICKER_METHODS = [
  ["driver", "Current shutter / driver", "Keep the station’s existing calibrated camera settings."],
  ["adaptive", "Adaptive shutter", "Set exposure and a short average once when applied, then keep those settings until you apply again. Linux V4L2."],
  ["adaptive_average", "Adaptive filtering", "Check lighting once when applied, then keep raw video or 90-frame averaging until you apply again."],
  ["normal", "Normal · raw", "Use the camera image without software flicker correction."],
  ["driver_50", "Driver · 50 Hz", "Ask the camera driver to compensate for 50 Hz lighting."],
  ["driver_60", "Driver · 60 Hz", "Ask the camera driver to compensate for 60 Hz lighting."],
  ["shutter_100", "Shutter · 1/100 s", "An exact 10 ms exposure for 50 Hz lighting. Linux V4L2."],
  ["shutter_50", "Shutter · 1/50 s", "An exact 20 ms exposure for 50 Hz lighting. Linux V4L2."],
  ["average_30", "Average · 30 frames", "Smooth moving bands over 30 frames. Keep the jewellery still while the filter settles."],
  ["average_90", "Average · 90 frames", "Stronger smoothing over 90 frames. Moving objects may leave trails while the filter settles."],
  ["row", "Row correction", "Reduce horizontal brightness bands. This can also change natural shading."],
  ["hybrid", "Hybrid · average + rows", "Combine 90-frame averaging and horizontal row correction."],
  ["temporal_average", "Legacy temporal filter", "Use the station’s saved averaging and brightness correction configuration."],
];

function Icon({ name }: { name: "settings" | "reconnect" | "close" }) {
  return <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
    {name === "reconnect" ? <><path d="M20 7v5h-5"/><path d="M19.1 16a8 8 0 1 1 .4-8L20 12"/></> : name === "close" ? <path d="m6 6 12 12M18 6 6 18"/> : <><path d="m9.5 3-.6 2.4-2 .9-2.2-.7-2 3.4 1.7 1.8v2.4L2.7 15l2 3.4 2.2-.7 2 .9.6 2.4h4.9l.6-2.4 2-.9 2.2.7 2-3.4-1.7-1.8v-2.4l1.7-1.8-2-3.4-2.2.7-2-.9-.6-2.4z"/><circle cx="12" cy="12" r="3"/></>}
  </svg>;
}

const COMBINED_NECK_LABEL = "Chain / Necklace";
const NECK_JEWELLERY_NOTE = "This jewellery type may include necklace, haram, kasu mala, dollar chain, mangalsutra, pendant chain, and similar neck jewellery.";
const EAR_NOSE_LABEL = "Earrings / Nosepin";
const EAR_NOSE_NOTE = "Usually earrings; this category may also include a nosepin or nose ornament.";

async function requestJson(url: string, init?: RequestInit) {
  const response = await fetch(url, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers || {}) },
  });
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    throw new Error(payload.detail || "The request could not be completed");
  }
  return response.json();
}

function StatusDot({ ok }: { ok: boolean }) {
  return <span className={`statusDot ${ok ? "ok" : "bad"}`} />;
}

function WeightTable({ weights, gross }: { weights?: Record<string, any>; gross?: number }) {
  const w = weights || { gross_g: gross, note: "Stone and net estimates unavailable for this capture." };
  const grams = (value: any) => typeof value === "number" && Number.isFinite(value) ? `${value.toFixed(2)} g` : "Unavailable";
  const range = (low: any, high: any) => low == null || high == null ? "Unavailable" : low === high ? grams(low) : `${grams(low)} – ${grams(high)}`;
  return <div className="weightSummary">
    <table className="weightTable">
      <caption>Jewellery weights (grams)</caption>
      <thead><tr><th scope="col">Weight</th><th scope="col">Value</th><th scope="col">Range</th></tr></thead>
      <tbody>
        <tr><th scope="row">Gross (scale)</th><td>{grams(w.gross_g)}</td><td>—</td></tr>
        <tr><th scope="row">Stone (estimated average)</th><td>{grams(w.stone_g)}</td><td>{range(w.stone_min_g, w.stone_max_g)}</td></tr>
        <tr><th scope="row">Net (estimated)</th><td>{grams(w.net_g)}</td><td>{range(w.net_min_g, w.net_max_g)}</td></tr>
      </tbody>
    </table>
    <p>{w.note}</p>
  </div>;
}

function ResultImage({ src, alt, label }: { src?: string | null; alt: string; label?: string }) {
  if (!src) return null;
  return (
    <figure className="resultFigure">
      <a href={src} target="_blank" rel="noreferrer" title="Open full-size image" aria-label={`${label || alt}: open full-size image in a new tab`}>
        <img className="resultImage" src={src} alt={alt} />
        <span className="imageZoomHint">View full size</span>
      </a>
      {label && <figcaption>{label}</figcaption>}
    </figure>
  );
}

export default function App() {
  const [live, setLive] = useState<any>({ camera: {}, scale: {} });
  const [settings, setSettings] = useState<any>(null);
  const [hardwarePlatform, setHardwarePlatform] = useState("");
  const [labels, setLabels] = useState<string[]>([]);
  const [active, setActive] = useState<CaptureState | null>(null);
  const [busy, setBusy] = useState(false);
  const [shuttingDown, setShuttingDown] = useState(false);
  const [message, setMessage] = useState("Place the jewellery and check the weight.");
  const [error, setError] = useState("");
  const [setupOpen, setSetupOpen] = useState(false);
  const [settingsSection, setSettingsSection] = useState<"flicker" | "areas" | "rotation" | "scale">("flicker");
  const [flickerMode, setFlickerMode] = useState("driver");
  const [scaleModel, setScaleModel] = useState("embsys_wm");
  const [cameraRotation, setCameraRotation] = useState(-90);
  const [settingsSaving, setSettingsSaving] = useState(false);
  const [settingsNotice, setSettingsNotice] = useState("");
  const [drawingTarget, setDrawingTarget] = useState<"processing" | "apriltag" | null>(null);
  const [draftRois, setDraftRois] = useState<{ processing: Roi | null; apriltag: Roi | null }>({ processing: null, apriltag: null });
  const dragRef = useRef<{ x: number; y: number; target: "processing" | "apriltag" } | null>(null);
  const settingsButtonRef = useRef<HTMLButtonElement>(null);
  const closeSettingsRef = useRef<HTMLButtonElement>(null);
  const [rois, setRois] = useState<{ processing: Roi | null; apriltag: Roi | null }>({
    processing: null,
    apriltag: null,
  });
  const [itemCorrections, setItemCorrections] = useState<Record<number, string>>({});
  const [captureTab, setCaptureTab] = useState<CaptureTab>("jewellery");
  const [tareMode, setTareMode] = useState<TareMode>("pledge");
  const [tareCaptures, setTareCaptures] = useState<Partial<Record<TareMode, CaptureState>>>({});
  const [currentTime, setCurrentTime] = useState(() => new Date());
  const [videoSrc, setVideoSrc] = useState("/api/video");
  const [cameraReconnecting, setCameraReconnecting] = useState(false);
  const [markerRefreshing, setMarkerRefreshing] = useState(false);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const cameraAreaRef = useRef<HTMLDivElement>(null);
  const cameraImageRef = useRef<HTMLImageElement>(null);
  const shutdownRequestedRef = useRef(false);
  const videoRetryRef = useRef<number | null>(null);
  const cameraWasConnectedRef = useRef(false);
  const markerRefreshAttemptedRef = useRef(false);

  useEffect(() => {
    const timer = window.setInterval(() => setCurrentTime(new Date()), 1000);
    return () => window.clearInterval(timer);
  }, []);

  useEffect(() => {
    const connected = Boolean(live.camera?.connected);
    if (connected && !cameraWasConnectedRef.current) {
      setVideoSrc(`/api/video?reconnect=${Date.now()}`);
    }
    cameraWasConnectedRef.current = connected;
  }, [live.camera?.connected]);

  useEffect(() => () => {
    if (videoRetryRef.current !== null) window.clearTimeout(videoRetryRef.current);
  }, []);

  function retryVideoStream() {
    if (shutdownRequestedRef.current || videoRetryRef.current !== null) return;
    videoRetryRef.current = window.setTimeout(() => {
      videoRetryRef.current = null;
      setVideoSrc(`/api/video?retry=${Date.now()}`);
    }, 2000);
  }

  async function reconnectCamera() {
    if (cameraReconnecting || shutdownRequestedRef.current) return;
    setCameraReconnecting(true);
    setError("");
    setMessage("Reconnecting the camera...");
    setLive((current: any) => ({
      ...current,
      camera: { ...(current.camera || {}), connected: false, status: "Camera reconnecting" },
    }));
    try {
      await requestJson("/api/camera/reconnect", { method: "POST" });
      setVideoSrc(`/api/video?manual-reconnect=${Date.now()}`);
      setMessage("Camera reconnect requested. The live view will return when the camera is available.");
    } catch (reason: any) {
      setError(reason.message);
    } finally {
      setCameraReconnecting(false);
    }
  }

  useEffect(() => {
    requestJson("/api/settings").then((data) => {
      setSettings(data.settings);
      setHardwarePlatform(data.platform);
      setFlickerMode(data.settings.camera.anti_flicker_mode);
      setScaleModel(data.settings.scale.model || "embsys_wm");
      setCameraRotation(data.settings.camera.rotation % 360 === 90 ? 90 : -90);
      setLabels(data.labels || []);
      setRois(data.settings.rois || { processing: null, apriltag: null });
    }).catch((reason) => setError(reason.message));
  }, []);

  async function refreshMarkerRoi(automatic = false) {
    if (markerRefreshing || shutdownRequestedRef.current) return;
    setMarkerRefreshing(true);
    if (!automatic) {
      setError("");
      setMessage("Detecting the AprilTag and refreshing its area...");
    }
    try {
      const data = await requestJson("/api/settings/rois/apriltag/refresh", { method: "POST" });
      setSettings(data.settings);
      setRois((current) => ({ ...current, apriltag: data.roi }));
      setError("");
      setMessage(automatic ? "Marker area detected automatically." : "Marker area refreshed.");
      return data;
    } catch (reason: any) {
      if (!automatic || !rois.apriltag) setError(reason.message);
      if (!automatic) setMessage("Could not refresh the marker area.");
    } finally {
      setMarkerRefreshing(false);
    }
  }

  useEffect(() => {
    if (!settings || !live.camera?.connected || markerRefreshAttemptedRef.current) return;
    markerRefreshAttemptedRef.current = true;
    void refreshMarkerRoi(true);
  }, [settings, live.camera?.connected]);

  useEffect(() => {
    let disposed = false;
    let socket: WebSocket | null = null;
    let reconnectTimer: number | null = null;
    const protocol = location.protocol === "https:" ? "wss" : "ws";

    const applyLiveState = (payload: any) => {
      if (disposed) return;
      setLive(payload);
      setActive((current) => {
        if (current && payload.latest?.id === current.id) return payload.latest;
        if (!current && payload.latest && ["processing", "awaiting_confirmation", "rejected"].includes(payload.latest.status)) {
          return payload.latest;
        }
        return current;
      });
    };

    const connectSocket = () => {
      if (disposed || shutdownRequestedRef.current) return;
      socket = new WebSocket(`${protocol}://${location.host}/ws/live`);
      socket.onmessage = (event) => applyLiveState(JSON.parse(event.data));
      socket.onerror = () => socket?.close();
      socket.onclose = () => {
        if (disposed || shutdownRequestedRef.current) return;
        setLive((current: any) => ({
          ...current,
          camera: { ...(current.camera || {}), connected: false },
          scale: { ...(current.scale || {}), connected: false, weight_g: null },
        }));
        reconnectTimer = window.setTimeout(connectSocket, 1000);
      };
    };

    const pollLiveState = () => {
      if (disposed || shutdownRequestedRef.current) return;
      requestJson(`/api/live?poll=${Date.now()}`, { cache: "no-store" })
        .then(applyLiveState)
        .catch(() => setLive((current: any) => ({
          ...current,
          camera: { ...(current.camera || {}), connected: false },
          scale: { ...(current.scale || {}), connected: false, weight_g: null },
        })));
    };

    connectSocket();
    pollLiveState();
    const pollTimer = window.setInterval(pollLiveState, 1000);
    return () => {
      disposed = true;
      window.clearInterval(pollTimer);
      if (reconnectTimer !== null) window.clearTimeout(reconnectTimer);
      socket?.close();
    };
  }, []);

  const videoContentRect = useCallback(() => {
    const area = cameraAreaRef.current;
    const image = cameraImageRef.current;
    if (!area) return null;
    const bounds = area.getBoundingClientRect();
    const sourceWidth = image?.naturalWidth || 0;
    const sourceHeight = image?.naturalHeight || 0;
    if (!sourceWidth || !sourceHeight) {
      return { bounds, left: 0, top: 0, width: bounds.width, height: bounds.height };
    }
    const scale = Math.min(bounds.width / sourceWidth, bounds.height / sourceHeight);
    const width = sourceWidth * scale;
    const height = sourceHeight * scale;
    return {
      bounds,
      left: (bounds.width - width) / 2,
      top: (bounds.height - height) / 2,
      width,
      height,
    };
  }, []);

  const drawRois = useCallback(() => {
    const canvas = canvasRef.current;
    const area = cameraAreaRef.current;
    const video = videoContentRect();
    if (!canvas || !area || !video) return;
    const rect = area.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.round(rect.width * dpr);
    canvas.height = Math.round(rect.height * dpr);
    canvas.style.width = `${rect.width}px`;
    canvas.style.height = `${rect.height}px`;
    const context = canvas.getContext("2d");
    if (!context) return;
    context.scale(dpr, dpr);
    context.clearRect(0, 0, rect.width, rect.height);
    const render = (roi: Roi | null, color: string) => {
      if (!roi) return;
      const x = video.left + roi.x * video.width;
      const y = video.top + roi.y * video.height;
      const width = roi.width * video.width;
      const height = roi.height * video.height;
      context.strokeStyle = color;
      context.lineWidth = 1;
      context.strokeRect(x, y, width, height);
    };
    const shown = setupOpen && settingsSection === "areas" ? draftRois : rois;
    render(shown.apriltag, "#6bc9aa");
    if (setupOpen && settingsSection === "areas") render(shown.processing, "#edbc63");
  }, [rois, draftRois, setupOpen, settingsSection, videoContentRect]);

  useEffect(() => {
    drawRois();
    const observer = new ResizeObserver(drawRois);
    if (cameraAreaRef.current) observer.observe(cameraAreaRef.current);
    window.addEventListener("orientationchange", drawRois);
    return () => {
      observer.disconnect();
      window.removeEventListener("orientationchange", drawRois);
    };
  }, [drawRois, setupOpen]);

  function toggleSettings() {
    if (!setupOpen) {
      setFlickerMode(settings?.camera?.anti_flicker_mode || "driver");
      setCameraRotation(settings?.camera?.rotation % 360 === 90 ? 90 : -90);
      setScaleModel(settings?.scale?.model || "embsys_wm");
      setDraftRois(rois);
      setSettingsNotice("");
    }
    setDrawingTarget(null);
    dragRef.current = null;
    setSetupOpen(value => !value);
  }

  useEffect(() => {
    if (!setupOpen) return;
    closeSettingsRef.current?.focus();
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setSetupOpen(false);
        setDrawingTarget(null);
        dragRef.current = null;
        settingsButtonRef.current?.focus();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [setupOpen]);

  async function saveFlicker() {
    setSettingsSaving(true);
    setSettingsNotice("");
    try {
      const data = await requestJson("/api/settings/camera/flicker", { method: "PUT", body: JSON.stringify({ mode: flickerMode }) });
      setSettings(data.settings);
      setSettingsNotice("Saved. The camera is reconnecting with your selected method.");
    } catch (reason: any) { setSettingsNotice(`Could not save: ${reason.message}`); }
    finally { setSettingsSaving(false); }
  }

  async function saveAreas() {
    setSettingsSaving(true);
    try {
      const data = await requestJson("/api/settings/rois", { method: "PUT", body: JSON.stringify(draftRois) });
      setSettings(data);
      setRois(data.rois);
      setDrawingTarget(null);
      setSettingsNotice(settings?.camera?.anti_flicker_mode === "adaptive" ? "Camera areas saved. Adaptive metering restarts if its area changed." : "Camera areas saved.");
    } catch (reason: any) { setSettingsNotice(`Could not save: ${reason.message}`); }
    finally { setSettingsSaving(false); }
  }

  async function saveScale() {
    if (settingsSaving) return;
    setSettingsSaving(true);
    setSettingsNotice("");
    try {
      const data = await requestJson("/api/settings/scale", { method: "PUT", body: JSON.stringify({ model: scaleModel }) });
      setSettings(data.settings);
      setLive((current: any) => ({ ...current, scale: data.scale }));
      setSettingsNotice("Scale selection saved. Connecting now; it will reconnect automatically after unplugging and on startup.");
    } catch (reason: any) { setSettingsNotice(`Could not apply scale: ${reason.message}`); }
    finally { setSettingsSaving(false); }
  }

  function checkCaptureUnit() {
    if (live.scale?.connected && live.scale?.unit === "ct") {
      window.alert("ct capture is not allowed. Switch the scale to grams (g) and try again.");
      return false;
    }
    return true;
  }

  async function saveRotation() {
    if (settingsSaving) return;
    setSettingsSaving(true);
    setSettingsNotice("");
    try {
      const data = await requestJson("/api/settings/camera/rotation", { method: "PUT", body: JSON.stringify({ rotation: cameraRotation }) });
      setSettings(data.settings);
      setRois(data.settings.rois);
      setDraftRois(data.settings.rois);
      setDrawingTarget(null);
      dragRef.current = null;
      setSettingsNotice("Rotation applied and saved. It will be restored next time you open the app.");
    } catch (reason: any) { setSettingsNotice(`Could not apply rotation: ${reason.message}`); }
    finally { setSettingsSaving(false); }
  }

  function pointerPosition(event: { clientX: number; clientY: number }) {
    const video = videoContentRect();
    if (!video || !video.width || !video.height) return null;
    return { x: Math.max(0, Math.min(1, (event.clientX - video.bounds.left - video.left) / video.width)),
      y: Math.max(0, Math.min(1, (event.clientY - video.bounds.top - video.top) / video.height)) };
  }

  async function shutdownBackend() {
    if (!window.confirm("Shut down the backend safely? The camera and scale will be disconnected.")) return;
    shutdownRequestedRef.current = true;
    setShuttingDown(true);
    setError("");
    setMessage("Safely shutting down the camera, scale, and backend...");
    try {
      await requestJson("/api/shutdown", { method: "POST" });
    } catch (reason: any) {
      shutdownRequestedRef.current = false;
      setShuttingDown(false);
      setError(reason.message);
    }
  }

  async function capture() {
    if (!checkCaptureUnit()) return;
    setBusy(true);
    setError("");
    setMessage("Capturing jewellery and weight...");
    try {
      const captured = await requestJson("/api/captures", { method: "POST" });
      setActive(captured);
      setMessage("Checking the item type...");
      const classified = await requestJson(`/api/captures/${captured.id}/item-type`, { method: "POST" });
      setActive(classified);
      if (classified.status === "rejected") {
        setItemCorrections({});
        setMessage(classified.rejection_reason);
        return;
      }
      setItemCorrections(Object.fromEntries(
        (classified.classification.items || []).map((item: any) => [item.index, item.predicted_label]),
      ));
      setMessage(`Detected ${classified.classification.count || 0} jewel(s). Please confirm each item type.`);
    } catch (reason: any) {
      setError(reason.message);
      setMessage("Please recapture after placing gold jewellery on the testbed and removing foreign objects.");
    } finally {
      setBusy(false);
    }
  }

  async function captureTare() {
    if (!checkCaptureUnit()) return;
    setBusy(true);
    setError("");
    setMessage(`Capturing ${tareMode} tare weight...`);
    try {
      const captured = await requestJson("/api/tare-captures", {
        method: "POST",
        body: JSON.stringify({ mode: tareMode }),
      });
      setTareCaptures((current) => ({ ...current, [tareMode]: captured }));
      setMessage(`${tareMode === "pledge" ? "Pledge" : "Release"} tare saved. ${captured.result?.packet?.message || "Packet check unavailable."}`);
    } catch (reason: any) {
      setError(reason.message);
      setMessage("Ready to try the tare capture again.");
    } finally {
      setBusy(false);
    }
  }

  function classificationPayload() {
    return {
      items: (active?.classification?.items || []).map((item: any) => ({
        index: item.index,
        label: itemCorrections[item.index] || item.predicted_label,
        learn: (itemCorrections[item.index] || item.predicted_label) !== item.predicted_label
          || String(item.predicted_label).trim().toLowerCase() === "not gold jewelry",
      })),
    };
  }

  async function recapture() {
    if (!checkCaptureUnit()) return;
    setBusy(true);
    setError("");
    try {
      if (active?.status === "awaiting_confirmation") {
        setMessage("Saving manual corrections for gallery matching...");
        await requestJson(`/api/captures/${active.id}/corrections`, {
          method: "POST",
          body: JSON.stringify(classificationPayload()),
        });
      }
      await capture();
    } catch (reason: any) {
      setError(reason.message);
      setMessage("Corrections could not be saved. Please retry recapture.");
    } finally {
      setBusy(false);
    }
  }

  async function confirmItems() {
    if (!active) return;
    setBusy(true);
    setError("");
    try {
      const next = await requestJson(`/api/captures/${active.id}/confirm`, {
        method: "POST",
        body: JSON.stringify(classificationPayload()),
      });
      setActive(next);
      setMessage("Every detected jewel is being checked separately.");
    } catch (reason: any) {
      setError(reason.message);
    } finally {
      setBusy(false);
    }
  }

  function startNew() {
    setActive(null);
    setItemCorrections({});
    setError("");
    setMessage("Place the next jewellery item and check the weight.");
  }

  const stage = useMemo(() => {
    if (!active) return "capture";
    if (active.status === "rejected") return "rejected";
    if (active.status === "awaiting_confirmation") return "confirm";
    if (["processing"].includes(active.status)) return "processing";
    if (active.status === "complete") return "results";
    if (active.status === "failed") return "failed";
    return "capture";
  }, [active]);

  const weight = live.scale?.connected ? (live.scale?.value ?? live.scale?.weight_g) : undefined;
  const weightUnit = live.scale?.unit || "g";
  const weightPrecision = live.scale?.model === "essae_jx500" ? 3 : 2;
  const result = active?.result || {};
  const media = active?.media || {};
  const detectedItems = active?.classification?.items || [];
  const predictedNotGoldItems = detectedItems.filter(
    (item: any) => String(item.predicted_label || "").trim().toLowerCase() === "not gold jewelry",
  );
  const selectedNotGoldItems = detectedItems.filter(
    (item: any) => String(itemCorrections[item.index] || item.predicted_label || "").trim().toLowerCase() === "not gold jewelry",
  );
  const hasNotGoldPrediction = predictedNotGoldItems.length > 0;
  const canConfirmItems = detectedItems.length > 0
    && selectedNotGoldItems.length === 0
    && !hasNotGoldPrediction;
  const tareCapture = tareCaptures[tareMode];
  const cameraAdapting = Boolean(live.camera?.connected && live.camera?.capture_ready === false);
  const cameraReady = Boolean(
    live.camera?.connected
    && typeof live.camera?.frame_age_ms === "number"
    && live.camera.frame_age_ms < 2000
    && live.camera?.capture_ready !== false
  );
  const liveDate = currentTime.toLocaleDateString("en-IN", {
    day: "2-digit", month: "2-digit", year: "numeric",
  });
  const liveTime = currentTime.toLocaleTimeString("en-IN", {
    hour: "2-digit", minute: "2-digit", second: "2-digit",
  });

  return (
    <div className="appShell">
      <header className="topbar">
        <div className="brandBlock">
          <img src="/brand/company" alt="EMBSYS" className="companyLogo" />
          <div className="brandDivider" />
          <div>
            <h1>Jewellery Capture</h1>
            <p>Weight and jewellery details</p>
          </div>
        </div>
        <div className="topbarActions">
          <button
            className="shutdownButton"
            onClick={shutdownBackend}
            disabled={shuttingDown || busy || stage === "processing"}
            title={stage === "processing" ? "Wait for the current analysis to finish" : "Safely stop the backend"}
          >
            {shuttingDown ? "Shutting down..." : "Shut down"}
          </button>

        </div>
      </header>

      <main className={`workspace ${setupOpen ? "setupWorkspace" : ""} ${stage === "results" && captureTab === "jewellery" ? "resultsWorkspace" : ""}`}>
        <section className="cameraCard">
          <div className="cameraHeader">
            <div className="cameraTelemetry"><StatusDot ok={Boolean(live.camera?.connected)} /><span>{live.camera?.connected ? "LIVE" : "OFFLINE"}</span><span className="fpsLabel">FPS <strong>{live.camera?.stream_fps > 0 ? live.camera.stream_fps.toFixed(1) : "—"}</strong></span></div>
            <div className="cameraHeaderActions">
              <button className={`iconButton ${cameraReconnecting ? "isSpinning" : ""}`} title="Reconnect camera" aria-label="Reconnect camera" onClick={reconnectCamera} disabled={cameraReconnecting || settingsSaving}><Icon name="reconnect" /></button>
              <button ref={settingsButtonRef} className={`iconButton ${setupOpen ? "selected" : ""}`} title="Settings" aria-label="Settings" aria-expanded={setupOpen} aria-controls="station-settings" onClick={toggleSettings}><Icon name="settings" /></button>
            </div>
          </div>
          <div
            className="cameraArea"
            ref={cameraAreaRef}
            style={live.camera?.resolution?.[0] && live.camera?.resolution?.[1]
              ? { aspectRatio: `${live.camera.resolution[0]} / ${live.camera.resolution[1]}` }
              : undefined}
          >
            <img
              ref={cameraImageRef}
              src={videoSrc}
              alt="Live jewellery camera"
              onLoad={drawRois}
              onError={retryVideoStream}
            />
            <canvas
              ref={canvasRef}
              className={`roiCanvas ${setupOpen && settingsSection === "areas" && drawingTarget ? "active" : ""}`}
              aria-label="Draw camera area"
              onPointerDown={event => {
                if (!event.isPrimary || !drawingTarget || !setupOpen || settingsSection !== "areas") return;
                const point = pointerPosition(event);
                if (!point) return;
                dragRef.current = { ...point, target: drawingTarget };
                event.currentTarget.setPointerCapture(event.pointerId);
              }}
              onPointerMove={event => {
                const start = dragRef.current;
                const point = pointerPosition(event);
                if (!start || !point) return;
                const roi = { x: Math.min(start.x, point.x), y: Math.min(start.y, point.y), width: Math.abs(point.x - start.x), height: Math.abs(point.y - start.y) };
                if (roi.width > .005 && roi.height > .005) setDraftRois(current => ({ ...current, [start.target]: roi }));
              }}
              onPointerUp={event => { dragRef.current = null; if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId); }}
              onPointerCancel={() => { dragRef.current = null; }}
            />
            <div className="liveWeightOverlay" aria-label="Live scale weight">
              <span>Live weight</span>
              <strong>{typeof weight === "number" ? weight.toFixed(weightPrecision) : "--"}<small>{weightUnit}</small></strong>
            </div>
            {!cameraReady && !cameraAdapting && (
              <div className="cameraUnavailable" role="status">
                <strong>Camera unavailable</strong>
                <span>Waiting for automatic reconnection</span>
                <button className="secondary" onClick={reconnectCamera} disabled={cameraReconnecting}>
                  {cameraReconnecting ? "Reconnecting..." : "Reconnect now"}
                </button>
              </div>
            )}
          </div>

        </section>

        <aside className="controlCard">
          <div className="deviceStrip">
            <span><StatusDot ok={cameraReady} />{live.camera?.anti_flicker?.status === "UNAVAILABLE" ? "Method unavailable" : cameraAdapting ? "Adjusting lighting..." : cameraReady ? "Camera ready" : "Camera unavailable"}</span>
            <span title={live.scale?.error || live.scale?.port || ""}><StatusDot ok={Boolean(live.scale?.connected)} />{live.scale?.connected ? "Scale connected" : live.scale?.status || "Scale reconnecting"}</span>
          </div>

          {setupOpen && <section id="station-settings" className="settingsPanel" aria-label="Station settings">
            <div className="settingsHeading"><div><span className="eyebrow">Station preferences</span><h2>Settings</h2></div><button ref={closeSettingsRef} className="iconButton" onClick={() => { toggleSettings(); settingsButtonRef.current?.focus(); }} aria-label="Close settings" title="Close settings"><Icon name="close" /></button></div>
            <div className="settingsLayout">
              <nav className="settingsNav" aria-label="Settings categories">
                <button aria-current={settingsSection === "flicker" ? "page" : undefined} onClick={() => { setSettingsSection("flicker"); setDrawingTarget(null); setSettingsNotice(""); }}>Camera flicker</button>
                <button aria-current={settingsSection === "areas" ? "page" : undefined} onClick={() => { setSettingsSection("areas"); setSettingsNotice(""); }}>Camera areas</button>
                <button aria-current={settingsSection === "rotation" ? "page" : undefined} onClick={() => { setSettingsSection("rotation"); setDrawingTarget(null); dragRef.current = null; setSettingsNotice(""); }}>Camera rotation</button>
                <button aria-current={settingsSection === "scale" ? "page" : undefined} onClick={() => { setSettingsSection("scale"); setDrawingTarget(null); dragRef.current = null; setSettingsNotice(""); }}>Weight scale</button>
              </nav>
              <div className="settingsContent">
                {!settings ? <p role="status">Loading settings…</p> : settingsSection === "flicker" ? <>
                  <h3>Flicker removal</h3><p>Choose the method that works best with your lighting. Watch the live camera as it settles.</p>
                  <label htmlFor="flicker-method">Correction method</label>
                  <select id="flicker-method" value={flickerMode} disabled={settingsSaving} onChange={event => { setFlickerMode(event.target.value); setSettingsNotice(""); }}>{FLICKER_METHODS.filter(([value]) => hardwarePlatform === "linux" || !["adaptive", "shutter_100", "shutter_50"].includes(value)).map(([value, label]) => <option key={value} value={value}>{label}</option>)}</select>
                  <p className="methodDescription">{FLICKER_METHODS.find(([value]) => value === flickerMode)?.[2]}</p>
                  <div className="filterStatus" role="status"><strong>{live.camera?.anti_flicker?.status === "UNAVAILABLE" ? "Method unavailable" : cameraAdapting ? "Filter settling" : cameraReady ? "Camera ready" : "Waiting for camera"}</strong><span>{live.camera?.anti_flicker?.method || live.camera?.exposure?.requested_shutter || FLICKER_METHODS.find(([value]) => value === settings.camera.anti_flicker_mode)?.[1]}</span>
                    {live.camera?.anti_flicker?.target_frames > 1 && <span>{live.camera.anti_flicker.averaged_frames} / {live.camera.anti_flicker.target_frames} frames · keep jewellery still</span>}
                    {(live.camera?.anti_flicker?.detail || live.camera?.anti_flicker?.adaptive?.error || live.camera?.error) && <span className="errorText">{live.camera.anti_flicker?.detail || live.camera.anti_flicker?.adaptive?.error || live.camera.error}</span>}
                  </div>
                  <button className="primary" disabled={settingsSaving || (busy || stage === "processing")} onClick={saveFlicker}>{settingsSaving ? "Applying…" : "Apply method"}</button>
                  {(busy || stage === "processing") && <p>Camera changes are available when the current capture finishes. Analysis continues while settings are open.</p>}
                </> : settingsSection === "scale" ? <>
                  <h3>Weight scale</h3>
                  <p>Select the scale connected to this station. Only readings in grams can be captured.</p>
                  <label htmlFor="scale-model">Scale model</label>
                  <select id="scale-model" value={scaleModel} disabled={settingsSaving} onChange={event => { setScaleModel(event.target.value); setSettingsNotice(""); }}>
                    <option value="embsys_wm">EMBSYS WM</option>
                    <option value="essae_jx500">ESSAE JX500</option>
                  </select>
                  <p>ESSAE JX500 finds its USB port automatically, including when the port number changes. Its decoded USB output must be connected.</p>
                  <div className="filterStatus" role="status"><strong>{live.scale?.status || "Waiting for scale"}</strong><span>{live.scale?.port || "Searching for USB scale"}</span>{live.scale?.error && <span>{live.scale.error}</span>}</div>
                  <button className="primary" disabled={settingsSaving || busy || stage === "processing"} onClick={saveScale}>{settingsSaving ? "Applying..." : "Apply scale"}</button>
                </> : settingsSection === "rotation" ? <>
                  <h3>Camera rotation</h3>
                  <p>Choose the camera orientation. Both options keep the full-resolution image at 1080 × 1920.</p>
                  <label htmlFor="camera-rotation">Orientation</label>
                  <select id="camera-rotation" value={cameraRotation} disabled={settingsSaving} onChange={event => { setCameraRotation(Number(event.target.value)); setSettingsNotice(""); }}>
                    <option value={90}>Turn right (+90°)</option>
                    <option value={-90}>Turn left (−90°)</option>
                  </select>
                  <p className="methodDescription">Directions are relative to the camera's original image. Apply to update the live view and save this orientation for future sessions.</p>
                  <button className="primary" disabled={settingsSaving || markerRefreshing || cameraReconnecting || busy || stage === "processing"} onClick={saveRotation}>{settingsSaving ? "Applying…" : "Apply rotation"}</button>
                  {(busy || stage === "processing") && <p>Camera changes are available when the current capture finishes.</p>}
                </> : <>
                  <h3>Camera areas</h3><p>The testbed is detected automatically at capture time. Draw a marker search area or a lighting metering area on the live camera.</p>
                  <div className="areaOption"><strong>Marker search area</strong><p>Include the entire AprilTag with a little space around it.</p><button className="secondary" disabled={settingsSaving || !live.camera?.connected} aria-pressed={drawingTarget === "apriltag"} onClick={() => setDrawingTarget("apriltag")}>Draw marker area</button><button className="textButton" disabled={markerRefreshing || !cameraReady || settingsSaving || busy || stage === "processing"} onClick={async () => { const data = await refreshMarkerRoi(false); if (data) setDraftRois(current => ({ ...current, apriltag: data.roi })); }}>{markerRefreshing ? "Detecting…" : "Auto-detect and save marker"}</button></div>
                  <div className="areaOption"><strong>Lighting metering area</strong><p>Used by adaptive shutter control. Leave empty to meter the centre of the image.</p><button className="secondary" disabled={settingsSaving || !live.camera?.connected} aria-pressed={drawingTarget === "processing"} onClick={() => setDrawingTarget("processing")}>Draw metering area</button><button className="textButton" disabled={settingsSaving} onClick={() => setDraftRois(current => ({ ...current, processing: null }))}>Use default area</button></div>
                  {drawingTarget && <p className="drawingHint" role="status">Drag on the live camera to draw the {drawingTarget === "apriltag" ? "marker search" : "lighting metering"} area. Then save your areas.</p>}
                  <button className="primary" disabled={settingsSaving || busy || stage === "processing"} onClick={saveAreas}>{settingsSaving ? "Saving…" : "Save camera areas"}</button>
                </>}
                {settingsNotice && <p className="settingsNotice" role="status">{settingsNotice}</p>}
              </div>
            </div>
          </section>}
          <div className="workflowPanel" hidden={setupOpen}>
          <div className="captureTabs" role="tablist" aria-label="Capture process">
            <button
              role="tab"
              aria-selected={captureTab === "jewellery"}
              className={captureTab === "jewellery" ? "selected" : ""}
              onClick={() => setCaptureTab("jewellery")}
            >Jewellery capture</button>
            <button
              role="tab"
              aria-selected={captureTab === "tare"}
              className={captureTab === "tare" ? "selected" : ""}
              onClick={() => setCaptureTab("tare")}
              disabled={["confirm", "processing"].includes(stage)}
            >Tare weight</button>
          </div>

          <div className="stepBody">
            {captureTab === "jewellery" && stage === "capture" && (
              <div className="captureView">
                <span className="eyebrow">Ready to capture</span>
                <h2>Place one or more jewels</h2>
                <p>Keep every jewel on the testbed, leave a visible gap between them, and keep the marker clearly visible.</p>
                <div className="weightCard">
                  <span>Current weight</span>
                  <strong>{typeof weight === "number" ? `${weight.toFixed(weightPrecision)} ${weightUnit}` : "Waiting..."}</strong>
                  <time dateTime={currentTime.toISOString()}>{liveDate} · {liveTime}</time>
                </div>
                <button className="primary captureButton" onClick={capture} disabled={busy || !cameraReady || !live.scale?.connected || !rois.apriltag}>
                  {busy ? "Please wait..." : "Capture jewellery"}
                </button>
                {!rois.apriltag && <p className="hint">Keep the AprilTag visible so its area can be detected automatically.</p>}
              </div>
            )}

            {captureTab === "jewellery" && stage === "rejected" && (
              <div className="rejectedReview">
                <span className="eyebrow errorText">Recapture required</span>
                <h2>Other objects detected with jewellery</h2>
                <p role="alert">{active?.rejection_reason}</p>
                <ResultImage src={media.evidence} alt="Capture containing other objects with jewellery" label="Captured evidence" />
                <button className="primary" onClick={recapture} disabled={busy || !cameraReady || !live.scale?.connected || !rois.apriltag}>
                  {busy ? "Please wait..." : "Recapture image"}
                </button>
              </div>
            )}

            {captureTab === "jewellery" && stage === "confirm" && (
              <div className="confirmationView">
                <span className="eyebrow">{detectedItems.length} jewel{detectedItems.length === 1 ? "" : "s"} identified</span>
                <h2>{hasNotGoldPrediction ? "Non-gold object detected" : "Confirm each jewel type"}</h2>
                <div className="confirmationContent">
                  <ResultImage src={media.evidence} alt="Captured jewellery" label="Captured evidence" />
                  <div className="confirmationPanel">
                    <div className="detectedItemList">
                      {detectedItems.map((item: any) => (
                        <div className="detectedItem" key={item.index}>
                          <div>
                            <strong>Jewel {item.index}</strong>
                            <span>{Math.round(Number(item.confidence || 0) * 100)}% confidence</span>
                          </div>
                          <select
                            aria-label={`Jewel ${item.index} type`}
                            value={itemCorrections[item.index] || item.predicted_label}
                            disabled={String(item.predicted_label).trim().toLowerCase() === "not gold jewelry"}
                            onChange={(event) => setItemCorrections((current) => ({ ...current, [item.index]: event.target.value }))}
                          >
                            {labels.map((label) => <option key={label} value={label}>{label}</option>)}
                          </select>
                          {(itemCorrections[item.index] || item.predicted_label) === COMBINED_NECK_LABEL && (
                            <p className="jewelleryTypeNote">{NECK_JEWELLERY_NOTE}</p>
                          )}
                          {(itemCorrections[item.index] || item.predicted_label) === EAR_NOSE_LABEL && (
                            <p className="jewelleryTypeNote">{EAR_NOSE_NOTE}</p>
                          )}
                        </div>
                      ))}
                    </div>
                    {hasNotGoldPrediction && (
                      <div className="notGoldGate" role="alert">
                        <strong>Analysis is blocked</strong>
                        <p>
                          {predictedNotGoldItems.length === 1
                            ? `Jewel ${predictedNotGoldItems[0].index} was classified as Not Gold Jewelry.`
                            : `Jewels ${predictedNotGoldItems.map((item: any) => item.index).join(", ")} were classified as Not Gold Jewelry.`}
                          {" "}This classification cannot be overwritten. Remove the object and recapture; its Not Gold label will be saved for gallery matching.
                        </p>
                      </div>
                    )}
                    <div className="confirmationActions">
                      <button
                        className="secondary"
                        onClick={recapture}
                        disabled={busy || !cameraReady || !live.scale?.connected || !rois.apriltag}
                      >
                        {busy ? "Please wait..." : "Recapture image"}
                      </button>
                      <button className="primary" onClick={confirmItems} disabled={busy || !canConfirmItems}>
                        Confirm all and analyze
                      </button>
                    </div>
                    <p className="confirmationHint">
                      {selectedNotGoldItems.length
                        ? "Not Gold Jewelry cannot proceed to analysis. Recapture saves Not Gold labels and your manual corrections for gallery matching, then takes a new image."
                        : "Review every detected jewel before starting the detailed analysis."}
                    </p>
                  </div>
                </div>
              </div>
            )}

            {captureTab === "jewellery" && stage === "processing" && (
              <div className="progressView">
                <span className="eyebrow">Please wait</span>
                <div className="countdownRing" style={{ "--progress": `${active?.job?.percent || 0}%` } as any}>
                  <strong>{active?.job?.seconds_remaining ?? 0}</strong>
                  <span>seconds</span>
                </div>
                <h2>{active?.job?.message || "Checking your jewellery"}</h2>
                <div className="progressTrack"><span style={{ width: `${active?.job?.percent || 0}%` }} /></div>
                <p>The result will appear automatically.</p>
              </div>
            )}

            {captureTab === "jewellery" && stage === "results" && (
              <div className="resultsView">
                <div className="resultsHeading">
                  <div>
                    <span className="eyebrow successText">Analysis completed</span>
                    <h2>Jewellery results</h2>
                    <p>{result.count} jewel{result.count === 1 ? "" : "s"} captured and analyzed</p>
                  </div>
                  <div className="resultsHeadingAside">
                    <div className="resultWeightBadge">
                      <span>Captured weight</span>
                      <strong>{active?.weight_g != null ? `${active.weight_g.toFixed(2)} g` : "Unavailable"}</strong>
                    </div>
                    <div className="resultsTopActions">
                      <a className="primary linkButton" href={`${media.result_image}?download=true`}>Download image</a>
                      <a className="primary linkButton" href={media.pdf} target="_blank" rel="noreferrer">Open PDF</a>
                      <button className="secondary" onClick={startNew}>Next capture</button>
                    </div>
                  </div>
                </div>
                <ResultImage
                  src={media.result_image}
                  alt="Captured jewellery with jewel details"
                  label="Captured image and details"
                />
                {result.count === 1 && (result.items?.[0]?.route?.stones || result.items?.[0]?.stones || result.stones) && <WeightTable weights={result.weights} gross={active?.weight_g} />}
                <div className="itemResults">
                  {(result.items || []).map((item: any, itemPosition: number) => {
                    const itemMedia = (media.items || [])[itemPosition] || {};
                    return (
                      <section className="itemResultCard" key={item.index}>
                        <div className="itemResultHeader">
                          <span className="itemNumber">{item.index}</span>
                          <div>
                            <small>Jewellery type</small>
                            <h3>{item.label}</h3>
                            {item.label === COMBINED_NECK_LABEL && <p className="jewelleryTypeNote">{NECK_JEWELLERY_NOTE}</p>}
                            {item.label === EAR_NOSE_LABEL && <p className="jewelleryTypeNote">{EAR_NOSE_NOTE}</p>}
                          </div>
                          <span className="completeBadge">Complete</span>
                        </div>
                        <div className="summaryGrid compact">
                          {item.dimensions?.applicable === false && <div className="wide"><span>Dimensions unavailable</span><strong>{item.dimensions.reason}</strong></div>}
                          {item.dimensions?.outer_diameter_mm != null && <div><span>Outer diameter</span><strong>{Number(item.dimensions.outer_diameter_mm).toFixed(2)} mm</strong></div>}
                          {item.dimensions?.inner_diameter_mm != null && <div><span>Inner diameter</span><strong>{Number(item.dimensions.inner_diameter_mm).toFixed(2)} mm</strong></div>}
                          {item.beads && <div><span>Bead analysis</span><strong>{item.beads.beads_detected ? "Beads detected" : "Beads not detected"}</strong></div>}
                          {item.beads && <div><span>Bead count</span><strong>{item.beads.count ?? item.beads.detections?.length ?? "Unavailable"}</strong></div>}
                          {item.beads?.summary && <div className="wide"><span>Bead insights</span><strong>{item.beads.summary}</strong></div>}
                          {item.stones?.stone_area_mm2 != null && <div><span>Detected stone area</span><strong>{Number(item.stones.stone_area_mm2).toFixed(2)} mm²</strong></div>}
                          {item.stones && <div className={`wide riskTag risk${item.stones.risk_level || "NONE"}`}><span>Stone analysis</span><strong>{({ HIGH: "High stones region", MEDIUM: "Medium stones", LOW: "Low stones", NONE: "No stones" } as Record<string, string>)[item.stones.risk_level || "NONE"] || "No stones"}</strong></div>}
                        </div>
                        {(item.errors || []).map((warning: any) => <p className="itemWarning" key={warning.stage}>{warning.stage}: {warning.message}</p>)}
                        <div className="resultGallery">
                          <ResultImage src={itemMedia.dimensions} alt={`Jewel ${item.index} size result`} label={item.dimensions?.applicable === false ? "Ring image — dimensions unavailable" : "Dimension analysis"} />
                          <ResultImage src={itemMedia.beads} alt={`Jewel ${item.index} bead result`} label="Bead analysis" />
                          <ResultImage src={itemMedia.stones} alt={`Jewel ${item.index} stone result`} label="Stone analysis" />
                        </div>
                      </section>
                    );
                  })}
                </div>
              </div>
            )}

            {captureTab === "jewellery" && stage === "failed" && (
              <>
                <span className="eyebrow errorText">Needs attention</span>
                <h2>The check could not be completed</h2>
                <p>Please check the item position and capture it again.</p>
                <button className="primary" onClick={startNew}>Try another capture</button>
              </>
            )}

            {captureTab === "tare" && (
              <>
                <span className="eyebrow">Tare weight capture</span>
                <h2>Capture the packed jewel</h2>
                <p>This saves the packet image with the current scale weight, printed packet number, and barcode check.</p>
                <div className="tareModeButtons" role="group" aria-label="Tare capture mode">
                  <button className={tareMode === "pledge" ? "selected" : ""} onClick={() => setTareMode("pledge")}>Pledge tare</button>
                  <button className={tareMode === "release" ? "selected" : ""} onClick={() => setTareMode("release")}>Release tare</button>
                </div>
                <div className="weightCard">
                  <span>Current tare weight</span>
                  <strong>{typeof weight === "number" ? `${weight.toFixed(weightPrecision)} ${weightUnit}` : "Waiting..."}</strong>
                  <time dateTime={currentTime.toISOString()}>{liveDate} · {liveTime}</time>
                </div>
                <button className="primary captureButton" onClick={captureTare} disabled={busy || !cameraReady || !live.scale?.connected}>
                  {busy ? "Please wait..." : `Capture ${tareMode} tare`}
                </button>
                {tareCapture && (
                  <div className="tareResult">
                    <span className="eyebrow successText">Saved {tareMode} tare</span>
                    <ResultImage src={tareCapture.media?.evidence} alt={`${tareMode} tare packet`} />
                    <div className="tarePacketDetails">
                      <strong>Packet number: {tareCapture.result?.packet?.packet_number || "Not read"}</strong>
                      <span>Barcode: {tareCapture.result?.packet?.barcode || "Not read"}</span>
                      <span>{tareCapture.result?.packet?.status === "MATCH" ? "Barcode matched" : tareCapture.result?.packet?.message || "Barcode not verified"}</span>
                    </div>
                    {tareCapture.media?.result_image && (
                      <a className="primary linkButton" href={`${tareCapture.media.result_image}?download=true`}>Download image</a>
                    )}
                    <div className="tareMeta">
                      <strong>{Number(tareCapture.weight_g).toFixed(2)} g</strong>
                      <time dateTime={tareCapture.captured_at}>
                        {new Date(tareCapture.captured_at).toLocaleString("en-IN")}
                      </time>
                    </div>
                  </div>
                )}
              </>
            )}
          </div>

          </div>
          <div className="messageBar" role="status">{message}</div>
          {error && <div className="errorBar">{error}</div>}
        </aside>
      </main>
      <footer>Live date &amp; time: {liveDate} · {liveTime} · One operator station</footer>
    </div>
  );
}
