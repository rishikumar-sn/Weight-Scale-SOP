import { useCallback, useEffect, useMemo, useRef, useState } from "react";

type Roi = { x: number; y: number; width: number; height: number };
type CaptureTab = "jewellery" | "tare";
type TareMode = "pledge" | "release";
type CaptureState = Record<string, any>;

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
      <a href={src} target="_blank" rel="noreferrer" title="Open full-size image">
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
  const [labels, setLabels] = useState<string[]>([]);
  const [active, setActive] = useState<CaptureState | null>(null);
  const [busy, setBusy] = useState(false);
  const [shuttingDown, setShuttingDown] = useState(false);
  const [message, setMessage] = useState("Place the jewellery and check the weight.");
  const [error, setError] = useState("");
  const [setupOpen, setSetupOpen] = useState(false);
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
        if (!current && payload.latest && ["processing", "awaiting_confirmation"].includes(payload.latest.status)) {
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
    render(rois.apriltag, "#6bc9aa");
  }, [rois, videoContentRect]);

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
    setBusy(true);
    setError("");
    setMessage("Capturing jewellery and weight...");
    try {
      const captured = await requestJson("/api/captures", { method: "POST" });
      setActive(captured);
      setMessage("Checking the item type...");
      const classified = await requestJson(`/api/captures/${captured.id}/item-type`, { method: "POST" });
      setActive(classified);
      setItemCorrections(Object.fromEntries(
        (classified.classification.items || []).map((item: any) => [item.index, item.predicted_label]),
      ));
      setMessage(`Detected ${classified.classification.count || 0} jewel(s). Please confirm each item type.`);
    } catch (reason: any) {
      setError(reason.message);
      setMessage("Ready to try again.");
    } finally {
      setBusy(false);
    }
  }

  async function captureTare() {
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

  async function recapture() {
    setItemCorrections({});
    await capture();
  }

  async function confirmItems() {
    if (!active) return;
    const items = active.classification?.items || [];
    setBusy(true);
    setError("");
    try {
      const next = await requestJson(`/api/captures/${active.id}/confirm`, {
        method: "POST",
        body: JSON.stringify({
          items: items.map((item: any) => ({
            index: item.index,
            label: itemCorrections[item.index] || item.predicted_label,
            learn: (itemCorrections[item.index] || item.predicted_label) !== item.predicted_label,
          })),
        }),
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
    if (active.status === "awaiting_confirmation") return "confirm";
    if (["processing"].includes(active.status)) return "processing";
    if (active.status === "complete") return "results";
    if (active.status === "failed") return "failed";
    return "capture";
  }, [active]);

  const predicted = active?.classification?.predicted_label || "";
  const isNotGold = predicted.trim().toLowerCase() === "not gold jewelry";
  const weight = live.scale?.weight_g;
  const result = active?.result || {};
  const media = active?.media || {};
  const detectedItems = active?.classification?.items || [];
  const tareCapture = tareCaptures[tareMode];
  const cameraReady = Boolean(
    live.camera?.connected
    && typeof live.camera?.frame_age_ms === "number"
    && live.camera.frame_age_ms < 2000
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
          <img src="/brand/client" alt="Bank logo" className="clientLogo" />
        </div>
      </header>

      <main className={`workspace ${setupOpen ? "setupWorkspace" : ""} ${stage === "results" && captureTab === "jewellery" ? "resultsWorkspace" : ""}`}>
        <section className={`cameraCard ${setupOpen ? "setupOpen" : ""}`}>
          <div className="cameraHeader">
            <span><StatusDot ok={cameraReady} /> {cameraReady ? `Camera${live.camera?.stream_fps > 0 ? ` · ${live.camera.stream_fps.toFixed(1)} fps` : ""}` : (live.camera?.status || "Camera unavailable")}</span>
            <div className="cameraHeaderActions">
              <button className="textButton" onClick={reconnectCamera} disabled={cameraReconnecting}>
                {cameraReconnecting ? "Reconnecting..." : "Reconnect camera"}
              </button>
              <button className="textButton" onClick={() => setSetupOpen((value) => !value)}>
                {setupOpen ? "Close setup" : "Camera setup"}
              </button>
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
              className="roiCanvas"
            />
            <div className="liveWeightOverlay" aria-label="Live scale weight">
              <span>Live weight</span>
              <strong>{typeof weight === "number" ? weight.toFixed(2) : "--"}<small>g</small></strong>
            </div>
            {!cameraReady && (
              <div className="cameraUnavailable" role="status">
                <strong>Camera unavailable</strong>
                <span>Waiting for automatic reconnection</span>
                <button className="secondary" onClick={reconnectCamera} disabled={cameraReconnecting}>
                  {cameraReconnecting ? "Reconnecting..." : "Reconnect now"}
                </button>
              </div>
            )}
          </div>
          {setupOpen && (
            <div className="roiTools">
              <div>
                <strong>Automatic testbed area</strong>
                <p>The segmentation model finds the testbed at capture time. Only the marker area needs refreshing.</p>
              </div>
              <button className="textButton" onClick={() => refreshMarkerRoi(false)} disabled={!cameraReady || markerRefreshing}>
                {markerRefreshing ? "Detecting marker..." : "Refresh marker area"}
              </button>
            </div>
          )}
        </section>

        <aside className="controlCard">
          <div className="deviceStrip">
            <span><StatusDot ok={cameraReady} />{cameraReady ? "Camera ready" : "Camera unavailable"}</span>
            <span><StatusDot ok={Boolean(live.scale?.connected)} />Scale {live.scale?.port || ""}</span>
          </div>

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
                  <strong>{typeof weight === "number" ? `${weight.toFixed(2)} g` : "Waiting..."}</strong>
                  <time dateTime={currentTime.toISOString()}>{liveDate} · {liveTime}</time>
                </div>
                <button className="primary captureButton" onClick={capture} disabled={busy || !cameraReady || !live.scale?.connected || !rois.apriltag}>
                  {busy ? "Please wait..." : "Capture jewellery"}
                </button>
                {!rois.apriltag && <p className="hint">Keep the AprilTag visible so its area can be detected automatically.</p>}
              </div>
            )}

            {captureTab === "jewellery" && stage === "confirm" && (
              <div className="confirmationView">
                <span className="eyebrow">{detectedItems.length} jewel{detectedItems.length === 1 ? "" : "s"} identified</span>
                <h2>{isNotGold ? "Not a gold jewel" : "Confirm each jewel type"}</h2>
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
                    <div className="confirmationActions">
                      <button
                        className="secondary"
                        onClick={recapture}
                        disabled={busy || !cameraReady || !live.scale?.connected || !rois.apriltag}
                      >
                        {busy ? "Please wait..." : "Recapture image"}
                      </button>
                      <button className="primary" onClick={confirmItems} disabled={busy || !detectedItems.length}>Confirm all and analyze</button>
                    </div>
                    <p className="confirmationHint">Review every detected jewel before starting the detailed analysis.</p>
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
                          {item.dimensions?.outer_diameter_mm != null && <div><span>Outer diameter</span><strong>{Number(item.dimensions.outer_diameter_mm).toFixed(2)} mm</strong></div>}
                          {item.dimensions?.inner_diameter_mm != null && <div><span>Inner diameter</span><strong>{Number(item.dimensions.inner_diameter_mm).toFixed(2)} mm</strong></div>}
                          {item.beads && <div><span>Bead analysis</span><strong>{item.beads.beads_detected ? "Beads detected" : "Beads not detected"}</strong></div>}
                          {item.beads && <div><span>Bead count</span><strong>{item.beads.count ?? item.beads.detections?.length ?? "Unavailable"}</strong></div>}
                          {item.beads?.summary && <div className="wide"><span>Bead insights</span><strong>{item.beads.summary}</strong></div>}
                          {item.stones?.stone_area_mm2 != null && <div><span>Detected stone area</span><strong>{Number(item.stones.stone_area_mm2).toFixed(2)} mm²</strong></div>}
                          {item.stones && <div className={`wide riskTag risk${item.stones.risk_level || "NONE"}`}><span>Stone analysis</span><strong>{item.stones.risk_status}</strong></div>}
                        </div>
                        {(item.errors || []).map((warning: any) => <p className="itemWarning" key={warning.stage}>{warning.stage}: {warning.message}</p>)}
                        <div className="resultGallery">
                          <ResultImage src={itemMedia.dimensions} alt={`Jewel ${item.index} size result`} label="Dimension analysis" />
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
                  <strong>{typeof weight === "number" ? `${weight.toFixed(2)} g` : "Waiting..."}</strong>
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

          <div className="messageBar">{message}</div>
          {error && <div className="errorBar">{error}</div>}
        </aside>
      </main>
      <footer>Live date &amp; time: {liveDate} · {liveTime} · One operator station</footer>
    </div>
  );
}
