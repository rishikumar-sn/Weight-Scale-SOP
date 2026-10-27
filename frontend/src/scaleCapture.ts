type ScaleCaptureContext = {
  setupOpen: boolean;
  busy: boolean;
  settingsSaving: boolean;
  cameraReady: boolean;
  captureTab: "jewellery" | "tare";
  stage: string;
  ageMs: number | null | undefined;
};

export function scaleCaptureAction(context: ScaleCaptureContext): "capture" | "recapture" | "tare" | null {
  if (context.setupOpen || context.busy || context.settingsSaving || !context.cameraReady
      || context.ageMs == null || context.ageMs >= 3000) return null;
  if (context.captureTab === "tare") return "tare";
  if (context.stage === "confirm") return "recapture";
  if (["capture", "rejected", "failed", "results"].includes(context.stage)) return "capture";
  return null;
}
