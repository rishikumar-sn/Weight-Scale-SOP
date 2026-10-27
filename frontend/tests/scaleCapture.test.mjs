import assert from "node:assert/strict";
import test from "node:test";
import { scaleCaptureAction } from "../src/scaleCapture.ts";

const ready = { setupOpen: false, busy: false, settingsSaving: false, cameraReady: true,
  captureTab: "jewellery", stage: "capture", ageMs: 0 };

test("a scale press during confirmation recaptures and preserves previous corrections", () => {
  assert.equal(scaleCaptureAction({ ...ready, stage: "confirm" }), "recapture");
});

test("a scale press can capture again from rejection, failure, and completed results", () => {
  for (const stage of ["capture", "rejected", "failed", "results"]) {
    assert.equal(scaleCaptureAction({ ...ready, stage }), "capture");
  }
});

test("setup, busy camera, and processing cannot be interrupted by a scale press", () => {
  for (const change of [{ setupOpen: true }, { busy: true }, { settingsSaving: true },
    { cameraReady: false }, { stage: "processing" }, { stage: "preparing" },
    { ageMs: null }, { ageMs: 3000 }]) {
    assert.equal(scaleCaptureAction({ ...ready, ...change }), null);
  }
});

test("tare presses stay on the selected tare process", () => {
  assert.equal(scaleCaptureAction({ ...ready, captureTab: "tare", stage: "results" }), "tare");
});
