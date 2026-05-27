const keyInput = document.getElementById("apiKey");
const storedKey = localStorage.getItem("kyc-identity-api-key") || "";
if (keyInput) keyInput.value = storedKey;

let stream = null;
let frameTimer = null;
let frameBusy = false;
const sessionId = crypto.randomUUID();

function headers() {
  const out = {};
  const key = keyInput?.value?.trim();
  if (key) {
    localStorage.setItem("kyc-identity-api-key", key);
    out["X-API-Key"] = key;
  }
  return out;
}

function pretty(value) {
  return JSON.stringify(value, null, 2);
}

function challengeParams() {
  const params = new URLSearchParams({ session_id: sessionId });
  document.querySelectorAll(".challenge:checked").forEach(input => params.append("challenge", input.value));
  return params;
}

async function postForm(path, form) {
  const res = await fetch(path, { method: "POST", headers: headers(), body: form });
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
  return data;
}

function drawDocument(file, objects = []) {
  const canvas = document.getElementById("documentCanvas");
  const ctx = canvas.getContext("2d");
  const img = new Image();
  const url = URL.createObjectURL(file);
  img.onload = () => {
    const scale = Math.min(1, 1200 / Math.max(img.naturalWidth, img.naturalHeight));
    canvas.width = Math.max(1, Math.round(img.naturalWidth * scale));
    canvas.height = Math.max(1, Math.round(img.naturalHeight * scale));
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    ctx.lineWidth = 2;
    ctx.font = "14px Inter, sans-serif";
    for (const obj of objects) {
      const p = obj.box?.pixel;
      if (!p) continue;
      const color = obj.label === "face" ? "#38bdf8" : obj.label === "id_card" ? "#22c55e" : "#f59e0b";
      const x = p.x * scale, y = p.y * scale, w = p.width * scale, h = p.height * scale;
      ctx.strokeStyle = color;
      ctx.fillStyle = color;
      ctx.strokeRect(x, y, w, h);
      ctx.fillText(`${obj.label} ${Math.round((obj.confidence || 0) * 100)}%`, x + 4, Math.max(16, y - 18));
    }
    URL.revokeObjectURL(url);
  };
  img.src = url;
}

document.getElementById("documentRun").addEventListener("click", async () => {
  const file = document.getElementById("documentFile").files[0];
  if (!file) return;
  const result = document.getElementById("documentResult");
  result.textContent = "Running OCR...";
  const mode = document.getElementById("documentMode").value;
  const params = new URLSearchParams({
    values_only: mode === "values" ? "true" : "false",
    detect_objects: mode === "values" ? "false" : "true",
    include_stats: "true",
    retry: "true",
    accuracy_mode: "accurate",
  });
  const forcedType = document.getElementById("forceDocType").checked ? document.getElementById("documentType").value.trim() : "";
  if (forcedType) params.set("document_type", forcedType);
  const form = new FormData();
  form.append("file", file);
  try {
    const data = await postForm(`/ocr?${params}`, form);
    drawDocument(file, data.objects || []);
    result.textContent = pretty(mode === "values" ? { document_type: data.document_type, values: data.values, meta: data.meta } : data);
  } catch (error) {
    result.textContent = error.message;
  }
});

document.getElementById("portraitRun").addEventListener("click", async () => {
  const file = document.getElementById("portraitFile").files[0];
  if (!file) return;
  const result = document.getElementById("portraitResult");
  result.textContent = "Analyzing portrait...";
  const form = new FormData();
  form.append("file", file);
  try {
    result.textContent = pretty(await postForm("/identity/api/portrait", form));
  } catch (error) {
    result.textContent = error.message;
  }
});

async function snapshotBlob() {
  const video = document.getElementById("webcam");
  const canvas = document.getElementById("frameCanvas");
  canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);
  return new Promise(resolve => canvas.toBlob(resolve, "image/jpeg", 0.9));
}

document.getElementById("cameraStart").addEventListener("click", async () => {
  stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 }, audio: false });
  const video = document.getElementById("webcam");
  video.srcObject = stream;
  await video.play();
  document.getElementById("livenessStart").disabled = false;
});

async function sendFrame() {
  if (frameBusy) return;
  frameBusy = true;
  try {
    const form = new FormData();
    form.append("file", await snapshotBlob(), "frame.jpg");
    const data = await postForm(`/identity/api/liveness/frame?${challengeParams()}`, form);
    const state = data.liveness_state || {};
    document.getElementById("faceSignal").textContent = data.face_detected ? "Detected" : "Missing";
    document.getElementById("blinkSignal").textContent = `${state.blink_count || 0}`;
    document.getElementById("headSignal").textContent = data.head_position || "unknown";
    document.getElementById("riskSignal").textContent = state.risk_status || "checking";
    document.getElementById("livenessResult").textContent = pretty(data);
  } finally {
    frameBusy = false;
  }
}

document.getElementById("livenessStart").addEventListener("click", () => {
  if (frameTimer) clearInterval(frameTimer);
  frameTimer = setInterval(sendFrame, 300);
  document.getElementById("livenessFinish").disabled = false;
});

document.getElementById("livenessFinish").addEventListener("click", async () => {
  if (frameTimer) clearInterval(frameTimer);
  frameTimer = null;
  try {
    const data = await postForm(`/identity/api/liveness/complete?${challengeParams()}`, new FormData());
    document.getElementById("livenessResult").textContent = pretty(data);
  } catch (error) {
    document.getElementById("livenessResult").textContent = error.message;
  }
});
