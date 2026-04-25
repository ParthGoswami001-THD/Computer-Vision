const POINT_LABELS = ["TL", "TR", "BL", "BR"];

const appState = {
  sourceImage: null,
  sourceSize: [0, 0],
  presets: [],
  srcPoints: [],
  dstPoints: [],
  status: "",
  mathText: "",
  previewImage: null,
  hasResult: false,
};

let animTimer = null;
let picker = {
  image: null,
  points: [],
  scaleX: 1,
  scaleY: 1,
};

function byId(id) {
  return document.getElementById(id);
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    credentials: "same-origin",
    ...options,
  });
  const payload = await response.json();
  if (!response.ok || !payload.ok) {
    throw new Error(payload.error || `Request failed: ${response.status}`);
  }
  return payload;
}

function setBusy(isBusy) {
  ["applyPointsBtn", "applyPresetBtn", "runBtn", "animateBtn", "saveBtn", "openPickerBtn"].forEach((id) => {
    byId(id).disabled = isBusy;
  });
}

function showError(err) {
  const text = err instanceof Error ? err.message : String(err);
  byId("statusBox").value = `Error: ${text}`;
}

function clearAnimation() {
  if (animTimer) {
    clearInterval(animTimer);
    animTimer = null;
  }
}

function buildPointEditor(containerId, prefix) {
  const container = byId(containerId);
  container.innerHTML = "";

  for (let i = 0; i < 4; i += 1) {
    const row = document.createElement("div");
    row.className = "point-row";

    const label = document.createElement("label");
    label.textContent = `${prefix}${i + 1} ${POINT_LABELS[i]}`;

    const xInput = document.createElement("input");
    xInput.type = "number";
    xInput.step = "0.1";
    xInput.id = `${containerId}-${i}-x`;

    const yInput = document.createElement("input");
    yInput.type = "number";
    yInput.step = "0.1";
    yInput.id = `${containerId}-${i}-y`;

    row.append(label, xInput, yInput);
    container.appendChild(row);
  }
}

function setPointInputs(containerId, points) {
  for (let i = 0; i < 4; i += 1) {
    byId(`${containerId}-${i}-x`).value = Number(points[i][0]).toFixed(1);
    byId(`${containerId}-${i}-y`).value = Number(points[i][1]).toFixed(1);
  }
}

function readPointInputs(containerId) {
  const out = [];
  for (let i = 0; i < 4; i += 1) {
    const x = Number(byId(`${containerId}-${i}-x`).value);
    const y = Number(byId(`${containerId}-${i}-y`).value);
    if (!Number.isFinite(x) || !Number.isFinite(y)) {
      throw new Error(`Invalid point at row ${i + 1}.`);
    }
    out.push([x, y]);
  }
  return out;
}

function updatePresetSelect(presets) {
  const select = byId("presetSelect");
  const current = select.value;
  select.innerHTML = "";

  presets.forEach((name) => {
    const opt = document.createElement("option");
    opt.value = name;
    opt.textContent = name;
    select.appendChild(opt);
  });

  if (presets.includes(current)) {
    select.value = current;
  } else if (presets.length > 0) {
    select.value = presets[0];
  }
}

function renderState(payload) {
  if (payload.source_image) {
    appState.sourceImage = payload.source_image;
  }

  appState.sourceSize = payload.source_size;
  appState.presets = payload.presets;
  appState.srcPoints = payload.src_points;
  appState.dstPoints = payload.dst_points;
  appState.status = payload.status;
  appState.mathText = payload.math_text;
  appState.previewImage = payload.preview_image;
  appState.hasResult = Boolean(payload.has_result);

  updatePresetSelect(appState.presets);
  setPointInputs("srcEditor", appState.srcPoints);
  setPointInputs("dstEditor", appState.dstPoints);

  byId("statusBox").value = appState.status;
  byId("mathBox").value = appState.mathText;
  byId("previewImage").src = appState.previewImage;
}

async function loadState() {
  setBusy(true);
  try {
    const payload = await api("/api/state");
    renderState(payload);
  } catch (err) {
    showError(err);
  } finally {
    setBusy(false);
  }
}

async function uploadImage(file) {
  clearAnimation();
  setBusy(true);
  try {
    const fd = new FormData();
    fd.append("image", file);
    const payload = await api("/api/upload", { method: "POST", body: fd });
    renderState(payload);
  } catch (err) {
    showError(err);
  } finally {
    setBusy(false);
  }
}

async function applyPreset() {
  clearAnimation();
  setBusy(true);
  try {
    const payload = await api("/api/apply-preset", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name: byId("presetSelect").value }),
    });
    renderState(payload);
  } catch (err) {
    showError(err);
  } finally {
    setBusy(false);
  }
}

async function applyManualPoints() {
  clearAnimation();
  setBusy(true);
  try {
    await applyManualPointsCore();
  } catch (err) {
    showError(err);
  } finally {
    setBusy(false);
  }
}

async function applyManualPointsCore() {
  const srcPoints = readPointInputs("srcEditor");
  const dstPoints = readPointInputs("dstEditor");

  const payload = await api("/api/set-points", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ src_points: srcPoints, dst_points: dstPoints }),
  });
  renderState(payload);
}

async function runTransform() {
  clearAnimation();
  setBusy(true);
  try {
    await applyManualPointsCore();
    setBusy(true);
    const payload = await api("/api/run", { method: "POST" });
    renderState(payload);
  } catch (err) {
    showError(err);
  } finally {
    setBusy(false);
  }
}

async function runAnimation() {
  clearAnimation();
  setBusy(true);
  try {
    await applyManualPointsCore();
    setBusy(true);

    const payload = await api("/api/animate", { method: "POST" });
    byId("statusBox").value = payload.status;

    const frames = payload.frames || [];
    if (!frames.length) {
      throw new Error("No frames were generated.");
    }

    let index = 0;
    byId("previewImage").src = frames[0];
    animTimer = setInterval(() => {
      index += 1;
      if (index >= frames.length) {
        clearAnimation();
        byId("statusBox").value = "Animation finished.";
        return;
      }
      byId("previewImage").src = frames[index];
    }, 130);
  } catch (err) {
    showError(err);
  } finally {
    setBusy(false);
  }
}

async function saveResult() {
  setBusy(true);
  try {
    const defaultName = "";
    const filename = window.prompt("Optional filename (for example result.png)", defaultName);
    const payload = await api("/api/save", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ filename: filename || null }),
    });
    byId("statusBox").value = payload.status;
    await loadGallery();
  } catch (err) {
    showError(err);
  } finally {
    setBusy(false);
  }
}

function openPickerModal() {
  if (!appState.sourceImage) {
    showError("Source image is not available yet.");
    return;
  }

  picker.points = [];
  picker.image = new Image();
  picker.image.onload = () => {
    drawPickerCanvas();
    byId("pickerModal").classList.remove("hidden");
    byId("pickerStatus").textContent = "Clicked: 0 / 4";
  };
  picker.image.src = appState.sourceImage;
}

function closePickerModal() {
  byId("pickerModal").classList.add("hidden");
}

function drawPickerCanvas() {
  const canvas = byId("pickerCanvas");
  const ctx = canvas.getContext("2d");

  const maxWidth = Math.min(window.innerWidth * 0.9, 980);
  const maxHeight = Math.min(window.innerHeight * 0.65, 640);
  const ratio = Math.min(maxWidth / picker.image.width, maxHeight / picker.image.height, 1);

  canvas.width = Math.round(picker.image.width * ratio);
  canvas.height = Math.round(picker.image.height * ratio);

  picker.scaleX = picker.image.width / canvas.width;
  picker.scaleY = picker.image.height / canvas.height;

  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(picker.image, 0, 0, canvas.width, canvas.height);

  const colors = ["#ff4040", "#25b85a", "#3478f6", "#d09a00"];
  picker.points.forEach((p, idx) => {
    const x = p[0] / picker.scaleX;
    const y = p[1] / picker.scaleY;

    ctx.beginPath();
    ctx.arc(x, y, 7, 0, Math.PI * 2);
    ctx.fillStyle = colors[idx];
    ctx.fill();
    ctx.lineWidth = 2;
    ctx.strokeStyle = "#fff";
    ctx.stroke();

    ctx.font = "bold 13px Segoe UI";
    ctx.fillStyle = colors[idx];
    ctx.fillText(`S${idx + 1}`, x + 11, y - 10);
  });
}

function onPickerCanvasClick(event) {
  if (picker.points.length >= 4) {
    return;
  }

  const canvas = byId("pickerCanvas");
  const rect = canvas.getBoundingClientRect();
  const xCanvas = event.clientX - rect.left;
  const yCanvas = event.clientY - rect.top;

  const xImage = xCanvas * picker.scaleX;
  const yImage = yCanvas * picker.scaleY;

  picker.points.push([xImage, yImage]);
  byId("pickerStatus").textContent = `Clicked: ${picker.points.length} / 4`;
  drawPickerCanvas();
}

function resetPickerPoints() {
  picker.points = [];
  byId("pickerStatus").textContent = "Clicked: 0 / 4";
  drawPickerCanvas();
}

async function applyPickerPoints() {
  if (picker.points.length !== 4) {
    showError("Please click exactly four points.");
    return;
  }

  clearAnimation();
  setBusy(true);
  try {
    const payload = await api("/api/set-source-points", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        src_points: picker.points,
        move_destination: byId("relativeDst").checked,
      }),
    });

    renderState(payload);
    closePickerModal();
  } catch (err) {
    showError(err);
  } finally {
    setBusy(false);
  }
}

async function loadGallery() {
  const list = byId("galleryList");
  list.innerHTML = "";

  try {
    const payload = await api("/api/gallery");
    const files = payload.files || [];

    if (!files.length) {
      const li = document.createElement("li");
      li.textContent = "No result images found.";
      list.appendChild(li);
      byId("galleryInfo").textContent = "Save a result to preview it here.";
      byId("galleryImage").removeAttribute("src");
      return;
    }

    files.forEach((name) => {
      const li = document.createElement("li");
      li.textContent = name;
      li.addEventListener("click", () => {
        [...list.children].forEach((n) => n.classList.remove("active"));
        li.classList.add("active");
        byId("galleryImage").src = `/gallery/${encodeURIComponent(name)}`;
        byId("galleryInfo").textContent = name;
      });
      list.appendChild(li);
    });
  } catch (err) {
    showError(err);
  }
}

function setupTabs() {
  const buttons = [...document.querySelectorAll(".tab-btn")];
  const panels = [...document.querySelectorAll(".tab-panel")];

  buttons.forEach((btn) => {
    btn.addEventListener("click", () => {
      buttons.forEach((b) => b.classList.remove("active"));
      panels.forEach((p) => p.classList.remove("active"));
      btn.classList.add("active");
      byId(btn.dataset.tab).classList.add("active");

      if (btn.dataset.tab === "tab-gallery") {
        loadGallery();
      }
    });
  });
}

function wireEvents() {
  byId("uploadInput").addEventListener("change", (event) => {
    const file = event.target.files && event.target.files[0];
    if (file) {
      uploadImage(file);
    }
  });

  byId("applyPresetBtn").addEventListener("click", applyPreset);
  byId("applyPointsBtn").addEventListener("click", applyManualPoints);
  byId("runBtn").addEventListener("click", runTransform);
  byId("animateBtn").addEventListener("click", runAnimation);
  byId("saveBtn").addEventListener("click", saveResult);

  byId("openPickerBtn").addEventListener("click", openPickerModal);
  byId("pickerCloseBtn").addEventListener("click", closePickerModal);
  byId("pickerResetBtn").addEventListener("click", resetPickerPoints);
  byId("pickerApplyBtn").addEventListener("click", applyPickerPoints);
  byId("pickerCanvas").addEventListener("click", onPickerCanvasClick);

  byId("refreshGalleryBtn").addEventListener("click", loadGallery);

  window.addEventListener("resize", () => {
    if (!byId("pickerModal").classList.contains("hidden") && picker.image) {
      drawPickerCanvas();
    }
  });
}

async function bootstrap() {
  buildPointEditor("srcEditor", "S");
  buildPointEditor("dstEditor", "D");
  setupTabs();
  wireEvents();
  await loadState();
  await loadGallery();
}

bootstrap();
