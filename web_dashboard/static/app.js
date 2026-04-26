const state = {
  upload: null,
  analysis: null,
  runtimeMode: "tools",
  gemmaModel: "gemma4:26b",
  analysisStrategy: "tools_only",
  activeTab: "image",
  activeSubtab: "border",
};

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => Array.from(document.querySelectorAll(selector));
const escapeHtml = (value) =>
  String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");

const tabs = {
  image: $("#imagePanel"),
  analysis: $("#analysisPanel"),
  cad: $("#cadPanel"),
};

const subtabs = {
  border: $("#borderSubpanel"),
  gdt: $("#gdtSubpanel"),
  projections: $("#projectionsSubpanel"),
  callouts: $("#calloutsSubpanel"),
  view3d: $("#view3dSubpanel"),
};

function setTab(tabName) {
  state.activeTab = tabName;
  $$(".tab").forEach((button) => button.classList.toggle("is-active", button.dataset.tab === tabName));
  Object.entries(tabs).forEach(([name, panel]) => panel.classList.toggle("is-active", name === tabName));
}

function setSubtab(tabName) {
  state.activeSubtab = tabName;
  $$(".subtab").forEach((button) => button.classList.toggle("is-active", button.dataset.subtab === tabName));
  Object.entries(subtabs).forEach(([name, panel]) => panel.classList.toggle("is-active", name === tabName));
}

function setRuntimeMode(mode) {
  state.runtimeMode = mode;
  if (mode === "tools") {
    state.analysisStrategy = "tools_only";
  } else if (state.analysisStrategy === "tools_only") {
    state.analysisStrategy = "gemma_teacher_calibrated";
  }
  $("#analysisStrategySelect").value = state.analysisStrategy;
  $$(".mode-button").forEach((button) => button.classList.toggle("is-active", button.dataset.mode === mode));
  updateModelBadge();
  renderRuntimeSummary();
}

function setAnalysisStrategy(strategy) {
  state.analysisStrategy = strategy;
  state.runtimeMode = strategy === "tools_only" ? "tools" : "gemma";
  $$(".mode-button").forEach((button) => button.classList.toggle("is-active", button.dataset.mode === state.runtimeMode));
  updateModelBadge();
  renderRuntimeSummary();
}

function setGemmaModel(model) {
  state.gemmaModel = model;
  updateModelBadge();
  renderRuntimeSummary();
}

function updateModelBadge() {
  $("#modelBadge").textContent = state.upload
    ? state.runtimeMode === "gemma"
      ? `${state.gemmaModel} ready`
      : "Tools ready"
    : "No drawing loaded";
}

function currentFilename() {
  return state.upload?.filename || "<none>";
}

function annotationMasks() {
  if (state.analysis?.annotationMasks?.length) {
    return state.analysis.annotationMasks.map((item) => (item.shape === "line" ? item : item.crop || item)).filter(Boolean);
  }
  return [...(state.analysis?.gdt || []), ...(state.analysis?.callouts || [])]
    .map((item) => item.crop)
    .filter(Boolean);
}

function setFeedback(area, proposed, issue = "<issue>", extra = {}) {
  const fields = {
    filename: currentFilename(),
    area,
    proposed,
    issue,
    ...extra,
  };
  $("#feedbackText").value = Object.entries(fields)
    .filter(([, value]) => value !== undefined && value !== null && value !== "")
    .map(([key, value]) => `${key}=${value}`)
    .join(" | ");
}

function attachFeedback(root = document) {
  root.querySelectorAll("[data-feedback]").forEach((node) => {
    if (node.dataset.feedbackBound) return;
    node.dataset.feedbackBound = "true";
    node.addEventListener("click", (event) => {
      event.stopPropagation();
      if (node.dataset.suppressNextClick === "true") {
        node.dataset.suppressNextClick = "false";
        return;
      }
      setFeedback(
        node.dataset.feedback,
        node.dataset.proposed || node.textContent.trim(),
        "<issue>",
        {
          confidence: node.dataset.confidence,
          crop: node.dataset.crop,
        }
      );
    });
  });
}

async function uploadImage(file) {
  const response = await fetch("/api/upload", {
    method: "POST",
    headers: {
      "Content-Type": file.type || "application/octet-stream",
      "X-Filename": file.name,
    },
    body: file,
  });
  if (!response.ok) {
    throw new Error(await response.text());
  }
  return response.json();
}

async function onImageSelected(event) {
  const file = event.target.files?.[0];
  if (!file) return;

  $("#currentFile").textContent = "Uploading...";
  const upload = await uploadImage(file);
  state.upload = upload;
  state.analysis = null;

  const image = $("#originalImage");
  image.onload = () => renderEmptyAnalysis();
  image.src = upload.url;
  $(".image-stage").classList.add("has-image");
  $("#currentFile").textContent = upload.filename;
  updateModelBadge();
  $("#processButton").disabled = false;
  setFeedback("image", "original drawing loaded", "<issue>");
  setTab("analysis");
}

function updateProgress(phase, value) {
  const progress = $(`#${phase}Progress`);
  const percent = $(`#${phase}Percent`);
  progress.value = value;
  percent.textContent = `${Math.round(value)}%`;
}

function animateProgress() {
  updateProgress("segmenting", 0);
  updateProgress("cad", 0);
  return new Promise((resolve) => {
    let tick = 0;
    const timer = setInterval(() => {
      tick += 1;
      updateProgress("segmenting", Math.min(100, tick * 7));
      if (tick > 10) updateProgress("cad", Math.min(100, (tick - 10) * 8));
      if (tick >= 23) {
        clearInterval(timer);
        updateProgress("segmenting", 100);
        updateProgress("cad", 100);
        resolve();
      }
    }, 120);
  });
}

async function processDrawing() {
  if (!state.upload) return;
  $("#processButton").disabled = true;
  $$(".mode-button").forEach((button) => (button.disabled = true));
  $("#gemmaModelSelect").disabled = true;
  $("#analysisStrategySelect").disabled = true;
  const mode = state.runtimeMode;
  $("#modelBadge").textContent = mode === "gemma" ? `${state.gemmaModel} processing` : "Tools processing";
  $("#runtimeSummary").innerHTML = `<div class="runtime-row"><strong>${escapeHtml(strategyLabel(state.analysisStrategy))}</strong><span>${mode === "gemma" ? escapeHtml(state.gemmaModel) : "algorithmic-tools"}</span><span>Processing ${escapeHtml(state.upload.filename)}</span></div>`;
  const progressDone = animateProgress();
  const response = await fetch("/api/analyze", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      id: state.upload.id,
      filename: state.upload.filename,
      imageUrl: state.upload.url,
      mode,
      gemmaModel: state.gemmaModel,
      analysisStrategy: state.analysisStrategy,
    }),
  });
  const analysis = await response.json();
  await progressDone;
  state.analysis = analysis;
  $("#modelBadge").textContent = analysis.status;
  renderAnalysis();
  $("#processButton").disabled = false;
  $$(".mode-button").forEach((button) => (button.disabled = false));
  $("#gemmaModelSelect").disabled = false;
  $("#analysisStrategySelect").disabled = false;
}

function cropData(crop) {
  return `${Math.round(crop.x * 100)},${Math.round(crop.y * 100)},${Math.round(crop.w * 100)},${Math.round(crop.h * 100)}`;
}

function preciseCropData(crop) {
  return [crop.x, crop.y, crop.w, crop.h].map((value) => value.toFixed(4)).join(",");
}

function renderCrop(container, crop, feedback, proposed, confidence, options = {}) {
  container.innerHTML = "";
  container.dataset.feedback = feedback;
  container.dataset.proposed = proposed;
  container.dataset.confidence = confidence;
  container.dataset.crop = cropData(crop);

  const fit = options.fit || "cover";
  if (fit === "contain") {
    const canvas = document.createElement("canvas");
    canvas.setAttribute("aria-label", proposed);
    container.appendChild(canvas);
    const image = new Image();
    image.onload = () => {
      const box = container.getBoundingClientRect();
      const cropWidth = Math.max(1, image.naturalWidth * crop.w);
      const cropHeight = Math.max(1, image.naturalHeight * crop.h);
      const canvasWidth = Math.max(260, Math.round(box.width || container.clientWidth || 420));
      const canvasHeight = Math.max(170, Math.round(box.height || container.clientHeight || 240));
      canvas.width = canvasWidth;
      canvas.height = canvasHeight;
      const context = canvas.getContext("2d");
      context.fillStyle = "#fffefa";
      context.fillRect(0, 0, canvasWidth, canvasHeight);
      const scale = Math.min(canvasWidth / cropWidth, canvasHeight / cropHeight);
      const drawWidth = cropWidth * scale;
      const drawHeight = cropHeight * scale;
      const dx = (canvasWidth - drawWidth) / 2;
      const dy = (canvasHeight - drawHeight) / 2;
      context.drawImage(
        image,
        crop.x * image.naturalWidth,
        crop.y * image.naturalHeight,
        cropWidth,
        cropHeight,
        dx,
        dy,
        drawWidth,
        drawHeight
      );
      context.save();
      context.beginPath();
      context.rect(dx, dy, drawWidth, drawHeight);
      context.clip();
      (options.masks || []).forEach((mask) => {
        if (mask.shape === "line" && mask.line) {
          drawLineMask(context, mask.line, crop, image, scale, dx, dy);
          return;
        }
        const maskCrop = mask.crop || mask;
        const ix0 = Math.max(crop.x, maskCrop.x);
        const iy0 = Math.max(crop.y, maskCrop.y);
        const ix1 = Math.min(crop.x + crop.w, maskCrop.x + maskCrop.w);
        const iy1 = Math.min(crop.y + crop.h, maskCrop.y + maskCrop.h);
        if (ix1 <= ix0 || iy1 <= iy0) return;
        const mx = dx + ((ix0 - crop.x) * image.naturalWidth * scale);
        const my = dy + ((iy0 - crop.y) * image.naturalHeight * scale);
        const mw = (ix1 - ix0) * image.naturalWidth * scale;
        const mh = (iy1 - iy0) * image.naturalHeight * scale;
        context.fillStyle = "#fffefa";
        context.fillRect(mx, my, mw, mh);
        context.strokeStyle = "#c7462d";
        context.setLineDash([5, 4]);
        context.strokeRect(mx, my, mw, mh);
        context.setLineDash([]);
      });
      context.restore();
      context.strokeStyle = "#c7462d";
      context.lineWidth = 2;
      context.strokeRect(dx + 1, dy + 1, Math.max(1, drawWidth - 2), Math.max(1, drawHeight - 2));
      if (options.rectangleTool) {
        installRectangleTool(container, canvas, crop, { dx, dy, drawWidth, drawHeight }, feedback, proposed, confidence);
      }
    };
    image.src = state.upload.url;
    return;
  }

  const image = document.createElement("img");
  image.alt = proposed;
  image.onload = () => {
    const box = container.getBoundingClientRect();
    const scale = Math.max(box.width / (image.naturalWidth * crop.w), box.height / (image.naturalHeight * crop.h));
    image.style.width = `${image.naturalWidth * scale}px`;
    image.style.height = `${image.naturalHeight * scale}px`;
    image.style.left = `${-image.naturalWidth * crop.x * scale}px`;
    image.style.top = `${-image.naturalHeight * crop.y * scale}px`;
  };
  image.src = state.upload.url;
  container.appendChild(image);
}

function drawLineMask(context, line, crop, image, scale, dx, dy) {
  const x1 = dx + ((line.x1 - crop.x) * image.naturalWidth * scale);
  const y1 = dy + ((line.y1 - crop.y) * image.naturalHeight * scale);
  const x2 = dx + ((line.x2 - crop.x) * image.naturalWidth * scale);
  const y2 = dy + ((line.y2 - crop.y) * image.naturalHeight * scale);
  const strokeWidth = Math.max(5, Math.min(12, (line.width || 0.006) * image.naturalWidth * scale));
  context.lineCap = "round";
  context.lineJoin = "round";
  context.strokeStyle = "#fffefa";
  context.lineWidth = strokeWidth;
  context.setLineDash([]);
  context.beginPath();
  context.moveTo(x1, y1);
  context.lineTo(x2, y2);
  context.stroke();
  context.strokeStyle = "#c7462d";
  context.lineWidth = 2;
  context.setLineDash([5, 4]);
  context.beginPath();
  context.moveTo(x1, y1);
  context.lineTo(x2, y2);
  context.stroke();
  context.setLineDash([]);
}

function installRectangleTool(container, canvas, crop, draw, feedback, proposed, confidence) {
  container.classList.add("has-rectangle-tool");
  const existing = container.querySelector(".draw-selection");
  if (existing) existing.remove();

  const selection = document.createElement("div");
  selection.className = "draw-selection";
  container.appendChild(selection);

  let start = null;
  let active = false;

  const canvasPoint = (event) => {
    const rect = canvas.getBoundingClientRect();
    return {
      x: ((event.clientX - rect.left) / Math.max(1, rect.width)) * canvas.width,
      y: ((event.clientY - rect.top) / Math.max(1, rect.height)) * canvas.height,
    };
  };

  const clampToDrawing = (point) => ({
    x: Math.min(draw.dx + draw.drawWidth, Math.max(draw.dx, point.x)),
    y: Math.min(draw.dy + draw.drawHeight, Math.max(draw.dy, point.y)),
  });

  const paintSelection = (a, b) => {
    const canvasRect = canvas.getBoundingClientRect();
    const containerRect = container.getBoundingClientRect();
    const scaleX = canvasRect.width / Math.max(1, canvas.width);
    const scaleY = canvasRect.height / Math.max(1, canvas.height);
    const left = Math.min(a.x, b.x) * scaleX + canvasRect.left - containerRect.left;
    const top = Math.min(a.y, b.y) * scaleY + canvasRect.top - containerRect.top;
    const width = Math.abs(b.x - a.x) * scaleX;
    const height = Math.abs(b.y - a.y) * scaleY;
    selection.style.display = "block";
    selection.style.left = `${left}px`;
    selection.style.top = `${top}px`;
    selection.style.width = `${width}px`;
    selection.style.height = `${height}px`;
  };

  const toImageCrop = (a, b) => {
    const x0 = (Math.min(a.x, b.x) - draw.dx) / Math.max(1, draw.drawWidth);
    const y0 = (Math.min(a.y, b.y) - draw.dy) / Math.max(1, draw.drawHeight);
    const x1 = (Math.max(a.x, b.x) - draw.dx) / Math.max(1, draw.drawWidth);
    const y1 = (Math.max(a.y, b.y) - draw.dy) / Math.max(1, draw.drawHeight);
    return {
      absolute: {
        x: crop.x + Math.max(0, Math.min(1, x0)) * crop.w,
        y: crop.y + Math.max(0, Math.min(1, y0)) * crop.h,
        w: Math.max(0, Math.min(1, x1) - Math.max(0, Math.min(1, x0))) * crop.w,
        h: Math.max(0, Math.min(1, y1) - Math.max(0, Math.min(1, y0))) * crop.h,
      },
      local: {
        x: Math.max(0, Math.min(1, x0)),
        y: Math.max(0, Math.min(1, y0)),
        w: Math.max(0, Math.min(1, x1) - Math.max(0, Math.min(1, x0))),
        h: Math.max(0, Math.min(1, y1) - Math.max(0, Math.min(1, y0))),
      },
    };
  };

  canvas.addEventListener("pointerdown", (event) => {
    if (event.button !== 0) return;
    event.preventDefault();
    event.stopPropagation();
    active = true;
    start = clampToDrawing(canvasPoint(event));
    selection.style.display = "none";
    canvas.setPointerCapture(event.pointerId);
  });

  canvas.addEventListener("pointermove", (event) => {
    if (!active || !start) return;
    event.preventDefault();
    const current = clampToDrawing(canvasPoint(event));
    paintSelection(start, current);
  });

  canvas.addEventListener("pointerup", (event) => {
    if (!active || !start) return;
    event.preventDefault();
    event.stopPropagation();
    active = false;
    const end = clampToDrawing(canvasPoint(event));
    const width = Math.abs(end.x - start.x);
    const height = Math.abs(end.y - start.y);
    if (width < 6 || height < 6) {
      selection.style.display = "none";
      return;
    }
    container.dataset.suppressNextClick = "true";
    paintSelection(start, end);
    const marked = toImageCrop(start, end);
    setFeedback("projection-rectangle", `${proposed} marked error rectangle`, "<issue>", {
      confidence,
      crop: cropData(crop),
      rect: cropData(marked.absolute),
      rectExact: preciseCropData(marked.absolute),
      rectWithinProjection: preciseCropData(marked.local),
      originalArea: feedback,
    });
  });

  canvas.addEventListener("pointercancel", () => {
    active = false;
    start = null;
    selection.style.display = "none";
  });
}

function renderMask(container, masks, feedback, proposed, confidence) {
  container.innerHTML = "";
  container.dataset.feedback = feedback;
  container.dataset.proposed = proposed;
  container.dataset.confidence = confidence;
  container.dataset.crop = masks.map(cropData).join(";");

  const image = new Image();
  image.onload = () => {
    const ratio = image.naturalWidth / image.naturalHeight || 1;
    const width = Math.min(980, Math.max(420, Math.round(container.clientWidth || 640)));
    const height = Math.round(width / ratio);
    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    const context = canvas.getContext("2d");
    context.fillStyle = "#fffefa";
    context.fillRect(0, 0, width, height);
    context.strokeStyle = "#2e343b";
    context.lineWidth = 2;

    masks.forEach((mask) => {
      const sx = mask.x * image.naturalWidth;
      const sy = mask.y * image.naturalHeight;
      const sw = mask.w * image.naturalWidth;
      const sh = mask.h * image.naturalHeight;
      const dx = mask.x * width;
      const dy = mask.y * height;
      const dw = mask.w * width;
      const dh = mask.h * height;
      context.drawImage(image, sx, sy, sw, sh, dx, dy, dw, dh);
      context.strokeRect(dx, dy, dw, dh);
    });

    container.appendChild(canvas);
  };
  image.src = state.upload.url;
}

function renderEmptyAnalysis() {
  $("#borderMask").innerHTML = "";
  $("#titleBlockCrop").innerHTML = "";
  renderRuntimeSummary();
  ["titleFields", "gdtList", "projectionGrid", "calloutGrid", "view3dContent"].forEach((id) => {
    $(`#${id}`).innerHTML = `<div class="empty-analysis">No analysis result</div>`;
  });
  $("#cadSummary").innerHTML = `<strong>CAD status</strong><span>No drawing processed</span>`;
}

function renderAnalysis() {
  renderRuntimeSummary();
  renderTitleBlock();
  renderGdt();
  renderProjections();
  renderCallouts();
  render3d();
  renderCad();
  attachFeedback(document);
}

function renderRuntimeSummary() {
  const target = $("#runtimeSummary");
  if (!state.analysis) {
    target.innerHTML = `
      <div class="runtime-row">
        <strong>${state.runtimeMode === "gemma" ? "Gemma mode" : "Tools mode"}</strong>
        <span>${escapeHtml(strategyLabel(state.analysisStrategy))}</span>
        <span>${state.runtimeMode === "gemma" ? `${state.gemmaModel} full-image inference will run during Process` : "Deterministic segmentation tools only"}</span>
      </div>`;
    return;
  }
  const mode = state.analysis.runtimeMode || state.runtimeMode;
  const gemma = state.analysis.gemma;
  if (mode !== "gemma") {
    target.innerHTML = `
      <button class="runtime-row" data-feedback="runtime-mode" data-proposed="algorithmic tools mode" data-confidence="">
        <strong>Tools mode</strong>
        <span>${escapeHtml(strategyLabel(state.analysis.analysisStrategy || "tools_only"))}</span>
        <span>Algorithmic segmentation only</span>
        <span>${escapeHtml(state.analysis.status)}</span>
      </button>`;
    return;
  }
  if (!gemma) {
    target.innerHTML = `
      <button class="runtime-row is-error" data-feedback="gemma-runtime" data-proposed="Gemma result missing">
        <strong>Gemma mode</strong>
        <span>No Gemma result returned</span>
      </button>`;
    return;
  }
  const score = gemma.score || {};
  const parsed = gemma.ok ? "parsed" : "not parsed";
  const count = score.totalExpected !== undefined
    ? `${score.totalFound || 0}/${score.totalExpected || 0}`
    : score.total_expected !== undefined
      ? `${score.total_found || 0}/${score.total_expected || 0}`
      : "unscored";
  target.innerHTML = `
    <button class="runtime-row ${gemma.ok ? "" : "is-error"}" data-feedback="gemma-runtime" data-proposed="Gemma ${escapeHtml(parsed)} callout count ${escapeHtml(count)}" data-confidence="">
      <strong>Gemma mode</strong>
      <span>${escapeHtml(strategyLabel(gemma.strategy || state.analysis.analysisStrategy || state.analysisStrategy))}</span>
      <span>${escapeHtml(gemma.model || state.analysis.model || "gemma4")}</span>
      <span>${escapeHtml(parsed)} · callouts ${escapeHtml(count)}</span>
      ${gemma.error ? `<span>${escapeHtml(gemma.error)}</span>` : ""}
    </button>`;
}

function strategyLabel(strategy) {
  const labels = {
    tools_only: "Tools only",
    gemma_image_only: "Gemma image only",
    gemma_tools_briefing: "Gemma tools briefing",
    gemma_candidate_review: "Gemma candidate review",
    gemma_teacher_calibrated: "Gemma teacher calibrated",
  };
  return labels[strategy] || strategy || "Unknown strategy";
}

function renderTitleBlock() {
  const border = state.analysis.border || {
    confidence: "",
    present: false,
    masks: [
      { x: 0, y: 0, w: 1, h: 0.055 },
      { x: 0, y: 0.945, w: 1, h: 0.055 },
      { x: 0, y: 0, w: 0.045, h: 1 },
      { x: 0.955, y: 0, w: 0.045, h: 1 },
    ],
  };
  const block = state.analysis.titleBlock;
  if (border.present === false || !border.masks?.length) {
    $("#borderMask").innerHTML = `<button class="empty-analysis" data-feedback="border-mask" data-proposed="no border detected" data-confidence="${border.confidence ?? ""}">No border detected</button>`;
  } else {
    renderMask(
      $("#borderMask"),
      border.masks,
      "border-mask",
      "border masked as sheet edge",
      border.confidence
    );
  }
  if (block.present === false) {
    $("#titleBlockCrop").innerHTML = `<button class="empty-analysis" data-feedback="title-block-mask" data-proposed="no title block detected" data-confidence="${block.confidence ?? ""}">No title block detected</button>`;
  } else {
    renderCrop($("#titleBlockCrop"), block.crop, "title-block-mask", "title block isolated", block.confidence ?? "", {
      fit: "contain",
    });
  }
  const candidateNotes = block.candidate?.notes?.length
    ? `<button class="field-row" data-feedback="title-block-detector" data-proposed="${escapeHtml(block.candidate.notes.join("; "))}" data-confidence="${block.confidence ?? ""}">
        <span>Detector notes</span>
        <strong>${escapeHtml(block.candidate.notes.join("; "))}</strong>
        <span class="confidence">${Math.round((block.confidence ?? 0) * 100)}%</span>
      </button>`
    : "";
  $("#titleFields").innerHTML = candidateNotes + block.fields
    .map(
      (field) => `
        <button class="field-row" data-feedback="title-block-field" data-proposed="${escapeHtml(field.label)}: ${escapeHtml(field.value)}" data-confidence="${field.confidence}">
          <span>${escapeHtml(field.label)}</span>
          <strong>${escapeHtml(field.value)}</strong>
          <span class="confidence">${Math.round(field.confidence * 100)}%</span>
        </button>`
    )
    .join("");
}

function symbolSvg(symbol, value) {
  const mark =
    symbol === "position"
      ? `<circle cx="42" cy="43" r="18" fill="none" stroke="currentColor" stroke-width="4"/><line x1="42" y1="16" x2="42" y2="70" stroke="currentColor" stroke-width="3"/><line x1="15" y1="43" x2="69" y2="43" stroke="currentColor" stroke-width="3"/>`
      : `<path d="M18 56 L74 28" fill="none" stroke="currentColor" stroke-width="5"/>`;
  const safeValue = escapeHtml(value);
  return `
    <svg viewBox="0 0 280 90" role="img" aria-label="${safeValue}">
      <rect x="4" y="14" width="272" height="62" fill="none" stroke="currentColor" stroke-width="3"/>
      <line x1="88" y1="14" x2="88" y2="76" stroke="currentColor" stroke-width="3"/>
      <line x1="182" y1="14" x2="182" y2="76" stroke="currentColor" stroke-width="3"/>
      ${mark}
      <text x="104" y="52" font-size="21" font-family="monospace" fill="currentColor">${escapeHtml(value.split("|").slice(1).join("|").trim() || value)}</text>
    </svg>`;
}

function renderGdt() {
  $("#gdtList").innerHTML = state.analysis.gdt
    .map(
      (item, index) => `
        <article class="gdt-row">
          <div class="gdt-original crop-view" id="gdtCrop${index}" data-feedback="gdt-original-crop" data-proposed="${escapeHtml(item.label || item.kind || item.id)} original crop" data-confidence="${item.confidence}" data-crop="${cropData(item.crop)}"></div>
          <button class="gdt-vector" data-feedback="gdt-vector-reconstruction" data-proposed="${escapeHtml(item.value || item.kind || "unclassified callout")}" data-confidence="${item.confidence}">
            ${symbolSvg(item.symbol || "unclassified", item.value || item.kind || "unclassified")}
          </button>
          <button class="gdt-meta" data-feedback="gdt-recognition" data-proposed="${escapeHtml(item.label || item.kind || item.id)}: ${escapeHtml(item.value || "unclassified")}" data-confidence="${item.confidence}">
            <strong>${escapeHtml(item.label || item.kind || item.id)}</strong>
            <span>${escapeHtml(item.value || item.symbol || "unclassified")}</span>
            <span class="confidence">${Math.round(item.confidence * 100)}%</span>
          </button>
        </article>`
    )
    .join("");
  state.analysis.gdt.forEach((item, index) => {
    renderCrop($(`#gdtCrop${index}`), item.crop, "gdt-original-crop", `${item.label} original crop`, item.confidence, {
      fit: "contain",
    });
  });
}

function renderProjections() {
  $("#projectionGrid").innerHTML = state.analysis.projections
    .map(
      (projection, index) => `
        <article class="projection-card">
          <div class="crop-view" id="projectionCrop${index}" data-feedback="projection-crop" data-proposed="${escapeHtml(projection.label)}" data-confidence="${projection.confidence}" data-crop="${cropData(projection.crop)}"></div>
          <button class="projection-label" data-feedback="projection-label" data-proposed="${escapeHtml(projection.label)} axis ${escapeHtml(projection.axis)}" data-confidence="${projection.confidence}" data-crop="${cropData(projection.crop)}">
            <strong>${escapeHtml(projection.label)}</strong>
            <span>${escapeHtml(projection.segmentationMode || projection.axis)}</span>
            <span class="confidence">${Math.round(projection.confidence * 100)}%</span>
          </button>
        </article>`
    )
    .join("");
  state.analysis.projections.forEach((projection, index) => {
    renderCrop($(`#projectionCrop${index}`), projection.crop, "projection-crop", projection.label, projection.confidence, {
      fit: "contain",
      masks: annotationMasks(),
      rectangleTool: true,
    });
  });
}

function renderCallouts() {
  const toolCallouts = state.analysis.callouts
    .map(
      (callout) => `
        <button class="callout-card" data-feedback="2d-callout" data-proposed="${escapeHtml(callout.label)}: ${escapeHtml(callout.value)}" data-confidence="${callout.confidence}" data-crop="${callout.crop ? cropData(callout.crop) : ""}">
          <span>${escapeHtml(callout.view ? `${callout.view} ${callout.kind}` : callout.label)}</span>
          <strong>${escapeHtml(callout.value)}</strong>
          <span class="confidence">${Math.round(callout.confidence * 100)}%</span>
        </button>`
    )
    .join("");
  const gemmaCallouts = gemmaViewCallouts();
  const gemmaSection = gemmaCallouts.length
    ? `
      <section class="gemma-callout-section">
        <h3>Gemma inference</h3>
        <div class="callout-grid">
          ${gemmaCallouts
            .map(
              (callout) => `
                <button class="callout-card" data-feedback="gemma-callout" data-proposed="${escapeHtml(callout.view)} ${escapeHtml(callout.kind)}: ${escapeHtml(callout.text)}">
                  <span>${escapeHtml(callout.view)} ${escapeHtml(callout.kind)}</span>
                  <strong>${escapeHtml(callout.text || callout.id || "unread")}</strong>
                  <span class="confidence">${escapeHtml(callout.reason || "Gemma full-image inference")}</span>
                </button>`
            )
            .join("")}
        </div>
      </section>`
    : "";
  $("#calloutGrid").innerHTML = toolCallouts + gemmaSection;
}

function gemmaViewCallouts() {
  const views = state.analysis?.gemma?.parsedResponse?.views;
  if (!views || typeof views !== "object") return [];
  return Object.entries(views).flatMap(([view, items]) =>
    Array.isArray(items)
      ? items.map((item) => ({
          view,
          id: item.id || "",
          kind: item.kind || "callout",
          text: item.text || item.value || "",
          reason: item.reason || "",
        }))
      : []
  );
}

function render3d() {
  const perspective = state.analysis.projections.find((projection) => projection.id === "perspective");
  if (!perspective) {
    $("#view3dContent").innerHTML = `
      <button class="empty-analysis" data-feedback="3d-view-missing" data-proposed="no 3D view detected">
        No 3D view detected
      </button>`;
    return;
  }
  $("#view3dContent").innerHTML = `
    <article class="projection-card">
      <div class="crop-view" id="perspectiveCrop"></div>
      <button class="projection-label" data-feedback="3d-view-label" data-proposed="${escapeHtml(perspective.label)} axis ${escapeHtml(perspective.axis)}" data-confidence="${perspective.confidence}">
        <strong>${escapeHtml(perspective.label)}</strong>
        <span>${escapeHtml(perspective.axis)}</span>
        <span class="confidence">${Math.round(perspective.confidence * 100)}%</span>
      </button>
    </article>`;
  renderCrop($("#perspectiveCrop"), perspective.crop, "3d-view-crop", perspective.label, perspective.confidence, {
    fit: "contain",
    masks: annotationMasks(),
  });
}

function renderCad() {
  const cad = state.analysis.cad;
  $("#cadSummary").innerHTML = `
    <strong>CAD status</strong>
    <span>${escapeHtml(cad.strategy)}</span>
    <span class="confidence">confidence ${Math.round(cad.confidence * 100)}%</span>
    <button data-feedback="cad-strategy" data-proposed="${escapeHtml(cad.strategy)}" data-confidence="${cad.confidence}">Flag CAD strategy</button>
    <button data-feedback="cad-output" data-proposed="${escapeHtml(cad.stepFile || "no STEP file produced")}">Flag STEP output</button>`;
}

function quickTemplate(template) {
  const templates = {
    "missing-callout": ["missing-callout", "agent omitted a drawing callout"],
    "bad-crop": ["segmentation", "crop bounds are incorrect"],
    "wrong-symbol": ["gdt", "recognized symbol or value is incorrect"],
    "wrong-axis": ["projection-label", "projection axis label is incorrect"],
    "bad-cad": ["cad", "constructed CAD does not match drawing"],
  };
  const [area, proposed] = templates[template] || ["dashboard", template];
  setFeedback(area, proposed, "<issue>");
}

$$(".tab").forEach((button) => button.addEventListener("click", () => setTab(button.dataset.tab)));
$$(".subtab").forEach((button) => button.addEventListener("click", () => setSubtab(button.dataset.subtab)));
$$(".mode-button").forEach((button) => button.addEventListener("click", () => setRuntimeMode(button.dataset.mode)));
$("#gemmaModelSelect").addEventListener("change", (event) => setGemmaModel(event.target.value));
$("#analysisStrategySelect").addEventListener("change", (event) => setAnalysisStrategy(event.target.value));
$("#imageInput").addEventListener("change", onImageSelected);
$("#processButton").addEventListener("click", processDrawing);
$("#copyFeedback").addEventListener("click", async () => {
  await navigator.clipboard.writeText($("#feedbackText").value);
  $("#copyFeedback").textContent = "Copied";
  setTimeout(() => ($("#copyFeedback").textContent = "Copy"), 1000);
});
$$("[data-template]").forEach((button) => button.addEventListener("click", () => quickTemplate(button.dataset.template)));

renderEmptyAnalysis();
