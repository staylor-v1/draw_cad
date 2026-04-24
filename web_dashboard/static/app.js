const state = {
  upload: null,
  analysis: null,
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

function currentFilename() {
  return state.upload?.filename || "<none>";
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
  $("#modelBadge").textContent = "Gemma 4 ready";
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
  $("#modelBadge").textContent = "Gemma 4 processing";
  const progressDone = animateProgress();
  const response = await fetch("/api/analyze", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      id: state.upload.id,
      filename: state.upload.filename,
      imageUrl: state.upload.url,
    }),
  });
  const analysis = await response.json();
  await progressDone;
  state.analysis = analysis;
  $("#modelBadge").textContent = analysis.status;
  renderAnalysis();
  $("#processButton").disabled = false;
}

function cropData(crop) {
  return `${Math.round(crop.x * 100)},${Math.round(crop.y * 100)},${Math.round(crop.w * 100)},${Math.round(crop.h * 100)}`;
}

function renderCrop(container, crop, feedback, proposed, confidence) {
  container.innerHTML = "";
  container.dataset.feedback = feedback;
  container.dataset.proposed = proposed;
  container.dataset.confidence = confidence;
  container.dataset.crop = cropData(crop);

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

function renderEmptyAnalysis() {
  ["titleFields", "gdtList", "projectionGrid", "calloutGrid", "view3dContent"].forEach((id) => {
    $(`#${id}`).innerHTML = `<div class="empty-analysis">No analysis result</div>`;
  });
  $("#cadSummary").innerHTML = `<strong>CAD status</strong><span>No drawing processed</span>`;
}

function renderAnalysis() {
  renderTitleBlock();
  renderGdt();
  renderProjections();
  renderCallouts();
  render3d();
  renderCad();
  attachFeedback(document);
}

function renderTitleBlock() {
  const block = state.analysis.titleBlock;
  renderCrop($("#titleBlockCrop"), block.crop, "border-title-block", "title block crop", "");
  $("#titleFields").innerHTML = block.fields
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
          <div class="gdt-original crop-view" id="gdtCrop${index}" data-feedback="gdt-original-crop" data-proposed="${escapeHtml(item.label)} original crop" data-confidence="${item.confidence}" data-crop="${cropData(item.crop)}"></div>
          <button class="gdt-vector" data-feedback="gdt-vector-reconstruction" data-proposed="${escapeHtml(item.value)}" data-confidence="${item.confidence}">
            ${symbolSvg(item.symbol, item.value)}
          </button>
          <button class="gdt-meta" data-feedback="gdt-recognition" data-proposed="${escapeHtml(item.label)}: ${escapeHtml(item.value)}" data-confidence="${item.confidence}">
            <strong>${escapeHtml(item.label)}</strong>
            <span>${escapeHtml(item.value)}</span>
            <span class="confidence">${Math.round(item.confidence * 100)}%</span>
          </button>
        </article>`
    )
    .join("");
  state.analysis.gdt.forEach((item, index) => {
    renderCrop($(`#gdtCrop${index}`), item.crop, "gdt-original-crop", `${item.label} original crop`, item.confidence);
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
            <span>${escapeHtml(projection.axis)}</span>
            <span class="confidence">${Math.round(projection.confidence * 100)}%</span>
          </button>
        </article>`
    )
    .join("");
  state.analysis.projections.forEach((projection, index) => {
    renderCrop($(`#projectionCrop${index}`), projection.crop, "projection-crop", projection.label, projection.confidence);
  });
}

function renderCallouts() {
  $("#calloutGrid").innerHTML = state.analysis.callouts
    .map(
      (callout) => `
        <button class="callout-card" data-feedback="2d-callout" data-proposed="${escapeHtml(callout.label)}: ${escapeHtml(callout.value)}" data-confidence="${callout.confidence}">
          <span>${escapeHtml(callout.label)}</span>
          <strong>${escapeHtml(callout.value)}</strong>
          <span class="confidence">${Math.round(callout.confidence * 100)}%</span>
        </button>`
    )
    .join("");
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
  renderCrop($("#perspectiveCrop"), perspective.crop, "3d-view-crop", perspective.label, perspective.confidence);
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
$("#imageInput").addEventListener("change", onImageSelected);
$("#processButton").addEventListener("click", processDrawing);
$("#copyFeedback").addEventListener("click", async () => {
  await navigator.clipboard.writeText($("#feedbackText").value);
  $("#copyFeedback").textContent = "Copied";
  setTimeout(() => ($("#copyFeedback").textContent = "Copy"), 1000);
});
$$("[data-template]").forEach((button) => button.addEventListener("click", () => quickTemplate(button.dataset.template)));

renderEmptyAnalysis();
