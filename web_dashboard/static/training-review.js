const grid = document.querySelector("#candidate-grid");
const template = document.querySelector("#candidate-template");
const searchInput = document.querySelector("#search");
const previewFilter = document.querySelector("#preview-filter");
const confidenceFilter = document.querySelector("#confidence-filter");
const sourceFilter = document.querySelector("#source-filter");
const visibleCount = document.querySelector("#visible-count");
const selectedCount = document.querySelector("#selected-count");
const saveStatus = document.querySelector("#save-status");
const saveButton = document.querySelector("#save-selection");
const selectVisibleButton = document.querySelector("#select-visible");
const rejectVisibleButton = document.querySelector("#reject-visible");
const clearVisibleButton = document.querySelector("#clear-visible");
const copyButton = document.querySelector("#copy-selected");
const dialog = document.querySelector("#candidate-dialog");
const dialogTitle = document.querySelector("#dialog-title");
const dialogSource = document.querySelector("#dialog-source");
const dialogPreview = document.querySelector("#dialog-preview");
const dialogPath = document.querySelector("#dialog-path");
const dialogOpen = document.querySelector("#dialog-open");
const dialogSelect = document.querySelector("#dialog-select");
const dialogReject = document.querySelector("#dialog-reject");
const dialogDuplicate = document.querySelector("#dialog-duplicate");
const rejectedCount = document.querySelector("#rejected-count");
const duplicateCount = document.querySelector("#duplicate-count");

let candidates = [];
let visibleCandidates = [];
let dispositions = new Map();
let activeCandidate = null;
const DISPOSITIONS = ["use", "reject", "duplicate"];

function formatBytes(bytes) {
  if (!Number.isFinite(bytes)) return "";
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function escapeText(value) {
  return String(value ?? "");
}

function candidateMatches(candidate) {
  const query = searchInput.value.trim().toLowerCase();
  const haystack = [
    candidate.name,
    candidate.modelSlug,
    candidate.originalMember,
    candidate.extension,
    candidate.confidence,
  ]
    .join(" ")
    .toLowerCase();
  if (query && !haystack.includes(query)) return false;
  if (previewFilter.value !== "all" && candidate.previewKind !== previewFilter.value) return false;
  if (confidenceFilter.value !== "all" && candidate.confidence !== confidenceFilter.value) return false;
  if (sourceFilter.value !== "all" && candidate.modelSlug !== sourceFilter.value) return false;
  return true;
}

function updateCounts() {
  const counts = dispositionCounts();
  visibleCount.textContent = String(visibleCandidates.length);
  selectedCount.textContent = String(counts.use);
  rejectedCount.textContent = String(counts.reject);
  duplicateCount.textContent = String(counts.duplicate);
}

function dispositionCounts() {
  const counts = { use: 0, reject: 0, duplicate: 0 };
  for (const value of dispositions.values()) {
    if (value in counts) counts[value] += 1;
  }
  return counts;
}

function buildPreview(candidate) {
  const holder = document.createElement("div");
  holder.className = "preview";
  if (candidate.previewKind === "image") {
    const img = document.createElement("img");
    img.src = candidate.url;
    img.alt = candidate.name;
    img.loading = "lazy";
    holder.appendChild(img);
    return holder;
  }
  if (candidate.previewKind === "pdf") {
    const frame = document.createElement("iframe");
    frame.src = `${candidate.url}#toolbar=0&navpanes=0&view=FitH`;
    frame.title = candidate.name;
    holder.appendChild(frame);
    return holder;
  }
  const filePreview = document.createElement("div");
  filePreview.className = "file-preview";
  filePreview.innerHTML = `<div><strong>${escapeText(candidate.extension).toUpperCase()}</strong><span>Open source file for review</span></div>`;
  holder.appendChild(filePreview);
  return holder;
}

function syncDialogSelection() {
  if (!activeCandidate) return;
  const disposition = dispositions.get(activeCandidate.path);
  dialogSelect.textContent = disposition === "use" ? "Using" : "Use";
  dialogReject.textContent = disposition === "reject" ? "Rejected" : "Reject";
  dialogDuplicate.textContent = disposition === "duplicate" ? "Duplicate" : "Duplicate";
  dialogSelect.classList.toggle("primary", disposition === "use");
  dialogReject.classList.toggle("danger", disposition === "reject");
  dialogDuplicate.classList.toggle("warning", disposition === "duplicate");
}

function setDisposition(path, disposition) {
  if (dispositions.get(path) === disposition) dispositions.delete(path);
  else dispositions.set(path, disposition);
}

function selectedPaths() {
  return [...dispositions.entries()]
    .filter(([, disposition]) => disposition === "use")
    .map(([path]) => path)
    .sort();
}

function dispositionObject() {
  return Object.fromEntries([...dispositions.entries()].sort(([a], [b]) => a.localeCompare(b)));
}

function openCandidate(candidate) {
  activeCandidate = candidate;
  dialogTitle.textContent = candidate.name;
  dialogSource.textContent = `${candidate.modelSlug} | ${candidate.extension.toUpperCase()} | ${candidate.confidence}`;
  dialogPath.textContent = candidate.sourceUrl
    ? `source=${candidate.sourceUrl} | file=${candidate.path}`
    : candidate.path;
  dialogOpen.href = candidate.url;
  dialogPreview.replaceChildren(buildPreview(candidate).firstElementChild);
  syncDialogSelection();
  if (typeof dialog.showModal === "function") dialog.showModal();
  else dialog.setAttribute("open", "");
}

function restoreScrollPosition(x, y) {
  window.scrollTo(x, y);
  requestAnimationFrame(() => window.scrollTo(x, y));
}

function render({ preserveScroll = false } = {}) {
  const scrollPosition = preserveScroll ? { x: window.scrollX, y: window.scrollY } : null;
  grid.replaceChildren();
  visibleCandidates = candidates.filter(candidateMatches);
  for (const candidate of visibleCandidates) {
    const node = template.content.firstElementChild.cloneNode(true);
    const checkboxes = [...node.querySelectorAll("input[data-disposition]")];
    const preview = buildPreview(candidate);
    const title = node.querySelector(".candidate-title");
    const subtitle = node.querySelector(".candidate-subtitle");
    const tags = node.querySelector(".candidate-tags");
    const disposition = dispositions.get(candidate.path);

    node.dataset.path = candidate.path;
    node.classList.toggle("selected", disposition === "use");
    node.classList.toggle("rejected", disposition === "reject");
    node.classList.toggle("duplicate", disposition === "duplicate");
    title.textContent = candidate.name;
    subtitle.textContent = `${candidate.modelSlug} | ${formatBytes(candidate.sizeBytes)}`;

    tags.replaceChildren(
      makeTag(candidate.extension.toUpperCase()),
      makeTag(candidate.previewKind),
      makeTag(candidate.confidence, candidate.confidence),
      makeTag(disposition || "unreviewed", disposition || ""),
    );

    node.querySelector(".preview").replaceWith(preview);
    node.querySelector(".candidate-disposition").addEventListener("click", (event) => {
      event.stopPropagation();
    });
    for (const checkbox of checkboxes) {
      checkbox.checked = disposition === checkbox.dataset.disposition;
      checkbox.addEventListener("change", () => {
        setDisposition(candidate.path, checkbox.dataset.disposition);
        updateCounts();
        saveStatus.textContent = "Unsaved changes";
        render({ preserveScroll: true });
      });
    }
    node.addEventListener("dblclick", (event) => {
      if (event.target.closest("input, iframe")) return;
      setDisposition(candidate.path, "use");
      saveStatus.textContent = "Unsaved changes";
      render({ preserveScroll: true });
    });
    node.addEventListener("click", (event) => {
      if (event.target.closest("input, iframe")) return;
      openCandidate(candidate);
    });
    grid.appendChild(node);
  }
  updateCounts();
  if (scrollPosition) restoreScrollPosition(scrollPosition.x, scrollPosition.y);
}

function makeTag(label, className = "") {
  const tag = document.createElement("span");
  tag.className = ["tag", className].filter(Boolean).join(" ");
  tag.textContent = label;
  return tag;
}

function populateSources() {
  const sources = [...new Set(candidates.map((candidate) => candidate.modelSlug).filter(Boolean))].sort();
  for (const source of sources) {
    const option = document.createElement("option");
    option.value = source;
    option.textContent = source;
    sourceFilter.appendChild(option);
  }
}

async function saveSelection() {
  saveButton.disabled = true;
  saveStatus.textContent = "Saving...";
  try {
    const response = await fetch("/api/training-selection", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ dispositions: dispositionObject() }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "Selection save failed");
    saveStatus.textContent = `Saved ${payload.reviewedCount} reviewed at ${payload.updatedAt}`;
  } catch (error) {
    saveStatus.textContent = error.message;
  } finally {
    saveButton.disabled = false;
  }
}

async function copySelection() {
  const text = selectedPaths().join("\n");
  await navigator.clipboard.writeText(text);
  saveStatus.textContent = `Copied ${selectedPaths().length} use paths`;
}

async function load() {
  saveStatus.textContent = "Loading candidates...";
  const response = await fetch("/api/training-candidates");
  const payload = await response.json();
  candidates = payload.candidates || [];
  const savedDispositions = payload.selection?.dispositions || {};
  dispositions = new Map(Object.entries(savedDispositions));
  populateSources();
  saveStatus.textContent = payload.selection?.updatedAt
    ? `Loaded saved selection from ${payload.selection.updatedAt}`
    : "No saved selection";
  render();
}

for (const control of [searchInput, previewFilter, confidenceFilter, sourceFilter]) {
  control.addEventListener("input", render);
}

selectVisibleButton.addEventListener("click", () => {
  for (const candidate of visibleCandidates) dispositions.set(candidate.path, "use");
  saveStatus.textContent = "Unsaved changes";
  render({ preserveScroll: true });
});

rejectVisibleButton.addEventListener("click", () => {
  for (const candidate of visibleCandidates) dispositions.set(candidate.path, "reject");
  saveStatus.textContent = "Unsaved changes";
  render({ preserveScroll: true });
});

clearVisibleButton.addEventListener("click", () => {
  for (const candidate of visibleCandidates) dispositions.delete(candidate.path);
  saveStatus.textContent = "Unsaved changes";
  render({ preserveScroll: true });
});

saveButton.addEventListener("click", saveSelection);
copyButton.addEventListener("click", copySelection);
dialogSelect.addEventListener("click", () => {
  if (!activeCandidate) return;
  setDisposition(activeCandidate.path, "use");
  syncDialogSelection();
  saveStatus.textContent = "Unsaved changes";
  render({ preserveScroll: true });
});

dialogReject.addEventListener("click", () => {
  if (!activeCandidate) return;
  setDisposition(activeCandidate.path, "reject");
  syncDialogSelection();
  saveStatus.textContent = "Unsaved changes";
  render({ preserveScroll: true });
});

dialogDuplicate.addEventListener("click", () => {
  if (!activeCandidate) return;
  setDisposition(activeCandidate.path, "duplicate");
  syncDialogSelection();
  saveStatus.textContent = "Unsaved changes";
  render({ preserveScroll: true });
});

dialog.addEventListener("click", (event) => {
  if (event.target === dialog) dialog.close();
});

document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && dialog.open) dialog.close();
});

load().catch((error) => {
  saveStatus.textContent = error.message;
});
