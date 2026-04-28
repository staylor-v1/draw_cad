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
const clearVisibleButton = document.querySelector("#clear-visible");
const copyButton = document.querySelector("#copy-selected");
const dialog = document.querySelector("#candidate-dialog");
const dialogTitle = document.querySelector("#dialog-title");
const dialogSource = document.querySelector("#dialog-source");
const dialogPreview = document.querySelector("#dialog-preview");
const dialogPath = document.querySelector("#dialog-path");
const dialogOpen = document.querySelector("#dialog-open");
const dialogSelect = document.querySelector("#dialog-select");

let candidates = [];
let visibleCandidates = [];
let selected = new Set();
let activeCandidate = null;

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
  visibleCount.textContent = String(visibleCandidates.length);
  selectedCount.textContent = String(selected.size);
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
  const isSelected = selected.has(activeCandidate.path);
  dialogSelect.textContent = isSelected ? "Selected" : "Use";
  dialogSelect.classList.toggle("primary", isSelected);
}

function openCandidate(candidate) {
  activeCandidate = candidate;
  dialogTitle.textContent = candidate.name;
  dialogSource.textContent = `${candidate.modelSlug} | ${candidate.extension.toUpperCase()} | ${candidate.confidence}`;
  dialogPath.textContent = candidate.path;
  dialogOpen.href = candidate.url;
  dialogPreview.replaceChildren(buildPreview(candidate).firstElementChild);
  syncDialogSelection();
  if (typeof dialog.showModal === "function") dialog.showModal();
  else dialog.setAttribute("open", "");
}

function render() {
  grid.replaceChildren();
  visibleCandidates = candidates.filter(candidateMatches);
  for (const candidate of visibleCandidates) {
    const node = template.content.firstElementChild.cloneNode(true);
    const checkbox = node.querySelector("input");
    const checkLabel = node.querySelector(".candidate-check span");
    const preview = buildPreview(candidate);
    const title = node.querySelector(".candidate-title");
    const subtitle = node.querySelector(".candidate-subtitle");
    const tags = node.querySelector(".candidate-tags");

    node.dataset.path = candidate.path;
    checkbox.checked = selected.has(candidate.path);
    node.classList.toggle("selected", checkbox.checked);
    checkLabel.textContent = checkbox.checked ? "Selected" : "Use";
    title.textContent = candidate.name;
    subtitle.textContent = `${candidate.modelSlug} | ${formatBytes(candidate.sizeBytes)}`;

    tags.replaceChildren(
      makeTag(candidate.extension.toUpperCase()),
      makeTag(candidate.previewKind),
      makeTag(candidate.confidence, candidate.confidence),
    );

    node.querySelector(".preview").replaceWith(preview);
    node.querySelector(".candidate-check").addEventListener("click", (event) => {
      event.stopPropagation();
    });
    checkbox.addEventListener("change", () => {
      if (checkbox.checked) selected.add(candidate.path);
      else selected.delete(candidate.path);
      node.classList.toggle("selected", checkbox.checked);
      checkLabel.textContent = checkbox.checked ? "Selected" : "Use";
      updateCounts();
      saveStatus.textContent = "Unsaved changes";
    });
    node.addEventListener("click", (event) => {
      if (event.target.closest("input, iframe")) return;
      openCandidate(candidate);
    });
    grid.appendChild(node);
  }
  updateCounts();
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
      body: JSON.stringify({ selected: [...selected] }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "Selection save failed");
    saveStatus.textContent = `Saved ${payload.count} at ${payload.updatedAt}`;
  } catch (error) {
    saveStatus.textContent = error.message;
  } finally {
    saveButton.disabled = false;
  }
}

async function copySelection() {
  const text = [...selected].sort().join("\n");
  await navigator.clipboard.writeText(text);
  saveStatus.textContent = `Copied ${selected.size} paths`;
}

async function load() {
  saveStatus.textContent = "Loading candidates...";
  const response = await fetch("/api/training-candidates");
  const payload = await response.json();
  candidates = payload.candidates || [];
  selected = new Set(payload.selection?.selected || []);
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
  for (const candidate of visibleCandidates) selected.add(candidate.path);
  saveStatus.textContent = "Unsaved changes";
  render();
});

clearVisibleButton.addEventListener("click", () => {
  for (const candidate of visibleCandidates) selected.delete(candidate.path);
  saveStatus.textContent = "Unsaved changes";
  render();
});

saveButton.addEventListener("click", saveSelection);
copyButton.addEventListener("click", copySelection);
dialogSelect.addEventListener("click", () => {
  if (!activeCandidate) return;
  if (selected.has(activeCandidate.path)) selected.delete(activeCandidate.path);
  else selected.add(activeCandidate.path);
  syncDialogSelection();
  saveStatus.textContent = "Unsaved changes";
  render();
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
