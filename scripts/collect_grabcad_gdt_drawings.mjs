#!/usr/bin/env node
import fs from "node:fs/promises";
import path from "node:path";

const ROOT = process.cwd();
const DEFAULT_OUT = path.join(ROOT, "training_data", "gdt_grabcad");
const DRAWING_EXTENSIONS = new Set([
  ".pdf", ".dwg", ".dxf", ".idw", ".slddrw", ".drw", ".prt",
  ".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp",
]);

function parseArgs(argv) {
  const args = {
    port: 9223,
    tagUrl: "https://grabcad.com/library/tag/gd-t",
    outDir: DEFAULT_OUT,
    maxModels: 0,
    download: false,
    fromManifest: false,
    urlList: "",
    onlyWithFileCandidates: false,
    slugs: [],
    settleMs: 2200,
  };
  for (let i = 2; i < argv.length; i += 1) {
    const arg = argv[i];
    const next = argv[i + 1];
    if (arg === "--port") args.port = Number(next), i += 1;
    else if (arg === "--tag-url") args.tagUrl = next, i += 1;
    else if (arg === "--out-dir") args.outDir = path.resolve(next), i += 1;
    else if (arg === "--max-models") args.maxModels = Number(next), i += 1;
    else if (arg === "--download") args.download = true;
    else if (arg === "--from-manifest") args.fromManifest = true;
    else if (arg === "--url-list") args.urlList = path.resolve(next), i += 1;
    else if (arg === "--only-with-file-candidates") args.onlyWithFileCandidates = true;
    else if (arg === "--slug") args.slugs.push(next), i += 1;
    else if (arg === "--settle-ms") args.settleMs = Number(next), i += 1;
    else if (arg === "--help") {
      console.log(`Usage: node scripts/collect_grabcad_gdt_drawings.mjs [--download] [--max-models N]\n\nRequires a logged-in Chrome session running with --remote-debugging-port=9223.`);
      process.exit(0);
    }
  }
  return args;
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function jsonGet(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`GET ${url} failed with ${response.status}`);
  return response.json();
}

class Cdp {
  constructor(webSocketUrl) {
    this.webSocketUrl = webSocketUrl;
    this.id = 0;
    this.pending = new Map();
    this.events = [];
  }

  async open() {
    this.ws = new WebSocket(this.webSocketUrl);
    this.ws.onmessage = (event) => {
      const msg = JSON.parse(event.data);
      if (msg.id && this.pending.has(msg.id)) {
        const { resolve, reject } = this.pending.get(msg.id);
        this.pending.delete(msg.id);
        if (msg.error) reject(new Error(`${msg.error.message}: ${msg.error.data ?? ""}`));
        else resolve(msg);
      } else if (msg.method) {
        this.events.push(msg);
      }
    };
    await new Promise((resolve, reject) => {
      this.ws.onopen = resolve;
      this.ws.onerror = reject;
    });
  }

  send(method, params = {}) {
    const id = ++this.id;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.ws.send(JSON.stringify({ id, method, params }));
    });
  }

  async evaluate(expression, returnByValue = true) {
    const result = await this.send("Runtime.evaluate", {
      expression,
      awaitPromise: true,
      returnByValue,
    });
    if (result.result.exceptionDetails) {
      throw new Error(result.result.exceptionDetails.text);
    }
    return result.result.result.value;
  }

  async close() {
    if (this.ws) this.ws.close();
  }
}

async function findPage(port, tagUrl) {
  const tabs = await jsonGet(`http://127.0.0.1:${port}/json/list`);
  const exact = tabs.find((tab) => tab.type === "page" && tab.url === tagUrl);
  const tag = tabs.find((tab) => tab.type === "page" && tab.url.includes("/library/tag/gd-t"));
  const grabcad = tabs.find((tab) => tab.type === "page" && tab.url.includes("grabcad.com"));
  return exact ?? tag ?? grabcad ?? tabs.find((tab) => tab.type === "page");
}

async function navigate(page, url, settleMs) {
  await page.send("Page.navigate", { url });
  await sleep(settleMs);
}

function normalizeModelUrl(raw) {
  try {
    const url = new URL(raw);
    if (url.hostname !== "grabcad.com") return null;
    const parts = url.pathname.split("/").filter(Boolean);
    if (parts.length !== 2 || parts[0] !== "library") return null;
    if (parts[1] === "tag" || parts[1] === "new") return null;
    return `https://grabcad.com/library/${parts[1]}`;
  } catch {
    return null;
  }
}

async function discoverModels(page, tagUrl, settleMs) {
  await navigate(page, tagUrl, settleMs);
  const seenHeights = [];
  const byUrl = new Map();
  for (let step = 0; step < 18; step += 1) {
    const batch = await page.evaluate(`(() => {
      const links = [...document.querySelectorAll('a[href*="/library/"]')].map((a) => {
        const card = a.closest('article, li, .modelCard, .model-card, .card, .library-card, .model, .item, .thumbnail, .result, div');
        const img = card?.querySelector('img');
        return {
          href: a.href,
          text: a.innerText.trim(),
          title: (a.getAttribute('title') || '').trim(),
          cardText: (card?.innerText || '').trim().replace(/\\s+/g, ' ').slice(0, 500),
          image: img?.src || ''
        };
      });
      return { y: window.scrollY, height: document.body.scrollHeight, links };
    })()`);
    for (const item of batch.links) {
      const modelUrl = normalizeModelUrl(item.href);
      if (!modelUrl) continue;
      const existing = byUrl.get(modelUrl) ?? { url: modelUrl, titles: new Set(), cardTexts: new Set(), images: new Set() };
      for (const text of [item.text, item.title]) {
        if (text && !/^download/i.test(text)) existing.titles.add(text);
      }
      if (item.cardText) existing.cardTexts.add(item.cardText);
      if (item.image) existing.images.add(item.image);
      byUrl.set(modelUrl, existing);
    }
    seenHeights.push(batch.height);
    await page.evaluate("window.scrollTo(0, document.body.scrollHeight)");
    await sleep(settleMs);
    if (seenHeights.length >= 4) {
      const last = seenHeights.slice(-4);
      if (last.every((height) => height === last[0])) break;
    }
  }

  return [...byUrl.values()].map((item) => ({
    url: item.url,
    slug: new URL(item.url).pathname.split("/").pop(),
    title: [...item.titles].sort((a, b) => b.length - a.length)[0] || [...item.cardTexts][0]?.split(" by ")[0] || item.slug,
    cardText: [...item.cardTexts].sort((a, b) => b.length - a.length)[0] || "",
    previewImages: [...item.images],
  }));
}

async function inspectModel(page, model, settleMs) {
  await navigate(page, model.url, settleMs);
  await page.evaluate("window.scrollTo(0, 0)");
  await sleep(600);
  const details = await page.evaluate(`(() => {
    const visibleText = document.body.innerText.replace(/\\s+/g, ' ').trim();
    const fileRows = [...document.querySelectorAll('a, button, [role="button"], li, tr, .file, .files, .download, .downloads')].map((el) => ({
      tag: el.tagName,
      text: (el.innerText || el.textContent || '').trim().replace(/\\s+/g, ' ').slice(0, 300),
      href: el.href || '',
      cls: typeof el.className === 'string' ? el.className : '',
      id: el.id || ''
    })).filter((x) => x.text || x.href);
    const title = document.querySelector('h1')?.innerText.trim() || document.title.replace(/ \\| GrabCAD.*/, '');
    const author = [...document.querySelectorAll('a[href^="/"], a[href^="https://grabcad.com/"]')]
      .map((a) => ({ text: a.innerText.trim(), href: a.href }))
      .find((a) => a.href.match(/^https:\\/\\/grabcad\\.com\\/[A-Za-z0-9._-]+$/) && a.text && !/log out|settings/i.test(a.text));
    return { title, author, visibleText: visibleText.slice(0, 5000), fileRows };
  })()`);
  const possibleFiles = [];
  const filePattern = /[\w .()[\]-]+\.(pdf|dwg|dxf|idw|slddrw|drw|jpg|jpeg|png|webp|tif|tiff|bmp)\b/ig;
  let match;
  while ((match = filePattern.exec(details.visibleText)) !== null) possibleFiles.push(match[0].trim());
  for (const row of details.fileRows) {
    filePattern.lastIndex = 0;
    while ((match = filePattern.exec(`${row.text} ${row.href}`)) !== null) possibleFiles.push(match[0].trim());
  }
  const uniqueFiles = [...new Set(possibleFiles)].sort();
  const orthographicCandidate = uniqueFiles.some((name) => DRAWING_EXTENSIONS.has(path.extname(name).toLowerCase())) ||
    /drawing|orthographic|blueprint|drafting|dimension|gd[&-]?t|geometric|toleranc/i.test(details.visibleText);
  return {
    ...model,
    pageTitle: details.title,
    author: details.author,
    possibleDrawingFiles: uniqueFiles,
    orthographicCandidate,
    pageEvidence: details.visibleText.slice(0, 1200),
  };
}

async function clickDownload(page, model, downloadDir, settleMs) {
  const before = new Set(await fs.readdir(downloadDir).catch(() => []));
  const clickResult = await page.evaluate(`(() => {
    const nodes = [...document.querySelectorAll('a, button, [role="button"]')];
    const candidates = nodes.map((el, index) => ({
      index,
      text: (el.innerText || el.textContent || '').trim(),
      aria: el.getAttribute('aria-label') || '',
      title: el.getAttribute('title') || '',
      href: el.href || '',
      visible: !!(el.offsetWidth || el.offsetHeight || el.getClientRects().length)
    })).filter((x) => x.visible && /download/i.test([x.text, x.aria, x.title, x.href].join(' ')));
    const best = candidates.find((x) => /download files/i.test(x.text)) || candidates[0];
    if (!best) return { clicked: false, reason: 'no visible download control', candidates };
    nodes[best.index].scrollIntoView({ block: 'center' });
    nodes[best.index].click();
    return { clicked: true, chosen: best, candidates };
  })()`);
  if (!clickResult.clicked) return { ...clickResult, downloadedFiles: [] };
  await sleep(settleMs);
  const modalResult = await page.evaluate(`(() => {
    const nodes = [...document.querySelectorAll('a, button, [role="button"], input[type="submit"]')];
    const candidates = nodes.map((el, index) => ({
      index,
      text: (el.innerText || el.value || el.textContent || '').trim(),
      aria: el.getAttribute('aria-label') || '',
      title: el.getAttribute('title') || '',
      href: el.href || '',
      visible: !!(el.offsetWidth || el.offsetHeight || el.getClientRects().length)
    })).filter((x) => x.visible && /(download|agree|accept|continue)/i.test([x.text, x.aria, x.title, x.href].join(' ')));
    const best = candidates.find((x) => /(agree|accept|continue|download)/i.test(x.text)) || null;
    if (!best) return { clicked: false, candidates };
    nodes[best.index].click();
    return { clicked: true, chosen: best, candidates };
  })()`);
  await sleep(Math.max(6000, settleMs * 2));
  const after = await fs.readdir(downloadDir).catch(() => []);
  const downloadedFiles = after.filter((name) => !before.has(name));
  return { ...clickResult, modalResult, downloadedFiles, model: model.url };
}

async function main() {
  const args = parseArgs(process.argv);
  const rawDir = path.join(args.outDir, "raw_archives");
  await fs.mkdir(rawDir, { recursive: true });
  await fs.mkdir(path.join(args.outDir, "manifests"), { recursive: true });

  const version = await jsonGet(`http://127.0.0.1:${args.port}/json/version`);
  const browser = new Cdp(version.webSocketDebuggerUrl);
  await browser.open();
  await browser.send("Browser.setDownloadBehavior", {
    behavior: "allow",
    downloadPath: rawDir,
    eventsEnabled: true,
  });

  const tab = await findPage(args.port, args.tagUrl);
  if (!tab) throw new Error(`No Chrome page target found on port ${args.port}`);
  const page = new Cdp(tab.webSocketDebuggerUrl);
  await page.open();
  await page.send("Runtime.enable");
  await page.send("Page.enable");

  let discovered;
  if (args.urlList) {
    const urls = (await fs.readFile(args.urlList, "utf8"))
      .split(/\r?\n/)
      .map((line) => line.trim())
      .filter(Boolean);
    discovered = urls.map((url) => ({
      url,
      slug: new URL(url).pathname.split("/").pop(),
      title: new URL(url).pathname.split("/").pop(),
      cardText: "",
      previewImages: [],
      possibleDrawingFiles: [],
    }));
  } else if (args.fromManifest) {
    const existing = JSON.parse(await fs.readFile(path.join(args.outDir, "manifest.json"), "utf8"));
    discovered = existing.models.map((model) => ({
      url: model.url,
      slug: model.slug,
      title: model.pageTitle || model.title,
      cardText: model.cardText || model.pageEvidence || "",
      previewImages: model.previewImages || [],
      possibleDrawingFiles: model.possibleDrawingFiles || [],
    }));
  } else {
    discovered = await discoverModels(page, args.tagUrl, args.settleMs);
  }
  let selected = args.maxModels > 0 ? discovered.slice(0, args.maxModels) : discovered;
  if (args.slugs.length) {
    const wanted = new Set(args.slugs);
    selected = selected.filter((model) => wanted.has(model.slug));
  }
  if (args.onlyWithFileCandidates && !args.urlList) {
    selected = selected.filter((model) => model.possibleDrawingFiles?.length || /\.(pdf|dwg|dxf|idw|slddrw|drw|jpe?g|png|webp|tiff?|bmp)\b/i.test(model.cardText || ""));
  }
  const inspected = [];
  const downloads = [];
  for (const model of selected) {
    const info = await inspectModel(page, model, args.settleMs);
    inspected.push(info);
    console.log(`[inspect] ${info.slug}: ${info.possibleDrawingFiles.length} candidate file names`);
    if (args.download) {
      if (args.onlyWithFileCandidates && !info.possibleDrawingFiles.length) {
        downloads.push({ model: info.url, skipped: true, reason: "no drawing file names visible on model page" });
        console.log(`[download] ${info.slug}: skipped; no drawing file names visible`);
        continue;
      }
      const result = await clickDownload(page, info, rawDir, args.settleMs);
      downloads.push(result);
      console.log(`[download] ${info.slug}: ${result.downloadedFiles?.join(", ") || result.reason || "clicked"}`);
    }
  }

  const manifest = {
    source: args.tagUrl,
    collectedAt: new Date().toISOString(),
    chromePort: args.port,
    discoveredCount: discovered.length,
    processedCount: inspected.length,
    downloadEnabled: args.download,
    rawArchiveDir: rawDir,
    models: inspected,
    downloads,
  };
  const manifestPath = path.join(args.outDir, "manifest.json");
  await fs.writeFile(manifestPath, `${JSON.stringify(manifest, null, 2)}\n`);
  const queue = inspected.map((m) => `${m.url} | ${m.pageTitle || m.title} | files=${m.possibleDrawingFiles.join(", ")}`).join("\n");
  await fs.writeFile(path.join(args.outDir, "orthographic_download_queue.txt"), `${queue}\n`);
  await fs.writeFile(path.join(args.outDir, "README.md"), `# GrabCAD GD&T Candidate Drawings\n\nSource tag page: ${args.tagUrl}\n\nCollected through a user-authenticated temporary Chrome session. GrabCAD projects are user-contributed; verify each item's license and reuse terms before using outside local evaluation.\n\n- Manifest: \`manifest.json\`\n- Raw archives: \`raw_archives/\`\n- Manual queue: \`orthographic_download_queue.txt\`\n\nThe collector prioritizes 2D orthographic drawing assets such as PDF, DWG, DXF, IDW, SLDDRW, DRW, and raster drawing sheets. Full project archives may include 3D CAD files; downstream extraction should keep only drawing-sheet candidates for the GD&T segmentation corpus.\n`);

  await page.close();
  await browser.close();
  console.log(JSON.stringify({ manifestPath, discovered: discovered.length, processed: inspected.length, rawDir }, null, 2));
}

main().catch((error) => {
  console.error(error.stack || error.message);
  process.exit(1);
});
