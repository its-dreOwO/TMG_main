"use strict";

const MAX_BYTES = 4 * 1024 * 1024;
const COLD_START_HINT_MS = 6000;

const $ = (id) => document.getElementById(id);
const form = $("form"), fileInput = $("file"), drop = $("drop");
const submit = $("submit"), statusEl = $("status");
const result = $("result"), counts = $("counts");
const viewBtn = $("view"), download = $("download"), again = $("again");

let file = null;
let pdfUrl = null;

// Start the model loading while the person picks a file (cold start).
fetch("api/warm", { cache: "no-store" }).catch(() => {});

function setStatus(text, isError = false) {
  statusEl.textContent = text;
  statusEl.classList.toggle("error", isError);
}

function pick(f) {
  file = null;
  submit.disabled = true;
  if (!f) return;
  const isPdf = f.type === "application/pdf" || f.name.toLowerCase().endsWith(".pdf");
  if (!isPdf) return setStatus("Vui lòng chọn một file PDF.", true);
  if (f.size > MAX_BYTES) return setStatus("File quá lớn. Giới hạn là 4 MB.", true);
  file = f;
  $("drop-title").textContent = f.name;
  $("drop-hint").textContent = `${(f.size / 1024 / 1024).toFixed(2)} MB · bấm để chọn file khác`;
  submit.disabled = false;
  setStatus("");
}

fileInput.addEventListener("change", () => pick(fileInput.files[0]));
["dragenter", "dragover"].forEach((t) => drop.addEventListener(t, (e) => {
  e.preventDefault();
  drop.classList.add("over");
}));
["dragleave", "drop"].forEach((t) => drop.addEventListener(t, (e) => {
  e.preventDefault();
  drop.classList.remove("over");
}));
drop.addEventListener("drop", (e) => pick(e.dataTransfer.files[0]));

function outputName(name) {
  return name.replace(/\.pdf$/i, "") + "_danh_dau.pdf";
}

function base64ToBlob(b64) {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new Blob([bytes], { type: "application/pdf" });
}

function renderCounts(c) {
  const n = (k) => Math.max(0, Number(c && c[k]) | 0);
  const high = n("Cao"), mid = n("Trung bình"), low = n("Thấp");
  counts.replaceChildren();
  if (!high && !mid && !low) {
    counts.textContent = "Không phát hiện điều khoản rủi ro nào.";
    return;
  }
  const strong = document.createElement("strong");
  strong.textContent = String(high);
  counts.append(strong, ` rủi ro Cao · ${mid} Trung bình · ${low} Thấp`);
}

async function readError(res) {
  try {
    const body = await res.json();
    if (body && body.error) return body.error;
  } catch (_) { /* not JSON, e.g. a Cloudflare timeout page */ }
  if (res.status === 524 || res.status === 504) return "Máy chủ xử lý quá lâu. Vui lòng thử lại.";
  return `Có lỗi xảy ra (mã ${res.status}). Vui lòng thử lại.`;
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  if (!file) return;

  // Open the viewer tab now, while the click still counts as a user action;
  // browsers block window.open() after an asynchronous wait.
  let tab = window.open("", "_blank");
  if (tab) tab.document.title = "Đang phân tích…";

  submit.disabled = true;
  fileInput.disabled = true;
  setStatus("Đang phân tích…");
  const hint = setTimeout(
    () => setStatus("Đang khởi động mô hình, lần đầu có thể mất khoảng 30 giây…"), COLD_START_HINT_MS);

  const params = new URLSearchParams({ contract_type: $("contract-type").value });
  if ($("party").value) params.set("party", $("party").value);

  try {
    const res = await fetch(`api/analyze?${params}`, {
      method: "POST",
      headers: { "content-type": "application/pdf" },
      body: file,
    });
    if (!res.ok) throw new Error(await readError(res));
    const data = await res.json();

    if (pdfUrl) URL.revokeObjectURL(pdfUrl);
    pdfUrl = URL.createObjectURL(base64ToBlob(data.pdf));
    download.href = pdfUrl;
    download.download = outputName(file.name);
    renderCounts(data.counts);

    if (tab && !tab.closed) tab.location.href = pdfUrl;
    form.hidden = true;
    result.hidden = false;
    setStatus("");
  } catch (err) {
    if (tab && !tab.closed) tab.close();
    const msg = err instanceof TypeError ? "Không kết nối được máy chủ. Vui lòng thử lại." : err.message;
    setStatus(msg, true);
    submit.disabled = false;
  } finally {
    clearTimeout(hint);
    fileInput.disabled = false;
  }
});

viewBtn.addEventListener("click", () => {
  if (pdfUrl) window.open(pdfUrl, "_blank");
});

again.addEventListener("click", () => {
  result.hidden = true;
  form.hidden = false;
  form.reset();
  file = null;
  submit.disabled = true;
  $("drop-title").textContent = "Chọn file PDF";
  $("drop-hint").textContent = "hoặc kéo thả vào đây · tối đa 4 MB, 30 trang";
  setStatus("");
});
