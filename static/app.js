const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));

const state = {
  file: null,
  result: null,
  loadingTimer: null,
};

function setView(name) {
  $$(".view").forEach((view) => {
    const active = view.id === `view-${name}`;
    view.classList.toggle("active", active);
    // CSS 里有 [hidden] { display: none !important; }，只切 class 不够，必须同步 hidden。
    view.hidden = !active;
  });
  $$(".nav-btn").forEach((button) => button.classList.toggle("active", button.dataset.view === name));
  window.scrollTo({ top: 0, behavior: "smooth" });
  if (name === "learn") {
    history.replaceState(null, "", "#learn");
  } else if (name === "extract") {
    history.replaceState(null, "", "#extract");
  } else {
    history.replaceState(null, "", window.location.pathname);
  }
}

$$("[data-view]").forEach((element) => {
  element.addEventListener("click", (event) => {
    event.preventDefault();
    setView(element.dataset.view);
  });
});

async function refreshHealth() {
  const pill = $("#healthPill");
  const text = $("#healthText");
  try {
    const response = await fetch("/api/health", { cache: "no-store" });
    const health = await response.json();
    if (health.model_loaded) {
      pill.classList.add("ok");
      pill.classList.remove("bad");
      text.textContent = "模型已就绪";
    } else {
      text.textContent = "模型正在加载";
      setTimeout(refreshHealth, 1800);
    }
  } catch (error) {
    pill.classList.add("bad");
    pill.classList.remove("ok");
    text.textContent = "服务未连接";
  }
}
refreshHealth();

function setSteps(active, doneThrough = 0) {
  $$(".step").forEach((step) => {
    const number = Number(step.dataset.step);
    step.classList.toggle("done", number <= doneThrough);
    step.classList.toggle("active", number === active);
  });
}

function humanSize(bytes) {
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

function setFile(file) {
  if (!file) return;
  if (!/\.wav$/i.test(file.name) && file.type !== "audio/wav" && file.type !== "audio/x-wav") {
    showLabError("请上传 WAV 文件。当前演示不直接接受 MP3。");
    return;
  }
  state.file = file;
  $("#fileName").textContent = file.name;
  $("#fileMeta").textContent = humanSize(file.size);
  $("#fileChip").hidden = false;
  $("#embedBtn").disabled = false;
  $("#labError").hidden = true;
  setSteps(2, 1);
}

function showLabError(message) {
  const box = $("#labError");
  box.textContent = message;
  box.hidden = false;
  box.scrollIntoView({ behavior: "smooth", block: "center" });
}

function clearLabError() {
  $("#labError").hidden = true;
}

$("#dropzone").addEventListener("click", (event) => {
  if (event.target.closest("button")) return;
  $("#audioInput").click();
});
$("#dropzone").addEventListener("keydown", (event) => {
  if (event.key === "Enter" || event.key === " ") {
    event.preventDefault();
    $("#audioInput").click();
  }
});
$("#audioInput").addEventListener("change", (event) => {
  setFile(event.target.files?.[0]);
});

["dragenter", "dragover"].forEach((name) => {
  $("#dropzone").addEventListener(name, (event) => {
    event.preventDefault();
    $("#dropzone").classList.add("dragging");
  });
});
["dragleave", "drop"].forEach((name) => {
  $("#dropzone").addEventListener(name, (event) => {
    event.preventDefault();
    $("#dropzone").classList.remove("dragging");
  });
});
$("#dropzone").addEventListener("drop", (event) => {
  setFile(event.dataTransfer.files?.[0]);
});

async function loadDemo(autoStart = false) {
  try {
    $("#demoBtn").disabled = true;
    $("#heroDemoBtn").disabled = true;
    const response = await fetch("/assets/demo.wav", { cache: "no-store" });
    if (!response.ok) throw new Error("示例音频加载失败");
    const blob = await response.blob();
    const file = new File([blob], "示例音频_4.0s.wav", { type: "audio/wav" });
    setFile(file);
    if (autoStart) {
      await startEmbed();
    } else {
      $("#inputCard").scrollIntoView({ behavior: "smooth", block: "start" });
    }
  } catch (error) {
    showLabError(`示例音频没加载出来：${error.message}`);
  } finally {
    $("#demoBtn").disabled = false;
    $("#heroDemoBtn").disabled = false;
  }
}

$("#demoBtn").addEventListener("click", (event) => {
  event.stopPropagation();
  loadDemo(false);
});
$("#heroDemoBtn").addEventListener("click", () => loadDemo(true));

$("#randomSecret").addEventListener("click", () => {
  const digits = selectedPayload() / 4;
  const bytes = new Uint8Array(2);
  crypto.getRandomValues(bytes);
  const value = ((bytes[0] << 8) | bytes[1]).toString(16).toUpperCase().padStart(4, "0");
  $("#secretInput").value = value.slice(0, digits);
  $("#secretInput").focus();
});

function selectedPayload() {
  return Number($("#payloadSelect").value || 8);
}

function syncSecretLength() {
  const digits = selectedPayload() / 4;
  const input = $("#secretInput");
  input.maxLength = digits;
  input.value = input.value.toUpperCase().replace(/[^0-9A-F]/g, "").slice(0, digits);
  $("#secretHint").textContent = `正好 ${digits} 位十六进制（0–9 / A–F）= ${selectedPayload()} bit。不是 ${digits} 个汉字的容量。`;
}

$("#payloadSelect").addEventListener("change", syncSecretLength);

$("#secretInput").addEventListener("input", (event) => {
  const digits = selectedPayload() / 4;
  event.target.value = event.target.value.toUpperCase().replace(/[^0-9A-F]/g, "").slice(0, digits);
});

$$('input[name="mode"]').forEach((radio) => {
  radio.addEventListener("change", () => {
    const mode = $('input[name="mode"]:checked').value;
    $$(".mode-card").forEach((card) => card.classList.toggle("active", card.dataset.modeCard === mode));
    const note = $("#modeNote");
    if (mode === "direct") {
      note.innerHTML = "<b>直接换抽屉：</b>像把“第 17 格”改成“第 900 格”。声音变化更小，但格子使用次数会变，直方图容易看出来。";
    } else {
      note.innerHTML = "<b>换座位，不换书单：</b>只交换已有编号，书目和数量完全不变。代价是要多带 24 个填充位，恢复率略低，极端情况下会装不下。";
    }
  });
});

$$('input[name="bandwidth"]').forEach((radio) => {
  radio.addEventListener("change", () => {
    const bandwidth = $('input[name="bandwidth"]:checked').value;
    $$(".quality-pill").forEach((pill) => pill.classList.toggle("active", pill.dataset.qualityCard === bandwidth));
  });
});

function selectedBandwidth() {
  const checked = $('input[name="bandwidth"]:checked');
  return checked ? checked.value : "24";
}

function startLoadingAnimation() {
  const steps = $$("#loadingSteps .load-step");
  steps.forEach((step) => step.classList.remove("active", "done"));
  steps[0].classList.add("active");
  const titles = [
    "先让 EnCodec 把声音拆成索引…",
    "正在把口令写进 L0 抽屉…",
    "正在把改过的索引还原成 WAV…",
    "正在落成 PCM16、重读、再编码…",
    "正在去交织、纠错、重新取口令…",
  ];
  let index = 0;
  $("#loadingTitle").textContent = titles[0];
  state.loadingTimer = setInterval(() => {
    if (index < steps.length - 1) {
      steps[index].classList.remove("active");
      steps[index].classList.add("done");
      index += 1;
      steps[index].classList.add("active");
      $("#loadingTitle").textContent = titles[index];
    }
  }, 1000);
}

function stopLoadingAnimation() {
  if (state.loadingTimer) {
    clearInterval(state.loadingTimer);
    state.loadingTimer = null;
  }
  $$("#loadingSteps .load-step").forEach((step) => {
    step.classList.remove("active");
    step.classList.add("done");
  });
}

async function responseError(response) {
  try {
    const data = await response.json();
    if (typeof data.detail === "string") return data.detail;
    if (data.detail) return JSON.stringify(data.detail);
  } catch (error) {
    /* ignore */
  }
  return `请求失败（HTTP ${response.status}）`;
}

async function startEmbed() {
  clearLabError();
  if (!state.file) {
    showLabError("先选一段 WAV，或者点“用示例音频”。");
    return;
  }
  const secret = $("#secretInput").value.trim();
  const key = $("#keyInput").value.trim();
  const mode = $('input[name="mode"]:checked').value;
  const payload = selectedPayload();
  const digits = payload / 4;
  if (!new RegExp(`^[0-9A-F]{${digits}}$`).test(secret)) {
    showLabError(`口令必须正好是 ${digits} 位十六进制（${payload} bit），例如 ${payload === 16 ? "A7C3" : "C0"}。`);
    return;
  }
  if (key.length < 4) {
    showLabError("演示密钥至少 4 个字符。");
    return;
  }

  $("#resultPanel").hidden = true;
  $("#loadingCard").hidden = false;
  $("#embedBtn").disabled = true;
  setSteps(3, 2);
  startLoadingAnimation();
  $("#loadingCard").scrollIntoView({ behavior: "smooth", block: "center" });

  const form = new FormData();
  form.append("audio", state.file, state.file.name);
  form.append("mode", mode);
  form.append("secret", secret);
  form.append("key", key);
  form.append("bandwidth", selectedBandwidth());
  form.append("payload", String(payload));
  // nonce 只影响“换一种合法置换再试”的抽取顺序，接收端不需要它。
  form.append("nonce", Math.random().toString(36).slice(2, 12));

  try {
    const response = await fetch("/api/embed", { method: "POST", body: form });
    if (!response.ok) throw new Error(await responseError(response));
    const result = await response.json();
    stopLoadingAnimation();
    $("#loadingCard").hidden = true;
    state.result = result;
    renderResult(result, key);
    setSteps(5, 5);
  } catch (error) {
    stopLoadingAnimation();
    $("#loadingCard").hidden = true;
    $("#embedBtn").disabled = false;
    setSteps(2, 1);
    showLabError(error.message);
  }
}

$("#embedBtn").addEventListener("click", startEmbed);

function renderBits(container, bits, compareBits = null) {
  container.innerHTML = "";
  bits.forEach((bit, index) => {
    const span = document.createElement("span");
    span.className = "bit";
    if (bit === 1) span.classList.add("one");
    if (compareBits && compareBits[index] !== bit) span.classList.add("mismatch");
    span.textContent = bit;
    span.title = `第 ${index + 1} 位：${bit}`;
    container.appendChild(span);
  });
}

function metric(label, value, sub, tone = "") {
  return `<div class="metric ${tone}"><div class="label">${label}</div><div class="value">${value}</div><div class="sub">${sub}</div></div>`;
}

function pct(value) {
  return `${(value * 100).toFixed(1)}%`;
}

/* 把「codec 自身失真」和「藏法引入的失真」并排摆出来，避免把 codec 的锅算到藏法头上。 */
function renderPercept(result) {
  const box = $("#perceptBox");
  if (!box) return;
  const emb = result.perceptual;
  const floor = result.perceptual_source;
  if (!emb || !floor) {
    box.innerHTML = "";
    return;
  }
  const rows = [
    ["对数谱距离 LSD（dB，越小越好）", floor.lsd_db.toFixed(2), emb.lsd_db.toFixed(2)],
    ["分段 SNR（dB，越大越好）", floor.seg_snr_mean_db.toFixed(1), emb.seg_snr_mean_db.toFixed(1)],
    ["掩码超阈比例（粗估，越小越好）", pct(floor.audible_fraction), pct(emb.audible_fraction)],
  ];
  const gap = emb.seg_snr_mean_db - floor.seg_snr_mean_db;
  const verdict = gap >= 0
    ? `嵌入带来的噪声比 codec 自身的噪声低 <b>${gap.toFixed(1)} dB</b>：先听 ①→②，再听 ②→③，若第二段几乎听不出差别，说明嵌入被 codec 的失真盖住了。`
    : `嵌入带来的噪声比 codec 自身的噪声<b>高 ${Math.abs(gap).toFixed(1)} dB</b>：②→③ 的差异会比 ①→② 更刺耳，这一段能听出来主要是藏法造成的，不是 codec。`;
  box.innerHTML = `
    <div class="percept-head"><span>客观量</span><span>① 原始 → ② 重建（codec）</span><span>② 重建 → ③ 藏字（嵌入）</span></div>
    ${rows.map((r) => `<div class="percept-row"><span>${r[0]}</span><b>${r[1]}</b><b>${r[2]}</b></div>`).join("")}
    <div class="percept-note">${verdict}<br>
      配对方式：<b>${result.pairing === "nearest" ? "最近邻码字" : "随机码字"}</b>。
      STOI 只测可懂度，对“咔哒、金属感、发闷”这类失真几乎不敏感，所以不要用 STOI 证明“听不出来”。</div>`;
}

function renderResult(result, key) {
  const exact = result.exact;
  const fellBack = Boolean(result.fallback_from);
  const banner = $("#resultBanner");
  banner.classList.toggle("failed", !exact);
  $("#failureActions").hidden = exact && !fellBack;
  $("#switchDirect").hidden = fellBack;
  $("#resultIcon").textContent = exact ? "✓" : "!";
  $("#resultTitle").textContent = exact
    ? (fellBack ? "你选的那条没读回，已换另一条交付。" : "口令完整取回来了。")
    : "这次没有完整取回。";
  $("#resultSubtitle").textContent = exact
    ? (fellBack
        ? result.fallback_note
        : "先保存成 PCM16 WAV，再重新编码、去交织、纠错；这一轮真的走完了。")
    : "流程跑完了，但纠错后仍有比特没对上。这种音频本身就不适合这种藏法，不是你操作错了。";
  $("#resultCode").textContent = result.recovered_hex;

  const totalBits = result.message_bits.length;
  const mismatchCount = result.message_bits.reduce((sum, bit, index) => {
    const recoveredBit = (parseInt(result.recovered_hex, 16) >> (totalBits - 1 - index)) & 1;
    return sum + (bit === recoveredBit ? 0 : 1);
  }, 0);
  const recoveredBits = Array.from({ length: totalBits }, (_, index) => (parseInt(result.recovered_hex, 16) >> (totalBits - 1 - index)) & 1);

  const metrics = [
    metric(
      "纠错后结果",
      exact ? "完全恢复" : "部分错误",
      exact ? `${totalBits} / ${totalBits} 比特一致` : `${totalBits - mismatchCount} / ${totalBits} 比特一致`,
      exact ? "good" : "bad",
    ),
    metric(
      "发送前自检",
      `第 ${result.attempts} 次通过`,
      result.mode === "hpp"
        ? `换座位有失败率；发送端最多试 ${result.max_attempts} 种合法置换，确认能读回才交付`
        : "直接替换不重试，一次成型",
      result.attempts > 1 ? "warn" : "good",
    ),
    metric(
      "重编码后原始比特存活率",
      pct(result.raw_bit_accuracy),
      "纠错前，索引还能保住的原始比特",
      result.raw_bit_accuracy > 0.82 ? "good" : "warn",
    ),
    metric(
      "拿错钥匙会读到",
      result.wrong_key.hex,
      `和真口令的比特重合 ${pct(result.wrong_key.bit_accuracy)}`,
      "warn",
    ),
    metric(
      "STOI 语音可懂度",
      Number.isFinite(result.stoi) ? result.stoi.toFixed(3) : "—",
      "只测可懂度，对咔哒/金属感不敏感；不能当“听不出来”的证据",
      Number.isFinite(result.stoi) && result.stoi > 0.98 ? "good" : "warn",
    ),
    metric(
      "分段 SNR（嵌入 vs 重建）",
      result.perceptual ? `${result.perceptual.seg_snr_mean_db.toFixed(1)} dB` : "—",
      result.perceptual_source
        ? `codec 地板 ${result.perceptual_source.seg_snr_mean_db.toFixed(1)} dB；要高于地板才可能被掩蔽`
        : "缺少基线",
      result.perceptual && result.perceptual_source
        && result.perceptual.seg_snr_mean_db > result.perceptual_source.seg_snr_mean_db
        ? "good" : "bad",
    ),
    metric(
      "发送端直方图 L1",
      result.sender_hist_l1.toFixed(3),
      result.mode === "hpp" ? "HPP：严格保持，应为 0" : "直接替换：>0 就有分布痕迹",
      result.mode === "hpp" || result.sender_hist_l1 < 0.08 ? "good" : "warn",
    ),
    metric(
      "接收端再编码后 L1",
      result.receiver_hist_l1.toFixed(3),
      "重编码会重新分配索引；这个值才更接近真实链路",
      result.receiver_hist_l1 < 0.2 ? "good" : "warn",
    ),
    metric(
      "被改动的载波",
      `${result.changed_tokens} / ${result.carrier_positions}`,
      `改动比例 ${pct(result.changed_ratio)} · 共 ${result.frame_count} 帧`,
      "",
    ),
    metric(
      "交付音质档位",
      `${result.bandwidth_kbps} kbps · ${result.quantizers} 层 RVQ`,
      "只影响重建有多像原声；第 0 层码字与层数无关，换档不影响提取",
      "",
    ),
    metric(
      "载荷速率",
      `${result.payload_bps.toFixed(2)} bit/s`,
      `${result.payload_bits || totalBits} bit / ${result.duration_seconds.toFixed(2)} 秒`,
      "",
    ),
  ];
  $("#metricGrid").innerHTML = metrics.join("");

  $("#sentHex").textContent = result.message_hex;
  $("#recoveredHex").textContent = result.recovered_hex;
  renderBits($("#sentBits"), result.message_bits);
  renderBits($("#recoveredBits"), recoveredBits);
  const asciiText = result.ascii_hint ? `可读字符：${result.ascii_hint}` : "可读字符：—";
  $("#asciiHint").textContent = asciiText;
  $("#bitExplain").textContent = exact
    ? "16 个比特全部对上。注意：这只说明“这条链路能取回”，不说明“别人检测不到”。"
    : `有 ${mismatchCount} 个比特没回来。可能是音频太短、索引分布太偏，或者重编码恰好把这几个位置擦掉了。换一条音频或改用直接替换再试。`;

  $("#frameChip").textContent = `${result.frame_count} 帧 · L0`;
  $("#changedChip").textContent = `${result.changed_tokens} / ${result.carrier_positions} 格被换`;

  $("#tokenGrid").innerHTML = result.token_preview
    .map(
      (item) => `<div class="token-cell ${item.changed ? "changed" : ""}">
        <span class="t-frame">帧 ${item.frame}<b class="t-bit">${item.bit}</b></span>
        <span class="t-old">${item.cover} →</span> <span class="t-new">${item.stego}</span>
      </div>`,
    )
    .join("");

  $("#coverAudio").src = result.files.cover;
  $("#stegoAudio").src = result.files.stego;
  $("#sourceAudio").src = result.files.source;
  renderPercept(result);
  $("#waveformImage").src = result.files.waveform;
  $("#histogramImage").src = result.files.histogram;
  $("#divergenceImage").src = result.files.divergence;
  $("#downloadStego").href = result.files.stego;
  $("#downloadStego").setAttribute("download", `stego_${result.message_hex}.wav`);
  $("#downloadCover").href = result.files.cover;
  $("#downloadCover").setAttribute("download", "cover.wav");

  const security = $("#securityNote");
  if (result.mode === "hpp") {
    security.innerHTML = `<b>HPP 做到的是：</b>发送端索引多重集严格不变，所以“数格子”的直方图攻击会掉到接近随机。
      但本次接收端 L1 仍有 ${result.receiver_hist_l1.toFixed(3)}；历史留出集上仍有直方图 AUC 0.598、LFCC 0.574、Nes2Net 偏移 0.645。
      这次只是单文件演示，不能拿来证明安全。`;
  } else {
    security.innerHTML = `<b>直接替换的软肋：</b>某些索引会突然出现得更多。历史留出集上索引直方图 AUC 0.921、LFCC 0.708。
      本次发送端 L1 为 ${result.sender_hist_l1.toFixed(3)}。它能“取回”，但不是“隐身”。`;
  }

  $("#resultPanel").hidden = false;
  $("#embedBtn").disabled = false;
  $("#resultPanel").scrollIntoView({ behavior: "smooth", block: "start" });
  $("#goExtract").dataset.session = result.session;
}

$("#againBtn").addEventListener("click", () => {
  $("#resultPanel").hidden = true;
  setSteps(2, 1);
  $("#inputCard").scrollIntoView({ behavior: "smooth", block: "start" });
});

/* 失败时的两条出路：换一组随机再试，或退回一定读得回的直接替换。 */
$("#retryEmbed").addEventListener("click", () => {
  $("#resultPanel").hidden = true;
  startEmbed();
});

$("#switchDirect").addEventListener("click", () => {
  const radio = document.querySelector('input[name="mode"][value="direct"]');
  if (radio) {
    radio.checked = true;
    radio.dispatchEvent(new Event("change", { bubbles: true }));
  }
  $("#resultPanel").hidden = true;
  startEmbed();
});

$("#goExtract").addEventListener("click", async () => {
  if (!state.result) return;
  setView("extract");
  $("#extractMode").value = state.result.mode;
  $("#extractKey").value = $("#keyInput").value;
  $("#expectedSecret").value = state.result.message_hex;
  try {
    const blob = await fetch(state.result.files.stego).then((response) => response.blob());
    const file = new File([blob], `stego_${state.result.message_hex}.wav`, { type: "audio/wav" });
    const transfer = new DataTransfer();
    transfer.items.add(file);
    $("#extractFile").files = transfer.files;
    $("#extractExplain").textContent = "已自动把刚生成的藏字 WAV 放到这里。直接点“从这段音频里读”即可。";
  } catch (error) {
    /* 用户仍可手动选择下载后的文件 */
  }
});

/* ---------- 只做提取 ---------- */
$("#extractBtn").addEventListener("click", async () => {
  const errorBox = $("#extractError");
  errorBox.hidden = true;
  $("#extractResult").hidden = true;
  const file = $("#extractFile").files?.[0];
  if (!file) {
    errorBox.textContent = "先选一个藏字 WAV。";
    errorBox.hidden = false;
    return;
  }
  const mode = $("#extractMode").value;
  const key = $("#extractKey").value.trim();
  const expected = $("#expectedSecret").value.trim();
  if (key.length < 4) {
    errorBox.textContent = "演示密钥至少 4 个字符。";
    errorBox.hidden = false;
    return;
  }
  const button = $("#extractBtn");
  const original = button.textContent;
  button.disabled = true;
  button.textContent = "正在再编码、取索引、纠错…";
  const form = new FormData();
  form.append("audio", file, file.name);
  form.append("mode", mode);
  form.append("key", key);
  form.append("expected", expected);
  form.append("payload", String($("#extractPayload").value || 8));
  try {
    const response = await fetch("/api/extract", { method: "POST", body: form });
    if (!response.ok) throw new Error(await responseError(response));
    const result = await response.json();
    $("#extractCode").textContent = result.message_hex;
    const extractEyebrow = $("#extractEyebrow");
    if (extractEyebrow) extractEyebrow.textContent = `读到的 ${result.payload_bits || result.message_bits.length} bit`;
    renderBits($("#extractBits"), result.message_bits);
    const ascii = result.ascii_hint ? `可读字符：${result.ascii_hint}` : "可读字符：—";
    $("#extractExplain").textContent = `读到 ${result.positions} 个载波位置，模式为 ${result.mode_label}，纠错方案 ${result.scheme}。${ascii}`;
    const box = $("#expectedBox");
    const checkedBits = Number(result.payload_bits || result.message_bits.length);
    if (result.expected) {
      box.hidden = false;
      box.classList.toggle("bad", !result.expected.exact);
      box.textContent = result.expected.exact
        ? `和期望口令 ${result.expected.expected_hex} 完全一致：${checkedBits} / ${checkedBits} 比特。`
        : `和期望口令 ${result.expected.expected_hex} 不一致；比特重合 ${(result.expected.bit_accuracy * 100).toFixed(1)}%。先检查模式、密钥和音频有没有被裁剪/转码。`;
    } else {
      box.hidden = false;
      box.classList.remove("bad");
      box.textContent = "没有填写期望口令，所以只能显示读到的内容，无法自动判断对错。";
    }
    $("#extractResult").hidden = false;
    $("#extractResult").scrollIntoView({ behavior: "smooth", block: "center" });
  } catch (error) {
    errorBox.textContent = error.message;
    errorBox.hidden = false;
  } finally {
    button.disabled = false;
    button.textContent = original;
  }
});

if (window.location.hash === "#learn") setView("learn");
if (window.location.hash === "#extract") setView("extract");
