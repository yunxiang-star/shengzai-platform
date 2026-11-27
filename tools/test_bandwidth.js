// 端到端验证：三档交付音质 + 跨档提取。
// 用法：node tools/test_bandwidth.js
const fs = require("fs");
const path = require("path");

const BASE = "http://127.0.0.1:8765";
const wav = fs.readFileSync(path.join(__dirname, "..", "assets", "demo.wav"));
const SECRET = "C0DE";

async function embed(mode, bandwidth) {
  const form = new FormData();
  form.append("audio", new Blob([wav], { type: "audio/wav" }), "demo.wav");
  form.append("mode", mode);
  form.append("secret", SECRET);
  form.append("key", "demo-2026");
  form.append("bandwidth", String(bandwidth));
  form.append("nonce", Math.random().toString(36).slice(2, 10));
  const response = await fetch(`${BASE}/api/embed`, { method: "POST", body: form });
  const body = await response.json();
  if (!response.ok) throw new Error(`${mode}/${bandwidth}: ${JSON.stringify(body).slice(0, 200)}`);
  return body;
}

async function extractFromUrl(url, mode, bandwidth) {
  const audio = Buffer.from(await (await fetch(BASE + url)).arrayBuffer());
  const form = new FormData();
  form.append("audio", new Blob([audio], { type: "audio/wav" }), "stego.wav");
  form.append("mode", mode);
  form.append("key", "demo-2026");
  form.append("expected", SECRET);
  if (bandwidth) form.append("bandwidth", String(bandwidth));
  const response = await fetch(`${BASE}/api/extract`, { method: "POST", body: form });
  const body = await response.json();
  if (!response.ok) throw new Error(`extract ${mode}: ${JSON.stringify(body).slice(0, 200)}`);
  return body;
}

(async () => {
  let failures = 0;
  for (const bandwidth of [6, 12, 24]) {
    for (const mode of ["direct", "hpp"]) {
      const result = await embed(mode, bandwidth);
      const stegoUrl = result.files.stego;
      const sameBand = await extractFromUrl(stegoUrl, mode, bandwidth);
      const crossBand = await extractFromUrl(stegoUrl, mode, 24);
      const okSame = sameBand.message_hex === SECRET;
      const okCross = crossBand.message_hex === SECRET;
      if (!okSame || !okCross) failures += 1;
      console.log(
        `${String(bandwidth).padStart(2)} kbps ${mode.padEnd(6)} ` +
          `Q=${String(result.quantizers).padStart(2)} embed=${result.exact ? "ok" : "FAIL"} ` +
          `segSNR=${result.perceptual.seg_snr_mean_db.toFixed(1)} ` +
          `floor=${result.perceptual_source.seg_snr_mean_db.toFixed(1)} ` +
          `stoi=${result.stoi.toFixed(3)} ` +
          `extract@${bandwidth}=${sameBand.message_hex}${okSame ? "" : "(FAIL)"} ` +
          `extract@24=${crossBand.message_hex}${okCross ? "" : "(FAIL)"} ` +
          `fallback=${result.fallback_from || "-"}`
      );
    }
  }
  console.log(failures === 0 ? "\nALL OK" : `\n${failures} FAILURES`);
  process.exit(failures === 0 ? 0 : 1);
})();
