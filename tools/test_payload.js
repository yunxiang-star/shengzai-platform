// 端到端回归：两种模式 × 两种口令长度 × 交付档位。
// 用法：node tools/test_payload.js
const fs = require("fs");
const path = require("path");

const BASE = "http://127.0.0.1:8765";
const wav = fs.readFileSync(path.join(__dirname, "..", "assets", "demo.wav"));
const CASES = [
  { mode: "direct", payload: 16, secret: "C0DE" },
  { mode: "direct", payload: 8, secret: "C0" },
  { mode: "hpp", payload: 16, secret: "C0DE" },
  { mode: "hpp", payload: 8, secret: "C0" },
];

async function embed(testCase, bandwidth) {
  const form = new FormData();
  form.append("audio", new Blob([wav], { type: "audio/wav" }), "demo.wav");
  form.append("mode", testCase.mode);
  form.append("secret", testCase.secret);
  form.append("key", "demo-2026");
  form.append("bandwidth", String(bandwidth));
  form.append("payload", String(testCase.payload));
  form.append("nonce", Math.random().toString(36).slice(2, 10));
  const response = await fetch(`${BASE}/api/embed`, { method: "POST", body: form });
  const body = await response.json();
  if (!response.ok) throw new Error(`${testCase.mode}: ${JSON.stringify(body).slice(0, 160)}`);
  return body;
}

async function extract(url, testCase, bandwidth) {
  const audio = Buffer.from(await (await fetch(BASE + url)).arrayBuffer());
  const form = new FormData();
  form.append("audio", new Blob([audio], { type: "audio/wav" }), "stego.wav");
  form.append("mode", testCase.mode);
  form.append("key", "demo-2026");
  form.append("expected", testCase.secret);
  form.append("bandwidth", String(bandwidth));
  form.append("payload", String(testCase.payload));
  const response = await fetch(`${BASE}/api/extract`, { method: "POST", body: form });
  const body = await response.json();
  if (!response.ok) throw new Error(`extract: ${JSON.stringify(body).slice(0, 160)}`);
  return body;
}

(async () => {
  let failures = 0;
  for (const testCase of CASES) {
    const result = await embed(testCase, 24);
    const back = await extract(result.files.stego, testCase, 24);
    const ok = result.exact && back.message_hex === testCase.secret && back.expected?.exact;
    if (!ok) failures += 1;
    console.log(
      `${testCase.mode.padEnd(6)} ${String(testCase.payload).padStart(2)} bit ` +
        `carriers=${String(result.carrier_positions).padStart(2)} ` +
        `changed=${String(result.changed_tokens).padStart(2)} ` +
        `seg=${result.perceptual.seg_snr_mean_db.toFixed(1)} ` +
        `lsd=${result.perceptual.lsd_db.toFixed(2)} ` +
        `embed=${result.exact ? "ok" : "FAIL"} -> ${result.recovered_hex} ` +
        `extract=${back.message_hex}${ok ? "" : " FAIL"}`,
    );
  }
  console.log(failures === 0 ? "\nALL OK" : `\n${failures} FAILURES`);
  process.exit(failures === 0 ? 0 : 1);
})();
