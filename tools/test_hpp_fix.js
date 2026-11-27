// 回归测试：用之前失败过的口令，验证自检重试与降级标注。
const fs = require("fs");
const path = require("path");

const BASE = "http://127.0.0.1:8765";
const APP = path.resolve(__dirname, "..");

async function waitReady() {
  for (let i = 0; i < 120; i += 1) {
    try {
      const r = await fetch(`${BASE}/api/health`);
      if (r.ok) {
        const j = await r.json();
        if (j.model_loaded) return j;
      }
    } catch (e) {}
    await new Promise((r) => setTimeout(r, 1000));
  }
  throw new Error("health timeout");
}

async function embed(secret, mode, nonce) {
  const wav = fs.readFileSync(path.join(APP, "assets", "demo.wav"));
  const form = new FormData();
  form.append("audio", new Blob([wav], { type: "audio/wav" }), "demo.wav");
  form.append("mode", mode);
  form.append("secret", secret);
  form.append("key", "demo-2026");
  form.append("nonce", nonce);
  const r = await fetch(`${BASE}/api/embed`, { method: "POST", body: form });
  if (!r.ok) throw new Error(`embed ${r.status} ${await r.text()}`);
  return r.json();
}

async function extractFromSession(session, mode, expected) {
  const url = `${BASE}/outputs/${session}/stego.wav`;
  const buf = Buffer.from(await (await fetch(url)).arrayBuffer());
  const form = new FormData();
  form.append("audio", new Blob([buf], { type: "audio/wav" }), "stego.wav");
  form.append("mode", mode);
  form.append("key", "demo-2026");
  form.append("expected", expected);
  const r = await fetch(`${BASE}/api/extract`, { method: "POST", body: form });
  if (!r.ok) throw new Error(`extract ${r.status} ${await r.text()}`);
  return r.json();
}

(async () => {
  const health = await waitReady();
  console.log("health", JSON.stringify(health));

  const cases = [
    { secret: "46FF", mode: "hpp" },   // 用户这次失败的口令
    { secret: "1391", mode: "hpp" },   // 已知难解
    { secret: "C4DD", mode: "hpp" },   // 已知难解
    { secret: "C0DE", mode: "hpp" },
    { secret: "C0DE", mode: "direct" },
  ];

  for (const item of cases) {
    const t0 = Date.now();
    const res = await embed(item.secret, item.mode, Math.random().toString(36).slice(2, 12));
    const ms = Date.now() - t0;
    console.log(
      `${item.mode} ${item.secret} -> ${res.recovered_hex} exact=${res.exact} ` +
      `tries=${res.attempts}/${res.max_attempts} fallback=${res.fallback_from} ` +
      `mode=${res.mode} stoi=${res.stoi.toFixed(3)} ${ms}ms`
    );
    if (res.exact) {
      const ex = await extractFromSession(res.session, res.mode, res.message_hex);
      console.log(
        `   re-extract from saved WAV: ${ex.message_hex} matches=${ex.expected.exact} ` +
        `positions=${ex.positions}`
      );
    }
  }
})().catch((e) => {
  console.error("FAILED", e.message);
  process.exit(1);
});
