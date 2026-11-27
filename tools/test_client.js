const fs = require("fs");
const path = require("path");

const base = "http://127.0.0.1:8765";
const demoPath = path.resolve(__dirname, "..", "assets", "demo.wav");

async function main() {
  const page = await fetch(base + "/");
  const html = await page.text();
  console.log("home", page.status, html.includes("语音隐写演示台"));
  const css = await fetch(base + "/static/style.css");
  console.log("css", css.status, (await css.text()).length);
  const js = await fetch(base + "/static/app.js");
  console.log("js", js.status, (await js.text()).length);

  const bytes = fs.readFileSync(demoPath);
  for (const mode of ["direct", "hpp"]) {
    const form = new FormData();
    form.append("audio", new Blob([bytes], { type: "audio/wav" }), "demo.wav");
    form.append("mode", mode);
    form.append("secret", "C0DE");
    form.append("key", "demo-2026");
    const started = Date.now();
    const response = await fetch(base + "/api/embed", { method: "POST", body: form });
    const text = await response.text();
    if (!response.ok) throw new Error(`${mode} embed ${response.status} ${text}`);
    const data = JSON.parse(text);
    console.log("embed", mode, response.status, {
      exact: data.exact,
      recovered: data.recovered_hex,
      raw: Number(data.raw_bit_accuracy.toFixed(3)),
      stoi: Number(data.stoi.toFixed(3)),
      senderL1: Number(data.sender_hist_l1.toFixed(3)),
      receiverL1: Number(data.receiver_hist_l1.toFixed(3)),
      changed: data.changed_tokens,
      carriers: data.carrier_positions,
      ms: Date.now() - started,
    });
    for (const [name, url] of Object.entries(data.files)) {
      const asset = await fetch(base + url);
      if (!asset.ok) throw new Error(`${mode} file ${name} ${asset.status}`);
    }
    const stego = await fetch(base + data.files.stego);
    const stegoBytes = Buffer.from(await stego.arrayBuffer());
    const extractForm = new FormData();
    extractForm.append("audio", new Blob([stegoBytes], { type: "audio/wav" }), "stego.wav");
    extractForm.append("mode", mode);
    extractForm.append("key", "demo-2026");
    extractForm.append("expected", "C0DE");
    const extractResponse = await fetch(base + "/api/extract", { method: "POST", body: extractForm });
    const extractText = await extractResponse.text();
    if (!extractResponse.ok) throw new Error(`${mode} extract ${extractResponse.status} ${extractText}`);
    const extractData = JSON.parse(extractText);
    console.log("extract", mode, {
      hex: extractData.message_hex,
      expectedExact: extractData.expected?.exact,
      positions: extractData.positions,
    });
  }
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
