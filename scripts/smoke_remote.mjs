const BASE = "https://eric-fithian.github.io/deslop-machine-public/v4";
const FILES = [
  "config.json",
  "tokenizer.json",
  "onnx/model_q4f16.onnx",
  "onnx/model_q4f16.onnx_data",
];

async function main() {
  for (const rel of FILES) {
    const url = `${BASE}/${rel}`;
    const res = await fetch(url, { method: "HEAD" });
    if (!res.ok) {
      throw new Error(`${url} returned ${res.status}`);
    }
    console.log(`${rel} ${res.headers.get("content-length") || "ok"}`);
  }
}

await main();
