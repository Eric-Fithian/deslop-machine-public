import { createServer } from "node:http";
import { existsSync } from "node:fs";
import { readFile } from "node:fs/promises";
import { extname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { chromium, firefox } from "playwright";

const ROOT = fileURLToPath(new URL("../", import.meta.url));
const DRAFT = "Recent advances in machine learning have led to significant improvements.";
const TYPES = {
  ".css": "text/css",
  ".html": "text/html",
  ".js": "text/javascript",
  ".json": "application/json",
  ".onnx": "application/octet-stream",
  ".onnx_data": "application/octet-stream",
};

async function main() {
  const server = await serve();
  const url = `http://127.0.0.1:${server.address().port}/`;
  const engine = process.argv[2] || "chromium";
  const browserType = engine === "firefox" ? firefox : chromium;
  const browser = await browserType.launch({
    args: ["--enable-unsafe-webgpu", "--enable-webgpu-developer-features"],
  });
  const page = await browser.newPage();
  page.setDefaultTimeout(180000);
  page.on("console", (msg) => {
    console.error(`browser ${msg.type()}: ${msg.text()}`);
  });
  page.on("pageerror", (err) => {
    throw err;
  });
  await page.goto(url, { waitUntil: "domcontentloaded" });
  await page.waitForFunction(() => {
    const el = document.getElementById("status");
    return el && (el.textContent.includes("Ready") || el.textContent.includes("Error") || el.textContent.includes("failed"));
  });
  const ready = await page.textContent("#status");
  if (!ready.includes("Ready")) {
    throw new Error(`expected Ready, got ${ready}`);
  }
  await page.fill("#draft", DRAFT);
  await page.click("#run");
  await page.waitForFunction(() => {
    const text = document.getElementById("status").textContent;
    return text && text.includes("tok/s");
  }, { timeout: 180000 });
  const revised = await page.textContent("#revised");
  const status = await page.textContent("#status");
  console.log(ready);
  console.log(status);
  console.log(revised);
  await browser.close();
  server.close();
  if (!revised.trim() || revised.includes("Paste an AI paragraph")) {
    throw new Error("page produced an empty revision");
  }
}

function serve() {
  return new Promise((resolve) => {
    const server = createServer(async (req, res) => {
      const rel = req.url === "/" ? "index.html" : decodeURIComponent(req.url.split("?")[0].slice(1));
      const file = join(ROOT, rel);
      if (!file.startsWith(ROOT) || !existsSync(file)) {
        res.writeHead(404);
        res.end();
        return;
      }
      const body = await readFile(file);
      res.writeHead(200, {
        "Content-Type": TYPES[extname(file)] || "application/octet-stream",
        "Content-Length": String(body.length),
        "Cross-Origin-Opener-Policy": "same-origin",
        "Cross-Origin-Embedder-Policy": "require-corp",
        "Cross-Origin-Resource-Policy": "same-origin",
      });
      res.end(body);
    });
    server.listen(0, "127.0.0.1", () => resolve(server));
  });
}

await main();
