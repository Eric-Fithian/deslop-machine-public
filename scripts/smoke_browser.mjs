import { chromium } from "playwright";

const URL = "https://eric-fithian.github.io/deslop-machine-public/";
const DRAFT = "Recent advances in machine learning have led to significant improvements.";

async function main() {
  const browser = await chromium.launch({
    channel: "chrome",
    args: ["--enable-unsafe-webgpu", "--enable-webgpu-developer-features"],
  });
  const page = await browser.newPage();
  page.setDefaultTimeout(300000);
  page.setDefaultNavigationTimeout(300000);
  page.on("console", (msg) => {
    console.error(`browser ${msg.type()}: ${msg.text()}`);
  });
  page.on("pageerror", (err) => {
    throw err;
  });
  page.on("requestfailed", (req) => {
    console.error(`failed ${req.failure()?.errorText} ${req.url()}`);
  });
  page.on("response", (res) => {
    if (res.status() >= 400) {
      console.error(`${res.status()} ${res.url()}`);
    }
  });
  await page.goto(URL, { waitUntil: "domcontentloaded" });
  try {
    await page.waitForFunction(() => {
      const el = document.getElementById("status");
      return el && (el.textContent.includes("Ready") || el.textContent.includes("Error") || el.textContent.includes("failed") || el.textContent.includes("needs WebGPU"));
    });
  } catch (err) {
    console.error(`status: ${await page.textContent("#status")}`);
    throw err;
  }
  const ready = await page.textContent("#status");
  if (!ready.includes("Ready")) {
    throw new Error(`expected Ready, got ${ready}`);
  }
  await page.fill("#draft", DRAFT);
  await page.click("#run");
  await page.waitForFunction(() => {
    const text = document.getElementById("status").textContent;
    return text && text.includes("tok/s");
  }, { timeout: 300000 });
  const revised = await page.textContent("#revised");
  const status = await page.textContent("#status");
  console.log(ready);
  console.log(status);
  console.log(revised);
  await browser.close();
  if (!revised.trim()) {
    throw new Error("page produced an empty revision");
  }
}

await main();
