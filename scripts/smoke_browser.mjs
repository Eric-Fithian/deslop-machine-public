import { chromium } from "playwright";

const URL = "https://eric-fithian.github.io/deslop-machine-public/";
const DRAFT = "Recent advances in machine learning have led to significant improvements.";

async function main() {
  const browser = await chromium.launch();
  const page = await browser.newPage();
  page.on("pageerror", (err) => {
    throw err;
  });
  await page.goto(URL, { waitUntil: "domcontentloaded" });
  await page.waitForFunction(() => !document.getElementById("run").disabled, {
    timeout: 180000,
  });
  const ready = await page.textContent("#status");
  if (!ready.includes("Ready")) {
    throw new Error(`expected Ready, got ${ready}`);
  }
  await page.fill("#draft", DRAFT);
  await page.click("#run");
  await page.waitForFunction(() => {
    const text = document.getElementById("revised").textContent;
    return text && text.length > 20;
  }, { timeout: 120000 });
  const revised = await page.textContent("#revised");
  console.log(ready);
  console.log(revised);
  await browser.close();
  if (!revised.trim()) {
    throw new Error("page produced an empty revision");
  }
}

await main();
