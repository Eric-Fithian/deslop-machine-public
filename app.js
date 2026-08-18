const draft = document.getElementById("draft");
const revised = document.getElementById("revised");
const run = document.getElementById("run");
const status = document.getElementById("status");

const PLACEHOLDER = "Paste an AI paragraph.";
revised.textContent = PLACEHOLDER;
revised.classList.add("empty");

const worker = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });

worker.onmessage = (event) => {
  const msg = event.data;
  if (msg.type === "progress") {
    status.textContent = msg.text;
    return;
  }
  if (msg.type === "ready") {
    status.textContent = msg.text;
    run.disabled = false;
    return;
  }
  if (msg.type === "token") {
    if (revised.classList.contains("empty")) {
      revised.textContent = "";
      revised.classList.remove("empty");
    }
    revised.textContent += msg.text;
    return;
  }
  if (msg.type === "done") {
    revised.classList.remove("generating");
    run.disabled = false;
    const tps = msg.seconds > 0 ? (msg.tokens / msg.seconds) : 0;
    status.textContent = `${tps.toFixed(1)} tok/s`;
    return;
  }
  if (msg.type === "error") {
    revised.classList.remove("generating");
    run.disabled = false;
    status.textContent = msg.text;
  }
};

worker.onerror = (event) => {
  run.disabled = false;
  revised.classList.remove("generating");
  status.textContent = event.message || "The model worker failed.";
};

run.addEventListener("click", rewrite);
draft.addEventListener("keydown", (event) => {
  if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
    event.preventDefault();
    rewrite();
  }
});

function rewrite() {
  const text = draft.value.trim();
  if (!text || run.disabled) {
    return;
  }
  revised.textContent = "";
  revised.classList.remove("empty");
  revised.classList.add("generating");
  run.disabled = true;
  status.textContent = "Rewriting…";
  worker.postMessage({ type: "generate", text });
}

worker.postMessage({ type: "load" });
