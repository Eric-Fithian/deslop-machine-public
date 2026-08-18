const draft = document.getElementById("draft");
const revised = document.getElementById("revised");
const run = document.getElementById("run");
const pauseBtn = document.getElementById("pause");
const clearBtn = document.getElementById("clear");
const status = document.getElementById("status");
const temp = document.getElementById("temp");
const tempVal = document.getElementById("temp-val");

const PLACEHOLDER = "Paste an AI paragraph.";
let acceptTokens = false;
let paused = false;

showPlaceholder();

const worker = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });

worker.onmessage = (event) => {
  const msg = event.data;
  if (msg.type === "progress") {
    status.textContent = msg.text;
    return;
  }
  if (msg.type === "ready") {
    status.textContent = msg.text;
    setIdle();
    return;
  }
  if (msg.type === "token") {
    if (!acceptTokens) {
      return;
    }
    if (revised.classList.contains("empty")) {
      revised.textContent = "";
      revised.classList.remove("empty");
    }
    revised.textContent += msg.text;
    return;
  }
  if (msg.type === "paused") {
    setPaused();
    status.textContent = "Paused";
    return;
  }
  if (msg.type === "cleared") {
    showPlaceholder();
    setIdle();
    return;
  }
  if (msg.type === "done") {
    setIdle();
    const tps = msg.seconds > 0 ? (msg.tokens / msg.seconds) : 0;
    status.textContent = `${tps.toFixed(1)} tok/s`;
    return;
  }
  if (msg.type === "error") {
    setIdle();
    status.textContent = msg.text;
  }
};

worker.onerror = (event) => {
  setIdle();
  status.textContent = event.message || "The model worker failed.";
};

temp.addEventListener("input", () => {
  tempVal.textContent = Number(temp.value).toFixed(1);
});
run.addEventListener("click", rewrite);
pauseBtn.addEventListener("click", togglePause);
clearBtn.addEventListener("click", clearOutput);
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
  setGenerating();
  status.textContent = "Rewriting…";
  worker.postMessage({ type: "generate", text, temperature: readTemperature() });
}

function togglePause() {
  if (pauseBtn.disabled) {
    return;
  }
  if (paused) {
    setGenerating();
    status.textContent = "Rewriting…";
    worker.postMessage({ type: "resume", temperature: readTemperature() });
    return;
  }
  worker.postMessage({ type: "pause" });
}

function clearOutput() {
  if (clearBtn.disabled) {
    return;
  }
  acceptTokens = false;
  worker.postMessage({ type: "clear" });
  showPlaceholder();
  setIdle();
}

function setIdle() {
  acceptTokens = false;
  paused = false;
  run.disabled = false;
  pauseBtn.disabled = true;
  pauseBtn.textContent = "Pause";
  clearBtn.disabled = revised.classList.contains("empty");
  revised.classList.remove("generating");
}

function setGenerating() {
  acceptTokens = true;
  paused = false;
  run.disabled = true;
  pauseBtn.disabled = false;
  pauseBtn.textContent = "Pause";
  clearBtn.disabled = false;
  revised.classList.add("generating");
}

function setPaused() {
  paused = true;
  run.disabled = false;
  pauseBtn.disabled = false;
  pauseBtn.textContent = "Resume";
  clearBtn.disabled = revised.classList.contains("empty");
  revised.classList.remove("generating");
}

function showPlaceholder() {
  revised.textContent = PLACEHOLDER;
  revised.classList.add("empty");
  revised.classList.remove("generating");
}

function readTemperature() {
  const value = Number(temp.value);
  if (!Number.isFinite(value) || value < 0) {
    throw new Error(`temperature must be a finite number >= 0, got ${temp.value}`);
  }
  return value;
}

worker.postMessage({ type: "load" });
