import {
  AutoModelForCausalLM,
  AutoTokenizer,
  InterruptableStoppingCriteria,
  TextStreamer,
  env,
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0";

const PROMPT = "### Draft:\n{ai}\n\n### Revised:\n";
const STOPS = ["### Draft:", "\n\n###"];
const MAX_NEW_TOKENS = 256;

const root = self.location.pathname.replace(/worker\.js$/, "");
env.allowLocalModels = true;
env.allowRemoteModels = false;
env.useBrowserCache = false;
env.localModelPath = root;
env.backends.onnx.wasm.simd = true;
env.backends.onnx.wasm.numThreads = self.navigator?.hardwareConcurrency || 4;

let tokenizer = null;
let model = null;
let job = null;

self.onunhandledrejection = (event) => {
  post({ type: "error", text: String(event.reason?.message || event.reason) });
};

self.onmessage = async (event) => {
  const msg = event.data;
  if (msg.type === "load") {
    await load();
    return;
  }
  if (msg.type === "generate") {
    await startGenerate(msg.text, msg.temperature);
    return;
  }
  if (msg.type === "pause") {
    pause();
    return;
  }
  if (msg.type === "resume") {
    await resume(msg.temperature);
    return;
  }
  if (msg.type === "clear") {
    clearJob();
    return;
  }
  throw new Error(`unknown worker message ${msg.type}`);
};

async function load() {
  post({ type: "progress", text: "Loading the 0.6B model…" });
  await requireWebGpuF16();
  tokenizer = await AutoTokenizer.from_pretrained("v4");
  model = await AutoModelForCausalLM.from_pretrained("v4", {
    device: "webgpu",
    dtype: "q4f16",
    progress_callback: onProgress,
  });
  post({ type: "ready", text: "Ready. Qwen3-0.6B on GPU (q4f16)." });
}

async function startGenerate(text, temperature) {
  if (!tokenizer || !model) {
    throw new Error("model is not loaded");
  }
  if (job?.interrupt) {
    job.interrupt.interrupt();
  }
  job = {
    prompt: PROMPT.replace("{ai}", text),
    full: "",
    emitted: 0,
    tokens: 0,
    remaining: MAX_NEW_TOKENS,
    temperature: readTemperature(temperature),
    started: performance.now(),
    interrupt: null,
  };
  await runGenerate();
}

function pause() {
  if (!job?.interrupt) {
    return;
  }
  job.interrupt.interrupt();
}

async function resume(temperature) {
  if (!job) {
    throw new Error("nothing to resume");
  }
  if (job.interrupt) {
    throw new Error("generation is already running");
  }
  job.temperature = readTemperature(temperature);
  job.tokens = 0;
  job.started = performance.now();
  await runGenerate();
}

function clearJob() {
  if (job?.interrupt) {
    job.interrupt.interrupt();
  }
  job = null;
  post({ type: "cleared" });
}

async function runGenerate() {
  const current = job;
  if (!current) {
    throw new Error("no generation job");
  }
  if (current.remaining < 1) {
    finish(current, false);
    return;
  }
  current.interrupt = new InterruptableStoppingCriteria();
  const inputs = tokenizer(current.prompt + current.full);
  const streamer = new TextStreamer(tokenizer, {
    skip_prompt: true,
    callback_function: (piece) => {
      if (current !== job) {
        return;
      }
      current.tokens += 1;
      current.full += piece;
      const visible = firstStop(current.full);
      if (visible.length < current.full.length) {
        current.full = visible;
        current.interrupt.interrupt();
      }
      if (visible.length > current.emitted) {
        post({ type: "token", text: visible.slice(current.emitted) });
        current.emitted = visible.length;
      }
    },
  });
  const generated = await model.generate({
    ...inputs,
    max_new_tokens: current.remaining,
    do_sample: current.temperature > 0,
    temperature: current.temperature > 0 ? current.temperature : 1,
    streamer,
    stopping_criteria: current.interrupt,
  });
  if (current !== job) {
    return;
  }
  const used = generated.dims.at(-1) - inputs.input_ids.dims.at(-1);
  if (used < 0) {
    throw new Error(`generate shrank the sequence by ${-used} tokens`);
  }
  current.remaining -= used;
  const interrupted = current.interrupt.interrupted;
  current.interrupt = null;
  finish(current, interrupted && current.remaining > 0);
}

function finish(current, paused) {
  const seconds = (performance.now() - current.started) / 1000;
  if (paused) {
    post({ type: "paused", tokens: current.tokens, seconds });
    return;
  }
  job = null;
  post({ type: "done", tokens: current.tokens, seconds });
}

function readTemperature(value) {
  if (typeof value !== "number" || !Number.isFinite(value) || value < 0) {
    throw new Error(`temperature must be a finite number >= 0, got ${value}`);
  }
  return value;
}

function firstStop(text) {
  let end = text.length;
  for (const stop of STOPS) {
    const at = text.indexOf(stop);
    if (at !== -1 && at < end) {
      end = at;
    }
  }
  return text.slice(0, end);
}

async function requireWebGpuF16() {
  if (!self.navigator?.gpu) {
    throw new Error("This demo needs WebGPU. Use Chrome or Firefox on a recent GPU.");
  }
  const adapter = await self.navigator.gpu.requestAdapter();
  if (!adapter?.features.has("shader-f16")) {
    throw new Error("This demo needs WebGPU shader-f16. Chrome or Firefox on an M-series Mac works.");
  }
}

function onProgress(info) {
  if (!info || info.status !== "progress") {
    return;
  }
  const name = info.file ? info.file.split("/").pop() : "model";
  const pct = info.progress == null ? "" : ` ${Math.round(info.progress)}%`;
  post({ type: "progress", text: `Loading ${name}${pct}` });
}

function post(payload) {
  self.postMessage(payload);
}
