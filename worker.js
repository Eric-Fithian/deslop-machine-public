import {
  AutoModelForCausalLM,
  AutoTokenizer,
  TextStreamer,
  env,
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0";

const PROMPT = "### Draft:\n{ai}\n\n### Revised:\n";
const STOPS = ["### Draft:", "\n\n###"];
const MAX_NEW_TOKENS = 256;

const root = self.location.pathname.replace(/worker\.js$/, "");
env.allowLocalModels = true;
env.allowRemoteModels = false;
env.useBrowserCache = true;
env.localModelPath = root;
env.backends.onnx.wasm.simd = true;
env.backends.onnx.wasm.numThreads = self.navigator?.hardwareConcurrency || 4;

let tokenizer = null;
let model = null;
let backend = "wasm";
let dtype = "q4";

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
    await generate(msg.text, msg.temperature);
    return;
  }
  throw new Error(`unknown worker message ${msg.type}`);
};

async function load() {
  post({ type: "progress", text: "Loading the 135M model…" });
  backend = await pickDevice();
  dtype = backend === "webgpu" ? "q4f16" : "q4";
  tokenizer = await AutoTokenizer.from_pretrained("v3");
  model = await AutoModelForCausalLM.from_pretrained("v3", {
    device: backend,
    dtype,
    progress_callback: onProgress,
  });
  const label = backend === "webgpu" ? "GPU" : "CPU";
  post({ type: "ready", text: `Ready. SmolLM2-135M on ${label} (${dtype}).` });
}

async function generate(text, temperature) {
  if (!tokenizer || !model) {
    throw new Error("model is not loaded");
  }
  const temp = readTemperature(temperature);
  const prompt = PROMPT.replace("{ai}", text);
  const inputs = tokenizer(prompt);
  let full = "";
  let emitted = 0;
  let tokens = 0;
  const started = performance.now();
  const streamer = new TextStreamer(tokenizer, {
    skip_prompt: true,
    callback_function: (piece) => {
      tokens += 1;
      full += piece;
      const visible = firstStop(full);
      if (visible.length > emitted) {
        post({ type: "token", text: visible.slice(emitted) });
        emitted = visible.length;
      }
    },
  });
  await model.generate({
    ...inputs,
    max_new_tokens: MAX_NEW_TOKENS,
    do_sample: temp > 0,
    temperature: temp > 0 ? temp : 1,
    streamer,
  });
  const seconds = (performance.now() - started) / 1000;
  post({ type: "done", tokens, seconds });
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

async function pickDevice() {
  if (!self.navigator?.gpu) {
    return "wasm";
  }
  const adapter = await self.navigator.gpu.requestAdapter();
  if (adapter?.features.has("shader-f16")) {
    return "webgpu";
  }
  return "wasm";
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
