import {
  AutoModelForCausalLM,
  AutoTokenizer,
  TextStreamer,
  env,
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1";

const PROMPT = "### Draft:\n{ai}\n\n### Revised:\n";
const STOPS = ["### Draft:", "\n\n###"];
const MAX_NEW_TOKENS = 256;

const root = self.location.pathname.replace(/worker\.js$/, "");
env.allowLocalModels = true;
env.allowRemoteModels = false;
env.useBrowserCache = true;
env.localModelPath = root;

let tokenizer = null;
let model = null;

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
    await generate(msg.text);
    return;
  }
  throw new Error(`unknown worker message ${msg.type}`);
};

async function load() {
  post({ type: "progress", text: "Loading the 135M model…" });
  const device = await pickDevice();
  tokenizer = await AutoTokenizer.from_pretrained("model");
  model = await AutoModelForCausalLM.from_pretrained("model", {
    device,
    dtype: "q4",
    progress_callback: onProgress,
  });
  const label = device === "webgpu" ? "GPU" : "CPU";
  post({ type: "ready", text: `Ready. SmolLM2-135M on ${label}.` });
}

async function generate(text) {
  if (!tokenizer || !model) {
    throw new Error("model is not loaded");
  }
  const prompt = PROMPT.replace("{ai}", text);
  const inputs = tokenizer(prompt);
  let full = "";
  let emitted = 0;
  const streamer = new TextStreamer(tokenizer, {
    skip_prompt: true,
    callback_function: (piece) => {
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
    do_sample: false,
    streamer,
  });
  post({ type: "done" });
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
  if (self.navigator?.gpu && await self.navigator.gpu.requestAdapter()) {
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
