import { AutoModelForCausalLM, AutoTokenizer, env } from "@huggingface/transformers";
import { fileURLToPath } from "node:url";

const root = fileURLToPath(new URL("../", import.meta.url));
env.allowLocalModels = true;
env.allowRemoteModels = false;
env.localModelPath = root;

const PROMPT = "### Draft:\nRecent advances in machine learning have led to significant improvements in code generation and mathematical reasoning. In this paper, we propose a novel method that achieves state-of-the-art results.\n\n### Revised:\n";
const NEW_TOKENS = Number(process.argv[2] || 64);

async function main() {
  const tokenizer = await AutoTokenizer.from_pretrained("v4");
  const model = await AutoModelForCausalLM.from_pretrained("v4", { dtype: "q4" });
  const inputs = tokenizer(PROMPT);
  const started = performance.now();
  const ids = await model.generate({
    ...inputs,
    max_new_tokens: NEW_TOKENS,
    do_sample: false,
  });
  const seconds = (performance.now() - started) / 1000;
  const used = ids.dims.at(-1) - inputs.input_ids.dims.at(-1);
  const text = tokenizer.decode(ids[0], { skip_special_tokens: true });
  const revised = text.split("### Revised:\n")[1] ?? "";
  console.log(revised);
  console.log(`node q4: ${used} tokens in ${seconds.toFixed(2)}s = ${(used / seconds).toFixed(1)} tok/s`);
  if (!revised.trim()) {
    throw new Error("bench produced an empty revision");
  }
}

await main();
