import { AutoModelForCausalLM, AutoTokenizer, env } from "@huggingface/transformers";
import { fileURLToPath } from "node:url";

const root = fileURLToPath(new URL("../", import.meta.url));
env.allowLocalModels = true;
env.allowRemoteModels = false;
env.localModelPath = root;

const PROMPT = "### Draft:\nRecent advances in machine learning have led to significant improvements.\n\n### Revised:\n";

async function main() {
  const tokenizer = await AutoTokenizer.from_pretrained("v4");
  const model = await AutoModelForCausalLM.from_pretrained("v4", { dtype: "q4" });
  const inputs = tokenizer(PROMPT);
  const ids = await model.generate({ ...inputs, max_new_tokens: 24, do_sample: false });
  const text = tokenizer.decode(ids[0], { skip_special_tokens: true });
  const revised = text.split("### Revised:\n")[1] ?? "";
  console.log(revised);
  if (!revised.trim()) {
    throw new Error("model produced an empty revision");
  }
}

await main();
