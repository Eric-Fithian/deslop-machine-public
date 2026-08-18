import { AutoModelForCausalLM, AutoTokenizer, env } from "@huggingface/transformers";

env.allowLocalModels = false;
env.allowRemoteModels = true;
env.remoteHost = "https://eric-fithian.github.io/deslop-machine-public/";
env.remotePathTemplate = "{model}/";

const PROMPT = "### Draft:\nRecent advances in machine learning have led to significant improvements.\n\n### Revised:\n";

async function main() {
  const tokenizer = await AutoTokenizer.from_pretrained("v2");
  const model = await AutoModelForCausalLM.from_pretrained("v2", { dtype: "q4" });
  const inputs = tokenizer(PROMPT);
  const ids = await model.generate({ ...inputs, max_new_tokens: 24, do_sample: false });
  const text = tokenizer.decode(ids[0], { skip_special_tokens: true });
  const revised = text.split("### Revised:\n")[1] ?? "";
  console.log(revised);
  if (!revised.trim()) {
    throw new Error("hosted model produced an empty revision");
  }
}

await main();
