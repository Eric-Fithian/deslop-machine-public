"""Generate a few tokens from the exported q4 ONNX to prove it runs."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "model"
PROMPT = "### Draft:\nRecent advances in machine learning have led to significant improvements.\n\n### Revised:\n"


def main():
    if "--check" in sys.argv:
        check()
        return
    from optimum.onnxruntime import ORTModelForCausalLM
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = ORTModelForCausalLM.from_pretrained(
        MODEL, subfolder="onnx", file_name="model_q4.onnx")
    inputs = tok(PROMPT, return_tensors="pt")
    out = model.generate(**inputs, max_new_tokens=24, do_sample=False)
    text = tok.decode(out[0], skip_special_tokens=True)
    revised = text.split("### Revised:\n", 1)[1]
    print(revised)
    if not revised.strip():
        raise SystemExit("model produced an empty revision")


def check():
    assert (MODEL / "onnx" / "model_q4.onnx").is_file()
    print("check ok")


if __name__ == "__main__":
    main()
