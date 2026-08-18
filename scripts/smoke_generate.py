"""Generate a few tokens from the fused genai CPU model to prove it runs."""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "_genai_cpu"
PROMPT = "### Draft:\nRecent advances in machine learning have led to significant improvements.\n\n### Revised:\n"


def main():
    if "--check" in sys.argv:
        check()
        return
    import onnxruntime_genai as og

    model = og.Model(str(MODEL))
    tok = og.Tokenizer(model)
    params = og.GeneratorParams(model)
    params.set_search_options(max_length=len(tok.encode(PROMPT)) + 24, do_sample=False)
    generator = og.Generator(model, params)
    generator.append_tokens(tok.encode(PROMPT))
    started = time.perf_counter()
    tokens = 0
    while not generator.is_done():
        generator.generate_next_token()
        tokens += 1
    seconds = time.perf_counter() - started
    text = tok.decode(generator.get_sequence(0))
    revised = text.split("### Revised:\n", 1)[1]
    print(revised)
    print(f"{tokens / seconds:.1f} tok/s ({tokens} tokens in {seconds:.2f}s)", file=sys.stderr)
    if not revised.strip():
        raise SystemExit("model produced an empty revision")


def check():
    assert (MODEL / "model.onnx").is_file()
    print("check ok")


if __name__ == "__main__":
    main()
