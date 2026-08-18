"""Merge the 135M LoRA and export ONNX files for transformers.js."""
import shutil
import subprocess
import sys
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
ADAPTER = ROOT / "_adapter_v5b_smollm2_135m"
MERGED = ROOT / "_merged"
EXPORT = ROOT / "_export"
MODEL = ROOT / "v2"
BASE = "HuggingFaceTB/SmolLM2-135M"
EXTRAS = [
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
]


def main():
    if "--check" in sys.argv:
        check()
        return
    if "--assemble" in sys.argv:
        assemble()
        return
    if not (ADAPTER / "adapter_model.safetensors").is_file():
        raise SystemExit(f"adapter weights missing: {ADAPTER}")
    merge()
    export()
    assemble()


def merge():
    if (MERGED / "config.json").is_file():
        print(f"reusing {MERGED}", file=sys.stderr)
        return
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    MERGED.mkdir(parents=True, exist_ok=True)
    print(f"loading {BASE}", file=sys.stderr)
    tok = AutoTokenizer.from_pretrained(BASE)
    base = AutoModelForCausalLM.from_pretrained(BASE)
    print(f"merging {ADAPTER.name}", file=sys.stderr)
    model = PeftModel.from_pretrained(base, ADAPTER)
    merged = model.merge_and_unload()
    print(f"saving {MERGED}", file=sys.stderr)
    merged.save_pretrained(MERGED)
    tok.save_pretrained(MERGED)


def export():
    if (EXPORT / "config.json").is_file() and any(EXPORT.glob("*.onnx")):
        print(f"reusing {EXPORT}", file=sys.stderr)
        return
    if EXPORT.exists():
        shutil.rmtree(EXPORT)
    print("exporting ONNX", file=sys.stderr)
    cli = Path(sys.executable).parent / "optimum-cli"
    if not cli.is_file():
        raise SystemExit(f"optimum-cli missing: {cli}")
    subprocess.run(
        [
            str(cli), "export", "onnx",
            "--model", str(MERGED),
            "--task", "text-generation-with-past",
            str(EXPORT),
        ],
        check=True,
    )


def assemble():
    from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer
    from onnxruntime.transformers.fusion_options import AttentionOpType, FusionOptions
    from onnxruntime.transformers.optimizer import optimize_model

    onnx_src = pick_onnx(EXPORT)
    fused = EXPORT / "model_fused.onnx"
    print(f"fusing {onnx_src.name}", file=sys.stderr)
    options = FusionOptions("qwen3")
    options.set_attention_op_type(AttentionOpType.GroupQueryAttention)
    optimized = optimize_model(
        str(onnx_src),
        model_type="qwen3",
        num_heads=9,
        hidden_size=576,
        optimization_options=options,
    )
    optimized.save_model_to_file(str(fused))
    count_ops(fused)
    if MODEL.exists():
        shutil.rmtree(MODEL)
    MODEL.mkdir(parents=True)
    onnx_dir = MODEL / "onnx"
    onnx_dir.mkdir()
    for name in tqdm(EXTRAS, desc="copy tokenizer"):
        src = EXPORT / name
        if src.is_file():
            shutil.copy2(src, MODEL / name)
        else:
            fallback = MERGED / name
            if fallback.is_file():
                shutil.copy2(fallback, MODEL / name)
    q4 = onnx_dir / "model_q4.onnx"
    print(f"quantizing {fused.name} -> {q4.name}", file=sys.stderr)
    quant = MatMulNBitsQuantizer(str(fused), bits=4, block_size=32, is_symmetric=True)
    quant.process()
    quant.model.save_model_to_file(str(q4))
    q4f16 = onnx_dir / "model_q4f16.onnx"
    print(f"quantizing {fused.name} -> {q4f16.name}", file=sys.stderr)
    fp16 = EXPORT / "model_fused_fp16.onnx"
    to_fp16(fused, fp16)
    quant16 = MatMulNBitsQuantizer(str(fp16), bits=4, block_size=32, is_symmetric=True)
    quant16.process()
    quant16.model.save_model_to_file(str(q4f16))
    print(f"-> {MODEL}", file=sys.stderr)
    for path in sorted(MODEL.rglob("*")):
        if path.is_file():
            print(f"  {path.relative_to(MODEL)} {path.stat().st_size}",
                  file=sys.stderr)
    config_path = MODEL / "config.json"
    if not config_path.is_file():
        raise SystemExit("model/config.json missing after assemble")
    if not q4.is_file():
        raise SystemExit(f"{q4} missing after assemble")
    if not q4f16.is_file():
        raise SystemExit(f"{q4f16} missing after assemble")
    patch_js_config(config_path)


def to_fp16(src, dest):
    import onnx
    from onnxruntime.transformers.float16 import convert_float_to_float16

    converted = convert_float_to_float16(str(src), keep_io_types=False)
    onnx.save(converted, str(dest))


def count_ops(path):
    import onnx

    model = onnx.load(str(path), load_external_data=True)
    ops = {}
    for node in model.graph.node:
        ops[node.op_type] = ops.get(node.op_type, 0) + 1
    print("fused ops:", file=sys.stderr)
    for name, n in sorted(ops.items(), key=lambda kv: -kv[1])[:20]:
        print(f"  {name} {n}", file=sys.stderr)
    fused = sum(ops.get(name, 0) for name in (
        "GroupQueryAttention", "MultiHeadAttention", "Attention"))
    print(f"  attention fused: {fused}", file=sys.stderr)


def patch_js_config(path):
    import json

    cfg = json.loads(path.read_text())
    cfg["transformers.js_config"] = {
        "dtype": "q4",
        "kv_cache_dtype": {"q4f16": "float16", "fp16": "float16"},
    }
    path.write_text(json.dumps(cfg, indent=2) + "\n")


def pick_onnx(folder):
    candidates = [
        folder / "model.onnx",
        folder / "decoder_model_merged.onnx",
        folder / "decoder_model.onnx",
    ]
    for path in candidates:
        if path.is_file():
            return path
    found = sorted(folder.glob("*.onnx"))
    if not found:
        raise SystemExit(f"no ONNX files in {folder}: {list(folder.iterdir())}")
    return found[0]


def check():
    assert ADAPTER.name == "_adapter_v5b_smollm2_135m"
    assert BASE == "HuggingFaceTB/SmolLM2-135M"
    print("check ok")


if __name__ == "__main__":
    main()
