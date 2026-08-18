"""Merge the 135M LoRA and export fused ONNX files for transformers.js."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
ADAPTER = ROOT / "_adapter_v5b_smollm2_135m"
MERGED = ROOT / "_merged"
GENAI_CPU = ROOT / "_genai_cpu"
GENAI_WEBGPU = ROOT / "_genai_webgpu"
MODEL = ROOT / "v3"
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
    # Keep embeddings as Gather. The builder otherwise emits
    # GatherBlockQuantized, which onnxruntime-web does not implement.
    common = ["nodes_to_exclude=/model/embed_tokens/Gather"]
    build_genai(GENAI_CPU, "cpu", ["--extra_options", *common])
    build_genai(GENAI_WEBGPU, "webgpu", [
        "--extra_options", *common, "enable_webgpu_graph=true",
    ])


def build_genai(dest, provider, extra):
    if (dest / "model.onnx").is_file():
        print(f"reusing {dest}", file=sys.stderr)
        return
    if dest.exists():
        shutil.rmtree(dest)
    print(f"building {dest.name} ({provider} int4)", file=sys.stderr)
    subprocess.run(
        [
            sys.executable, "-m", "onnxruntime_genai.models.builder",
            "-i", str(MERGED),
            "-o", str(dest),
            "-p", "int4",
            "-e", provider,
            *extra,
        ],
        check=True,
    )
    if not (dest / "model.onnx").is_file():
        raise SystemExit(f"genai builder produced no model.onnx in {dest}")


def assemble():
    fused_cpu = count_ops(GENAI_CPU / "model.onnx")
    fused_gpu = count_ops(GENAI_WEBGPU / "model.onnx")
    if fused_cpu < 30:
        raise SystemExit(f"cpu graph has {fused_cpu} fused attention ops")
    if fused_gpu < 30:
        raise SystemExit(f"webgpu graph has {fused_gpu} fused attention ops")
    if MODEL.exists():
        shutil.rmtree(MODEL)
    MODEL.mkdir(parents=True)
    onnx_dir = MODEL / "onnx"
    onnx_dir.mkdir()
    for name in tqdm(EXTRAS, desc="copy tokenizer"):
        src = MERGED / name
        if not src.is_file():
            raise SystemExit(f"missing {src}")
        shutil.copy2(src, MODEL / name)
    save_external(GENAI_CPU / "model.onnx", onnx_dir / "model_q4.onnx")
    save_external(GENAI_WEBGPU / "model.onnx", onnx_dir / "model_q4f16.onnx")
    patch_js_config(MODEL / "config.json")
    print(f"-> {MODEL}", file=sys.stderr)
    for path in sorted(MODEL.rglob("*")):
        if path.is_file():
            print(f"  {path.relative_to(MODEL)} {path.stat().st_size}",
                  file=sys.stderr)


def pin_kv_cache_dim(model, dim):
    # transformers.js allocates empty past as [B, kv_heads, 0, last_dim];
    # a symbolic last dim becomes 0 and GroupQueryAttention rejects it.
    for value in list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info):
        for axis in value.type.tensor_type.shape.dim:
            if axis.dim_param == "kv_cache_dim":
                axis.ClearField("dim_param")
                axis.dim_value = dim


def save_external(src, dest):
    import onnx
    from onnx.external_data_helper import convert_model_to_external_data

    print(f"writing {dest.name}", file=sys.stderr)
    model = onnx.load(str(src), load_external_data=True)
    pin_kv_cache_dim(model, 64)
    convert_model_to_external_data(
        model,
        all_tensors_to_one_file=True,
        location=f"{dest.name}_data",
        size_threshold=1024,
    )
    onnx.save(model, str(dest))
    data = dest.parent / f"{dest.name}_data"
    if not dest.is_file():
        raise SystemExit(f"{dest} missing after save")
    if not data.is_file():
        raise SystemExit(f"{data} missing after save")


def count_ops(path):
    import onnx

    model = onnx.load(str(path), load_external_data=False)
    ops = {}
    for node in model.graph.node:
        ops[node.op_type] = ops.get(node.op_type, 0) + 1
    print(f"fused ops ({path.parent.name}):", file=sys.stderr)
    for name, n in sorted(ops.items(), key=lambda kv: -kv[1])[:12]:
        print(f"  {name} {n}", file=sys.stderr)
    fused = sum(ops.get(name, 0) for name in (
        "GroupQueryAttention", "MultiHeadAttention", "Attention"))
    print(f"  attention fused: {fused}", file=sys.stderr)
    return fused


def patch_js_config(path):
    cfg = json.loads(path.read_text())
    cfg["transformers.js_config"] = {
        "dtype": "q4",
        "use_external_data_format": 1,
        "kv_cache_dtype": {"q4f16": "float16", "fp16": "float16"},
    }
    path.write_text(json.dumps(cfg, indent=2) + "\n")


def check():
    assert ADAPTER.name == "_adapter_v5b_smollm2_135m"
    assert BASE == "HuggingFaceTB/SmolLM2-135M"
    assert MODEL.name == "v3"
    print("check ok")


if __name__ == "__main__":
    main()
