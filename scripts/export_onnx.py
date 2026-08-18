"""Merge the 0.6B LoRA and export fused ONNX files for transformers.js."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
ADAPTER = ROOT / "_adapter_v5b_0.6b"
MERGED = ROOT / "_merged"
GENAI_CPU = ROOT / "_genai_cpu"
GENAI_WEBGPU = ROOT / "_genai_webgpu"
MODEL = ROOT / "v4"
BASE = "Qwen/Qwen3-0.6B-Base"
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
    cfg = json.loads((MERGED / "config.json").read_text())
    layers = cfg["num_hidden_layers"]
    head_dim = cfg.get("head_dim") or cfg["hidden_size"] // cfg["num_attention_heads"]
    fused_cpu = count_ops(GENAI_CPU / "model.onnx")
    fused_gpu = count_ops(GENAI_WEBGPU / "model.onnx")
    if fused_cpu < layers:
        raise SystemExit(f"cpu graph has {fused_cpu} fused attention ops, need {layers}")
    if fused_gpu < layers:
        raise SystemExit(f"webgpu graph has {fused_gpu} fused attention ops, need {layers}")
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
    save_external(GENAI_CPU / "model.onnx", onnx_dir / "model_q4.onnx", head_dim)
    save_external(
        GENAI_WEBGPU / "model.onnx",
        onnx_dir / "model_q4f16.onnx",
        head_dim,
        unfuse=True,
    )
    gqa_inputs = max_gqa_inputs(onnx_dir / "model_q4f16.onnx")
    if gqa_inputs > 14:
        raise SystemExit(f"webgpu GQA still has {gqa_inputs} inputs, browser ORT max is 14")
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


def unfuse_qk_norm(model, head_dim, q_hidden, k_hidden):
    # Browser ORT (transformers.js 4.2 / ORT 1.26) caps GQA at 14 inputs.
    # The genai WebGPU builder folds Qwen3 Q/K RMSNorm into inputs 14/15.
    from onnx import numpy_helper
    import numpy as np

    def add_shape(name, values):
        for init in model.graph.initializer:
            if init.name == name:
                return name
        model.graph.initializer.append(
            numpy_helper.from_array(np.asarray(values, dtype=np.int64), name=name)
        )
        return name

    shape_h = add_shape("/model/constants/INT64/[0, -1, 128]", [0, -1, head_dim])
    shape_q = add_shape("/model/constants/INT64/[0, -1, 2048]", [0, -1, q_hidden])
    shape_k = add_shape("/model/constants/INT64/[0, -1, 1024]", [0, -1, k_hidden])

    rebuilt = []
    n_unfused = 0
    for node in model.graph.node:
        if node.op_type != "GroupQueryAttention" or len(node.input) < 16:
            rebuilt.append(node)
            continue
        q_in, k_in = node.input[0], node.input[1]
        q_w, k_w = node.input[14], node.input[15]
        if not q_w or not k_w:
            raise SystemExit(f"{node.name} has empty Q/K norm weights")
        eps = 1e-6
        kept_attrs = []
        for attr in node.attribute:
            if attr.name == "qk_norm_epsilon":
                eps = attr.f
            else:
                kept_attrs.append(attr)
        prefix = node.name.removesuffix("/GroupQueryAttention")
        q_out = add_norm(rebuilt, f"{prefix}/q_norm", q_in, q_w, shape_h, shape_q, eps)
        k_out = add_norm(rebuilt, f"{prefix}/k_norm", k_in, k_w, shape_h, shape_k, eps)
        rest = list(node.input[2:9])
        del node.input[:]
        node.input.extend([q_out, k_out, *rest, "", "", ""])
        del node.attribute[:]
        node.attribute.extend(kept_attrs)
        rebuilt.append(node)
        n_unfused += 1
    if n_unfused == 0:
        raise SystemExit("no fused Q/K-norm GQA nodes to rewrite")
    del model.graph.node[:]
    model.graph.node.extend(rebuilt)
    print(f"unfused Q/K-norm on {n_unfused} GQA nodes", file=sys.stderr)


def add_norm(nodes, prefix, src, weight, shape_in, shape_out, eps):
    from onnx import helper

    inner = f"{prefix}/Reshape_1/output_0"
    normed = f"{prefix}/SimplifiedLayerNormalization/output_0"
    outer = f"{prefix}/Reshape_2/output_0"
    nodes.append(helper.make_node("Reshape", [src, shape_in], [inner], name=f"{prefix}/Reshape_1"))
    nodes.append(helper.make_node(
        "SimplifiedLayerNormalization",
        [inner, weight],
        [normed],
        name=f"{prefix}/SimplifiedLayerNormalization",
        epsilon=float(eps),
        axis=-1,
        stash_type=1,
    ))
    nodes.append(helper.make_node("Reshape", [normed, shape_out], [outer], name=f"{prefix}/Reshape_2"))
    return outer


def save_external(src, dest, head_dim, unfuse=False):
    import onnx
    from onnx.external_data_helper import convert_model_to_external_data

    print(f"writing {dest.name}", file=sys.stderr)
    model = onnx.load(str(src), load_external_data=True)
    if unfuse:
        cfg = json.loads((MERGED / "config.json").read_text())
        q_hidden = cfg["num_attention_heads"] * head_dim
        k_hidden = cfg["num_key_value_heads"] * head_dim
        unfuse_qk_norm(model, head_dim, q_hidden, k_hidden)
    pin_kv_cache_dim(model, head_dim)
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


def max_gqa_inputs(path):
    import onnx

    model = onnx.load(str(path), load_external_data=False)
    counts = [len(n.input) for n in model.graph.node if n.op_type == "GroupQueryAttention"]
    if not counts:
        raise SystemExit(f"no GroupQueryAttention in {path}")
    return max(counts)


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
    assert ADAPTER.name == "_adapter_v5b_0.6b"
    assert BASE == "Qwen/Qwen3-0.6B-Base"
    assert MODEL.name == "v4"
    print("check ok")


if __name__ == "__main__":
    main()
