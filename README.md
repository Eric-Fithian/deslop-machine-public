# deslop

A browser demo of the 0.6B deslop student. Paste an AI paragraph on the left; the model rewrites it on the right. Nothing leaves your machine after the first model download.

Site: https://eric-fithian.github.io/deslop-machine-public/

This is the Qwen3-0.6B-Base LoRA from [deslop-machine](https://github.com/giuliofrey/deslop-machine), merged and quantized for [transformers.js](https://huggingface.co/docs/transformers.js). The ONNX graph uses fused GroupQueryAttention. The live site ships only the q4f16 WebGPU weights, so Chrome or Firefox with `shader-f16` (an M-series Mac) is required. Held-out Pangram Human for this student is 93.5%.

```bash
python3 -m http.server 8080
```
