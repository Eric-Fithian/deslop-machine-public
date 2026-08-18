# deslop

A browser demo of the 135M deslop student. Paste an AI paragraph on the left; the model rewrites it on the right. Nothing leaves your machine after the first model download.

Site: https://eric-fithian.github.io/deslop-machine-public/

This is the SmolLM2-135M LoRA from [deslop-machine](https://github.com/giuliofrey/deslop-machine), merged and quantized for [transformers.js](https://huggingface.co/docs/transformers.js). The ONNX graph uses fused GroupQueryAttention. A browser with WebGPU and `shader-f16` (Chrome or Firefox on an M-series Mac) runs the q4f16 GPU weights; otherwise it falls back to q4 on the CPU. Held-out Pangram Human for this student is 82%.

```bash
python3 -m http.server 8080
```
