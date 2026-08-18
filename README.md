# deslop

A browser demo of the 135M deslop student. Paste an AI paragraph on the left; the model rewrites it on the right. Nothing leaves your machine after the first model download.

Site: https://eric-fithian.github.io/deslop-machine-public/

This is the SmolLM2-135M LoRA from [deslop-machine](https://github.com/giuliofrey/deslop-machine), merged and quantized for [transformers.js](https://huggingface.co/docs/transformers.js). It is the small end of the size sweep, not the shipped 8B method. Held-out Pangram Human for this student is 82%.

```bash
python3 -m http.server 8080
```
