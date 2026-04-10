---
title: CIFAR 10
emoji: 🐠
colorFrom: gray
colorTo: blue
sdk: gradio
sdk_version: 6.11.0
app_file: app.py
pinned: false
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

## Local inference

The model is now runnable locally from the saved weights and JSON config.

```bash
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
./.venv/bin/python predict_local.py img.jpg
```

This prints the top-5 CIFAR-10 classes for `img.jpg`.
If you want the browser UI instead, run `./.venv/bin/python app.py`.
