# CIFAR-10 Vision Stack

Next.js frontend + FastAPI backend for running the saved CIFAR-10 model locally.

## Run the full stack locally

Backend:

```bash
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
./.venv/bin/python app.py
```

Frontend:

```bash
cd frontend
npm install
cp .env.local.example .env.local
npm run dev
```

Set `NEXT_PUBLIC_API_URL` in `frontend/.env.local` if your backend is not on `http://localhost:8000`.

Deployment note:
- `model.weights.h5` must be committed as the real binary file, not a Git LFS pointer. Railway will crash if it receives the pointer text instead of the actual HDF5 data.

The API exposes:

- `GET /health`
- `GET /classes`
- `POST /predict`

For a quick command-line test without the browser, run:

```bash
./.venv/bin/python predict_local.py img.jpg
```
