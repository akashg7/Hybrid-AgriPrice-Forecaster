Given everything you've built — **PyTorch + PyTorch Lightning (TFT)**, **TensorFlow/Keras (EfficientNet-B0)**, and **LightGBM** all running inside one FastAPI hub — the deployment decision boils down to one constraint that most "free hosting" guides ignore: **RAM**. Loading three different ML frameworks simultaneously needs real headroom, and most free backend hosts (Render, Railway, Fly.io) cap free tiers at 512MB–1GB, which will **OOM-crash** this stack on the first request.

So here's the architecture I'm locking in, split by what each platform is actually good at:

**Frontend (Next.js) → Vercel (Hobby/free tier)**
Native Next.js support, global edge CDN, automatic HTTPS, unlimited free deployments for non-commercial/portfolio projects. This is what gives you fast load times worldwide — exactly what you want a recruiter to experience.

**Backend (FastAPI + TFT + LightGBM + EfficientNet) → Hugging Face Spaces (Docker SDK)**
This is the key call: HF Spaces' free **CPU Basic tier gives you 2 vCPU, 16GB RAM, and 50GB disk** — completely free, no card required. That's the only free option with enough memory to hold all three model engines in memory at once without crashing. It's also a natural fit on a resume — recruiters recognize HF Spaces as a real ML deployment target, not just a toy host.

Total cost: **$0**. No Railway trial credits expiring, no Fly.io billing surprises.

Now let's execute this in phases — same pattern as your Antigravity prompts. I'll walk you through **Phase 0 (prep)** and **Phase 1 (backend deploy)** now; ping me once the backend's live and we'll do **Phase 2 (frontend)** and **Phase 3 (keep-alive + resume polish)**.

---

## Phase 0: Pre-Deployment Prep

**1. Trim and CPU-pin your dependencies.** Your `requirements.txt` likely pulls full `torch` and `tensorflow`, which download CUDA binaries you don't need on a CPU-only Space — this alone can shave 2–3GB off your image and cut cold-start/build time significantly:

```txt
--extra-index-url https://download.pytorch.org/whl/cpu
torch
pytorch-lightning
pytorch-forecasting
tensorflow-cpu
lightgbm
scikit-learn
fastapi
uvicorn[standard]
pandas
numpy
python-multipart
```

**2. Add CORS to `AgriSense_Hub_Backend.py`** so the Vercel frontend can call it (add this near your app initialization):

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://your-project-name.vercel.app",  # update after Phase 2
        "http://localhost:3000",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)
```

**3. Load models once at startup, not per-request.** If your `Engines/` classes don't already do this, wrap each engine's model load in a singleton/cached pattern so `TFT_Engine`, `LGBM_Engine`, `Crop_Engine`, `Disease_Engine` load into memory **once** when the container boots, not on every `/predict` call. This is what keeps your 96%+ accurate TFT responding fast after the first request.

**4. Bundle `hierarchy_map.csv` and all `Modules/` weights into the repo** — HF Spaces' 50GB ephemeral disk handles this fine, no need for external storage.

---

## Phase 1: Deploy Backend to Hugging Face Spaces

**Step 1 — Create the Space**
Go to huggingface.co/new-space → name it (e.g., `agrisense-ai-backend`) → choose **Docker** as the SDK → set visibility to Public.

**Step 2 — Add this `README.md` frontmatter** (required for HF to recognize port/config):

```markdown
---
title: AgriSense AI Backend
emoji: 🌾
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# AgriSense AI — Hybrid Price Forecasting & Crop Intelligence Backend
FastAPI hub serving TFT, LightGBM, EfficientNet-B0, and Crop Recommendation engines.
```

**Step 3 — Add this `Dockerfile`** at the repo root:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN useradd -m -u 1000 appuser
RUN chown -R appuser:appuser /app
USER appuser

ENV PYTHONUNBUFFERED=1
EXPOSE 7860

CMD ["uvicorn", "AgriSense_Hub_Backend:app", "--host", "0.0.0.0", "--port", "7860"]
```

Note: `libgl1` / `libglib2.0-0` are there because OpenCV/TensorFlow image preprocessing for your Grad-CAM/disease detection pipeline typically needs them — skip this and you'll hit a cryptic `ImportError: libGL.so.1` during the build.

**Step 4 — Push your code**

```bash
git clone https://huggingface.co/spaces/your-username/agrisense-ai-backend
cd agrisense-ai-backend
# copy in your AgriSense_Hub_Backend.py, Engines/, Modules/, requirements.txt, Dockerfile, README.md
git add .
git commit -m "Initial backend deploy"
git push
```

HF will auto-build the Docker image — watch the **Logs** tab. First build will take a few minutes (downloading torch/tensorflow-cpu wheels); subsequent builds are cached and faster.

**Step 5 — Verify**
Once it's live at `https://your-username-agrisense-ai-backend.hf.space`, hit your endpoints directly:

```bash
curl -X POST https://your-username-agrisense-ai-backend.hf.space/api/lgbm/predict -d '{...}'
```

Confirm `/api/tft/predict`, `/api/crop/recommend`, and `/api/disease/detect` all respond before moving to Phase 2.

---

Once this is live and responding, tell me and I'll walk you through **Phase 2 (Vercel frontend deploy + wiring the API URL)** and **Phase 3 (keep-alive pinging so the demo never shows a cold-start lag to a recruiter, plus the exact line to put on your resume)**.