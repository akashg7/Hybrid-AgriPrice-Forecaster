# 🚀 AgriSense AI: Free Deployment Guide

Deploying a Full-Stack Machine Learning application requires special consideration because Deep Learning frameworks (PyTorch and TensorFlow) require significant RAM, which most free tiers do not provide. 

Here is the **best and 100% free** strategy to deploy AgriSense so you can put the live link on your resume.

---

## 🏗️ The Deployment Strategy

1. **Frontend (Next.js):** [Vercel](https://vercel.com/) (The absolute best for Next.js, created by the Next.js team).
2. **Backend (FastAPI + ML Models):** [Hugging Face Spaces](https://huggingface.co/spaces) (Docker Space). 
   > [!IMPORTANT]
   > **Why Hugging Face?** Standard free tiers like Render or Heroku only give you 512MB of RAM. Loading both PyTorch (TFT) and TensorFlow (EfficientNet) simultaneously will cause an "Out of Memory" (OOM) crash. Hugging Face Spaces gives you **16GB RAM and 2 vCPUs for FREE**, which is perfect for heavy AI backends.

---

## Step 1: Prepare Your Repository

Before deploying, ensure your code is pushed to GitHub.

1. Go to GitHub and create a new repository (e.g., `Hybrid-AgriPrice-Forecaster`).
2. Push your entire local repository to this GitHub repo.

---

## Step 2: Deploy the Backend (Hugging Face Spaces)

We will run your `AgriSense_Hub_Backend.py` as a Docker container on Hugging Face.

### 1. Create a `Dockerfile`
In the **root** of your repository, create a file named exactly `Dockerfile` (no extension) with the following content:

```dockerfile
# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies (required for some ML libraries like OpenCV)
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . .

# Hugging Face Spaces expose port 7860 by default
EXPOSE 7860

# Command to run the backend on port 7860
CMD ["uvicorn", "AgriSense_Hub_Backend:app", "--host", "0.0.0.0", "--port", "7860"]
```

### 2. Update CORS in `AgriSense_Hub_Backend.py`
Ensure your backend accepts requests from anywhere (which you already have setup correctly):
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Keeps it open for your Vercel frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### 3. Deploy to Hugging Face
1. Create an account on [Hugging Face](https://huggingface.co/).
2. Click on your profile picture -> **New Space**.
3. **Space Name:** `agrisense-backend`
4. **License:** MIT
5. **Space SDK:** Select **Docker** (Blank).
6. **Space Hardware:** Free (2 vCPU, 16GB RAM).
7. Click **Create Space**.
8. Connect your GitHub repository, or manually upload your files. Hugging Face will automatically detect the `Dockerfile` and start building your FastAPI server.
9. **Get your Backend URL:** Once it's running, click the three dots (`...`) in the top right -> **Embed this Space** -> Look for the Direct URL (it usually looks like `https://yourusername-agrisense-backend.hf.space`).

---

## Step 3: Deploy the Frontend (Vercel)

Now we need to connect the frontend to the deployed backend.

### 1. Update Frontend API Calls
In your Next.js `frontend/` directory, locate where you are making API calls to the backend (likely in your UI components or an `api.js` file).
Change `http://localhost:8000` to your new Hugging Face Backend URL.

> [!TIP]
> **Best Practice:** Use an environment variable. Create a `.env.local` file in your `frontend/` folder:
> `NEXT_PUBLIC_API_URL=https://yourusername-agrisense-backend.hf.space`
> Then replace hardcoded URLs with `process.env.NEXT_PUBLIC_API_URL`.

### 2. Deploy to Vercel
1. Create an account on [Vercel](https://vercel.com/) (Sign in with GitHub).
2. Click **Add New** -> **Project**.
3. Import your `Hybrid-AgriPrice-Forecaster` repository from GitHub.
4. **Important Configuration:**
   - **Framework Preset:** Next.js
   - **Root Directory:** Click Edit and select the `frontend` folder (since your Next.js app isn't in the root of the repo).
   - **Environment Variables:** Add `NEXT_PUBLIC_API_URL` and set it to your Hugging Face Backend URL.
5. Click **Deploy**.

Vercel will build your Next.js app and give you a live URL (e.g., `https://agrisense.vercel.app`).

---

## Step 4: Add to Your Resume!

You are completely done. You now have an enterprise-grade deployed architecture that costs $0/month.

**Example Resume Bullet Point:**
> Developed and deployed a hybrid agricultural forecasting platform (Next.js, FastAPI). Orchestrated a decoupled micro-engine architecture hosting Temporal Fusion Transformers and EfficientNet-B0 on Hugging Face Docker Spaces, serving a Vercel-hosted frontend via REST APIs.

**Links to include:**
* **Live Demo:** `https://agrisense.vercel.app`
* **Backend API Docs:** `https://yourusername-agrisense-backend.hf.space/docs` (FastAPI auto-generates this!)
