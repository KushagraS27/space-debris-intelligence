# Space Debris Intelligence - Deployment Guide

This repository contains two main deployment components:
1. **Interactive Streamlit Dashboard** (`src/app/dashboard.py`)
2. **FastAPI Calculation Engine** (`src/api.py`)
3. **Static Web Client** (`web_client/index.html`)

---

## Method 1: Streamlit Community Cloud (Recommended & Free - 2 Minutes)

Streamlit Community Cloud is the fastest, free zero-configuration hosting platform for this dashboard.

### Steps to Deploy:
1. Go to [share.streamlit.io](https://share.streamlit.io/) and click **"Sign in with GitHub"**.
2. Click the **"Create app"** button.
3. Select **"I already have an app"**.
4. Fill in the repository settings:
   - **Repository:** `KushagraS27/space-debris-intelligence`
   - **Branch:** `main`
   - **Main file path:** `src/app/dashboard.py`
   - **App URL (optional):** Choose a custom subdomain (e.g., `space-debris-intelligence.streamlit.app`)
5. Click **"Deploy!"**.

Streamlit Cloud will automatically detect `requirements.txt`, install dependencies, and provide a live public HTTPS URL shareable with anyone.

---

## Method 2: Render.com (For Dashboard or FastAPI Backend)

### Deploying Dashboard on Render:
1. Go to [render.com](https://render.com) and create an account.
2. Click **New +** $\rightarrow$ **Web Service**.
3. Connect your GitHub repository `KushagraS27/space-debris-intelligence`.
4. Configure settings:
   - **Environment:** `Python 3`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `streamlit run src/app/dashboard.py --server.port $PORT --server.address 0.0.0.0`
5. Click **Create Web Service**.

### Deploying FastAPI Backend on Render:
- **Build Command:** `pip install -r requirements.txt`
- **Start Command:** `uvicorn src.api:app --host 0.0.0.0 --port $PORT`

---

## Method 3: Hugging Face Spaces (Free Docker / Streamlit Hosting)

1. Go to [huggingface.co/spaces](https://huggingface.co/spaces) $\rightarrow$ **Create new Space**.
2. Select **Streamlit** SDK.
3. Connect your GitHub repo or push code to the Space repository.

---

## Method 4: Local / Intranet Run

To run locally and preview on your machine or network:

```powershell
# 1. Start the Streamlit Dashboard
streamlit run src/app/dashboard.py

# 2. In a separate terminal, start the FastAPI engine
uvicorn src.api:app --reload --port 8000
```
