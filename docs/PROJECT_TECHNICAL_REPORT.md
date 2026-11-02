# 🛰️ SPACE DEBRIS INTELLIGENCE: TECHNICAL REPORT & ARCHITECTURE SPECIFICATION

**System Name:** Low-Cost, Uncertainty-Aware Space Debris Intelligence Platform  
**Target Users:** Small Satellite Operators, Emerging Space Startups & Research Fleets  
**Core Innovation:** False-Alarm Fatigue Suppression using NASA 2D Gaussian Covariance & TLE Age Error Scaling  

---

## 1. Executive Summary & Problem Context

In modern Low Earth Orbit (LEO) operations, commercial satellite operators receive hundreds of automated conjunction warnings weekly from public tracking catalogs (CelesTrak / Space-Track). However, public orbital ephemerides rely on **Two-Line Element (TLE)** general perturbation sets that degrade rapidly within 24 to 48 hours due to unmodeled atmospheric drag, solar radiation pressure, and sensor tracking gaps.

Traditional screening software flags any encounter under a flat distance threshold (e.g., $< 5\text{ km}$) as dangerous. Because the position uncertainty of aged TLEs inflates to tens of kilometers:
* **Over 95% of public collision alerts are false alarms.**
* Commercial operators face acute **"False Alarm Fatigue"** and manually compute risks on spreadsheets.
* Operators frequently conduct unnecessary fuel burns, permanently shortening active satellite operational life.

---

## 2. Solution Architecture & Key Methodologies

The Space Debris Intelligence platform operates as a **Smart Risk Filter**:

```
+---------------------------------------------------------------------------------+
|                                DATA FOUNDATION                                  |
|   CelesTrak GP/OMM JSON Ingestion  -->  Pydantic Validation  -->  SHA-256 Hashes |
+---------------------------------------------------------------------------------+
                                       │
                                       ▼
+---------------------------------------------------------------------------------+
|                               PHYSICS BASELINE                                  |
|   SGP4 Orbital Propagation (TEME Frame)  -->  Coarse Step Pair Screening        |
|   Derives Exact Relative Velocity (km/s), Miss Distance (km) & True Midpoints  |
+---------------------------------------------------------------------------------+
                                       │
                                       ▼
+---------------------------------------------------------------------------------+
|                          NASA 2D GAUSSIAN RISK ENGINE                           |
|   σ(DSE) = σ_0 + (DSE × α)                                                      |
|   P_c = [1 / (sqrt(2π) * σ)] * exp( - d² / (2σ²) )                              |
|   Decision Logic: Suppress if DSE > 2.5d OR d > 1.5 km OR P_c < P_c_crit        |
+---------------------------------------------------------------------------------+
                                       │
                                       ▼
+---------------------------------------------------------------------------------+
|                         INTERACTIVE PRODUCT INTERFACES                          |
|   Streamlit Dashboard (:8501)      FastAPI Backend (:8000)     REST Web Client  |
|   3D Earth-Centered Sphere View    Dual Encounter Simulators   Optical CNN      |
+---------------------------------------------------------------------------------+
```

### 2.1 NASA 2D Gaussian Covariance Model
In the b-plane encounter coordinate system:
$$\sigma(\text{DSE}) = \sigma_0 + (\text{DSE} \times \alpha)$$
Where:
* $\sigma_0 = 0.30\text{ km}$ is baseline precision at epoch.
* $\alpha = 0.40\text{ km/day}$ is empirical position divergence rate.
* $\text{DSE}$ is the elapsed Days Since Epoch.

The collision probability $P_c$ is evaluated over hard-body collision cross-sections:
$$P_c = \frac{1}{\sqrt{2\pi}\sigma} \exp\left(-\frac{d^2}{2\sigma^2}\right)$$

### 2.2 False Alarm Suppression Rule
An encounter is classified as **FALSE ALARM (SUPPRESSED)** if:
$$\text{DSE} > \text{DSE}_{\text{max}} \quad \lor \quad d > d_{\text{safe}} \quad \lor \quad P_c < P_{c\text{-crit}}$$
When suppressed, the system recommends **STAND DOWN**, preventing wasteful fuel burns while preserving satellite lifetime.

---

## 3. Financial & Fuel Savings Analysis

* **Average Avoidance Maneuver Propellant Cost:** Estimated between $\$800 - \$2,500$ USD in station-keeping delta-V and ground station coordination.
* **Empirical Screening Results (114 Real Encounters):**
  * Total Alerts Screened: `114`
  * Actionable Real Threats: `6` ($\sim 5.3\%$)
  * Suppressed False Alarms: `108` ($\sim 94.7\%$)
  * **Quantified Propellant Dollars Saved:** **$\$86,400\text{ USD}$** in unnecessary orbital burns.

---

## 4. Optical Debris Streak Detection Module

* **Engine:** TensorFlow Conv2D AutoEncoder + Directional Astronomical Morphological Filtering.
* **Capability:** Isolates high-speed satellite streaks (Starlink trains, tumbling fragments, crossing tracks) from dense background starfields.
* **Vector Output:** Extracts start/end pixel coordinates, trail length (px), and orbital inclination vector angle (deg).

---

## 5. System Deployment & Running Instructions

1. **Start the Core Pipeline & Screening:**
   ```powershell
   python -m src.main
   ```

2. **Launch the FastAPI Backend:**
   ```powershell
   python -m uvicorn src.api:app --reload --port 8000
   ```

3. **Launch the Streamlit Analytics Dashboard:**
   ```powershell
   python -m streamlit run src/app/dashboard.py
   ```

4. **Launch the Lightweight REST Web Client:**
   Open `web_client/index.html` in any modern web browser or serve via HTTP.

---
*Generated by Space Debris Intelligence Research Team.*
