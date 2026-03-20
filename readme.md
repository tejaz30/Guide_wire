# KamaiKavach 🛡️  
**Kamai = Earning | Kavach = Shield**

Guidewire DEVTrails 2026 · AI-Powered Parametric Insurance for India's Gig Economy  
Persona: Food Delivery Partners (Zomato / Swiggy) · Platform: Android-first  

---

## TL;DR (For Judges)

- Gig workers lose 20–30% income due to external disruptions  
- KamaiKavach provides weekly parametric income insurance  
- Fully automated: Trigger → Payout → Fraud Check → UPI Transfer  
- Built on 5 ML models (premium, payout, trigger, underwriting, fraud)  
- Key innovation: eliminates basis risk + defeats GPS spoofing  

---

## Problem Context

Delivery partners face income loss due to:
- Heavy rain, floods  
- Severe air pollution  
- Extreme heat  
- Curfews / shutdowns  

When disruption happens:
- Work stops instantly  
- Earnings drop to zero  
- No safety net exists  

Scope: income protection only (no health, accident, or repair coverage)

---

## Persona

Ravi — Delivery Partner, Bengaluru  
- Earns ₹700–900/day  
- Lives week-to-week  
- No savings buffer  
- Uses Android + UPI  

### Scenarios

Heavy Rain  
Rain ≥ 15 mm/hr → delivery stops → payout triggered  

Curfew  
Zone inaccessible → system detects inactivity → payout  

Pollution  
AQI > 300 → worker compensated without exposure  

---

## System Overview

KamaiKavach is a fully automated parametric insurance system:

Onboarding → Risk Scoring → Weekly Premium → Active Policy  
↓  
Real-time Trigger Detection (Model 3)  
↓  
Payout Calculation (Model 2)  
↓  
Fraud Detection (Model 5)  
↓  
Instant UPI Transfer  

- No claims  
- No paperwork  
- No human intervention  

---

# AI / ML Architecture (Core of the System)

The system is built as five independent but coordinated models.

---

## Model 1 — Weekly Premium Calculation

Objective: Estimate expected weekly loss (pure premium)

Approach:
- LightGBM Regressor  
- Tweedie loss (p = 1.5)

Why:
- Insurance data is zero-inflated + heavy-tailed  
- Tweedie models compound Poisson-Gamma distribution  

Key Insight:
Risk is non-linear (zone × season × behavior interactions)

Output:
Premium_weekly = clip(exp(F(x)), ₹15, ₹120)

Explainability:
SHAP used to show breakdown:
“₹47 = ₹8 flood risk + ₹6 monsoon + ₹3 work hours”

---

## Model 2 — Payout Calculation

Objective: Compute payout when disruption occurs

Two-layer system:

Layer 1 — Parametric Formula
- Based on earnings and hours lost  
- 1-hour deductible  
- 70% cap  
- Tiered payout  

Layer 2 — ML Model
- LightGBM (Huber loss)  
- Predicts actual hours lost  

Why:
Standard parametric insurance has basis risk  
Same trigger ≠ same impact  

This model adjusts payout based on:
- Event severity  
- Worker profile  
- Zone conditions  

Result:
Parametric speed + ML accuracy

---

## Model 3 — Trigger Detection

Objective: Detect real-world disruptions

Stage 1 — Rule Engine
- Rain ≥ 15 mm/hr  
- AQI > 300  
- Temp > 42°C  
- Curfew active  
- Platform downtime  

Stage 2 — Sequence Model (LSTM)
- Uses time-series window (~3 hours)  
- Detects patterns, not spikes  

Decision:
Trigger = threshold OR LSTM probability ≥ 0.75  

Features:
- Rainfall intensity  
- AQI  
- Temperature  
- Traffic signals  
- Alert feeds  

Key Insight:
Captures multi-signal disruptions and avoids noise

---

## Model 4 — Underwriting Risk Scoring

Objective: Estimate probability of worker filing a claim

Approach:
- LightGBM classifier  
- Probability calibration (Platt scaling)

Output:
P(claim in next 4 weeks)

Usage:
- Risk tier assignment  
- Premium adjustment  
- Policy eligibility  

Why separate:
Premium = amount  
Risk score = probability  

This improves actuarial consistency

---

## Model 5 — Fraud Detection

Objective: Detect fraudulent claims in real time

Layer 1 — Signal Corroboration
- Cell tower triangulation  
- WiFi fingerprint  
- Accelerometer patterns  
- Platform activity  
- Zone history  

Layer 2 — Isolation Forest
- Detects anomalous behavior  
- Works without labelled fraud  

Layer 3 — DBSCAN Clustering
- Detects coordinated fraud rings  
- Identifies synchronized claim patterns  

Fraud Score combines:
- Signal mismatch  
- Behavioral anomaly  
- Cluster risk  

Decision Engine:
- Low → instant payout  
- Medium → auto payout + monitor  
- High → soft hold  
- Very high → manual review  

---

# Adversarial Defense & Anti-Spoofing Strategy

## Threat

Coordinated workers use GPS spoofing to fake presence in disruption zones.

---

## Differentiation

Real worker:
- Consistent across all physical signals  

Fraudster:
- Fails multiple signals simultaneously  

---

## Data Beyond GPS

- Cell tower location  
- WiFi BSSID  
- Motion sensors  
- Platform logs  
- Zone behavior  

---

## Core Principle

GPS alone is untrusted  
System relies on multi-signal validation  

---

## UX Balance

- No “fraud” label shown  
- Missing data ≠ fraud  
- Tiered verification  

Design choice:
Prefer false negatives over false positives  

---

## Parametric Triggers

| Trigger | Threshold |
|--------|----------|
| Rain | ≥ 15 mm/hr |
| Flood | Red alert |
| AQI | > 300 |
| Heat | > 42°C |
| Curfew | Active |
| Downtime | > 2 hrs |

- Checked every 15 minutes  
- Zone-specific  
- 4-hour cooldown  

---

## Weekly Premium Plans

| Plan | Premium | Daily Payout | Weekly Cap |
|------|--------|-------------|-----------|
| Basic | ₹39 | ₹300 | ₹900 |
| Standard | ₹59 | ₹500 | ₹1500 |
| Pro | ₹79 | ₹700 | ₹2100 |

Principles:
- Weekly pricing aligns with gig economy  
- 70% cap prevents moral hazard  
- Max 3 days/week  

---

## Platform Choice

- Android-first (majority users)  
- Native UPI support  
- Push notifications  
- Required for fraud signal collection  

---

## Tech Stack

| Layer | Technology |
|------|-----------|
| Mobile | React Native |
| Backend | FastAPI |
| Database | PostgreSQL |
| ML | LightGBM, PyTorch, sklearn |
| APIs | IMD, CPCB, Weather |
| Payments | Razorpay |
| Auth | Firebase |
| Hosting | AWS / Railway |

---

## Development Plan

Phase 1  
- Architecture  
- README  
- ML design  

Phase 2  
- Policy system  
- Trigger engine  
- Payout flow  

Phase 3  
- Fraud detection  
- ML refinement  
- Dashboard  

---

## MVP vs Full System

Phase 2 MVP:
- Rule-based triggers  
- Premium model  
- Basic fraud detection  

Phase 3:
- LSTM triggers  
- DBSCAN fraud rings  
- Advanced ML payout  

---

## Business Viability

- Target loss ratio: 60–70%  
- Weekly caps limit exposure  
- Fraud system reduces leakage  
- Sustainable under moderate disruption rates  

---

## Key Differentiators

1. Eliminates basis risk via ML payout  
2. Multi-signal anti-spoofing  
3. Weekly-first pricing model  
4. Fully automated claims  
5. Worker-friendly fraud handling  

---

## Requirement Mapping

| Requirement | Section |
|------------|--------|
| Persona | Persona |
| Workflow | System Overview |
| Weekly pricing | Premium |
| Parametric triggers | Triggers |
| AI/ML | ML Architecture |
| Fraud detection | Model 5 |
| Platform | Platform Choice |
| Tech stack | Tech Stack |
| Plan | Development Plan |

---

## Future Work

Graph-based fraud detection (Graph Neural Networks)  
Supply-side disruption detection (order volume drops)  

---

## Closing

KamaiKavach is not just insurance.  
It is income stability infrastructure for gig workers.

Built for Ravi. Built for millions like him.
