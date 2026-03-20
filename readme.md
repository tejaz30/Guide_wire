# KamaiKavach 🛡️
**Kamai = Earning | Kavach = Shield**

> Guidewire DEVTrails 2026 · AI-Powered Parametric Insurance for India's Gig Economy  
> **Persona:** Food Delivery Partners (Zomato / Swiggy) · **Platform:** Android Mobile App

> 📄 Full mathematical specifications for all ML models are documented in [`/docs/ml-architecture.md`](/docs/ml-architecture.md)

---

## TL;DR

| | |
|---|---|
| **Persona** | Zomato / Swiggy food delivery partners |
| **Core idea** | Weekly parametric income insurance — automatic payouts when external disruptions halt deliveries |
| **Key innovation 1** | ML-driven dynamic premium with zone-level risk (LightGBM + Tweedie) |
| **Key innovation 2** | Hybrid trigger system — rule-based threshold engine + LSTM pattern detector |
| **Key innovation 3** | 3-layer anti-fraud: multi-signal corroboration + Isolation Forest + DBSCAN ring detector |
| **Key innovation 4** | Basis risk elimination — payout proportional to actual hours lost, not a flat amount |
| **USP** | Zero-claim, zero-friction, instant payout (< 2 hrs). Worker never files anything. |

---

## The Problem

Food delivery partners on Zomato and Swiggy lose **20–30% of their monthly income** when external disruptions — heavy rain, floods, severe pollution, sudden curfews — halt deliveries. They have no safety net. When a red-alert day hits, a delivery partner loses an entire day's wages with zero recourse.

**KamaiKavach** provides automated, zero-paperwork income protection triggered by real-world disruption events, structured on a weekly premium cycle that matches how gig workers earn and spend.

> ⚠️ Strictly excluded: health, life, accident, and vehicle repair coverage. Income loss only.

---

## Key Differentiators

**1. Basis risk elimination**  
Standard parametric insurance pays a fixed amount when a threshold is crossed — regardless of actual impact. Our ML payout layer predicts actual hours lost per worker, per zone, per event. A worker in a flooded zone gets a different payout than one in a zone with light drizzle from the same trigger. No other team is solving this.

**2. Multi-signal anti-spoofing**  
GPS alone is obsolete. Our 3-layer fraud system (device signal corroboration + Isolation Forest anomaly detection + DBSCAN ring detector) makes GPS spoofing self-defeating — a fraudster would need to simultaneously fake cell tower location, WiFi network, accelerometer history, and platform activity to defeat it.

**3. Weekly-first design**  
Every financial and UX decision — premium cycle, payout caps, deductible structure, appeal flow — is designed around how gig workers actually earn and spend. Most insurance products are monthly products bolted onto a weekly persona. Ours is built weekly from the ground up.

**4. Zero-claim UX**  
The worker never initiates a claim, never fills a form, never uploads a photo. The platform detects, validates, and pays automatically. The only thing the worker sees is a push notification with a transfer confirmation.

---

## Persona & Scenarios

**Ravi — Zomato delivery partner, Bengaluru, age 26**  
Earns ₹700–900/day across HSR Layout and Koramangala zones. Lives week-to-week with no savings buffer. Android user, comfortable with UPI.

**Scenario 1 — Heavy Rain**  
IMD issues a Red Alert. Rainfall exceeds 15mm/hr in Ravi's zone. Without KamaiKavach: ₹700 lost, no recourse. With KamaiKavach: the trigger pipeline detects the alert, calculates Ravi's predicted hours lost via ML, and transfers ₹480 to his UPI account within 2 hours. No claim form. No photos. No calls.

**Scenario 2 — Bandh / Curfew**  
An unplanned bandh shuts restaurants and blocks roads in Ravi's zone. The platform health check API confirms order volume collapse. Covered hours of inactivity are automatically compensated — proportional to the actual disruption window, not a flat amount.

**Scenario 3 — Severe Air Pollution**  
AQI crosses 300 (Severe) for 4+ hours. CPCB AQI API triggers coverage. Ravi is compensated without being forced to ride in hazardous conditions. A worker in a lightly affected zone receives a lower payout than one in the heavily polluted core — basis risk is eliminated.

---

## Application Workflow

```
Onboarding → Risk Profiling → Weekly Policy Purchase → Active Coverage
     ↓                                                        ↓
Verify ID                                        Real-time trigger pipeline polls
Link delivery platform (OAuth)                   weather / AQI / traffic APIs
Declare operating zones                          every 15 minutes per zone
Choose weekly plan                                           ↓
Enable background signals (opt-in)              Trigger fires → Fraud score computed
                                                             ↓
                                        F < 0.20 → Instant payout (< 10 min)
                                        F < 0.45 → Auto-payout, 2hr watch
                                        F < 0.70 → Soft hold, 4hr review
                                        F ≥ 0.70 → Freeze, human reviewer
```

---

## Weekly Premium Model

**Why weekly?** Delivery partners earn and spend week-to-week. A monthly premium competes with rent. A weekly premium of ₹39–79 sits comfortably alongside fuel costs.

| Plan | Weekly Premium | Payout per Disruption Day | Max Weekly Payout |
|------|---------------|--------------------------|-------------------|
| Basic Kavach | ₹39 | up to ₹300 | ₹900 (3 days) |
| Standard Kavach | ₹59 | up to ₹500 | ₹1,500 (3 days) |
| Pro Kavach | ₹79 | up to ₹700 | ₹2,100 (3 days) |

- Auto-debit every Sunday night via UPI mandate
- Coverage: Monday 00:00 – Sunday 23:59 for declared zones
- Maximum 3 disruption days covered per week
- 70% earnings replacement ratio — a worker should never prefer not working over working

**Dynamic pricing:** Premium is computed weekly by a **LightGBM regressor with Tweedie loss** — the actuarial industry standard for zero-inflated insurance loss distributions. Inputs include zone flood frequency (5yr average), drainage quality, seasonal disruption rates, worker earnings profile, peak-hour ratio, zone elevation, and distance to waterways. The model captures non-linear risk interactions (a flood-prone zone during monsoon worked 12-hour shifts is multiplicatively riskier, not additively) that conventional GLMs cannot model. SHAP values decompose the premium into per-feature contributions, shown in-app as: *"Your premium this week is ₹47 — ₹8 above base because your zone has a high flood history, ₹6 because it's monsoon season."* This satisfies IRDAI transparency requirements.

Premium floor: ₹15/week. Premium cap: ₹120/week.

---

## Parametric Triggers

Payouts are triggered by **external, verifiable, objective events** — not worker-submitted claims.

The trigger pipeline runs as a **two-stage system**:

**Stage 1 — Rule-based threshold engine** (high recall guarantee):

| # | Trigger | Data Source | Threshold |
|---|---------|-------------|-----------|
| 1 | Heavy rainfall | IMD / OpenWeatherMap | ≥ 15 mm/hr |
| 2 | Flood alert | IMD public API | Red Alert (Level 3) |
| 3 | Severe air pollution | CPCB AQI API | AQI > 300 (Severe) |
| 4 | Extreme heat | OpenWeatherMap | Temperature > 42°C |
| 5 | Bandh / curfew | Govt alert feed + Google Maps Traffic | Active curfew OR zone traffic collapse |
| 6 | Platform downtime | Mock platform health API | Downtime > 2 hours |

**Stage 2 — LSTM sequence classifier** (precision filter): An LSTM monitors a sliding 3-hour window of multi-signal readings and detects disruption *patterns* rather than single threshold crossings. This handles multi-signal events (moderate rain + high AQI + traffic paralysis simultaneously) that individually fall below thresholds but collectively constitute a disruption. The combined trigger fires when either stage detects an event. A 4-hour cooldown per zone prevents multiple payouts from the same continuous event.

All triggers are **zone-specific** — matched against the worker's declared operating zone at policy creation time.

---

## Platform Justification: Mobile (Android-first)

- 95%+ of food delivery partners use Android as their primary device
- UPI payment integration is native and seamless on Android
- Push notifications for instant payout alerts require a native app
- Background signal collection (accelerometer, cell tower, WiFi) for fraud detection **requires native mobile APIs** — not achievable on web
- The **admin/insurer dashboard** (loss ratios, claim review queue, predictive analytics) is a separate lightweight web interface

---

## AI / ML Architecture

KamaiKavach runs five ML models in production. Full mathematical specifications, feature sets, training procedures, and architecture diagrams for all five models are in [`/docs/ml-architecture.md`](/docs/ml-architecture.md).

---

### Model 1 — Weekly Premium Calculator
**LightGBM Regressor · Tweedie loss (p = 1.5)**

Answers: *how much should this worker pay per week?*

Tweedie loss models the compound Poisson-Gamma structure of insurance claims — zero-inflated with a heavy right tail — correctly. LightGBM captures non-linear risk interactions automatically via tree splits. 25+ features across environmental risk, worker behaviour, platform dynamics, urban infrastructure, and temporal context. SHAP TreeExplainer provides per-feature premium breakdown shown in-app. Trained on 100k synthetic rows seeded from IMD zone-level disruption data. Retrained monthly.

---

### Model 2 — Payout Calculator
**Two-layer system: Parametric formula + GBT Regressor · Huber loss**

Answers: *when a trigger fires, exactly how much does this worker receive?*

**Layer 1 (deterministic):** Parametric formula computes gross payout from predicted hours lost, worker's hourly rate, a 1-hour deductible (prevents micro-claims), and a tiered rate (80% beyond 4 hours to control moral hazard). Hard cap at 70% of weekly earnings.

**Layer 2 (ML):** A GBT regressor predicts *actual hours lost* from trigger intensity, time of day, zone vulnerability, and worker profile. This eliminates **basis risk** — the core flaw of standard parametric insurance where a worker in a lightly affected zone gets the same payout as one in a heavily flooded zone. Huber loss (δ=1hr) handles worker behaviour heterogeneity: some push through mild rain, some stop immediately.

Result: parametric insurance's zero-touch speed, plus ML's precision in estimating actual impact. Payout computed and transferred within 2 hours of trigger.

---

### Model 3 — Parametric Trigger Detector
**Rule-based threshold engine + LSTM sequence classifier**

Answers: *has a disruption event just occurred in zone z at time t?*

The threshold engine guarantees no disruption is missed (high recall). The LSTM reduces false positives from sensor noise and detects multi-signal events that individually fall below thresholds. Architecturally isolated from worker profiles — it only answers whether an event happened, not who is affected. 4-hour cooldown deduplication per zone.

---

### Model 4 — Fraud Detection (3-Layer)
**Signal corroboration scorer + Isolation Forest + DBSCAN ring detector**

Answers: *is this claim genuine or fraudulent?*

**Layer 1 — Signal corroboration (S_corr):** Scores 5 independent device signals — cell tower triangulation, WiFi BSSID fingerprint, accelerometer/gyroscope motion pattern, platform activity log, and historical zone heatmap. A genuine stranded worker has a physical signature a GPS spoofer cannot replicate across all signals simultaneously. Cell tower location is determined by the telecom network — a GPS spoofing app cannot influence it. Discrepancy > 500m between GPS and cell tower = strong fraud signal.

**Layer 2 — Isolation Forest (s_iso):** Trained exclusively on legitimate claims. Detects anomalous claim behaviour — unusually fast filing, claimed hours far exceeding the ML prediction, sudden claim velocity spikes — without needing labelled fraud examples. Novel attack vectors not seen in training are flagged automatically.

**Layer 3 — DBSCAN ring detector (R):** Runs every 15 minutes during active disruptions. Clusters claims by account age, zone history, device fingerprint, and registration source. A Telegram-coordinated syndicate produces: accounts created around the same time, same device models, first-time claimers, claims arriving in seconds of each other. Genuine mass disruptions produce temporally spread, device-diverse, account-age-diverse patterns. DBSCAN finds the ring without needing labelled ring data.

**Combined fraud score:**
```
F_fraud = 1 - [0.50·S_corr + 0.30·(1 + s_iso)/2 + 0.20·(1 - R)]
```

**Tiered decision engine:**

| Fraud Score | Action | Worker sees |
|-------------|--------|-------------|
| F < 0.20 | Instant payout | *"Your claim is approved. ₹[X] transferred."* |
| F < 0.45 | Auto-payout, 2hr watch | *"Payment initiated. Confirming details in background."* |
| F < 0.70 | Soft hold, 4hr review | *"Verifying your claim due to network conditions. No action needed."* |
| F ≥ 0.70 | Freeze, human reviewer | *"Your claim needs a short review. We'll update you within 24 hours."* |

The word **"fraud"** never appears in the worker-facing UI. **Missing signal ≠ fraud signal** — a worker with poor connectivity in heavy rain may have incomplete signals; contradictory signals (GPS says Koramangala, cell tower says Whitefield) are fraud signals. Even when DBSCAN flags an entire batch as a ring, individual workers with S_corr = 1.0 bypass the ring flag and receive instant payout. Ring detection penalises the cluster, not the individual.

---

### Model 5 — Risk Profiling
**K-Means Clustering**

Segments workers into Low / Medium / High risk at onboarding based on zone disruption history, average earnings, and platform tenure. Determines the recommended plan tier and sets baseline fraud sensitivity thresholds for that worker.

---

## Adversarial Defense & Anti-Spoofing Strategy

### Threat Model

A coordinated syndicate of 500 delivery workers organised via Telegram uses GPS-spoofing apps to fake their location inside a declared disruption zone while resting at home, triggering mass false payouts and draining the liquidity pool. **Simple GPS verification is officially obsolete. KamaiKavach does not trust GPS alone.**

### How the architecture defeats it

A fraudster at home spoofing GPS will simultaneously fail at least three of the five Layer 1 corroboration checks: their cell tower triangulation will show a residential area; their WiFi BSSID will match their registered home router; their accelerometer will show couch-level stillness with no prior riding pattern. Defeating all five checks simultaneously requires physical presence in the disruption zone — which is exactly the condition that justifies a payout. **The architecture makes GPS spoofing self-defeating.**

The DBSCAN layer defeats the coordinated ring specifically. Five hundred workers filing simultaneously from the same zone with similar account ages, same device models, and no prior zone activity is exactly the cluster structure DBSCAN finds — without needing a single labelled fraud example.

### UX protection for honest workers

A genuine worker in heavy rain may have poor network connectivity — signals arrive late or incomplete. The 4-tier decision engine's soft-hold window gives delayed signals time to arrive before escalating to human review. Workers are never accused, never shown a rejection. The system is deliberately biased toward **false negatives** (paying a possibly fraudulent claim) over **false positives** (denying a genuine worker). The human cost of wrongly denying a gig worker's income protection exceeds the cost of one fraudulent payout.

---

## Payout Processing

Payouts are processed via **Razorpay Test Mode** simulating instant UPI transfers. The flow is fully automated:

1. Parametric trigger confirmed → claim auto-initiated
2. Fraud score computed from 3-layer model → decision tier assigned
3. Payout amount calculated by Model 2 (parametric formula + ML hours-lost prediction)
4. Razorpay payout API called with worker's registered UPI ID
5. Worker receives push notification: *"₹[X] transferred to your UPI account"*

No worker action required at any step. Target: money in account within 2 hours of trigger for F < 0.45 claims.

---

## Analytics Dashboard

**Worker Dashboard (in-app, mobile)**
- Active weekly policy status and coverage period
- Earnings protected this week / this month (cumulative)
- Live disruption alerts in operating zone
- Claim history with payout amounts, timestamps, and fraud tier
- Premium breakdown via SHAP: *"Why did my premium change this week?"*

**Insurer / Admin Dashboard (web)**
- Total active policies and weekly premium pool
- Live zone-level disruption heatmap
- Claim volume vs. payout ratio (loss ratio) by zone and week
- Fraud score distribution across recent claims
- Flagged claim queue (soft-hold + human review)
- Predictive disruption risk for next week by zone (IMD forecast feed)
- DBSCAN ring alerts with cluster visualisation

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| Mobile App (Worker) | React Native (Android-first) |
| Backend API | FastAPI (Python) |
| Database | PostgreSQL |
| ML Models | LightGBM, scikit-learn, PyTorch (LSTM) |
| Weather / AQI | OpenWeatherMap, CPCB AQI, IMD |
| News / Traffic | Govt alert feeds, Google Maps Traffic API |
| Payments | Razorpay Test Mode (mock UPI) |
| Admin Dashboard | React (Web) |
| Auth | Firebase Auth |
| Hosting | AWS EC2 / Railway (MVP) |

---

## Development Plan

| Phase | Timeline | Focus |
|-------|----------|-------|
| Phase 1 | Mar 4–20 | Architecture design, ML spec, README, wireframes, repo setup |
| Phase 2 | Mar 21–Apr 4 | Worker registration, policy creation, premium engine (Model 1), trigger pipeline (Model 3), claims flow, payout processing |
| Phase 3 | Apr 5–17 | Fraud detection (Model 4), payout calculator ML layer (Model 2), analytics dashboard, mock payout system, pitch deck, demo video |

> **MVP implementation note:** The architecture describes the full target system. Phase 2 will ship simplified versions where appropriate — LightGBM premium model only (SHAP explainability deferred), rule-based trigger engine only (LSTM added in Phase 3), and signal corroboration + Isolation Forest for fraud (DBSCAN ring detector added in Phase 3). This sequencing de-risks delivery while keeping the architecture extensible.

---

## Business Viability

| Assumption | Value | Rationale |
|---|---|---|
| Target loss ratio | 60–70% | Industry standard for parametric micro-insurance; sustainable at scale |
| Premium pool design | Covers 1-in-3 disruption weeks | Bengaluru/Chennai average ~15–18 disruption days/year across food delivery zones |
| Weekly premium range | ₹39–₹79 | <1% of weekly earnings for a ₹700/day worker — affordable without adverse selection |
| Fraud leakage reduction | Est. 60–75% vs GPS-only systems | 3-layer detection vs single-signal baseline |
| Max payout cap (70% replacement) | Preserves work incentive | Worker earning ₹800/day receives max ₹560/day — working always pays more |
| Deductible (1 hour) | Eliminates micro-claims | Prevents system abuse for short disruptions that don't materially impact income |

The weekly cap of 3 disruption days and the 70% replacement ratio are structural controls against adverse selection and moral hazard respectively — not arbitrary limits. A worker cannot collect more from KamaiKavach in a week than they would earn by working through mild conditions.

---

## Future Developments

**Relational fraud ring detection (GraphSAGE)**  
The current DBSCAN layer detects rings by feature similarity but cannot model relational structure — who referred whom, which accounts share device fingerprint chains. A sophisticated ring where each node looks legitimate in isolation will evade it. Direction: Graph Neural Network (GraphSAGE) on the worker social graph, where edges encode referral relationships, shared device fingerprints, and co-claim timestamps. A fraud ring manifests as a densely connected subgraph detectable at the network level even when individual nodes appear clean.

**Supply-side disruption trigger**  
All current triggers assume the worker cannot work. There is a structurally different failure mode: the worker *is* available, but orders do not exist because the supply side has collapsed. In March 2025, an LPG shortage forced restaurants across several Indian cities to scale back operations — delivery partners reported 25–30% earnings drops with no weather trigger firing. No existing parametric product would have paid out. Direction: a dual-signal trigger — zone-level order volume drop exceeding 35% over 48 hours, corroborated by a verified supply disruption signal (LPG bulletin, restaurant closure rate spike) — with proportional payout scaled to the depth of the volume drop.

---

*KamaiKavach — Built for Ravi. Built for every Ravi.*
