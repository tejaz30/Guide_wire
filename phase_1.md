
---

## Model — Weekly Premium Calculation

### General Description

The premium model answers one question: **how much should a Zomato/Swiggy delivery partner pay per week to be insured against income loss from external disruptions?**

We use a **Gradient Boosted Tree (GBT) regressor with Tweedie loss** — specifically LightGBM. This is the industry-standard approach for insurance pure premium modelling, combining the correct actuarial loss function (Tweedie) with a learner powerful enough to capture the non-linear risk interactions that matter in our problem: a worker in a flood-prone zone during monsoon season working 12-hour shifts faces a risk that is multiplicatively worse, not simply additive. Conventional GLMs cannot model this; GBT does so automatically via tree splits.
---

### Mathematical Formulation

#### The Target: Pure Premium

The training target is the **pure premium** — expected payout per unit of exposure:

$$\text{Pure Premium} = \frac{\mathbb{E}[\text{Total Claims}]}{\text{Exposure (weeks)}} = \mathbb{E}[N] \cdot \mathbb{E}[S \mid N > 0]$$

where $N$ is claim frequency (Poisson-distributed) and $S$ is claim severity (Gamma-distributed).

#### Why Tweedie Loss

Insurance loss distributions are **zero-inflated with a heavy right tail** — most weeks produce no claim, but when a monsoon hits, the payout can be large. The Tweedie family unifies this:

$$\text{Var}(Y) = \phi \cdot \mu^p, \quad p \in (1, 2)$$

- $p = 1$: Poisson (frequency only)
- $p = 2$: Gamma (severity only)
- $p = 1.5$: Tweedie — models the compound Poisson-Gamma, exactly the structure of insurance claims

The log-link ensures the predicted premium is always positive:

$$\log(\hat{\mu}_i) = F(x_i), \quad \hat{\mu}_i > 0 \;\forall\; x_i$$

where $F(x_i)$ is the GBT ensemble output.

#### Tweedie Deviance Loss (minimised during training)

$$\mathcal{L}_{\text{Tweedie}}(y, \hat{\mu}) = 2 \sum_{i=1}^{n} \left[ \frac{y_i \cdot \hat{\mu}_i^{1-p}}{1-p} - \frac{\hat{\mu}_i^{2-p}}{2-p} \right]$$

#### GBT Ensemble

The model learns an additive ensemble of $M$ regression trees:

$$F_M(x) = \sum_{m=1}^{M} \gamma_m h_m(x)$$

where each tree $h_m$ is fit to the **negative gradient** (pseudo-residuals) of the Tweedie deviance from the previous iteration:

$$r_i^{(m)} = -\left[\frac{\partial \mathcal{L}(y_i, F(x_i))}{\partial F(x_i)}\right]_{F = F_{m-1}}$$

#### SHAP Explainability

For a worker with feature vector $x$, the premium decomposition is:

$$\hat{\mu}(x) = \phi_0 + \sum_{j=1}^{p} \phi_j$$

where $\phi_0$ is the base rate and each $\phi_j$ is the SHAP value for feature $j$ — the marginal contribution of that feature to the premium, averaged over all possible feature orderings.

#### Feature Set and Premium Formula (Inference)

Features are grouped by dimension. Critically, **no live weather readings appear here** — only historical aggregates. This is what separates the premium model from the trigger model, which operates exclusively on real-time signals.

**Environmental risk (historical aggregates only)**

| Feature | Symbol | Description |
|---|---|---|
| Zone flood frequency (5yr avg) | $z_f$ | Disruption days per year in the worker's zone, 5-year average from IMD |
| Zone waterlogging score | $z_w$ | Drainage quality index 0–1 from BBMP/municipal records |
| Zone max rainfall P95 | $z_r$ | 95th percentile daily rainfall for zone (mm) — captures tail risk |
| Season disruption rate | $s_d$ | Fraction of weeks in current season historically with ≥1 trigger event, by zone |
| Heat stress days annual | $z_h$ | Days per year exceeding 42°C in the worker's operating zone |

**Worker behaviour & work pattern**

| Feature | Symbol | Description |
|---|---|---|
| Weekly earnings (4wk avg) | $w$ | Worker's average weekly earnings (₹) |
| Earnings coefficient of variation | $w_{cv}$ | CV of weekly earnings over 12 weeks — high CV = irregular worker |
| Hours per day | $h$ | Average daily hours on road |
| Peak hour ratio | $h_p$ | Fraction of hours worked during lunch/dinner rush (12–2pm, 7–10pm) |
| Weekend dependency | $h_w$ | Fraction of weekly earnings from weekends |
| Avg order distance (km) | $d_o$ | Mean delivery distance — longer routes = more outdoor exposure |
| Zone mobility index | $m_z$ | Number of distinct zones operated in per week |

**Platform dynamics**

| Feature | Symbol | Description |
|---|---|---|
| Platform | $p$ | Zomato = 0, Swiggy = 1 |
| Cancellation rate (3m) | $c_r$ | Order cancellation rate — affects platform rating and order allocation |
| Acceptance rate | $a_r$ | Fraction of offered orders accepted |
| Platform surge zone ratio | $p_s$ | Fraction of orders historically in surge-pricing zones |
| Order allocation rank | $r_o$ | Worker's percentile rank in order allocation in their zone |

**Urban infrastructure**

| Feature | Symbol | Description |
|---|---|---|
| Road connectivity score | $u_r$ | Passable alternate routes in zone during flooding (OSM graph) |
| Zone elevation (m) | $u_e$ | Mean elevation of zone — low-lying zones flood first |
| Distance to waterway (km) | $u_w$ | Distance from zone centroid to nearest river/canal |
| Restaurant density (zone) | $u_d$ | Zomato/Swiggy-listed restaurants in zone — low density = fewer pickup options |

**Socioeconomic profile**

| Feature | Symbol | Description |
|---|---|---|
| Tenure (weeks) | $t$ | Weeks on platform — experienced workers navigate disruptions better |
| Prior claims (3m) | $c$ | Number of claims filed in past 90 days |
| Earnings tier | $e_t$ | Quintile of weekly earnings among zone peers |
| Income source diversity | $i_d$ | Binary: other income sources inferred from low-hour weeks |

**Temporal**

| Feature | Symbol | Description |
|---|---|---|
| Season index | $s$ | 0 = normal, 0.6 = summer, 1.0 = monsoon |
| Week of year | $w_y$ | Week 1–52 capturing Diwali, IPL and other volume spikes |

Final weekly premium with floor and cap:

$$\text{Premium}_{\text{weekly}} = \text{clip}\left(e^{F_M(\mathbf{x})},\; ₹15,\; ₹120\right)$$

---

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    PREMIUM MODEL PIPELINE                       │
└─────────────────────────────────────────────────────────────────┘

  INPUTS                    TRAINING                  INFERENCE
  ──────                    ────────                  ─────────

  IMD Weather Data  ──►  Generate Synthetic   ──►  Worker Profile
  (zone risk,             Worker Dataset             (zone, earnings,
   season index)          (100k rows)                hours, claims)
                                │                         │
  Worker Profile  ──────────────┤                         ▼
  (earnings, hours,             ▼                  LightGBM Ensemble
   tenure, claims)      LightGBM Regressor          F_M(x) = Σ γ_m h_m
                         objective='tweedie'               │
                         power=1.5                         ▼
                         n_estimators=300          exp(F_M(x)) → ₹ Premium
                                │                         │
                                ▼                         ▼
                         SHAP TreeExplainer        SHAP Breakdown
                         (post-training)           "Zone: +₹8
                                                    Season: +₹6
                                                    Hours: +₹3"
```

---

### Suitability to the Situation

**1. Non-linear risk interaction capture.**
A Zomato rider in Velachery working 12 hours during monsoon season is not `zone_risk + season_risk + hours_risk`. The zone floods badly during monsoon specifically, and long hours mean more exposure during the worst hours. GBT discovers this via tree splits. A GLM would require manually engineering a `zone × season × hours` interaction term — and then choosing the right functional form. GBT finds it automatically.

**2. SHAP explainability satisfies regulatory and UX requirements.**
IRDAI-compliant insurance products must justify pricing to policyholders. SHAP values decompose the premium into per-feature contributions. In the worker's app, this appears as: *"Your premium this week is ₹47 — ₹8 above base because your zone has a high flood history, ₹6 because it's monsoon season."* This is not possible with a neural network.

**3. Production-viable with synthetic training data.**
No real gig worker insurance dataset exists for India. GBT with 300 trees trained on 100k synthetic rows (seeded from IMD zone-level disruption frequencies) generalises well enough for a pilot. Neural networks would require 500k+ rows to beat GBT on this problem.

---

## Model — Payout Calculation

### General Description

The payout model answers: **when a parametric trigger fires, how much should the worker receive?**

It is a **two-layer system**:

**Layer 1 — Parametric formula (deterministic).** When a trigger crosses threshold (e.g. rainfall > 15mm/hr), the formula runs automatically. No human involvement. This is the core of parametric insurance — zero-touch, instant, auditable.

**Layer 2 — GBT Impact Regressor (ML).** Instead of assuming the full disruption window = full income loss, a GBT regressor predicts the *actual hours lost* from trigger intensity, time of day, zone vulnerability, and worker profile. This eliminates **basis risk** — the core flaw of standard parametric insurance where a worker in a lightly affected zone gets the same payout as one in a heavily flooded zone.

The two layers together give you parametric insurance's speed and automation, plus ML's precision in estimating actual impact.

---

### Mathematical Formulation

#### Layer 1: Parametric Payout Formula

Coverage cap per week:

$$C = w \cdot 0.70$$

where $w$ is the worker's average weekly earnings. (70% replacement ratio prevents moral hazard — a worker should never prefer not working over working.)

Hourly rate insured:

$$r_h = \frac{w}{h_d \cdot 7}$$

where $h_d$ is average daily hours on road.

Tiered gross payout given predicted hours lost $\hat{h}$:

$$G(\hat{h}) = \begin{cases} \hat{h} \cdot r_h & \text{if } \hat{h} \leq 4 \\ 4 \cdot r_h + (\hat{h} - 4) \cdot 0.8 \cdot r_h & \text{if } \hat{h} > 4 \end{cases}$$

The 80% rate beyond 4 hours controls moral hazard for extended events — after half a workday, the marginal payout decreases.

Deductible (first hour is always worker's risk — prevents micro-claims):

$$D = r_h$$

Final net payout (with hard cap):

$$\boxed{P = \min\left(\max(0,\; G(\hat{h}) - D),\; C\right)}$$

#### Layer 2: GBT Hours-Lost Regressor

The ML model predicts $\hat{h}$ — actual hours lost — from trigger event features and worker profile:

$$\hat{h} = F_M(\mathbf{x}_{\text{event}})$$

**Huber loss** is used instead of MSE because some workers push through mild events (outliers in the hours-lost distribution):

$$\mathcal{L}_{\text{Huber}}(y, \hat{y}) = \begin{cases} \frac{1}{2}(y - \hat{y})^2 & \text{if } |y - \hat{y}| \leq \delta \\ \delta \left(|y - \hat{y}| - \frac{\delta}{2}\right) & \text{otherwise} \end{cases}$$

with $\delta = 1.0$ hour (residuals beyond 1 hour are penalised linearly, not quadratically).

GBT ensemble same structure as Model 1:

$$F_M(\mathbf{x}) = \sum_{m=1}^{M} \gamma_m h_m(\mathbf{x})$$

Pseudo-residuals for Huber:

$$r_i^{(m)} = \begin{cases} y_i - F_{m-1}(x_i) & \text{if } |y_i - F_{m-1}(x_i)| \leq \delta \\ \delta \cdot \text{sign}(y_i - F_{m-1}(x_i)) & \text{otherwise} \end{cases}$$

#### Synthetic Target Generation

Since no ground truth hours-lost data exists, we generate it from physical assumptions:

$$y_i = d_i \cdot \alpha(I_i) \cdot \beta(z_i) \cdot \epsilon_i$$

where:
- $d_i$ = trigger event duration (hours), from IMD/ERA5 event records
- $\alpha(I_i)$ = intensity impact factor: $\alpha = \min(1, I / I_{\max})$, where $I$ is rainfall mm/hr or AQI
- $\beta(z_i)$ = zone vulnerability multiplier (0.5–1.0, from historical flood scores)
- $\epsilon_i \sim \text{Beta}(2, 2)$ centred on 1 — adds realistic worker-level variation

#### Feature Set

Features here are split into two groups: **event context** (what happened) and **worker × urban context** (who was affected and where). The trigger model sees only the first two rows of event context — everything else is exclusive to this model.

**Event context (shared with trigger model — intensity only, not thresholds)**

| Feature | Symbol | Description |
|---|---|---|
| Rainfall intensity | $I_r$ | mm/hr at moment of trigger — continuous, not binary |
| AQI | $I_q$ | Air quality index value at trigger time |
| Temperature | $I_t$ | Degrees Celsius at trigger time |
| Event duration so far | $d$ | Hours since trigger threshold was first crossed |

**Worker behaviour at event moment**

| Feature | Symbol | Description |
|---|---|---|
| Peak hour ratio | $h_p$ | Fraction of this worker's usual hours in peak windows — determines income density of lost time |
| Consecutive work days | $c_w$ | Max consecutive days worked in last 4 weeks — financially pressured workers push through more |
| Hours per day std dev | $h_\sigma$ | Variability in daily hours — high variance = harder to predict actual impact |
| Avg order distance (km) | $d_o$ | Longer routes = more total outdoor time lost per hour of disruption |

**Urban infrastructure at event moment**

| Feature | Symbol | Description |
|---|---|---|
| Road connectivity score | $u_r$ | Passable alternate routes — low connectivity = total zone shutdown |
| Road type distribution | $u_t$ | Fraction of zone roads that are unpaved — become impassable faster |
| Public transport density | $u_p$ | Bus/metro stop density — higher = workers can reroute |
| Competitor density (zone) | $u_c$ | Active delivery workers in zone — high density = orders allocated to others first |
| Restaurant density (zone) | $u_d$ | Zomato/Swiggy restaurants in zone — closures reduce available pickups |

**Temporal context**

| Feature | Symbol | Description |
|---|---|---|
| Time of trigger | $\tau$ | Hour 0–23 — lunch/dinner rush disruptions cause 3× more income loss than off-peak |
| Day of week | $\delta_w$ | Weekday vs weekend — order mix and volume differ significantly |
| Is festival week | $f$ | Binary — Pongal, Diwali, Eid weeks have 40–60% higher order volume |
| IPL / cricket match day | $ipl$ | Binary — major match days spike order volume; disruption is costlier |
| Days to month end | $m_e$ | Near month-end workers are less likely to stop even in mild events (rent pressure) |

**Socioeconomic signals**

| Feature | Symbol | Description |
|---|---|---|
| Worker tenure | $t$ | Experienced workers know alternate routes and recover faster |
| Claim-to-disruption ratio | $r_c$ | Past claims / disruption events in zone — ratio > 1.5 flags over-claiming |
| Zone historical avg hours lost | $\bar{h}_z$ | Zone-level mean from past events — strong prior for this event |

---

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                     PAYOUT MODEL PIPELINE                       │
└─────────────────────────────────────────────────────────────────┘

  TRIGGER DETECTION         LAYER 2: ML              LAYER 1: FORMULA
  ─────────────────         ───────────              ────────────────

  IMD API / OpenAQ          GBT Regressor            Parametric Formula
  ──────────────────        (Huber loss)             ─────────────────
  Rainfall > 15mm/hr ──►   F_M(x_event)         ──► C  = w × 0.70
  AQI > 300          ──►   = predicted hours    │    r_h = w / (h_d × 7)
  Temp > 42°C        ──►     lost (e.g. 3.2)    │    G  = tiered formula
  Curfew alert       ──►         │               │    P  = min(G - D, C)
                                 │               │         │
  Worker Location ──────────►    │               │         ▼
  (GPS validation)               │               │    Fraud Check
                                 ▼               │    (Isolation Forest)
  Worker Profile ──────►  Event Features          │         │
  (earnings, hours,       + Worker Features       │         ▼
   zone, tenure)          → hours_lost            │    ₹ Payout via UPI
                                 └───────────────►┘    (within 2 hours)

  BASIS RISK ELIMINATED: Worker in light drizzle → 2.1 hrs predicted
                         Worker in heavy flood   → 6.8 hrs predicted
                         Same trigger, fair payout difference.
```

---

### Suitability to the Situation

**1. Eliminates basis risk — the core flaw of all parametric insurance.**
Standard parametric products pay a fixed amount when a threshold is crossed — regardless of actual impact. A worker in Velachery during a flood alert and a worker in OMR during the same alert face very different disruption realities. Our GBT layer predicts this difference. This is a genuine architectural innovation over conventional parametric insurance.

**2. Preserves zero-touch automation.**
The ML model runs in real time at trigger detection — no human adjuster, no claim form. The worker receives a push notification within minutes of the trigger and money within 2 hours. The ML layer doesn't slow this down; it runs in under 100ms.

**3. Huber loss is the right choice for worker behaviour heterogeneity.**
Some workers push through mild rain because they need the income. Some stop immediately. MSE would over-penalise these outliers and pull the model toward predicting everyone loses the full event window. Huber loss down-weights these extremes, producing a more calibrated central estimate.

**4. The deductible and tiered structure control moral hazard actuarially.**
The 1-hour deductible and 80% rate beyond 4 hours are not arbitrary — they ensure the expected payout is always less than the worker's actual earnings, removing the incentive to prefer a trigger event over working. This is standard actuarial practice adapted for a parametric context.

---

### Comparison with Alternative Models

**Why traditional indemnity claims lose:** The entire value proposition of this product is that a gig worker — who earns ₹600/day and cannot afford a day off — receives money the same day disruption occurs. Any process requiring form submission, documentation, or human review destroys this. Parametric automation is non-negotiable for this persona.

---
---

## Model — Parametric Trigger Detection

### General Description

The trigger model answers one question: **has an external disruption event just occurred that is severe enough to initiate an automatic claim for all active policyholders in the affected zone?**

This is a **multi-source real-time anomaly detection and threshold classification system** built as a two-stage pipeline:

**Stage 1 — Rule-based threshold engine.** Each disruption type has a hard threshold derived from actuarial calibration (e.g. rainfall > 15mm/hr, AQI > 300). When any threshold is crossed, Stage 2 fires immediately.

**Stage 2 — LSTM sequence classifier.** A Long Short-Term Memory network monitors the time-series of environmental signals continuously. Rather than waiting for a single reading to cross a threshold, the LSTM detects the *pattern* that precedes and constitutes a disruption — a rising sequence of rainfall over 90 minutes is a stronger trigger than one spike. It also handles multi-signal events (moderate rain + high AQI + traffic paralysis simultaneously) that individually fall below thresholds but together constitute a disruption.

The two stages are complementary: the threshold engine guarantees no disruption is missed (high recall), the LSTM improves precision by filtering false positives from noisy sensor readings and suppressing duplicate triggers from the same event.

This model is **architecturally separate from Models 1 and 2** — it uses only live real-time signals. It has no knowledge of individual worker profiles, earnings, or behaviour. It simply answers: did a trigger event happen in zone $z$ at time $t$?

---

### Mathematical Formulation

#### Stage 1: Threshold Classification

For each disruption type $k$, define a binary trigger indicator:

$$T_k(t, z) = \mathbf{1}\left[x_k(t, z) \geq \theta_k\right]$$

where $x_k(t, z)$ is the live sensor reading for signal $k$ in zone $z$ at time $t$, and $\theta_k$ is the calibrated threshold.

The zone-level trigger fires if any type crosses threshold:

$$T(t, z) = \max_k \; T_k(t, z) = \mathbf{1}\left[\exists\, k : x_k(t,z) \geq \theta_k\right]$$

**Calibrated thresholds** (derived from IMD disruption impact data):

| Trigger type | Signal $x_k$ | Threshold $\theta_k$ | Rationale |
|---|---|---|---|
| Heavy rain | Rainfall mm/hr | 15 mm/hr | IMD "heavy rain" classification; delivery platforms halt orders |
| Flood alert | IMD alert level | Level 3 (Red) | District-level Red alert from IMD public API |
| Severe pollution | AQI (PM2.5) | 300 (Severe) | CPCB "Severe" category; outdoor activity discouraged |
| Extreme heat | Temperature °C | 42°C | Wet-bulb safety threshold for outdoor workers |
| Curfew / Section 144 | Authority alert | Active = 1 | Binary from government API / news feed |
| Platform downtime | API response code | Downtime > 2hrs | Mock platform API health check |

#### Stage 2: LSTM Sequence Classifier

The LSTM takes a sliding window of $L = 12$ time steps (each 15 minutes = 3-hour window) of multi-signal readings:

$$\mathbf{X}_t = \left[ \mathbf{x}_{t-L+1}, \mathbf{x}_{t-L+2}, \ldots, \mathbf{x}_t \right] \in \mathbb{R}^{L \times d}$$

where $d$ is the number of input signals (see feature set below).

The LSTM hidden state update at each step $\tau$:

$$\mathbf{f}_\tau = \sigma\left(\mathbf{W}_f \mathbf{x}_\tau + \mathbf{U}_f \mathbf{h}_{\tau-1} + \mathbf{b}_f\right) \quad \text{(forget gate)}$$

$$\mathbf{i}_\tau = \sigma\left(\mathbf{W}_i \mathbf{x}_\tau + \mathbf{U}_i \mathbf{h}_{\tau-1} + \mathbf{b}_i\right) \quad \text{(input gate)}$$

$$\tilde{\mathbf{c}}_\tau = \tanh\left(\mathbf{W}_c \mathbf{x}_\tau + \mathbf{U}_c \mathbf{h}_{\tau-1} + \mathbf{b}_c\right) \quad \text{(candidate cell)}$$

$$\mathbf{c}_\tau = \mathbf{f}_\tau \odot \mathbf{c}_{\tau-1} + \mathbf{i}_\tau \odot \tilde{\mathbf{c}}_\tau \quad \text{(cell state)}$$

$$\mathbf{o}_\tau = \sigma\left(\mathbf{W}_o \mathbf{x}_\tau + \mathbf{U}_o \mathbf{h}_{\tau-1} + \mathbf{b}_o\right) \quad \text{(output gate)}$$

$$\mathbf{h}_\tau = \mathbf{o}_\tau \odot \tanh(\mathbf{c}_\tau) \quad \text{(hidden state)}$$

The final hidden state $\mathbf{h}_t$ is passed through a dense layer with sigmoid activation:

$$\hat{p}(t, z) = \sigma\left(\mathbf{w}^\top \mathbf{h}_t + b\right) \in [0, 1]$$

This is the **trigger probability** — the model's confidence that zone $z$ is experiencing a genuine disruption event at time $t$.

#### Combined Trigger Decision

The final trigger fires when either the threshold engine fires OR the LSTM confidence exceeds a tuned threshold $\alpha$:

$$\text{TRIGGER}(t, z) = T(t, z) \;\lor\; \left[\hat{p}(t, z) \geq \alpha\right]$$

where $\alpha = 0.75$ (tuned for high precision — we do not want false payouts).

#### Loss Function

Binary cross-entropy with class weighting to handle imbalance (disruption events are rare):

$$\mathcal{L}_{\text{BCE}} = -\frac{1}{n}\sum_{i=1}^{n} \left[ w_+ y_i \log \hat{p}_i + w_- (1-y_i)\log(1-\hat{p}_i) \right]$$

where $w_+ = \frac{n}{2 \cdot n_+}$ and $w_- = \frac{n}{2 \cdot n_-}$ are inverse-frequency class weights. Disruption events are rare (~15–20% of 15-minute windows in high-risk zones during monsoon), so $w_+ > w_-$.

#### Trigger Cooldown (Deduplication)

Once a trigger fires for zone $z$, a cooldown of $\Delta t = 4$ hours is applied before the next trigger can fire for the same zone. This prevents multiple payouts from the same continuous event:

$$\text{TRIGGER\_VALID}(t, z) = \text{TRIGGER}(t, z) \;\land\; \left(t - t_{\text{last}}^z > \Delta t\right)$$

---

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                  TRIGGER MODEL PIPELINE                         │
└─────────────────────────────────────────────────────────────────┘

  DATA INGESTION          STAGE 1                    STAGE 2
  ──────────────          ───────                    ───────

  IMD API  ──────────►  Threshold Engine         LSTM Classifier
  (rainfall, temp,      ─────────────────        ───────────────
   flood alerts)        Rain > 15mm/hr?  ──YES──►
                        AQI  > 300?      ──YES──►  Sliding window
  OpenAQ API ────────►  Temp > 42°C?    ──YES──►  X_t = [x_{t-11}
  (PM2.5, AQI)          Curfew active?  ──YES──►         ...
                        Platform down?  ──YES──►         x_t]
  Govt alert feed ───►            │                      │
                                  │ ANY YES               ▼
  Platform health ───►            │              LSTM hidden state h_t
  check API                       │                      │
                                  │                      ▼
                                  │              p̂(t,z) = σ(w·h_t + b)
                                  │                      │
                                  └──────────────────────┤
                                                         ▼
                                              TRIGGER(t,z) = T(t,z) OR p̂ ≥ 0.75
                                                         │
                                                         ▼
                                              Cooldown check (4hr window)
                                                         │
                                              ┌──────────┴──────────┐
                                              │                     │
                                           SUPPRESS             FIRE TRIGGER
                                        (same event)                │
                                                                     ▼
                                                         Identify all active
                                                         policyholders in zone z
                                                                     │
                                                                     ▼
                                                         Pass to Payout Model (M2)
```

---

### Suitability to the Situation

**1. The two-stage design balances recall and precision.**
For a parametric insurance product, a missed trigger (false negative) means a worker who deserved a payout gets nothing — a severe trust failure. A false positive means an unnecessary payout — a financial loss. The threshold engine provides a hard guarantee on recall: any reading above calibrated limits always fires. The LSTM layer then reduces false positives from sensor noise, spike readings, and sub-threshold multi-signal events. Neither stage alone is sufficient.

**2. LSTM captures the temporal signature of real disruption events.**
A single 15-minute rainfall reading of 16mm/hr could be a sensor glitch. A 90-minute sequence of [8, 11, 14, 18, 22, 19mm/hr] is unambiguously a developing storm. The LSTM's cell state $\mathbf{c}_\tau$ retains memory across the window, allowing it to distinguish genuine events from noise — something a threshold check on a single reading cannot do.

**3. Multi-signal fusion detects compound disruptions.**
Chennai's worst delivery disruptions often occur when moderate rain (12mm/hr, just below threshold) coincides with high AQI (260, just below threshold) and traffic signals failing. No single threshold fires. The LSTM sees all signals simultaneously across the window and can learn that this combination constitutes a real disruption — a critical capability for a multi-trigger parametric product.

**4. The cooldown mechanism prevents event fragmentation into multiple payouts.**
A 6-hour monsoon event should produce one payout, not 24 sequential trigger fires (one per 15-minute window). The 4-hour cooldown per zone ensures event continuity is respected. This is essential for pool solvency — without it, a single storm could exhaust the weekly premium pool.

---

### Feature Set

This model uses **only live real-time signals**. No historical aggregates, no worker profile features, no socioeconomic signals. This is the defining constraint that separates it from Models 1 and 2.

**Live environmental signals (15-minute cadence per zone)**

| Feature | Symbol | Source | Notes |
|---|---|---|---|
| Rainfall mm/hr | $x_r$ | IMD API / OpenWeatherMap | Primary heavy-rain trigger signal |
| Rainfall 3hr cumulative | $x_{r3}$ | Derived from API | Captures sustained vs spike rain |
| Temperature °C | $x_t$ | IMD API | Extreme heat trigger |
| Heat index (feels-like) | $x_{hi}$ | Derived: temp + humidity | More accurate than raw temperature for worker safety |
| PM2.5 (AQI) | $x_q$ | OpenAQ / CPCB API | Pollution trigger |
| PM10 | $x_{q2}$ | OpenAQ / CPCB API | Secondary pollution signal |
| Wind speed km/hr | $x_w$ | IMD API | High wind amplifies rain impact |
| Humidity % | $x_h$ | IMD API | Input to heat index; also affects delivery viability |
| Visibility km | $x_v$ | IMD API | Low visibility directly impairs two-wheeler delivery |

**Derived temporal signals (window-level features for LSTM)**

| Feature | Symbol | Description |
|---|---|---|
| Rainfall rate of change | $\Delta x_r$ | $x_r(t) - x_r(t-1)$ — rising vs falling intensity |
| AQI rate of change | $\Delta x_q$ | Rising AQI trend is more dangerous than static high value |
| Signals-above-50pct-threshold | $n_{50}$ | Count of signals currently above 50% of their trigger threshold — compound event indicator |
| Minutes since last rain | $\tau_r$ | Gap since last non-zero rainfall reading |

**Event / alert signals (binary)**

| Feature | Symbol | Source |
|---|---|---|
| IMD flood alert active (zone) | $a_f$ | IMD district alert API |
| Section 144 / curfew active | $a_c$ | Government press release feed / mock |
| Platform API health status | $a_p$ | Mock platform health endpoint |
| Traffic incident alert (zone) | $a_t$ | Google Maps / HERE Traffic API (mock) |

---

---

## Combined Runtime Architecture

Three models, three time horizons, one inference stack.

```
──────────────────────────────────────────────────────────────────────
FEATURE SEPARATION SUMMARY
──────────────────────────────────────────────────────────────────────

  Model 1 (Premium)   → Historical aggregates only. No live signals.
  Model 2 (Payout)    → Event intensity + worker profile + urban context.
  Model 3 (Trigger)   → Live real-time signals only. No worker data.

  Zero overlap between trigger features and premium features.
  Payout model receives trigger intensity but adds 20+ features
  the trigger model has no concept of.
```

---
---

## Model — Individual Risk Scoring (Underwriting)

### General Description

The underwriting model answers the question the insurer asks **before** selling a policy: *what is the probability that this specific delivery worker will file at least one claim in the next 4 weeks?*

This is distinct from Model 1 (premium pricing) in a precise way. Model 1 outputs a **rupee amount** — the price. Model 4 outputs a **probability** — the underlying risk score that justifies that price. In a mature system, the premium is a function of the risk score: a worker with P(claim) = 0.40 commands a higher premium than one with P(claim) = 0.08. Separating them means the insurer can also use the risk score independently — to decide whether to offer coverage at all, to set deductibles, or to flag workers who should not be auto-approved.

We use a **calibrated Gradient Boosted Classifier with Platt scaling** — specifically LightGBM with `objective='binary'`, followed by isotonic regression calibration to ensure the output is a true probability, not just a ranking score. Raw GBT outputs are well-ordered but poorly calibrated: a score of 0.7 does not mean 70% probability without calibration. For an insurance underwriting decision, a calibrated probability is legally and actuarially required.

---

### Mathematical Formulation

#### The Target: Individual Claim Probability

$$\hat{p}_i = P(\text{claim}_i = 1 \mid \mathbf{x}_i)$$

where $\mathbf{x}_i$ is the full feature vector for worker $i$ and $\hat{p}_i \in [0, 1]$ is the calibrated probability of filing at least one claim in the next 4 weeks.

#### Binary Cross-Entropy Loss

$$\mathcal{L}_{\text{BCE}} = -\frac{1}{n} \sum_{i=1}^{n} \left[ y_i \log \hat{p}_i + (1 - y_i) \log(1 - \hat{p}_i) \right]$$

With class weighting for imbalance. Let $n_+$ = number of claimants, $n_-$ = non-claimants:

$$w_+ = \frac{n}{2 \cdot n_+}, \quad w_- = \frac{n}{2 \cdot n_-}$$

The weighted loss becomes:

$$\mathcal{L}_{\text{weighted}} = -\frac{1}{n} \sum_{i=1}^{n} \left[ w_{y_i} \cdot y_i \log \hat{p}_i + w_{y_i} \cdot (1 - y_i) \log(1 - \hat{p}_i) \right]$$

#### GBT Ensemble (Binary Classification)

Same additive ensemble structure as Models 1–3:

$$F_M(\mathbf{x}) = \sum_{m=1}^{M} \gamma_m h_m(\mathbf{x})$$

The raw score is converted to a probability via the logistic sigmoid:

$$\hat{p}_{\text{raw}} = \sigma(F_M(\mathbf{x})) = \frac{1}{1 + e^{-F_M(\mathbf{x})}}$$

#### Platt Scaling (Probability Calibration)

Raw GBT probabilities are monotone but not calibrated — a score of 0.7 may correspond to a true empirical frequency of only 0.45. Platt scaling fits a logistic regression on the raw scores using a held-out calibration set:

$$\hat{p}_{\text{calibrated}} = \sigma(a \cdot F_M(\mathbf{x}) + b) = \frac{1}{1 + e^{-(a F_M(\mathbf{x}) + b)}}$$

where $a$ and $b$ are learned on the calibration fold to minimise the Brier score:

$$\text{Brier} = \frac{1}{n} \sum_{i=1}^{n} (\hat{p}_i - y_i)^2$$

A well-calibrated model has a reliability diagram (calibration curve) where predicted probabilities match empirical frequencies — if it says 0.30 for 100 workers, approximately 30 of them should actually claim.

#### Risk Tier Assignment

After calibration, workers are bucketed into underwriting tiers:

$$\text{Tier}(i) = \begin{cases} \text{Standard} & \hat{p}_i < 0.20 \\ \text{Elevated} & 0.20 \leq \hat{p}_i < 0.40 \\ \text{High} & 0.40 \leq \hat{p}_i < 0.60 \\ \text{Decline / surcharge} & \hat{p}_i \geq 0.60 \end{cases}$$

The premium from Model 1 is then modulated by the tier:

$$\text{Premium}_{\text{final}} = \text{Premium}_{\text{M1}} \times \text{TierMultiplier}(\hat{p}_i)$$

where TierMultiplier ∈ {0.90, 1.00, 1.20, 1.45} for Standard through High tiers.

#### SHAP for Underwriting Explainability

Each underwriting decision is explained via SHAP:

$$\hat{p}_{\text{calibrated}}(\mathbf{x}) = \text{base\_rate} + \sum_{j=1}^{d} \phi_j$$

where $\phi_j$ is the SHAP contribution of feature $j$ to the deviation from the population base claim rate. Required for regulatory adverse action notices.

---

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│              UNDERWRITING MODEL PIPELINE (MODEL 4)              │
└─────────────────────────────────────────────────────────────────┘

  INPUTS                TRAINING                    INFERENCE
  ──────                ────────                    ─────────

  Worker onboarding     Synthetic worker histories  New worker applies
  data (profile,        y=1 (claimed),              for policy
  platform history)     y=0 (no claim)                   │
         │                     │                         ▼
         │                     ▼                  Feature vector x_i
         │            LightGBM Classifier               │
         │            objective='binary'                 ▼
         │            scale_pos_weight=4.0      F_M(x) → σ(F_M(x))
         │                     │                = p_raw ∈ [0,1]
         │                     ▼                         │
         │            Platt Scaling                      ▼
         │            (isotonic regression          Platt Scaling
         │             on calibration fold)         a·F_M(x) + b
         │                     │                = p_calibrated
         │                     ▼                         │
         │            Brier score validation             ▼
         │            Calibration curve check    Risk Tier Assignment
         │                                       p < 0.20 → Standard
         │                                       p < 0.40 → Elevated
         │                                       p < 0.60 → High
         │                                       p ≥ 0.60 → Decline
         │                                                │
         └────────────────────────────────────────────────┤
                                                          ▼
                                                 Premium M1 × TierMultiplier
                                                 SHAP breakdown stored
                                                 Policy issued or declined
```

---

### Suitability to the Situation

**1. Calibrated probabilities are actuarially and legally required.**
An uncalibrated model that outputs a score of 0.72 is meaningless for pricing. The insurer needs to know that score corresponds to a 72% (or 45%, or 31%) empirical claim rate. Platt scaling ensures the output is a true probability, enabling the Tier × Multiplier pricing formula above. This is not optional for a product seeking IRDAI compliance.

**2. The underwriting model captures worker-level heterogeneity that the premium model averages out.**
Model 1 prices based on zone and season — two workers in the same zone in the same season pay the same base premium. Model 4 differentiates them: a veteran rider with 3 years of tenure and zero claim history in Adyar gets Standard tier (×0.90 multiplier), while a newly registered worker with an unverified zone history and two claims in 90 days in the same zone gets High tier (×1.20 multiplier). The premium model sets the floor; the underwriting model personalises it.

---

### Feature Set

The underwriting model uses the **richest feature set of all five models** because it must capture every dimension of individual risk — not just weather exposure, but platform behaviour, work patterns, financial pressure signals, and historical claim behaviour.

**Historical claim behaviour**

| Feature | Symbol | Description |
|---|---|---|
| Prior claims (3m) | $c_{3m}$ | Claims filed in past 90 days — strongest single predictor |
| Prior claims (12m) | $c_{12m}$ | Longer-window claim history — separates serial claimants from one-off events |
| Claim-to-disruption ratio | $r_c$ | Claims filed / disruption events in zone — ratio > 1.0 signals over-claiming |
| Days since last claim | $d_c$ | Recency — recent claimants have higher near-term probability |
| Claim severity history | $s_c$ | Average payout per claim — high severity claims are riskier to underwrite |

**Work pattern & exposure**

| Feature | Symbol | Description |
|---|---|---|
| Weekly earnings CV | $w_{cv}$ | Earnings volatility — high CV workers have more financial pressure to claim |
| Hours per day | $h$ | More hours = more exposure events per week |
| Peak hour ratio | $h_p$ | Concentration in high-volume windows = higher income at risk |
| Weekend dependency | $h_w$ | Weekend-heavy income = more volatile exposure |
| Avg order distance (km) | $d_o$ | Longer routes = more time in outdoor disruption conditions |
| Zone mobility index | $m_z$ | Single-zone workers are more disruption-concentrated |
| Consecutive work days | $c_w$ | Financial pressure indicator — unable to take voluntary rest days |

**Platform behaviour signals**

| Feature | Symbol | Description |
|---|---|---|
| Tenure (weeks) | $t$ | Experienced workers file fewer spurious claims |
| Cancellation rate (3m) | $c_r$ | High cancellation = lower platform standing = more financial pressure |
| Acceptance rate | $a_r$ | Low acceptance = selective worker = different risk profile |
| Order allocation rank | $r_o$ | Low-ranked workers get fewer orders — higher income volatility |
| Platform (Zomato / Swiggy) | $p$ | Operational and earnings structure differs between platforms |
| Account age vs tenure gap | $g_t$ | Account registered significantly before first order = possible dormant fraud prep |

**Zone & environmental exposure**

| Feature | Symbol | Description |
|---|---|---|
| Zone flood frequency (5yr) | $z_f$ | Historical disruption rate for worker's primary operating zone |
| Zone waterlogging score | $z_w$ | Drainage quality — structural flood risk |
| Zone elevation (m) | $u_e$ | Low elevation = first to flood |
| Season disruption rate | $s_d$ | Fraction of weeks in current season with historical triggers |
| Heat stress days annual | $z_h$ | Heat exposure days — relevant for extreme heat claims |

**Socioeconomic & onboarding signals**

| Feature | Symbol | Description |
|---|---|---|
| Earnings tier | $e_t$ | Quintile among zone peers — low earners have higher claim propensity |
| Income source diversity | $i_d$ | Single-income workers claim more aggressively |
| Onboarding zone match | $o_z$ | Does the worker's registered address match their operating zone? Mismatch = risk flag |
| KYC completeness score | $k$ | Fraction of identity verification steps completed — incomplete = elevated risk |
| Referral source | $r_s$ | How the worker was onboarded — direct vs referral vs bulk registration |

---
---

## Model — Fraud Detection (Multi-Signal Corroboration)

### General Description

#### The Threat: GPS Spoofing Ring

The fraud scenario this model is designed to defeat is the following coordinated attack:

> **500 delivery workers organised over a Telegram channel download a ₹0 GPS spoofing app. The app broadcasts false coordinates to the insurance platform, placing the worker's phone in a Red-alert weather zone. Since the parametric trigger checks GPS location to confirm the worker is in the disruption zone, 500 simultaneous spoofed locations trigger 500 automatic payouts — draining the liquidity pool in a single event.**

This attack works because the trigger model relies on a **single corroborating signal** (GPS location). GPS is trivially fakeable with a free app. The attacker does not need to understand the ML system — they just need to know that GPS in Zone X → payout.

The key insight the fraud model is built on: **a fraudster can spoof one signal, but spoofing five independent signals simultaneously — with all of them physically consistent with each other — is exponentially harder.** A genuine stranded worker has a physical and behavioural signature that a phone at home cannot replicate. The fraud detection architecture shifts from single-signal trust to **multi-signal corroboration**.

#### Model Architecture

The fraud detection system is a **three-layer pipeline**:

**Layer 1 — Signal corroboration engine (rule-based).** Five independent signals are checked for physical consistency. Each produces a binary or continuous corroboration score. No single signal can approve or reject — all five must be considered together.

**Layer 2 — Isolation Forest anomaly detector.** Trained on the feature vectors of known-legitimate claims. At inference time, claims that deviate from the normal distribution of genuine claims receive a high anomaly score. This catches individual fraud attempts that pass Layer 1's rules but are statistically unusual.

**Layer 3 — Clustering-based ring detector (DBSCAN).** Monitors for coordinated fraud by analysing claim patterns across all workers in real time. A genuine disruption produces a geographically distributed, temporally gradual claim pattern. A Telegram-coordinated ring produces an unnatural spike — many claims from the same zone in a narrow time window, from accounts with similar registration dates and no prior activity in that zone.

The output of all three layers feeds a **tiered confidence score** that determines whether the payout is instant, delayed pending verification, soft-held for review, or frozen for human investigation — without ever showing the word "fraud" to a genuine worker.

---

### Mathematical Formulation

#### Layer 1: Multi-Signal Corroboration Score

For a claim by worker $i$ at time $t$ in zone $z$, define five binary corroboration indicators:

$$c_1 = \mathbf{1}[\text{cell tower triangulation confirms zone } z]$$
$$c_2 = \mathbf{1}[\text{WiFi BSSID is not worker's registered home network}]$$
$$c_3 = \mathbf{1}[\text{accelerometer variance} > \theta_{\text{accel}}] \quad \text{(device is not stationary on a surface)}$$
$$c_4 = \mathbf{1}[\text{platform API shows login + order activity in zone } z \text{ within 2hr of trigger}]$$
$$c_5 = \mathbf{1}[\text{worker's historical heatmap includes zone } z \text{ in past 30 days}]$$

The composite corroboration score with learned weights $\mathbf{w}$:

$$S_{\text{corr}}(i) = \sum_{k=1}^{5} w_k \cdot c_k, \quad \sum_k w_k = 1$$

Weights are calibrated on historical labelled fraud data. Initial values: $w = [0.25, 0.20, 0.20, 0.20, 0.15]$ (cell tower most reliable, historical heatmap least reliable for new workers).

#### Layer 2: Isolation Forest Anomaly Score

The Isolation Forest assigns an anomaly score $s_{\text{iso}} \in [-1, 1]$ to each claim's feature vector, where values near $-1$ indicate anomalies:

$$s_{\text{iso}}(i) = 2^{-\frac{\mathbb{E}[h(\mathbf{x}_i)]}{c(n)}}$$

where $h(\mathbf{x}_i)$ is the average path length to isolate observation $\mathbf{x}_i$ across all trees, and $c(n) = 2H(n-1) - \frac{2(n-1)}{n}$ is the expected path length for a dataset of size $n$ (with $H$ the harmonic number).

The intuition: anomalous points (fraud) are isolated early — they fall in sparse regions of feature space and require fewer splits. Genuine claims cluster together and require many splits to isolate.

The Isolation Forest is trained **on legitimate claims only** — no fraud labels needed. This is critical because fraud patterns evolve; a model trained on known fraud patterns will miss novel attack vectors.

#### Layer 3: DBSCAN Coordinated Ring Detection

For each active trigger event, extract all simultaneous claims within a 30-minute window:

$$\mathcal{C}(t, z) = \{i : \text{claim filed in zone } z, |t_i - t| < 30\text{ min}\}$$

Apply DBSCAN clustering on claim feature vectors (registration age, historical zone activity, device fingerprint similarity):

$$\text{DBSCAN}(\mathcal{C}, \epsilon, \text{minPts})$$

where $\epsilon$ is the neighbourhood radius in feature space and $\text{minPts} = 5$ (a cluster of 5 or more similar accounts filing simultaneously is a ring signal).

Ring anomaly score for a cluster $\mathcal{K}$:

$$R(\mathcal{K}) = \frac{|\mathcal{K}|}{|\mathcal{C}(t,z)|} \cdot \mathbf{1}\left[\bar{t}_{\text{reg}}(\mathcal{K}) < 30 \text{ days}\right] \cdot \mathbf{1}\left[\bar{h}_{\text{zone}}(\mathcal{K}) < 0.1\right]$$

where:
- $|\mathcal{K}| / |\mathcal{C}|$ is the fraction of simultaneous claims in the cluster
- $\bar{t}_{\text{reg}}$ is mean account age of cluster members — new accounts are suspicious
- $\bar{h}_{\text{zone}}$ is mean historical activity frequency in zone $z$ — workers new to this zone are suspicious

$R(\mathcal{K}) > 0.4$ flags the entire cluster for human review.

#### Composite Fraud Score and Tiered Decision

Combining all three layers into a final fraud confidence score:

$$F_{\text{fraud}}(i) = 1 - \left[\alpha \cdot S_{\text{corr}}(i) + \beta \cdot \frac{1 + s_{\text{iso}}(i)}{2} + \gamma \cdot (1 - R(\mathcal{K}_i))\right]$$

where $\alpha + \beta + \gamma = 1$, default weights $\alpha = 0.50, \beta = 0.30, \gamma = 0.20$.

Higher $F_{\text{fraud}}$ = higher fraud probability. Tiered decision:

$$\text{Decision}(i) = \begin{cases} \text{Instant payout} & F_{\text{fraud}} < 0.20 \\ \text{Auto-payout, 2hr watch} & 0.20 \leq F_{\text{fraud}} < 0.45 \\ \text{Soft hold, 4hr review} & 0.45 \leq F_{\text{fraud}} < 0.70 \\ \text{Freeze, human review} & F_{\text{fraud}} \geq 0.70 \end{cases}$$

The worker-facing language for each tier — critically, no tier ever uses the word "fraud" or "rejected":

$$\text{UX message}(i) = \begin{cases} \text{"Payout of ₹225 sent to your UPI"} & \text{Instant} \\ \text{"Payout processing — arrives by [time]"} & \text{2hr watch} \\ \text{"Your claim is being verified — update within 4 hours"} & \text{Soft hold} \\ \text{"Verification in progress — our team will contact you"} & \text{Freeze} \end{cases}$$

---

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│            FRAUD DETECTION PIPELINE (MODEL 5)                   │
│            Threat: GPS Spoofing Ring via Telegram               │
└─────────────────────────────────────────────────────────────────┘

  CLAIM ARRIVES (from Model 3 trigger + Model 2 payout calculation)
          │
          ▼
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 1: Multi-Signal Corroboration Engine                     │
│                                                                 │
│  GPS location  ──────────────►  c1: Cell tower confirms zone?  │
│  (spoofable)                                                    │
│                                 c2: WiFi ≠ home network?       │
│  Device sensors ─────────────►  c3: Accelerometer moving?      │
│                                                                 │
│  Platform API ───────────────►  c4: Active orders before event?│
│                                                                 │
│  Worker history ─────────────►  c5: Works in this zone usually?│
│                                      │                         │
│  S_corr = Σ w_k · c_k ◄─────────────┘                         │
└─────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 2: Isolation Forest (Individual Anomaly)                 │
│                                                                 │
│  Trained on legitimate claims only (no fraud labels needed)     │
│                                                                 │
│  Feature vector x_i → avg path length h(x_i)                  │
│  s_iso = 2^(-E[h(x_i)] / c(n))                                 │
│                                                                 │
│  Short path = isolated = anomalous = high fraud signal          │
└─────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 3: DBSCAN Ring Detector (Coordinated Fraud)              │
│                                                                 │
│  All claims in zone z within 30-min window → C(t,z)            │
│                                                                 │
│  DBSCAN clusters by: account age, zone history,                 │
│  device fingerprint, registration source                        │
│                                                                 │
│  Cluster of 5+ new accounts, no zone history → R(K) > 0.4     │
│  → ENTIRE BATCH flagged, but honest workers in C(t,z)          │
│    with full corroboration (S_corr = 1.0) still auto-paid      │
└─────────────────────────────────────────────────────────────────┘
          │
          ▼
  F_fraud = 1 - [0.50·S_corr + 0.30·(1+s_iso)/2 + 0.20·(1-R)]
          │
          ▼
  ┌───────────────────────────────────────────┐
  │           TIERED DECISION ENGINE          │
  │                                           │
  │  F < 0.20  ──► Instant payout            │
  │  F < 0.45  ──► Auto-payout, 2hr watch    │
  │  F < 0.70  ──► Soft hold, 4hr review     │
  │  F ≥ 0.70  ──► Freeze, human reviewer    │
  │                                           │
  │  Worker NEVER sees "fraud" or "rejected"  │
  └───────────────────────────────────────────┘

  HONEST WORKER PROTECTION:
  Even if ring R(K) > 0.4 flags the whole batch,
  individual workers with S_corr = 1.0 (all 5 signals
  corroborated) bypass the ring flag and receive
  instant payout. Ring detection penalises the cluster,
  not the individual.
```

---

### Suitability to the Situation

**1. The GPS spoofing attack is defeated by the 5-signal corroboration design.**
A fraudster at home spoofing GPS will simultaneously fail at least three of the five corroboration checks: their cell tower triangulation will show a residential area, not the claimed disruption zone; their WiFi will match their registered home SSID; and their accelerometer will show couch-level stillness. Defeating all five checks simultaneously requires physical presence in the disruption zone — which is exactly the condition that justifies a payout. The architecture makes spoofing self-defeating.

**2. DBSCAN detects the Telegram ring specifically.**
The defining signature of a coordinated ring is a cluster of accounts filing simultaneously from the same zone, with similar registration dates and no prior activity in that zone. DBSCAN finds exactly this structure without needing labelled ring data. The 30-minute window and minPts=5 parameters are tuned to distinguish a genuine disruption (which produces a gradual, geographically dispersed claim arrival pattern) from a coordinated ring (which produces a spike).

**3. The UX language design prevents trust erosion.**
Genuine workers whose claims fall into "Soft hold" due to network issues or sensor anomalies must not feel accused. The tiered UX messages frame all non-instant decisions as verification delays, not fraud suspicion. This is critical for the product's retention: a single "your claim was rejected" message to an honest worker in a storm generates negative Telegram buzz faster than any marketing campaign can counteract.

---

### Feature Set

The fraud model uses the most heterogeneous feature set of all five models — it must capture physical device signals, behavioural patterns, network context, and population-level clustering signals simultaneously.

**Signal corroboration features (Layer 1)**

| Feature | Symbol | Source | What it detects |
|---|---|---|---|
| Cell tower zone match | $c_1$ | Telecom triangulation (mock) | Independent location verification — cannot be spoofed by GPS app |
| WiFi BSSID home match | $c_2$ | Device network info | Home WiFi connected = worker is at home |
| WiFi BSSID type | $c_2'$ | Device network info | Connected to unknown outdoor / commercial network = genuine outdoor presence |
| Accelerometer variance (5min) | $c_3$ | Device IMU | Near-zero variance = stationary on surface (couch/desk), not on a moving bike |
| Gyroscope tilt pattern | $c_3'$ | Device IMU | Phone lying flat vs handlebar-mounted tilt pattern |
| Battery charging status | $b$ | Device API | Charging at home vs discharging outdoors — subtle but consistent signal |
| Battery drain rate | $b'$ | Device API | Outdoor heat accelerates drain; home environment drain is slower |
| Platform login before event | $c_4$ | Mock platform API | Did the worker log in and accept orders before the disruption? |
| Last order timestamp | $c_4'$ | Mock platform API | Time since last order accepted — genuine workers have recent activity |
| Historical zone heatmap score | $c_5$ | Internal — past GPS logs | Fraction of past 30 days with activity in claimed zone |

**Claim behaviour features (Layer 2 — Isolation Forest inputs)**

| Feature | Symbol | Description |
|---|---|---|
| Claim-to-disruption ratio (3m) | $r_c$ | Claims / disruption events in zone — ratio > 1.5 is anomalous |
| Time-to-claim (minutes) | $t_c$ | How quickly after trigger did claim arrive — instant filing is suspicious |
| Claimed hours vs predicted hours | $\Delta h$ | Claimed hours lost vs Model 2's GBT prediction — large gap = anomaly |
| Prior claim velocity | $v_c$ | Claims per week over past 90 days — sudden spike is anomalous |
| Payout amount vs earnings ratio | $r_p$ | Claimed payout / weekly earnings — suspiciously high ratios flagged |
| Zone × time consistency | $z_t$ | Has this worker claimed this zone + this time-of-day combination before? |
| Device fingerprint consistency | $d_f$ | Same device UUID as prior claims — new device on claim day is suspicious |
| IP geolocation vs GPS | $\Delta_{ip}$ | Distance between IP-geolocated city and claimed GPS zone |

**Population-level ring detection features (Layer 3 — DBSCAN inputs)**

| Feature | Symbol | Description |
|---|---|---|
| Account registration age (days) | $a_r$ | New accounts cluster in fraud rings |
| Days since first order | $a_o$ | Dormant then suddenly active accounts |
| Zone activity history score | $h_z$ | Mean prior activity in claimed zone — genuine workers have history |
| Device model distribution | $d_m$ | Ring members often use same device model (bulk-purchased phones) |
| Referral chain depth | $r_d$ | Workers referred by the same account chain — network of linked accounts |
| Claim timestamp delta (batch) | $\Delta t_b$ | Time between consecutive claims in same zone — ring members file within seconds |
| Registration source | $r_s$ | Bulk registration via API vs organic onboarding |

---

### Comparison with Alternative Approaches

**Why a supervised classifier loses despite its simplicity:** Fraud labels arrive weeks after a fraud event is confirmed — legal and investigation processes take time. By the time the label exists, the attacker has changed tactics. A supervised model trained on last month's fraud patterns will miss this month's novel attack. The Isolation Forest trained on legitimate claims automatically flags anything that doesn't look like genuine behaviour, including attack vectors never seen before.
---

## Future Developments & Open Problems

### F1 — Relational Fraud Ring Detection

**Problem:** The current DBSCAN layer detects coordinated fraud rings by clustering on feature similarity. It cannot model the *relational structure* of a ring — who referred whom, which accounts share device fingerprint chains, and how payout proceeds flow across linked accounts. A sophisticated ring where each individual node looks legitimate in isolation will evade it.

**Direction:** Graph Neural Network (GraphSAGE) on the worker social graph, where edges encode referral relationships, shared device fingerprints, and co-claim timestamps. A fraud ring manifests as a densely connected subgraph — detectable at the network level even when individual nodes appear clean.

---

### F2 — Supply-Side Disruption Trigger

**Problem:** All current triggers assume the worker cannot work. There is a structurally different failure mode: the worker *is* available, but orders do not exist because the supply side of the platform has collapsed.

In March 2025, an LPG shortage forced restaurants and cloud kitchens across several Indian cities to scale back operations. Delivery partners on Zomato and Swiggy reported daily earnings falling 25–30% — not because they couldn't ride, but because kitchens weren't generating orders. No weather trigger fired. No existing parametric insurance product would have paid out.

**Direction:** A dual-signal trigger — zone-level order volume drop exceeding 35% over a 48-hour window, corroborated by a verified supply disruption signal (LPG supply bulletin, restaurant closure rate spike, or CNG station data) — with a proportional payout scaled to the depth of the volume drop rather than a binary full-day replacement.

---