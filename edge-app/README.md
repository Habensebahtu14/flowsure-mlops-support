# 🛡️ FlowSure — Edge Intent Classifier

AI-gestuurde klantenondersteuning die binnenkomende klantberichten automatisch
classificeert naar **intent**, **categorie** en **prioriteit**, zodat tickets direct
naar het juiste team worden gerouteerd.

Dit is het **edge-model**: een lichtgewicht, on-device classificatiemodel dat draait
zonder cloudverbinding of GPU. Het model is geëxporteerd naar **ONNX** en wordt
geserveerd via een **Streamlit**-webapp in een **Docker**-container, met een volledige
**CI/CD-pipeline** in GitHub Actions.

---

## 📋 Inhoudsopgave

- [Wat doet het project](#-wat-doet-het-project)
- [Architectuur](#-architectuur)
- [Projectstructuur](#-projectstructuur)
- [Het model](#-het-model)
- [Lokaal draaien](#-lokaal-draaien)
- [Draaien met Docker](#-draaien-met-docker)
- [Monitoring & logging](#-monitoring--logging)
- [Tests](#-tests)
- [CI/CD-pipeline](#-cicd-pipeline)
- [Data- & modelpijplijn](#-data--en-modelpijplijn-notebooks)

---

## 🎯 Wat doet het project

FlowSure ontvangt klantberichten (bijv. *"Where is my refund? It has been 2 weeks."*)
en bepaalt daarvoor automatisch:

| Output | Voorbeeld | Toelichting |
|--------|-----------|-------------|
| **Intent** | `get_refund` | Eén van de 27 herkende klantintenties |
| **Categorie** | `REFUND` | Eén van de 11 afdelingen/categorieën |
| **Prioriteit** | `medium` | `low` / `medium` / `high` voor routering & SLA |
| **Confidence** | `0.98` | Zekerheid van de voorspelling |
| **Top-3** | — | De drie meest waarschijnlijke intenties |
| **Inferentietijd** | `4.2 ms` | Latency per voorspelling |

De webapp biedt twee werkwijzen:

- **🎯 Classify Ticket** — één bericht classificeren, met confidence-balk, top-3 en
  intent/categorie/prioriteit-kaarten.
- **📊 Batch Classification** — een CSV uploaden (kolom `text`) of meerdere regels
  plakken; resultaat met samenvattende statistieken, grafieken (verdeling per categorie
  en prioriteit) en een downloadbare CSV.

---

## 🏗 Architectuur

```
Klantbericht
     │
     ▼
┌─────────────────────────────┐
│  Streamlit-webapp (app.py)  │   UI: single + batch classificatie
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  IntentClassifier            │   classifier.py
│  (ONNX Runtime)              │   - predict() / predict_batch()
└─────────────┬───────────────┘
              │
   ┌──────────┴───────────┐
   ▼                      ▼
baseline_model.onnx   intent_mapping.json
(TF-IDF + LogReg)     (intent → categorie + prioriteit)

Elke voorspelling wordt gelogd via monitoring.py (stdout + predictions.jsonl).
```

Het hele systeem draait in één Docker-container die als image naar Docker Hub wordt
gepusht via GitHub Actions.

---

## 📁 Projectstructuur

Deze app staat in de map **`edge-app/`** binnen de repo **`flowsure-mlops-support`**.

```
edge-app/
├── app/
│   ├── app.py                  # Streamlit-UI (single + batch tabs)
│   ├── classifier.py           # IntentClassifier — ONNX-inferentie
│   ├── monitoring.py           # Prediction logging (stdout + predictions.jsonl)
│   └── models/
│       ├── baseline_model.onnx # Geëxporteerd TF-IDF + LogReg model (~1.5 MB)
│       └── intent_mapping.json # intent → [categorie, prioriteit]
├── tests/
│   └── test_classifier.py      # Pytest-suite (laadt het echte ONNX-model)
├── .streamlit/config.toml      # Streamlit-configuratie
├── Dockerfile                  # Python 3.11-slim, non-root user, healthcheck-ready
├── docker-compose.yml          # Lokaal draaien + healthcheck + restart policy
├── requirements.txt            # Runtime-dependencies (pinned)
├── pytest.ini / conftest.py    # Test-configuratie
└── .env.example                # Voorbeeld-environmentvariabelen
```

> De data- en modelnotebooks horen bij hetzelfde MLOps-project maar staan **buiten deze
> repo** (zie [Data- & modelpijplijn](#-data--en-modelpijplijn-notebooks)).

---

## 🤖 Het model

- **Algoritme:** scikit-learn `Pipeline` — **TF-IDF + Logistic Regression**
- **Formaat:** geëxporteerd naar **ONNX**, geserveerd met **ONNX Runtime**
- **Grootte:** ~1.5 MB (ruim onder het edge-budget van 50 MB)
- **27 intents** verdeeld over **11 categorieën**: `ACCOUNT`, `CANCEL`, `CONTACT`,
  `DELIVERY`, `FEEDBACK`, `INVOICE`, `ORDER`, `PAYMENT`, `REFUND`, `SHIPPING`,
  `SUBSCRIPTION`.
- **Prioriteit** wordt per intent afgeleid via `intent_mapping.json` (bijv.
  `complaint`, `payment_issue`, `cancel_order` → `high`).

### Waarom TF-IDF + LogReg in plaats van DistilBERT?

Tijdens het modelleren is ook **DistilBERT** geëvalueerd. De baseline behaalde echter
**99,2% accuracy** (F1 ≥ 0,96), traint in seconden, is direct naar ONNX te converteren
en draait razendsnel op edge-hardware — terwijl DistilBERT op CPU-only hardware uren
training kostte. Daarom is het lichte model gekozen voor edge-deployment.

> ℹ️ De hoge accuracy komt mede doordat de Bitext-trainingsdata synthetisch is; in
> productie ligt de score realistisch gezien lager.

---

## 💻 Lokaal draaien

Vereist: **Python 3.11**

```bash
# vanuit de map "edge-app"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# start de app
cd app
streamlit run app.py
```

De app draait op **http://localhost:8501**.

---

## 🐳 Draaien met Docker

### Optie 1 — Direct van Docker Hub (snelste)

De image staat gepubliceerd op Docker Hub — je hoeft de repo niet te clonen of zelf te
bouwen:

```bash
docker pull habensebhatu/flowsure:latest
docker run -p 8501:8501 habensebhatu/flowsure:latest
```

Open vervolgens **http://localhost:8501**.

De image is **multi-arch** (`linux/amd64` + `linux/arm64`) en draait dus zowel op
Intel/AMD als op Apple Silicon. Voor een vaste, onveranderlijke versie kun je in plaats
van `:latest` een commit-tag gebruiken: `habensebhatu/flowsure:<git-sha>`.

### Optie 2 — Met Docker Compose (lokaal bouwen)

```bash
docker compose up --build
```

Open vervolgens **http://localhost:8501**.

De Compose-setup bevat een **healthcheck** (via `/_stcore/health`) en een
`restart: unless-stopped`-policy.

### Optie 3 — Met plain Docker (lokaal bouwen)

```bash
docker build -t flowsure:latest .
docker run -p 8501:8501 flowsure:latest
```

**Highlights van de Dockerfile:**
- `python:3.11-slim` als basis, met `libgomp1` + locales voor ONNX
- Gelaagde build: dependencies worden vóór de app-code gekopieerd (betere caching)
- Draait als **non-root** `appuser` (security best practice)

---

## 📈 Monitoring & logging

Elke voorspelling wordt gelogd voor monitoring-doeleinden (`monitoring.py`), naar twee
kanalen:

- **Systeemlogs (stdout)** — leesbare regels, zichtbaar via `docker logs`, voor live
  meekijken en foutopsporing. Voorbeeld:
  `INFO Prediction: intent=get_refund category=REFUND priority=medium confidence=0.98 (4.2ms)`
- **Gestructureerde logs** — `logs/predictions.jsonl`, één JSON-record per voorspelling,
  als databron voor drift-analyse.

Uit privacy-overweging wordt **geen ruwe klanttekst** gelogd, alleen metadata
(tekstlengte, intent, categorie, prioriteit, confidence, inferentietijd). Deze gelogde
voorspellingen vormen de input voor de drift-detectie (PSI) in de notebooks.

```bash
# Logs bekijken in een draaiende container
docker compose logs -f flowsure                                # systeemlogs (stdout)
docker compose exec flowsure cat /app/logs/predictions.jsonl   # gestructureerde logs
```

> 💡 Standaard zijn de logs vluchtig (verdwijnen bij het stoppen van de container). Om
> ze te bewaren over herstarts heen, mount je de logmap als volume
> (`./logs:/app/logs` in `docker-compose.yml`).

---

## 🧪 Tests

De testsuite (`tests/test_classifier.py`) laadt het **echte ONNX-model** (geen mocks) en
controleert onder andere:

- het model laadt correct en heeft > 0 intents;
- `predict()` geeft alle verwachte sleutels terug;
- confidence ligt tussen 0 en 1, prioriteit is `low`/`medium`/`high`;
- `top_3` bevat precies 3 voorspellingen;
- lege of whitespace-input gooit een `ValueError`;
- het modelbestand blijft onder het **edge-budget van 50 MB**;
- `predict_batch()` geeft één resultaat per invoer.

```bash
pip install -r requirements.txt pytest
pytest tests/ -v
```

---

## 🔄 CI/CD-pipeline

GitHub Actions (`.github/workflows/ci.yml`) draait bij elke push en pull request naar
`main`:

1. **Job `test`** — installeert dependencies en draait de volledige pytest-suite op
   Python 3.11.
2. **Job `build-and-push`** — draait **alleen** na een geslaagde testjob én bij een
   directe push naar `main` (niet bij PR's). Bouwt met **Buildx + QEMU** een
   **multi-arch image** (`linux/amd64` + `linux/arm64`) en pusht naar Docker Hub met
   twee tags:
   - `:latest` — laatste succesvolle build op `main`
   - `:<git-sha>` — onveranderlijke tag per commit (maakt rollbacks mogelijk)

> 🔐 Docker Hub-credentials staan als GitHub-secrets (`DOCKERHUB_USERNAME`,
> `DOCKERHUB_TOKEN`) — nooit hardcoded.

---


De **drift-detectie (PSI)** in de notebooks gebruikt de `predictions.jsonl` die deze
app genereert als bron voor model-monitoring (zie
[Monitoring & logging](#-monitoring--logging)).

Deze app serveert het resultaat van **Deel A**: het naar ONNX geëxporteerde edge-model.

---

*FlowSure MLOps · v1.1.0*
