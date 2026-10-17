# ClaimWise AI

A claims review assistant for hospital billing teams, built for the IBM Z Datathon (UTD AI BIZ Club).
It helps a billing specialist decide **which claims to look at first, why, and how to fix them**.

> **Decision support only.** Results require human review and are not a substitute for payer
> policy verification. A flag is not a determination that a claim will be denied or paid.
> "Billed value requiring review" is an estimate, not a guaranteed financial loss.

## What it does

| Tab | What it shows | Data | Method |
|---|---|---|---|
| Overview | Where denied dollars stand, top denial reasons, denial rate by payer, monthly trend | Real datathon files | Descriptive statistics |
| Appeal & rework queue | Denied claims ranked for follow-up. Click a row for the priority breakdown and the fix | Real datathon files | Documented priority score |
| Claim checker | Missing/inconsistent data and risk indicators. Click a row to see **why the score was given and how to fix it** | Loaded data or synthetic demo set | Documented rules and weights |
| Payer alerts | Payer rule updates and how many open denials they touch | **Synthetic feed** + real denial data | Counts of matching payer and denial codes |
| Notifications | Email alerts to administrators for chosen events | Computed from the above | Thresholds you set |
| Method & limits | Every rule, weight, formula and limitation | - | - |

**No machine learning is used.** We tested whether fields known before submission (billed amount,
payer, month) separate denied from paid claims. They do not (denial rate is about 10% in every
band, shown live on the Method tab), and the denial reason is only known after a denial, so a model
would be dishonest here.

## Be clear about what is real and what is not

* **Real (as supplied by the datathon):** the claims and denials files. All counts and dollar figures come from them.
* **Synthetic and labeled:** `data/sample_claims.csv` (demo claims with deliberate defects) and
  `data/payer_rule_updates.csv` (illustrative payer updates).
* **Not real-time and not a live integration:** payer alerts are read from a local file. You can simulate a
  new update arriving in the app. In production this table would be fed from payer bulletins.
* **Scoring numbers are our own design choices:** the risk levels (Low 1-24, Medium 25-49, High 50+), the
  review threshold (25), every rule weight, and the appeal-priority weights come from our team's judgment.
  They are **not** from an industry standard, a payer policy or a published model, and are not validated
  against real outcomes. A hospital would tune them with its own billing staff and data.

Two labels are always kept separate:
* **Confirmed data issue** - a fact visible in the record (for example, a missing patient ID).
* **Risk indicator** - a pattern associated with denials. Not a prediction.

## Run it (Mac or Linux)

From the project folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

python -m pytest              # automated tests
python run_pipeline.py        # validate data, print a summary, write CSVs to output/
streamlit run app.py          # open the dashboard (http://localhost:8501)
```

Python 3.9+ and Streamlit 1.50+ are required. Put `claims_and_billing.csv` and `denials.csv` in `data/`
(they are large and not committed).

### On Linux on IBM Z (s390x) - use the Dash dashboard

Streamlit cannot install on s390x (it needs `pyarrow`, which has no s390x build), so the same engine is also
provided with a pure-Python Dash dashboard (`app_dash.py`). The tests and `run_pipeline.py` run unchanged.

```bash
sudo apt install -y python3-venv python3-pip python3-pandas python3-numpy python3-pytest \
  python3-markupsafe python3-jinja2 python3-jsonschema
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install -r requirements-server.txt
python -m pytest
python app_dash.py --port 8050        # listens on 127.0.0.1 only
```
From your own computer, open a tunnel and browse to http://localhost:8050:
`ssh -L 8050:localhost:8050 -i <key.pem> linux1@<server-ip>`

Differences from the Streamlit version: no file-upload option, and the notification/simulation settings reset
on a browser refresh except the reviewer statuses and simulated updates, which are kept per browser session.

### On a remote Linux server with Streamlit (x86 only, no browser)

```bash
streamlit run app.py --server.headless true --server.port 8501
```
Then, on your own computer: `ssh -L 8501:localhost:8501 -i <key.pem> linux1@<server-ip>` and open http://localhost:8501.

## Email notifications

Messages contain **totals only** - no patient names, patient IDs or claim IDs. A real deployment would still
need privacy and compliance review. By default **nothing is sent**: "Send" saves a preview `.eml` file in
`output/outbox/`. To send for real, set these environment variables before starting the app (never put the
password in a file you commit):

```bash
export CLAIMWISE_SMTP_HOST=smtp.example.org
export CLAIMWISE_SMTP_PORT=587
export CLAIMWISE_SMTP_USER=your-account
export CLAIMWISE_SMTP_PASSWORD=your-password
export CLAIMWISE_FROM=claimwise@example.org
streamlit run app.py
```

## Project layout

```
app.py                       Streamlit dashboard (Mac / x86 Linux)
app_dash.py                  Dash dashboard (also runs on Linux on IBM Z / s390x)
requirements-server.txt      Dependencies for the Dash version
assets/style.css             Styles for the Dash version
src/plotly_charts.py         Charts for the Dash version
run_pipeline.py              Command-line run (validate, summarize, export)
.streamlit/config.toml       Theme (colors, light mode)
assets/favicon.png           Browser tab icon
src/claims_processor.py      Load, validate, clean, merge
src/risk_engine.py           Claim checker rules, weights, scores, breakdowns
src/appeal_queue.py          Denied-claim priority score and next-step guidance
src/financial_analytics.py   Totals, denial statistics, billed value requiring review
src/payer_alerts.py          Payer update feed and impact on open denials
src/notifications.py         Alert events, email builder and sender
src/charts.py                Overview charts
data/sample_claims.csv       Synthetic demo claims
data/payer_rule_updates.csv  Synthetic payer updates
tests/                       pytest unit tests
```

## Scoring reference

**Claim checker.** Score = sum of the weights of the rules that fire, capped at 100. A claim needs review if
it has any confirmed issue or a score of 25+. "Score if confirmed issues are corrected" = the indicators that remain.

**Appeal queue (open denials only).** Priority = min(denied amount / $5,000, 1) x 50
+ appeal stage (30 no appeal filed, 20 pending, 10 unclear) + min(days since denial / 90, 1) x 20.
Closed denials score 0. Payer appeal deadlines differ; age is only a rough urgency signal.

## Limitations

* The rules are generic examples, not any payer's actual policy.
* The data is synthetic; all numbers describe the loaded data only and are not forecasts.
* Next-step guidance is general and must be checked against the payer's policy.
* Reviewer status (Open / In review / Resolved) is kept only for the browser session.
