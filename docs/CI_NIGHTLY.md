# CI nightly · P5 false_pass gate

**Purpose**: catch any regression in `rule_false_pass_rate` (plan §八
red line, ≤ 0.005) the same day it lands. Closes
[`RISK_REGISTER.md`](../app/domain/gnn/RISK_REGISTER.md) §5 R5.

**Surface**:
- `scripts/gnn_eval_nightly.sh` — the unit of work (also runnable locally).
- `.github/workflows/gnn-eval-nightly.yml` — GitHub Actions wrapper (schedule + push + dispatch).
- `tests/scripts/test_gnn_eval_nightly.py` — verifies skip-on-missing path + happy path.

The script is the canonical interface; the workflow just supplies a
runtime and optionally fetches the dataset / checkpoint. Self-hosted
setups can skip the workflow and use cron — see §3.

---

## 1 · Exit codes (script + workflow)

| code | meaning | workflow behaviour |
|---|---|---|
| **0** | both splits passed the gate | ✅ green |
| **2** | hard failure (crash / bad args) | ❌ red, requires investigation |
| **3** | at least one split exceeded `false_pass_gate` | ❌ red, regression alert |
| **4** | skipped — dataset or checkpoint missing | ⚠️ green with warning annotation |

Code 4 exists so a fresh clone (no dataset, no ckpt) doesn't fail-loud
forever. Set `SKIP_IF_MISSING_DATA=0` to flip exit 4 to exit 2 if you
want strictness once your team has wired up artifact storage.

---

## 2 · GitHub Actions setup

The workflow runs:
- **Daily 03:00 UTC** on `cron`.
- **On push** when the rule comparator / GNN advisor / evaluator
  surface changes (paths watched in `gnn-eval-nightly.yml`).
- **Manual dispatch** via the Actions tab (lets you override
  `false_pass_gate` per run).

### Required (eventually) secrets

| secret | content | required? |
|---|---|---|
| `GNN_EVAL_DATASET_URL` | URL to a `*.tar.gz` of `datasets/circuit_compare/{labels,splits}` | **optional** (skip with exit 4 if absent) |
| `GNN_EVAL_CKPT_URL` | URL to `checkpoints/p3_followup_v2/best_f1.pt` (or whichever you want to gate against) | **optional** (skip with exit 4 if absent) |

While these secrets are unset the workflow still runs but emits a
single `::warning::` annotation and exits 0. This is the intended
"first wire-up" state — nothing fails, the team can land code freely,
and once dataset/ckpt storage is decided the secrets get filled in.

### Storage suggestions (not prescriptive)

- **GitHub Releases attachments** — fine for <2GB; URL is stable per
  release tag.
- **S3 / GCS / Azure Blob bucket + signed URL** — production-grade.
- **`huggingface-cli` / `gcsfuse`** — bigger, lets the workflow shell
  out instead of using `curl`.

### Disabling on a PR

Edit `gnn-eval-nightly.yml`'s `on:` block to drop `push` if it
generates too much CI load. The schedule + manual dispatch are usually
enough for a daily check.

---

## 3 · Self-hosted cron alternative

If the repo isn't on GitHub, drop `gnn_eval_nightly.sh` into cron
directly. The script is the same one CI calls.

```cron
# /etc/cron.d/labguardian-nightly  (or `crontab -e`)
# Run every day at 03:00 local time. Outputs go to /var/log/labguardian-nightly.log
0 3 * * * lab cd /opt/LabGuardian-Server && \
    LABEL_DIR=datasets/circuit_compare/labels \
    FALSE_PASS_GATE=0.005 \
    PYTHON=/opt/LabGuardian-Server/.venv/bin/python \
    bash scripts/gnn_eval_nightly.sh \
    >> /var/log/labguardian-nightly.log 2>&1
```

To alert on regression, wrap with a helper:

```bash
#!/usr/bin/env bash
set +e
bash scripts/gnn_eval_nightly.sh >> /var/log/labguardian-nightly.log 2>&1
rc=$?
case ${rc} in
  0|4) ;;                                            # ok / skipped
  3)   curl -fsS "$ALERT_WEBHOOK" -d "P5 false_pass regressed" ;;
  *)   curl -fsS "$ALERT_WEBHOOK" -d "P5 nightly crashed (exit ${rc})" ;;
esac
```

---

## 4 · Verifying the wiring locally

```bash
# Happy path (requires dataset + ckpt already on disk)
bash scripts/gnn_eval_nightly.sh && echo "✅ EXIT=$?"

# Skip path — should print friendly missing-artifact message + exit 4
bash scripts/gnn_eval_nightly.sh checkpoints/does_not_exist.pt
echo "EXIT=$?"            # → 4

# Hard-fail-on-missing path (production-grade)
SKIP_IF_MISSING_DATA=0 bash scripts/gnn_eval_nightly.sh checkpoints/does_not_exist.pt
echo "EXIT=$?"            # → 2
```

The same three behaviours are pinned in
`tests/scripts/test_gnn_eval_nightly.py` so any regression to the
exit-code contract gets caught by `pytest`.

---

## 5 · What CI does NOT cover (yet)

- **GraphMatcher runtime −50% measurement** (plan §八 the other hard
  gate) — needs P4.1 seed-mapping integration first. Tracked separately.
- **Real-student netlist domain adaptation** — needs an ingestion
  pipeline producing labelled real netlists. Out of scope for this
  wiring.
- **SEAL AUC drift** — currently logged in the metrics artifact but
  not gated. Easy to add a `--seal-f1-floor 0.92` flag later.
