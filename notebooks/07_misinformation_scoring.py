import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd().parents[0] if Path.cwd().name == "notebooks" else Path.cwd()))

import numpy as np
import pandas as pd
from src.misinfo_scoring import brier_score, composite_mis_risk_score
from src.utils import guard_against_leakage, load_config, resolve_path

config = load_config()
df = pd.read_csv(resolve_path(config["paths"]["train_csv"]))

# Guideline #1: mis_risk_label is evaluation-only ground truth, never a feature.
guard_against_leakage(df, feature_columns=["headline", "body_text", "source_domain"], config=config)

# %% [markdown]
# ## Compute the five signals + composite score for every article

# %%
cfg = config["misinfo_scoring"]
scored = df.apply(lambda r: composite_mis_risk_score(r, cfg["weights"], cfg["credible_domain_whitelist"]), axis=1)
scored_df = pd.json_normalize(scored)
result = pd.concat([df[["article_id", "headline", "source_domain", "mis_risk_label"]].reset_index(drop=True),
                     scored_df], axis=1)
result.head(10)

# %% [markdown]
# ## Calibration curve vs human-annotated `mis_risk_label` + Brier score

# %%
import matplotlib.pyplot as plt

labeled = result["mis_risk_label"].notna()
y_true = result.loc[labeled, "mis_risk_label"].values
y_pred = result.loc[labeled, "mis_risk_score"].values

bscore = brier_score(y_true, y_pred)
print(f"Brier score (n={labeled.sum()}): {bscore:.4f}  (0 = perfect calibration)")

bins = np.linspace(0, 1, 11)
bin_ids = np.digitize(y_pred, bins) - 1
bin_ids = np.clip(bin_ids, 0, 9)
calibration = pd.DataFrame({"bin": bin_ids, "y_true": y_true, "y_pred": y_pred}).groupby("bin").mean()

plt.figure(figsize=(6, 6))
plt.plot([0, 1], [0, 1], "k--", label="perfect calibration")
plt.plot(calibration["y_pred"], calibration["y_true"], marker="o", label="model")
plt.xlabel("Predicted Mis-Risk Score")
plt.ylabel("Observed human-labelled risk")
plt.title("Calibration curve")
plt.legend()
out_dir = resolve_path("artifacts")
out_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(out_dir / "misinfo_calibration_curve.png", dpi=150)
plt.show()

# %% [markdown]
# ## Top-10 highest-risk articles

# %%
result.sort_values("mis_risk_score", ascending=False).head(10)[
    ["article_id", "headline", "clickbait", "emotional_language",
     "source_credibility", "factual_density", "quote_authenticity", "mis_risk_score"]
]

# %% [markdown]
# **Note:** this synthetic dataset's `mis_risk_label` column appears only
# weakly related to the article text (it's practice data), so expect a
# modest correlation here. On real news data with genuine misinformation
# labels, tune `config["misinfo_scoring"]["weights"]` against a held-out
# validation set to improve calibration.
