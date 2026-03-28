# League of Legends - Win Prediction from Early Game Data

Predicting match outcomes from the first 10 minutes of a League of Legends game using structured match data and three different ML approaches. Built as a learning project to go through the full data science pipeline hands-on, from raw data to model comparison.

<img width="559" height="559" alt="image" src="https://github.com/user-attachments/assets/66524dfa-b8f3-4484-97f1-d66ba3f492a2" />

*A snapshot of my exploratory data analysis and model performance metrics.*

Dataset: [League of Legends Diamond Ranked Games (10 min)](https://www.kaggle.com/datasets/bobbyscience/league-of-legends-diamond-ranked-games-10-min) on Kaggle.
Dataset is already downloaded inside the repository.

---

## What this project covers

- exploratory data analysis and feature engineering on match statistics
- training and comparing three models: Logistic Regression, Random Forest, and a PyTorch neural network
- five concrete hypotheses tested against the data and evaluated after modeling

Three models were trained and used intentionally as a learning exercise to understand how they differ in behavior, not just in final accuracy.

--- 
## Setup

```bash
pip install -r requirements.txt
jupyter notebook notebook.ipynb
```

**requirements:** pandas, numpy, matplotlib, seaborn, scikit-learn, torch

---

## Notes

This is a learning project. The goal was to go through the full pipeline and test real hypotheses against the data, not to chase accuracy. The convergence of all three models to ~72% is itself the most important result: the first 10 minutes simply do not contain enough information to predict diamond-level outcomes with high confidence, and no amount of model complexity changes that.

---

## Hypotheses and Findings

### Hypothesis 1 — Gold difference is the strongest predictor of outcome
**confirmed.**

Gold difference ranked as the top feature by importance across all models. Teams with a lead exceeding ~2000 gold at 10 minutes win approximately 80% of the time, with win probability following a sigmoidal curve centered at 0.

<img width="1187" height="590" alt="image" src="https://github.com/user-attachments/assets/4bf247c1-05cd-49eb-a8f1-9bd1b2bbe1f7" />


---

### Hypothesis 2 — Objective control contributes independently of gold
**confirmed.**

Dragon and herald secured at 10 minutes correlate with win rate independently of gold lead. Dragon showed the highest impact of the three objectives tested, likely because first dragon indicates botlane dominance which compounds into broader map control.

<img width="1589" height="616" alt="image" src="https://github.com/user-attachments/assets/ae296024-d220-4226-b486-48a0d334f765" />


---

### hypothesis 3 — CS differential is a stronger signal than kill differential
**partially rejected.**

CS diff is more consistent as an income source, but kill differential ranked higher in Random Forest feature importance after redundant columns were removed. Kill leads at 10 minutes in diamond games carry more predictive weight than expected.

<img width="1483" height="1190" alt="image" src="https://github.com/user-attachments/assets/d6d5b53a-fe8e-4ff2-8d08-b801a0370def" />


<img width="1789" height="515" alt="image" src="https://github.com/user-attachments/assets/3e945bf0-2880-4eca-8874-70c506c91d34" />


---

### Hypothesis 4 — the gold/win probability relationship is sigmoidal
**confirmed.**

The logistic regression probability curve matched the binned bar chart from EDA exactly. Win probability is near 50% at even gold, accelerates sharply through the ±2000 range, and approaches 90%+ at large leads.

---

### Hypothesis 5 — a simple linear model is sufficient for this problem
**confirmed.**

Logistic Regression achieved the highest AUC (0.806) of all three models. Random Forest matched it without improvement. The neural network underperformed (AUC 0.786) due to insufficient data for its complexity. The performance ceiling of ~72% accuracy reflects irreducible variance from game factors not captured in the first 10 minutes.

<img width="889" height="690" alt="image" src="https://github.com/user-attachments/assets/04ba18a6-ce6f-467b-adb8-1ec8ac7776a5" />


| model | accuracy | AUC |
|---|---|---|
| Logistic Regression | 71.71% | 0.806 |
| Random Forest | 71.86% | 0.804 |
| Neural Network (PyTorch) | ~71.8% | 0.786 |

---

## Exploratory data analysis

Gold, CS, and kill differentials were engineered as new features (blue minus red) to capture team-level advantage rather than raw totals. several raw columns are highly correlated — gold, CS, and experience move together, which informed feature selection and helps explain why simpler models perform competitively.

<img width="1483" height="1190" alt="image" src="https://github.com/user-attachments/assets/d49b8258-ba31-40e3-be29-5e7dc5c37947" />


---

## Model details

### logistic regression
a linear baseline. fast to train, interpretable, and the strongest overall performer given the structure of the data.

<img width="742" height="590" alt="image" src="https://github.com/user-attachments/assets/51e6309f-4778-42cd-9d24-4a7d8dc7f92e" />


### Random Forest
Captures non-linear interactions between features without much tuning. matched logistic regression without exceeding it, which supports hypothesis 5.

Top 15 most important columns according to the RF model:
<img width="989" height="790" alt="image" src="https://github.com/user-attachments/assets/4c085ed6-5c95-4fb5-afc1-708bc30a32bc" />

### Neural Network (PyTorch)
A small feedforward network trained with binary cross entropy loss. included to get experience with the PyTorch training loop. The data is too small and too tabular for deep learning to have an edge here.

<img width="1389" height="495" alt="image" src="https://github.com/user-attachments/assets/e9150cd8-9258-44b1-bb5a-0e2303f18483" />


Loss decreases steadily. validation accuracy plateaus around 71-72% and oscillates slightly, consistent with the other models hitting the same ceiling.
