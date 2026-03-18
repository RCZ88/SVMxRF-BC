# 🧬 Breast Cancer Classifier: SVM vs Random Forest

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white) ![NumPy](https://img.shields.io/badge/NumPy-custom%20impl-013243?logo=numpy) ![Status](https://img.shields.io/badge/Status-Complete-brightgreen) ![Context](https://img.shields.io/badge/IB-Extended%20Essay-purple)

> **SVM achieved 97.9% recall on malignant cancer detection** — built from scratch without scikit-learn, evaluated with 5-fold stratified cross-validation.

A from-scratch Python implementation comparing Support Vector Machines and Random Forests on the Wisconsin Breast Cancer Dataset (569 samples). Built as the research core of my IB DP Computer Science Extended Essay. No sklearn shortcuts — both models implemented at the math level.

---

## 📊 Results

|Model|Accuracy|Precision|Recall|F1-Score|
|---|---|---|---|---|
|**SVM (RBF Kernel)**|**93.29%**|92.07%|**97.89%**|**94.79%**|
|Random Forest|90.01%|92.39%|90.92%|91.64%|

> SVM outperforms RF by ~3% on recall — the critical metric in cancer detection, where **false negatives cost lives**.  
> RF showed lower variance (std 0.011 vs 0.038), making it more stable but less sensitive.

---

## 🔧 What I Built

- **Custom Random Forest** — recursive tree building with Gini index splitting, row-wise bootstrapping, random feature subsets per tree, and parallel training via `joblib`
- **Custom SVM** — dual QP problem solved directly with Lagrange multipliers via `cvxopt.solvers.qp`, supporting linear, RBF, and polynomial kernels
- **Evaluation pipeline** — StratifiedKFold (5-fold), per-fold confusion matrices aggregated via micro-averaging for Accuracy, Precision, Recall, F1
- **Hyperparameter tuning** — grid search over RF (n_trees, max_depth, feature subset size) and SVM (C, gamma, kernel type)

---

## ⚙️ Tech Stack

|Layer|Tools|
|---|---|
|Core|Python 3.12, NumPy, Pandas|
|ML (custom)|Gini-based recursive splitting, QP dual solver|
|Optimization|`cvxopt` (SVM), `joblib` (RF parallelism)|
|Persistence|`pickle` for trained model export|
|Evaluation|StratifiedKFold, confusion matrix aggregation|

---

## 🧠 Key Technical Decisions

**NumPy over Pandas for computation**  
Switched mid-project after profiling bottlenecks. NumPy array ops gave 10–100x speed gains in split evaluation and bootstrapping — Pandas DataFrames have overhead that compounds across thousands of tree splits.

**Solving SVM from the dual formulation**  
Instead of calling `sklearn.svm.SVC`, I solved the dual QP problem directly using Lagrange multipliers and kernel functions. This forced a real understanding of support vectors, the margin boundary, and why RBF outperforms linear kernels on this dataset.

**StratifiedKFold over standard KFold**  
The dataset is imbalanced (63% benign / 37% malignant). Standard KFold risks skewed folds that inflate accuracy. Stratified splits preserve class ratios — essential for medical classification where minority class performance is what matters.

**Micro-averaged confusion matrix aggregation**  
Aggregated raw confusion matrices across all folds before computing metrics, rather than averaging per-fold metrics. This gives sample-weighted results that are more representative than mean-of-means.

---

## 🚀 How to Run

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
pip install numpy pandas cvxopt joblib
python main.py
```

To run hyperparameter tuning:

```python
hyperparameterTuning()
```

---

## 📚 What I Learned

- **Recall > Accuracy in medical ML.** A model that's 95% accurate but misses 20% of malignant cases is dangerous. Metric selection is a design decision, not an afterthought.
- **Custom implementations expose the math.** Writing Gini splitting and QP solving by hand revealed _why_ hyperparameter choices like gamma and max_depth have such disproportionate effects on overfitting.
- **Variance matters as much as performance.** SVM's higher recall came with higher standard deviation — in a production medical setting, RF's stability might actually be preferable depending on risk tolerance.

---

_Built for IB DP Computer Science Extended Essay. All models implemented from scratch — no scikit-learn._
