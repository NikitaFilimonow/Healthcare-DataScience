# A Critical Evaluation of Interpretable Machine Learning Models for Breast Cancer Diagnosis: Clinical Impact of Sensitivity

**Author:** Nikita Filimonow (nf434@cam.ac.uk)
MSt Healthcare Data Science, University of Cambridge — Module 5

---

## Abstract

Breast cancer remains a leading cause of cancer-related deaths among women worldwide. Fortunately, the growing evidence in the field of early diagnosis of breast cancer (stages I and II) already demonstrates potential to meaningfully change patient outcomes. Such earlier diagnosis would only be achievable if the solutions deployed in the diagnostic workflows are sensitive enough to correctly identify malignant tumours. Machine learning algorithms already support clinicians and pathologists with the identification of cancer in clinical settings, and this trend is unlikely to reverse. The purpose of this study was to critically evaluate the application of interpretable machine learning models (logistic regression and decision tree) for breast cancer diagnosis with a particular emphasis on sensitivity and clinical impact using Wisconsin Breast Cancer Diagnostic dataset (569 samples, 30 features). Both models were trained and optimised using GridSearchCV with stratified five-fold cross-validation. Performance of the models was evaluated using sensitivity, specificity, AUC-ROC, F1-score, accuracy and McNemar’s test. Logistic regression performed better than decision tree on all evaluation metrics, achieving a sensitivity of 0.976 and showing CV AUC-ROC of 0.996±0.005, whereas the decision tree revealed CV AUC-ROC of 0.956±0.023 and sensitivity of 0.905. Out of 42 malignant cases presented in the test set, logistic regression missed only 1 malignant case, while the decision tree missed 4. The difference in performance between the models was statistically significant (p=0.0312). Both models relied on well-known morphological features such as nuclear size, cell irregularity and perimeter, ensuring a great degree of interpretability. The findings suggest that logistic regression could be potentially considered as a promising baseline model for breast cancer classification. Further research and validation on different patient cohorts is needed to generate additional evidence. 

---

## Dataset

Wisconsin Breast Cancer Diagnostic dataset (569 samples, 30 features), built into scikit-learn and downloaded automatically when the script runs — no manual download required:

```python
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
```

External source: https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic

---

## Steps to Reproduce

```bash
git clone https://github.com/NikitaFilimonow/Healthcare-DataScience.git
cd Healthcare-DataScience/Module5
pip install scikit-learn==1.8.0 pandas==3.0.3 numpy==2.4.4 matplotlib==3.10.9 seaborn==0.13.2 statsmodels==0.14.6
python3 HDS_ML_NIKITAFILIMONOW_2606_Code.py
```

A fixed random seed (`random_state=42`) is applied throughout, so results are fully reproducible. Figures are saved to `figures/` and results to `results/`.

---

## Library Versions

| Library | Version |
|---|---|
| Python | 3.14 |
| scikit-learn | 1.8.0 |
| pandas | 3.0.3 |
| numpy | 2.4.4 |
| matplotlib | 3.10.9 |
| seaborn | 0.13.2 |
| statsmodels | 0.14.6 |

