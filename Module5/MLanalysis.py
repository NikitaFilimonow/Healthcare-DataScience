# =====================================================
# title: A critical evaluation of interpretable machine learning models
# for Breast Cancer diagnosis: Clinical Impact of Sensitivity     
# author: Nikita Filimonow, nf434@cam.ac.uk
# date: 03.06.2026
# =====================================================

# =====================================================
# 1. INITIAL SETUP   
# =====================================================
# 1.1 Importing required libraries for analysis 
import os
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.datasets import load_breast_cancer
# 1.2 Importing required models and evaluation metrics 
from sklearn.model_selection import (
    train_test_split, 
    GridSearchCV,
    cross_val_score,
    StratifiedKFold,
    learning_curve
)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    roc_auc_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay, 
    RocCurveDisplay,
    recall_score
)
from statsmodels.stats.contingency_tables import mcnemar
import warnings
warnings.filterwarnings('ignore')

# =========================================================
# 2. CONFIGURATION   
# =========================================================
# 2.1 Fixing random seed to ensure future reproducibility 
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
# Designating and creating directories for outputs
FIGURES_DIR = 'figures'
RESULTS_DIR = 'results'
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
# 2.2 Fixing 80/20 train/test ratio 
TEST_SIZE = 0.2
# 2.3 Defining cross-validations settings
CV_FOLDS = 5

# =========================================================
# 3. DATA LOADING 
# =========================================================

# 3.1 Loading dataset directly from sklearn
data = load_breast_cancer()

# 3.2 Converting to pandas DataFrame for easier manipulation 
df = pd.DataFrame(data.data, columns=data.feature_names)

# 3.3 Adding target column - 0 = malignant, 1 = bening 
df['target'] = data.target

# =========================================================
# 4. EXPLORATORY DATA ANALYSIS 
# =========================================================

# 4.1 Printing basic dataset information
print("=" *50)
print("Dataset Overview")
print("=" *50)
print(f"Shape : {df.shape}")
print(f"Missing values: {df.isnull().sum().sum()}")
print(f"Dtypes : {df.dtypes.value_counts().to_dict()}")
print(f"\nClass distribution:")
print(f"  Malignant (0): {(df['target']==0).sum()} ({(df['target']==0).mean():.1%})")
print(f"  Benign    (1): {(df['target']==1).sum()} ({(df['target']==1).mean():.1%})")
print(f"\nBasic statistics:")
print(df.describe().round(2))

# 4.2 Creating a class distribution plot
fig, ax = plt.subplots(figsize=(6, 4))
counts = df['target'].value_counts().sort_index()
ax.bar(['Malignant (0)', 'Benign (1)'], counts.values,
       color=['#E24B4A', '#1D9E75'], edgecolor='white')
ax.set_title('Class Distribution', fontsize=13)
ax.set_ylabel('Count')
for i, v in enumerate(counts.values):
    ax.text(i, v + 3, str(v), ha='center', fontsize=11)
plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/class_distribution.png', dpi=150, bbox_inches='tight')
plt.close()

# 4.3 Creating and visualising colleration heatmap to inform future selection rationale 
fig, ax = plt.subplots(figsize=(14, 12))
corr = df[list(data.feature_names)].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, cmap='coolwarm', center=0,
            annot=False, ax=ax, linewidths=0.2)
ax.set_title('Feature Correlation Matrix', fontsize=13)
plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()

# 4.4 Showing feature distribution by class focusing on the first 12 mean features
top_features = list(data.feature_names[:12])
fig, axes = plt.subplots(3, 4, figsize=(16, 10))
axes = axes.flatten()
for i, feature in enumerate(top_features):
    sns.histplot(data=df, x=feature, hue='target',
                 ax=axes[i], bins=25, alpha=0.6,
                 palette={0: '#E24B4A', 1: '#1D9E75'})
    axes[i].set_title(feature[:25], fontsize=9)
    axes[i].set_xlabel('')
    if axes[i].get_legend():
        axes[i].get_legend().remove()
fig.suptitle('Feature Distributions by Class (0=Malignant, 1=Benign)',
             fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/feature_distributions.png',
            dpi=150, bbox_inches='tight')
plt.close()

# =========================================================
# 5. DATA PREPROCESSING 
# =========================================================

# 5.1 Separating features and targets 
# Making x contain the 30 features  
# Making y contain the target column (0=malignant, 1=benign)
X = df[list(data.feature_names)].values
y = df['target'].values

# 5.2 Spliting data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(
 X, y,
 test_size=TEST_SIZE,
 random_state=RANDOM_STATE,
 stratify=y   
)

# 5.3 Printing the results
print("=" * 50)
print("DATA PREPROCESSING")
print("=" * 50)
print(f"Training set  : {x_train.shape[0]} samples")
print(f"Test set      : {x_test.shape[0]} samples")
print(f"Features      : {x_train.shape[1]}")
print(f"\nClass ratio in training set:")
print(f"  Malignant (0): {(y_train==0).sum()} ({(y_train==0).mean():.1%})")
print(f"  Benign    (1): {(y_train==1).sum()} ({(y_train==1).mean():.1%})")
print(f"\nClass ratio in test set:")
print(f"  Malignant (0): {(y_test==0).sum()} ({(y_test==0).mean():.1%})")
print(f"  Benign    (1): {(y_test==1).sum()} ({(y_test==1).mean():.1%})")

# 5.4 Performing feature scaling 
# Creating StandardScaler for logistic regression
# as features in the dataset range significantly 
# Preventing data leakage by applying scaler on training data only
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(x_train)
X_test_scaled = scaler.transform(x_test)


# 5.5 Printing the results
print(f"\nScaling applied: StandardScaler")
print(f"  Mean before scaling : {x_train.mean():.3f}")
print(f"  Mean after scaling  : {X_train_scaled.mean():.3f}")
print(f"  Std after scaling   : {X_train_scaled.std():.3f}")

# 5.6 Identifying class imbalance in the dataset
malignant_count = (y_train==0).sum()
malignant_pct = (y_train==0).mean()
benign_count = (y_train==1).sum()
benign_pct = (y_train==1).mean()

# 5.7 Printing the results
print("\nClass imbalance check:")
print(f"  Malignant (0): {malignant_count} ({malignant_pct:.1%})")
print(f"  Benign    (1): {benign_count} ({benign_pct:.1%})")
print(f"  Strategy  : class_weight='balanced' applied in model definitions")

# =========================================================
# 6. MODEL 1 - LOGISTIC REGRESSION 
# =========================================================

# 6.1 Defining hyperparameter grid
param_grid_lr = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear']
}

# 6.2 Establishing cross-validation strategy 
cv = StratifiedKFold(
n_splits=CV_FOLDS,
shuffle=True,
random_state=RANDOM_STATE
)

# 6.3 Conducting training with GridSearchCV and 
# identifying the best performing model 
lr_grid = GridSearchCV(
LogisticRegression(
    class_weight='balanced',
    max_iter=1000,
    random_state=RANDOM_STATE
),
param_grid=param_grid_lr,
cv=cv,
scoring='roc_auc',
n_jobs=1
)

lr_grid.fit(X_train_scaled, y_train)
lr_best = lr_grid.best_estimator_

# 6.4 Evaluating the best performing model on test set
y_pred_lr = lr_best.predict(X_test_scaled)
y_prob_lr = lr_best.predict_proba(X_test_scaled)[:, 1]

# 6.5 Checking cross-validation stability 
cv_scores_lr = cross_val_score(
lr_best, X_train_scaled, y_train,
cv=cv, scoring='roc_auc'
)

# 6.6 Printing the results
print(f"Best parameters : {lr_grid.best_params_}")
print(f"CV AUC-ROC      : {cv_scores_lr.mean():.3f} ± {cv_scores_lr.std():.3f}")
print(classification_report(y_test, y_pred_lr,
    target_names=['Malignant', 'Benign']))
print(f"AUC-ROC         : {roc_auc_score(y_test, y_prob_lr):.3f}")

# =========================================================
# 7. MODEL 2 - DECISION TREE 
# =========================================================

# 7.1 Defining hyperparameter grid
param_grid_dt = {
    'max_depth': [3, 4, 5, 6, 7, 10, None],
    'min_samples_split': [2, 5, 10, 20],
    'criterion': ['gini', 'entropy']
}

# 7.2 Conducting training with GridSearchCV and 
# identifying the best performing model 
dt_grid = GridSearchCV(
    DecisionTreeClassifier(
        class_weight='balanced', 
        random_state=RANDOM_STATE
    ), 
    param_grid=param_grid_dt,
    cv=cv,
    scoring='roc_auc',
    n_jobs=-1
)

dt_grid.fit(x_train, y_train)
dt_best = dt_grid.best_estimator_

# 7.3 Evaluating best performing model on test test 
y_pred_dt = dt_best.predict(x_test)
y_prob_dt = dt_best.predict_proba(x_test)[:, 1]

# 7.4 Defining hyperparameter grid
cv_scores_dt = cross_val_score(
    dt_best,
    x_train,
    y_train,
    cv=cv,
    scoring='roc_auc'
)

# 7.5 Printing the results 
print("=" * 50)
print("MODEL 2 - DECISION TREE")
print("=" * 50)
print(f"best_parameters : {dt_grid.best_params_}")
print(f"CV AUC-ROC.     : {cv_scores_dt.mean():.3f} ± {cv_scores_dt.std():.3f}")
print(classification_report(y_test, y_pred_dt,
       target_names=['Malignant', 'Benign']))
print(f"AUC-ROC         : {roc_auc_score(y_test, y_prob_dt):.3f}")

# =========================================================
# 8. MODEL EVALUATION AND PERFORMANCE METRICS
# =========================================================

# 8.1 Comparing models performance by plotting ROC curves 
fig, ax = plt.subplots(figsize=(8, 6))
RocCurveDisplay.from_predictions(
    y_test, y_prob_lr,
    name=f"Logistic Regression (AUC={roc_auc_score(y_test, y_prob_lr):.3f})",
    ax=ax, color='#3266ad'
)
RocCurveDisplay.from_predictions(
    y_test, y_prob_dt,
    name=f"Decision Tree (AUC={roc_auc_score(y_test, y_prob_dt):.3f})",
    ax=ax, color='#E24B4A'
)
ax.plot([0, 1], [0, 1], 'k--', linewidth=0.8, label='Random classifier')
ax.set_title('ROC Curves — Model Comparison', fontsize=13)
ax.set_xlabel('False Positive Rate (1 - Specificity)')
ax.set_ylabel('True Positive Rate (Sensitivity)')
ax.legend(loc='lower right')
plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/roc_curves.png', dpi=150, bbox_inches='tight')
plt.close()

# 8.2 Visualizing confusion matrices for prediction error comparison
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, preds, title in [
    (axes[0], y_pred_lr, 'Logistic Regression'),
    (axes[1], y_pred_dt, 'Decision Tree')
]:
    ConfusionMatrixDisplay(
        confusion_matrix(y_test, preds),
        display_labels=['Malignant', 'Benign']
    ).plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title(title, fontsize=12)
fig.suptitle('Confusion Matrices — Model Comparison', fontsize=13)
plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.close()

# 8.3 Evaluating feature importance for decision tree
pd.Series(dt_best.feature_importances_, index=data.feature_names)\
  .sort_values(ascending=True).tail(15)\
  .plot(kind='barh', figsize=(8, 6), color='#E24B4A', edgecolor='white',
        title='Feature Importance — Decision Tree (Top 15)')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/feature_importance_dt.png', dpi=150, bbox_inches='tight')
plt.close()

# 8.4 Identifying feature coefficients for logistic regression
pd.Series(abs(lr_best.coef_[0]), index=data.feature_names)\
  .sort_values(ascending=True).tail(15)\
  .plot(kind='barh', figsize=(8, 6), color='#3266ad', edgecolor='white',
        title='Feature Coefficients — Logistic Regression (Top 15, absolute)')
plt.xlabel('|Coefficient|')
plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/feature_importance_lr.png', dpi=150, bbox_inches='tight')
plt.close()

# 8.5 Analyzing sensitivity threshold
threshold_results = []
for t in [0.3, 0.4, 0.5, 0.6, 0.7]:
    y_pred_t = (y_prob_lr >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_t).ravel()
    threshold_results.append({
        'Threshold':        t,
        'Sensitivity':      round(tp / (tp + fn), 3),
        'Specificity':      round(tn / (tn + fp), 3),
        'Missed_Malignant': fn
    })
pd.DataFrame(threshold_results).to_csv(
    f'{RESULTS_DIR}/threshold_analysis.csv', index=False)

# 8.6 Conducting McNemar's test
correct_lr = (y_pred_lr == y_test)
correct_dt = (y_pred_dt == y_test)
b = (correct_lr & ~correct_dt).sum()
c = (~correct_lr & correct_dt).sum()
result = mcnemar([[0, b], [c, 0]], exact=True)
pd.DataFrame([{
    'b':           b,
    'c':           c,
    'p_value':     round(result.pvalue, 4),
    'significant': result.pvalue < 0.05
}]).to_csv(f'{RESULTS_DIR}/mcnemar_test.csv', index=False)

# 8.7 Saving models metrics for reference
pd.DataFrame({
    'Model':           ['Logistic Regression', 'Decision Tree'],
    'AUC_ROC':         [round(roc_auc_score(y_test, y_prob_lr), 3),
                        round(roc_auc_score(y_test, y_prob_dt), 3)],
    'CV_AUC_ROC_Mean': [round(cv_scores_lr.mean(), 3),
                        round(cv_scores_dt.mean(), 3)],
    'CV_AUC_ROC_Std':  [round(cv_scores_lr.std(),  3),
                        round(cv_scores_dt.std(),  3)],
    'Sensitivity':     [round(recall_score(y_test, y_pred_lr, pos_label=0), 3),
                        round(recall_score(y_test, y_pred_dt, pos_label=0), 3)],
    'Specificity':     [round(recall_score(y_test, y_pred_lr, pos_label=1), 3),
                        round(recall_score(y_test, y_pred_dt, pos_label=1), 3)],
}).to_csv(f'{RESULTS_DIR}/model_metrics.csv', index=False)

# 8.8 Printing the results
print("\n" + "=" * 50)
print("PRINTING THE RESULTS OF SECTION 8")
print("=" * 50)

metrics = pd.read_csv(f'{RESULTS_DIR}/model_metrics.csv')
print("\nMODEL PERFORMANCE ON TEST SET (n=114)")
print(metrics.to_string(index=False))

print("\nSENSITIVITY THRESHOLD ANALYSIS — Logistic Regression")
print(pd.read_csv(f'{RESULTS_DIR}/threshold_analysis.csv').to_string(index=False))

print("\nMcNEMAR'S TEST — LR vs DT")
mcn = pd.read_csv(f'{RESULTS_DIR}/mcnemar_test.csv').iloc[0]
print(f"LR correct, DT wrong : {int(mcn['b'])}")
print(f"LR wrong, DT correct : {int(mcn['c'])}")
print(f"p-value              : {mcn['p_value']}")
print(f"Interpretation       : {'Significant (p<0.05)' if mcn['significant'] else 'Not significant (p>=0.05)'}")

print("\nFIGURES SAVED")
for f in sorted(os.listdir(FIGURES_DIR)):
    print(f"  ✓ {f}")

print("\nRESULTS SAVED")
for f in sorted(os.listdir(RESULTS_DIR)):
    print(f"  ✓ {f}")

print("\n" + "=" * 50)
print("=" * 50)

# =========================================================
# 9. ADDITIONAL VISUALIZATION 
# =========================================================

# 9.1 Creating decision tree diagram
fig, ax = plt.subplots(figsize=(20, 10))
plot_tree(
    dt_best,
    feature_names=list(data.feature_names),
    class_names=['Malignant', 'Benign'],
    filled=True,
    rounded=True,
    fontsize=10,
    ax=ax
)
ax.set_title(f'Decision Tree Structure — max_depth={dt_best.max_depth}', fontsize=14)
plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/decision_tree.png', dpi=150, bbox_inches='tight')
plt.close()

# 9.2 Plotting learning curves
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, model, X_tr, name, color in [
    (axes[0], lr_best, X_train_scaled, 'Logistic Regression', '#3266ad'),
    (axes[1], dt_best, x_train,        'Decision Tree',       '#E24B4A')
]:
    train_sizes, train_scores, val_scores = learning_curve(
        model, X_tr, y_train,
        cv=cv,
        scoring='roc_auc',
        train_sizes=np.linspace(0.1, 1.0, 10),
        n_jobs=-1
    )
    train_mean = train_scores.mean(axis=1)
    train_std  = train_scores.std(axis=1)
    val_mean   = val_scores.mean(axis=1)
    val_std    = val_scores.std(axis=1)

    ax.plot(train_sizes, train_mean, color=color,
            label='Training score', linewidth=2)
    ax.fill_between(train_sizes,
                    train_mean - train_std,
                    train_mean + train_std,
                    alpha=0.15, color=color)
    ax.plot(train_sizes, val_mean, color=color,
            linestyle='--', label='Validation score', linewidth=2)
    ax.fill_between(train_sizes,
                    val_mean - val_std,
                    val_mean + val_std,
                    alpha=0.15, color=color)
    ax.set_title(f'Learning Curve — {name}', fontsize=12)
    ax.set_xlabel('Training set size')
    ax.set_ylabel('AUC-ROC')
    ax.legend(loc='lower right')
    ax.set_ylim([0.7, 1.05])
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/learning_curves.png', dpi=150, bbox_inches='tight')
plt.close()

# 9.3 Printing the results 
print("\n" + "=" * 50)
print("PRINTING THE RESULTS OF SECTION 9")
print("=" * 50)
print(f"\nAll figures in {FIGURES_DIR}/:")
for f in sorted(os.listdir(FIGURES_DIR)):
    print(f"  ✓ {f}")
print("=" * 50)

