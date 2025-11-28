#%%
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import seaborn as sns
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import OrdinalEncoder
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
#%%
df = pd.read_csv(r'bank-full.csv', sep=';')
#%%
pd.set_option('display.max_column', 20)
df.info()
df.head(5)
#%%
df.describe()
#%%
df = df.rename(columns={'contact': 'lc_contact_type', 'day': 'lc_day', 
                   'month': 'lc_month', 'duration': 'lc_duration',
                   'campaign': 'lc_contacts_count', 
                   'previous': 'pc_contacts_count',})
#%%
columns_of_interest = ['age', 'job', 'marital', 'education', 
                       'default', 'balance', 'housing', 'loan', 
                       'lc_contact_type', 'lc_day', 'lc_month',
                        'lc_duration', 'lc_contacts_count',
                        'pdays', 'pc_contacts_count', 'poutcome',
                        'y']
#%%
# Class balance
ax = plt.hist(x=df['y'])
plt.title('Checking for target variable imbalance')
plt.show()
#%%
fig, axes = plt.subplots(2, 2)
fig.set_figheight(10)
fig.set_figwidth(15)
order = ['may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec', 'jan',
        'feb', 'mar', 'apr']
ax = sns.countplot(df, x='lc_month', hue='y', ax=axes[0, 0], order=order)
axes[0, 0].set_xlabel('Month')
axes[0, 0].set_title('Conversion rate (Investigation for seasonality)')
ax = sns.countplot(df, x='marital', hue='y', ax=axes[0, 1])
axes[0, 1].set_xlabel('Marital status')
axes[0, 1].set_title('Marital status impact')
ax = sns.countplot(df, x='lc_contacts_count', hue='y', ax=axes[1, 0])
axes[1, 0].set_xlim(0, 10)
axes[1, 0].set_xlabel('Number of contacts with client')
axes[1, 0].set_title('Number of contacts impact')
ax = sns.histplot(df, x='age', ax=axes[1, 1])
axes[1, 1].set_xlabel('age')
axes[1, 1].set_title('Age distribution')
plt.show()
#%%
df[['marital', 'y']].groupby('marital').agg('mean')
#%%
df[['education', 'y']].groupby('education').agg('mean')
#%%
encoder = OrdinalEncoder()
df[['y']] = encoder.fit_transform(df[['y']])
#%%
# Fatique effect
pdays_bins = [i for i in range(0, 800, 20)]
df['pdays_bin'] = pd.cut(df.query('pdays>0')['pdays'], bins=39, labels=np.array(pdays_bins[1:])-20,
                        right=False)
conversion_rate = df.groupby('pdays_bin')['y'].mean().reset_index()

fig, axes = plt.subplots(1, 2)
fig.set_figwidth(15)
ax = sns.countplot(df.query('pdays_bin>0'), x='pdays_bin', hue='y', ax=axes[0])
ticks = axes[0].get_xticks()
axes[0].set_xticks(ticks[::5])
axes[0].set_xlabel('Days past since last campaign in bins')
axes[0].set_ylabel('Clients contacted per bin')
ax = sns.lineplot(conversion_rate, x='pdays_bin', y='y', ax=axes[1],
                  color='red')
axes[1].set_xlabel('Days past since last campaign in bins')
axes[1].set_ylabel('Conversion rate')
axes[1].axvspan(xmin=0, xmax=340, 
                color='red', alpha=0.1, label='Important part')
axes[1].legend()
plt.show()
#%%
# Age and balance
fig, axes = plt.subplots(1, 2)
fig.set_figwidth(10)
ax = sns.kdeplot(data=df[df['y']==1], y='age', x='balance', hue='y', fill=True,
                bin=100, ax=axes[0], palette='viridis')
ax = sns.kdeplot(data=df[df['y']==0], y='age', x='balance', hue='y', fill=True,
                bin=100, ax=axes[1])
axes[0].set_xlim(-3000, 11000)
axes[1].set_xlim(-3000, 11000)
plt.show()
#%%
# Loans
fig, axes = plt.subplots(2, 2)
fig.set_figwidth(14)
fig.set_figheight(14)
ax = sns.histplot(data=df, x='housing', ax=axes[0, 0], color='green')
ax = sns.histplot(data=df, x='loan', ax=axes[0, 1])
ax = sns.countplot(data=df, x='housing', hue='y', ax=axes[1, 0])
ax = sns.countplot(data=df, x='loan', hue='y', ax=axes[1, 1])
plt.show()
#%%
# Jobs, education and mean balances
fig, axes = plt.subplots(3, 1)
fig.set_figwidth(10)
fig.set_figheight(20)
ax = sns.countplot(data=df, y='job', hue='y', ax=axes[0])
balance_means = df.groupby('job')['balance'].agg(
    mean_balance='mean',
    count='count').reset_index()
ax = sns.barplot(data=balance_means, x='mean_balance', y='job', ax=axes[1],
                 color='skyblue')
ax = sns.countplot(df, x='education', hue='y', ax=axes[2], palette='icefire')
#%%
# Check statistics of retired sample group;
# Check conversion ratio in unknown, primary and teriary education groups
df.query('job=="retired"')['balance'].describe()
#%%
conversion_ratios = df.groupby('education')['y'].value_counts().unstack(fill_value=0)
conversion_ratios['conversion_ratio'] = conversion_ratios[1] / conversion_ratios[0]
conversion_ratios
#%%
# Call duration time
fig, axes = plt.subplots(2, 1)
fig.set_figheight(10)
ax = sns.histplot(df[df['y']==1], x='lc_duration', bins=150, color='orange', ax=axes[1])
ax.set_xlim(0, 2500)
axes[0].set_xlabel('Call duration [s]')
axes[1].set_xlabel('Call duration [s]')
axes[0].set_ylabel('Count (all outcomes)')
axes[1].set_ylabel('Count (positive outcomes)')
ax = sns.histplot(df[df['y']==1], x='lc_duration', bins=150, color='orange', ax=axes[0],
                  legend=True, label='Positive outcomes')
ax.set_xlim(0, 2500)
sns.histplot(df, x='lc_duration', bins=150, color='blue', ax=axes[0], 
            alpha=0.1, legend=True, label='All outcomes')
axes[0].legend()
fig.suptitle('Impact of call duration on outcome')
plt.show()
#%%
# Introduce conversion rate
df['pdays_bin'] = df['pdays_bin'].fillna(0)
df['conversion_rate'] = df.query('pdays>0').groupby('pdays_bin')['y'].mean()
df['conversion_rate'] = df['conversion_rate'].fillna(df['conversion_rate'].mean())
#%%
# Encode categorical features
categorical = ['job', 'marital', 'education', 'default', 'housing',
               'loan', 'lc_month', 'lc_contact_type', 'poutcome', 'pdays_bin']
df[categorical] = encoder.fit_transform(df[categorical])
#%%
# Balance transform
pt = PowerTransformer(method='yeo-johnson', standardize=True)
df['balance_tf'] = pt.fit_transform(df[['balance']])
#%%
# Correlation matrix
corr_matrix = df.corr()
fig = plt.figure()
ax = sns.heatmap(corr_matrix, cmap='mako', annot=True)
fig.set_figheight(15)
fig.set_figwidth(18)
#%%
# Baseline model
from sklearn.model_selection import train_test_split
X = df.drop(['y', 'balance', 'lc_duration'], axis=1)
y = df['y']
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, stratify=y,
                                                    shuffle=True, random_state=42)
#%%
pd.DataFrame(y_train).value_counts()
#%%
pd.DataFrame(y_test).value_counts()
#%%
y_train = pd.DataFrame(y_train)
y_test = pd.DataFrame(y_test)
#%%
# Calculate weight for the least class in train set
w = (y_train[y_train['y']==0].count(axis=0).sum() / y_train[y_train['y']==1].count(axis=0).sum())
#%%
# Create baseline model
log_regressor = LogisticRegression(class_weight={0:1, 1:w},
                                   max_iter=10000)
# %%
log_regressor.fit(X_train, y_train)
y_proba = log_regressor.predict_proba(X_test)[:, 1]
y_pred = (y_proba >= 0.5).astype(int)
# %%
# Baseline performance\
# Assume that missing true positive (potential customer) costs more
# than predicting false positive (waste of time and money on contact).
AUC = roc_auc_score(y_test, y_proba)
f1 = f1_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
print(f'Precision: {precision:.4f}; recall: {recall:.4f}; AUC: {AUC:.4f}; f1: {f1:.4f}')
# %%
# Create a weak classifier for AdaBoost
tree_model = DecisionTreeClassifier(max_depth=7, random_state=42, 
                                    class_weight={0:1, 1:w}, max_features=3)
# %%
# AdaBoost and XGBoost
xg_model = XGBClassifier(max_depth=3, n_estimators=100, scale_pos_weight=w,
                         random_state=42, learning_rate=0.2, eval_metric='auc')
ada_model = AdaBoostClassifier(estimator=tree_model, n_estimators=100, learning_rate=0.2,
                               random_state=42)
# %%
xg_model.fit(X_train, y_train)
y_proba_xg = xg_model.predict_proba(X=X_test)[:, 1]
y_pred_xg = (y_proba_xg >= 0.48).astype(int)
# %%
AUC_xg = roc_auc_score(y_test, y_proba_xg)
f1_xg = f1_score(y_test, y_pred_xg)
precision_xg = precision_score(y_test, y_pred_xg)
recall_xg = recall_score(y_test, y_pred_xg)
print(f'XGBOOST. Precision: {precision_xg:.4f}; recall: {recall_xg:.4f}; AUC: {AUC_xg:.4f}; f1: {f1_xg:.4f}')
# %%
ada_model.fit(X_train, y_train)
y_proba_ada = ada_model.predict_proba(X_test)[:, 1]
y_pred_ada = (y_proba_ada >= 0.48).astype(int)
# %%
AUC_ada = roc_auc_score(y_test, y_proba_ada)
f1_ada = f1_score(y_test, y_pred_ada)
precision_ada = precision_score(y_test, y_pred_ada)
recall_ada = recall_score(y_test, y_pred_ada)
print(f'AdaBoost. Precision: {precision_ada:.4f}; recall: {recall_ada:.4f}; AUC: {AUC_ada:.4f}; f1: {f1_ada:.4f}')
# %%
# Cross validation
def cv(model, threshold, X, y):
    skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
    auc_scores = []
    precisions = []
    recalls = []
    f1s = []

    for train_index, test_index in skf.split(X, y):
        X_train = X.iloc[train_index, :]
        X_test = X.iloc[test_index, :]
        y_train = y.iloc[train_index]
        y_test = y.iloc[test_index]

        y_test, X_test

        model.fit(X_train, y_train)

        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_proba>=threshold).astype(int)

        auc_scores.append(roc_auc_score(y_test, y_proba))
        f1s.append(f1_score(y_test, y_pred))
        precisions.append(precision_score(y_test, y_pred))
        recalls.append(recall_score(y_test, y_pred))

    print(f'Performance of {model} \n')
    print(f'AUC: {np.mean(auc_scores):.4f}; f1: {np.mean(f1s):.4f}; precision: {np.mean(precisions):.4f}; recall: {np.mean(recalls):.4f}')
# %%
# Validate
cv(xg_model, 0.4, X, y)