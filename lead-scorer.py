import pandas as pd
import numpy as np


df = pd.read_csv("data/leads-and-customers.csv")
df.head()

df['job_title'] = np.where(np.random.uniform(0, 1, len(df)) > 0.92, np.NaN, df.job_title)
df.job_title.isnull().value_counts()
df.job_title = df.job_title.fillna("UNK")

df['is_manager'] = df.job_title.str.contains("manager|director|supervisor", flags=re.IGNORECASE)
df.head()

# dummify our categorical columns (acquisition_channel, company_size, industry)
dummies = pd.get_dummies(df.acquisition_channel, prefix="acquisition_channel=")
df[dummies.columns] = dummies
pd.crosstab(df.acquisition_channel, df.converted, normalize='index')

dummies = pd.get_dummies(df.company_size, prefix="company_size=")
df[dummies.columns] = dummies
pd.crosstab(df.company_size, df.converted, normalize='index')

dummies = pd.get_dummies(df.industry, prefix="industry=")
df[dummies.columns] = dummies
pd.crosstab(df.industry, df.converted, normalize='index')


for f in ['is_manager', 'days_since_signup', 'visited_pricing', 'registered_for_webinar', 'attended_webinar', 'completed_form']:
    print f
    print pd.crosstab(df[f], df.converted, normalize='index')
    print "*"*80


# create a feature map. for each categorical variable, we need to
# exclude one of the options so we don't violate the dummy variable trap
features = [
    "is_manager",
    "days_since_signup",
    "completed_form",
    "visited_pricing",
    "registered_for_webinar",
    "attended_webinar",
    "acquisition_channel=_Cold Call",
    "acquisition_channel=_Cold Email",
    "acquisition_channel=_Organic Search",
    "acquisition_channel=_Paid Leads",
    # "acquisition_channel=_Paid Search",
    # "company_size=_1-10",
    "company_size=_1000-10000",
    "company_size=_10001+",
    "company_size=_101-250",
    "company_size=_11-50",
    "company_size=_251-1000",
    "company_size=_51-100",
    "industry=_Financial Services",
    "industry=_Furniture",
    "industry=_Heavy Manufacturing",
    "industry=_Scandanavion Design",
    # "industry=_Transportation",
    "industry=_Web & Internet"
]

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

lm = LogisticRegression()
lm.fit(df[features], df.converted)

rf = RandomForestClassifier()
rf.fit(df[features], df.converted)

from sklearn.metrics import classification_report, roc_curve

print classification_report(df.converted, lm.predict(df[features]))
fpr, tpr, thresholds = roc_curve(df.converted, lm.predict_proba(df[features])[:,1], pos_label=1)

print classification_report(df.converted, rf.predict(df[features]))
fpr, tpr, thresholds = roc_curve(df.converted, rf.predict_proba(df[features])[:,1], pos_label=1)

from ggplot import *
data = pd.DataFrame(dict(
    fpr=fpr,
    tpr=tpr,
    thresholds=thresholds
))

ggplot(data, aes(x='fpr', y='tpr')) + geom_line() + geom_abline() + coord_equal()

qplot(rf.predict_proba(df[features])[:,1])
probs = pd.Series(rf.predict_proba(df[features])[:,1])
df['grade'] = grade = pd.cut(probs, 5, labels=["F","D","C","B","A"])

lead_quality = df['grade'].value_counts()
lead_quality = lead_quality.reset_index().sort("index", ascending=False)

ggplot(lead_quality, aes(x='index', weight='grade')) + geom_bar()
