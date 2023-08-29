import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import xgboost as xgb
from sklearn.metrics import mean_squared_error
color_pal = sns.color_palette()
plt.style.use('fivethirtyeight')

df = pd.read_csv("../input/hourly-energy-consumption/PJME_hourly.csv")
df = df.set_index('Datetime')
df.index = pd.to_datetime(df.index)

# Graphic visualization along time
df.plot(style='o',
        figsize=(10, 5),
        color=color_pal[3],
        title='PJME Energy Use in MW')
plt.show()

df.query('PJME_MW < 19_000')['PJME_MW'] \
    .plot(style='o',
          figsize=(15, 5),
          color=color_pal[5],
          title='Outliers')

# Spliting our data into train and test data
train = df.loc[df.index < '01-01-2015']
test = df.loc[df.index >= '01-01-2015']

# Visualizind the splited data
fig, ax = plt.subplots(figsize=(15, 5))
train.plot(ax=ax, label='Training Set', title='Data Train/Test Split')
test.plot(ax=ax, label='Test Set')
ax.axvline('01-01-2015', color='black', ls='--')
ax.legend(['Training Set', 'Test Set'])
plt.show()

# Particular data -> 1 week of energy consumption
(df.loc[(df.index > '01-01-2010') & (df.index < '01-08-2010')]
    .plot(figsize=(15, 5), 
    title='Week Of Data'))
plt.show()
# Jan 01 -> weekday or weekend

# Feature creation:
def create_features(df):
    df = df.copy()
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek # Monday [0]
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    return df

# Visualize our features / target relationship]
fig, ax = plt.subplots(figsize=(10,8))
sns.boxplot(data=df, x='hour', y='PJME_MW') # Give an ideia about the distribution of the dataset
ax.set_title('MW by Hour')
# Results shown -> demonstrate a higher level of consumption on the evening / at night

fig, ax = plt.subplots(figsize=(10,8))
sns.boxplot(data=df, x='month', y='PJME_MW', palette='Blues') # Give an ideia about the distribution of the dataset
ax.set_title('MW by Month')

train = create_features(train)
test = create_features(test)

FEATURES = ['hour', 'dayofweek', 'quarter', 'month', 'year', 'dayofyear']
TARGET = 'PJME_MW'

# Create a feature dataset from our training dataset
X_train = train[FEATURES]
y_train = train[TARGET]

X_test = test[FEATURES]
y_test = test[TARGET]

# Importing a metric form sklearn
# MSE -> more penallty for any prediction that is way off
# Creating a regression model for prediction
reg = xgb.XGBRegressor(n_estimators=1000, # How many trees the boosted algorithm will create
                      early_stopping_rounds=50,
                      learning_rate=0.01)# See if the model has not improve, after 50 trees
reg.fit(X_train, y_train,
       eval_set=[(X_train, y_train), (X_test, y_test)], 
       verbose=100) # Show a series of mse validations
# Results shown -> the MSE is minimized until certain point (indicating 'model overfittng')
# Inputing a 'learning_rate' -> make sure to do not overfit too quickly
# [451] -> stopped after this because our teste validation started to get worst

# Features importance 
fi = pd.DataFrame(data=reg.feature_importances_,
            index=reg.feature_names_in_,
            columns=['importance'])
# Demonstrante data usage -> how does the machine is using our data
# When we have highly correlated features -> does not show how significant is each one the feature
fi.sort_values('importance').plot(kind='barh', title='Feature Importance')
plt.show()

# Forecast on test
test['prediction'] = reg.predict(X_test) # Show a array of predictions based on our tested data
test['prediction'] = reg.predict(X_test)
df.merge(test[['prediction']], how='left', left_index=True, right_index=True)
ax = df[['PJME_MW']].plot(figsize=(15,5))
df['prediction'].plot(ax=ax, style=',')
plt.legend(['Truth Data', 'Predictions'])
ax.set_title('Raw Data and Prediction')
plt.show()
