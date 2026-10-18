# %% importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error, mean_absolute_error
from sklearn.ensemble import StackingRegressor, RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline , make_pipeline
from sklearn.compose import ColumnTransformer 
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder ,  StandardScaler ,  OrdinalEncoder 
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBRegressor
from sklearn.base import clone
from sklearn.linear_model import RidgeCV ,Ridge
from geopy.distance import geodesic


# %% reading nessesary files 
# Read the data sets
orders = pd.read_csv("olist_orders_dataset.csv")
customers = pd.read_csv("olist_customers_dataset.csv")
order_items = pd.read_csv("olist_order_items_dataset.csv")
sellers = pd.read_csv("olist_sellers_dataset.csv")
products = pd.read_csv("olist_products_dataset.csv")
geolocation = pd.read_csv("olist_geolocation_dataset.csv")
category_translation = pd.read_csv("product_category_name_translation.csv")
# %%
# See the num of cols and rows in each data set
print('orders:', orders.shape)
print('customers:', customers.shape)
print('order_items:', order_items.shape)
print('sellers:', sellers.shape)
print('products:', products.shape)
print('geolocation:', geolocation.shape)
print('category_translation:', category_translation.shape)
# %%
# Spcify the cols we actually need from each dataset
orders_sel = orders[['order_id', 'customer_id', 'order_status', 'order_purchase_timestamp', 'order_delivered_customer_date']]
customers_sel = customers[['customer_id', 'customer_state', 'customer_zip_code_prefix']]
order_items_sel = order_items[['order_id', 'product_id', 'seller_id', 'freight_value', 'price']]
sellers_sel = sellers[['seller_id', 'seller_state', 'seller_zip_code_prefix']]
products_sel = products[['product_id', 'product_category_name', 'product_weight_g',
                         'product_length_cm', 'product_height_cm', 'product_width_cm']]
# %% taking the avg for every zip code to use it later for km calculations
geo_avg = geolocation.groupby('geolocation_zip_code_prefix', as_index=False)[['geolocation_lat', 'geolocation_lng']].mean()
geo_avg

# %%
# Here we added the transelated category col from the translated data set to the products dataset
products_sel = pd.merge(products_sel, category_translation, on='product_category_name', how='left')
products_sel

# %%
# Here we mearge the dataset we use left to keep all the orders
df = pd.merge(orders_sel, customers_sel, on='customer_id', how='left')
df = pd.merge(df, order_items_sel, on='order_id', how='left')
df = pd.merge(df, sellers_sel, on='seller_id', how='left')
df = pd.merge(df, products_sel, on='product_id', how='left')
df
# %%
# Adding custoumer geo location
df = pd.merge(df, geo_avg, left_on='customer_zip_code_prefix', right_on='geolocation_zip_code_prefix', how='left')
df = df.rename(columns={'geolocation_lat': 'customer_lat', 'geolocation_lng': 'customer_lng'})
df = df.drop(columns='geolocation_zip_code_prefix')
# %%
# Adding seller geo location
df = pd.merge(df, geo_avg, left_on='seller_zip_code_prefix', right_on='geolocation_zip_code_prefix', how='left')
df = df.rename(columns={'geolocation_lat': 'seller_lat', 'geolocation_lng': 'seller_lng'})
df = df.drop(columns='geolocation_zip_code_prefix')
df
# %%
df.shape
# %%
df.head(10)
# %%
df.columns
# %%
df.info()
# %%
df.dtypes
# %%
df.describe()
# %%
df.isnull().sum()
# %%
df.isnull().mean() * 100
# %%
df.duplicated().sum()
# %%
df['freight_value'].hist(bins=40)
# %%
df['product_weight_g'].hist(bins=40);
# %%
df['customer_state'].value_counts().plot(kind='bar');
# %%
df.plot(kind='scatter', x='customer_lng', y='customer_lat', alpha=0.05);
# %%
# change the date from str to datetime
df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])
df['order_delivered_customer_date'] = pd.to_datetime(df['order_delivered_customer_date'])
# %%
# Creat new col that have the count of days from the purchase date to the delivered date
# Delivery time in days = delivered date - purchase date
df['delivery_days'] = (df['order_delivered_customer_date'] - df['order_purchase_timestamp']).dt.total_seconds() / 86400
df[['order_purchase_timestamp', 'order_delivered_customer_date', 'delivery_days']]

# %%
df.columns
# %% needs to be in a function
# Convert degrees to radians
cust_lat = np.radians(df['customer_lat'])
cust_lng = np.radians(df['customer_lng'])
sell_lat = np.radians(df['seller_lat'])
sell_lng = np.radians(df['seller_lng'])

dlat = sell_lat - cust_lat
dlng = sell_lng - cust_lng

# Formula
a = np.sin(dlat / 2) ** 2 + np.cos(cust_lat) * np.cos(sell_lat) * np.sin(dlng / 2) ** 2

# Distance in km
df['dist_km'] = 2 * 6371 * np.arcsin(np.sqrt(a))
df[['customer_state', 'seller_state', 'dist_km']]
# %% Feature Engineering
# New col to indecate if the customer and seller in the same state
# 1 if the customer and the seller are in the same state, else 0
df['same_state'] = np.where(df['customer_state'] == df['seller_state'], 1, 0)
df['same_state']
# %%
# to calc the size of the shipment
# Volume = length x height x width
df['vol'] = df['product_length_cm'] * df['product_height_cm'] * df['product_width_cm']
df[['product_length_cm', 'product_height_cm', 'product_width_cm', 'vol']]

# %% another calculation for distance
def calculate_geodesic(row):
    if pd.isna(row['customer_lat']) or pd.isna(row['seller_lat']) or pd.isna(row['customer_lng']) or pd.isna(row['seller_lng']):
        return np.nan
    cust_loc = (row['customer_lat'], row['customer_lng'])
    sell_loc = (row['seller_lat'], row['seller_lng'])
    return geodesic(cust_loc, sell_loc).kilometers

df['dist_km'] = df.apply(calculate_geodesic, axis=1)

# %% 
df.columns

# %%
df.dtypes

# %% Droping the unnsessary colomns for modeling
bad = (
    ((df['order_status'] == 'delivered') & df['order_delivered_customer_date'].isna()) |
    ((df['order_status'] == 'canceled') & df['order_delivered_customer_date'].notna())
)

print(bad.sum())
# %%
df = df[~bad]

# %% drop all the rows where there is no product no category and no seller 
df = df.dropna(subset=['product_category_name', 'seller_id', 'product_id'], how='all')

# %% drop unnessary colomns
df = df.drop(columns=['product_category_name', 'product_length_cm','product_height_cm', 'product_width_cm'])

# %% rename the colomn product_category_name_english_x
df = df.rename(columns={'product_category_name_english_x': 'product_category_name_english'})
df = df.dropna(subset= 'product_category_name_english')

# %% map sellers id to sellernum once for all
all_sellers = sorted(df['seller_id'].unique())
seller_map = {old: f'seller {i}' for i, old in enumerate(all_sellers, start=1)}
df['seller_id_new'] = df['seller_id'].map(seller_map)
df

# %% delete the duplicated rows where costumer buy muliples of the same item on the same time
cols = ['product_category_name_english', 'product_weight_g', 'order_id',
        'seller_id', 'product_id', 'customer_id', 'order_purchase_timestamp']
df['quantity'] = df.groupby(cols)['order_id'].transform('size')
df = df.drop_duplicates(subset=cols, keep='first')
df['quantity'] = df['quantity'].fillna(1).astype(int)

# %% droping the nulls in y for training
df = df.dropna(subset=['delivery_days'])

# %% save to a new csv_cleaned
df.to_csv('olist_clean.csv', index=False)

# %% EDA 
corr_matrix  = df.corr(numeric_only=True)
lower_no_diag = corr_matrix.where(np.tril(np.ones(corr_matrix.shape), k=-1).astype(bool))
sns.heatmap(lower_no_diag, annot=True)
# %% functions needed
def evaluate(actual,pred):
    return mean_absolute_error(actual,pred), r2_score(actual,pred)

# %%
df.columns

# %% correlations with delevery days (bar chart and box plot)
# bar chart for numerical colomns
corr = df[list( ['freight_value',  'product_weight_g','dist_km', 'same_state', 'vol', 'quantity']) + ['delivery_days']].corr()['delivery_days'].drop('delivery_days').sort_values()
corr.plot(kind='barh', figsize=(6, 5), title='Correlation with delivery_days')
plt.show()

# box plot for binary and categorical colonms
for col in ['customer_state', 'seller_state', 'product_category_name_english', 'same_state']:
    plt.figure(figsize=(14, 5))
    sns.boxplot(data=df, x=col, y='delivery_days', showfliers=False)
    plt.xticks(rotation=90)
    plt.show()

# %%

df["month"] = df["order_purchase_timestamp"].dt.month
df["dow"]   = df["order_purchase_timestamp"].dt.dayofweek
df["day"] = df["order_purchase_timestamp"].dt.day
df["hour"]  = df["order_purchase_timestamp"].dt.hour

# %%
# %% Advanced Feature Engineering & Business Days Calculation
df['is_weekend'] = df['dow'].apply(lambda x: 1 if x >= 5 else 0)
df['weight_to_dist'] = df['product_weight_g'] / (df['dist_km'] + 1)

df['log_dist'] = np.log1p(df['dist_km'])
df['log_weight'] = np.log1p(df['product_weight_g'])
df['log_vol'] = np.log1p(df['vol'])

df['weight_x_dist'] = df['product_weight_g'] * df['dist_km']
df['log_weight_dist'] = np.log1p(df['weight_x_dist'])

# حساب تقريبي لأيام العمل الفعلي باستبعاد عطل نهاية الأسبوع من مدة الشحن
df['estimated_work_days'] = df['delivery_days'] * (1 - (df['is_weekend'] * 0.28))

# %% model features and split

X = df[[ 'customer_zip_code_prefix', 'freight_value','seller_state', 'customer_state' , 'seller_zip_code_prefix', 'product_weight_g','dist_km', 'same_state', 'vol', 'product_category_name_english', 'quantity' ,"month", "dow", "day", "hour"]]
y = df['delivery_days']

X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)

# %% Modeling using HistGradientBoostingRegressor 
# transforming

preprocess = ColumnTransformer(
    [('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan), ['customer_state', 'seller_state', 'product_category_name_english'])],
    remainder='passthrough')

model = Pipeline([
    ('prep', preprocess),
    ('hgb', HistGradientBoostingRegressor( l2_regularization=5.91, max_depth=12, min_samples_leaf=71,  learning_rate=0.05, max_iter=200, max_leaf_nodes=95, max_features=0.7,
        categorical_features=[True] * len(['customer_state', 'seller_state', 'product_category_name_english']) 
        + [False] * len([ 'customer_zip_code_prefix', 'freight_value' , 'seller_zip_code_prefix', 'product_weight_g','dist_km', 'same_state', 'vol', 'quantity',"month", "dow", "day", "hour"]),
        random_state=42))])

# # ----------------------------------------------------------------------------------------------
# # Base model and the forest tree model

# X = df[['customer_zip_code_prefix', 'freight_value','seller_state', 'customer_state' , 'seller_zip_code_prefix', 'product_weight_g','dist_km', 
# 'same_state', 'vol', 'product_category_name_english', 'quantity' ,"month", "dow", "day", "hour"]]

# y = df['delivery_days']

# X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)

# num_cols = ['freight_value', 'product_weight_g', 'vol', 'dist_km', 'quantity', 'same_state',
#             'customer_zip_code_prefix', 'seller_zip_code_prefix',
#             'month', 'dow', 'day', 'hour']

# cat_cols = ['customer_state', 'seller_state', 'product_category_name_english']

# num_pipe = make_pipeline(SimpleImputer(strategy='median'), StandardScaler())
# cat_pipe = make_pipeline(SimpleImputer(strategy='most_frequent'),
#                          OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan))

# preprocessing_pipe = ColumnTransformer([
#     ('num', num_pipe, num_cols),
#     ('cat', cat_pipe, cat_cols),])
# preprocessing_pipe
# rf_pipe = make_pipeline(preprocessing_pipe, RandomForestRegressor(random_state=42))

# #stepname__param
# param_grid = {'randomforestregressor__n_estimators': [50, 100],
#               'randomforestregressor__max_depth': [10, 15, 20],
#               'randomforestregressor__min_samples_leaf': [1, 5, 10]}

# random_grid_model = RandomizedSearchCV(rf_pipe, param_grid, n_iter=4, random_state=42)
# random_grid_model.fit(X_train, y_train)

# best_pipe = random_grid_model.best_estimator_

# y_pred = best_pipe.predict(X_train);    evaluate(y_train, y_pred)
# y_pred_val = best_pipe.predict(X_val);  evaluate(y_val, y_pred_val)

# # This is the base model

# lr_pipe = make_pipeline(preprocessing_pipe, LinearRegression())
# lr_pipe.fit(X_train, y_train)

# %% XGBoost Model (Fatema M part)
model_xgboost = Pipeline([
    ('prep', preprocess),
     ('xgb', XGBRegressor( n_estimators=500, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8,
        tree_method='hist', random_state=42,))])

model_xgboost

# %% model features and split for stacking (Nebras part)
features = [
    'customer_zip_code_prefix', 'freight_value', 'seller_state', 'customer_state', 
    'seller_zip_code_prefix', 'product_weight_g', 'dist_km', 'same_state', 
    'vol', 'product_category_name_english', 'quantity', 'month', 'dow', 'day', 
    'hour', 'is_weekend', 'weight_to_dist', 'log_dist', 'log_weight', 'log_vol',
    'weight_x_dist', 'log_weight_dist', 'estimated_work_days']

X_N = df[features]
y_N = df['delivery_days']

X_N_train_val, X_N_test, y_N_train_val, y_N_test = train_test_split(X_N, y_N, test_size=0.2, random_state=42)
X_N_train, X_N_val, y_N_train, y_N_val = train_test_split(X_N_train_val, y_N_train_val, test_size=0.2, random_state=42)

# %%
# Ultimate Stacking Model (3 Power Models + Ridge Final Estimator)
categorical_cols = ['customer_state', 'seller_state', 'product_category_name_english']
numerical_cols = [f for f in features if f not in categorical_cols]

preprocess = ColumnTransformer(
    [('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan), categorical_cols)],
    remainder='passthrough')

# دمج 3 نماذج عملاقة لتقليل الخطأ تماماً
estimators = [
    ('hgb', HistGradientBoostingRegressor(
        loss='absolute_error',
        l2_regularization=3.0, max_depth=12, min_samples_leaf=25,  
        learning_rate=0.04, max_iter=350, max_features=0.85,
        random_state=42)),
    ('rf', RandomForestRegressor(
        n_estimators=100, max_depth=16, min_samples_split=5, 
        random_state=42, n_jobs=-1)),
    ('et', ExtraTreesRegressor(
        n_estimators=120, max_depth=16, min_samples_split=5, 
        random_state=42, n_jobs=-1))]

base_stack = Pipeline([
    ('prep', preprocess),
    ('stacking', StackingRegressor(estimators=estimators, final_estimator=Ridge(alpha=1.0), n_jobs=-1))])

model_N = TransformedTargetRegressor(
    regressor=base_stack,
    func=np.log1p,
    inverse_func=np.expm1)

# %% New model comparision
models = {
    # name: (pipeline, X_train, y_train, X_test, y_test)
    'LinearRegression':     (lr_pipe, X_train,   y_train,   X_test,   y_test),
    'HistGradientBoosting': (model,   X_train,   y_train,   X_test,   y_test),
    'RandomForest':         (rf_pipe, X_train,   y_train,   X_test,   y_test),
    'XGBRegressor':         (model_xgboost, X_train,   y_train,   X_test,   y_test),
    'Stacking':             (model_N,   X_N_train, y_N_train, X_N_test, y_N_test),}

rows = []
for name, (m, Xtr, ytr, Xte, yte) in models.items():
    pipe = clone(m)                      # fresh copy, original object untouched
    pipe.fit(Xtr, ytr)
    rows.append({
        'Name': name,
        'Train_MAE': mean_absolute_error(ytr, pipe.predict(Xtr)),
        'Test_MAE': mean_absolute_error(yte, pipe.predict(Xte)),})

results = pd.DataFrame(rows).set_index('Name')
print(results.round(2))
ax = results.plot(kind='bar', figsize=(10, 8))

# numbers above the bars
for container in ax.containers:
    ax.bar_label(container, fmt='%.2f', fontsize=12, padding=3)

# remove the y axis and the top, left and right lines
ax.yaxis.set_visible(False)
for side in ['top', 'left', 'right']:
    ax.spines[side].set_visible(False)
ax.set_xlabel('')                      
plt.xticks(rotation=45, fontsize=14)
ax.legend(fontsize=12, frameon=False)
plt.tight_layout()
plt.savefig('model_mae_comparison.png', transparent=True, dpi=300, bbox_inches='tight')
plt.show()

# %% saving the best model (model_N)
joblib.dump(model_N, 'delivery_model.joblib')   # or `stack`, whichever model you pick
columns = list(model_N.feature_names_in_)
columns

# %%
df = pd.read_csv('olist_clean.csv')

customers = df.drop_duplicates('customer_zip_code_prefix').set_index('customer_zip_code_prefix')
customers = customers[['customer_state', 'customer_lat', 'customer_lng']].dropna()
customers.to_csv('app_customers.csv')

offers = df.groupby(['product_category_name_english', 'seller_id_new']).agg(
    seller_state=('seller_state', 'first'),
    seller_zip_code_prefix=('seller_zip_code_prefix', 'first'),
    seller_lat=('seller_lat', 'first'),
    seller_lng=('seller_lng', 'first'),
    price_min=('price', 'min'),
    price_max=('price', 'max'),
    products=('product_id', 'nunique'),
    freight_value=('freight_value', 'median'),
    product_weight_g=('product_weight_g', 'median'),
    vol=('vol', 'median'),).reset_index().dropna(subset=['seller_lat', 'seller_lng'])
offers.to_csv('app_offers.csv', index=False)
