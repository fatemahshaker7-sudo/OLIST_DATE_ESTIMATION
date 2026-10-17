# %% importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import datetime
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error, mean_absolute_error
from sklearn.ensemble import StackingRegressor, RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline , make_pipeline
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder , StandardScaler , OrdinalEncoder 

# %% reading nessesary files 
orders = pd.read_csv("Code/olist_orders_dataset.csv")
customers = pd.read_csv("Code/olist_customers_dataset.csv")
order_items = pd.read_csv("Code/olist_order_items_dataset.csv")
sellers = pd.read_csv("Code/olist_sellers_dataset.csv")
products = pd.read_csv("Code/olist_products_dataset.csv")
geolocation = pd.read_csv("Code/olist_geolocation_dataset.csv")
category_translation = pd.read_csv("Code/product_category_name_translation.csv")

# %%
orders_sel = orders[['order_id', 'customer_id', 'order_status', 'order_purchase_timestamp', 'order_delivered_customer_date']]
customers_sel = customers[['customer_id', 'customer_state', 'customer_zip_code_prefix']]
order_items_sel = order_items[['order_id', 'product_id', 'seller_id', 'freight_value', 'price']]
sellers_sel = sellers[['seller_id', 'seller_state', 'seller_zip_code_prefix']]
products_sel = products[['product_id', 'product_category_name', 'product_weight_g',
                         'product_length_cm', 'product_height_cm', 'product_width_cm']]

# %%
geo_avg = geolocation.groupby('geolocation_zip_code_prefix', as_index=False)[['geolocation_lat', 'geolocation_lng']].mean()

# %%
products_sel = pd.merge(products_sel, category_translation, on='product_category_name', how='left')

# %%
df = pd.merge(orders_sel, customers_sel, on='customer_id', how='left')
df = pd.merge(df, order_items_sel, on='order_id', how='left')
df = pd.merge(df, sellers_sel, on='seller_id', how='left')
df = pd.merge(df, products_sel, on='product_id', how='left')

# %%
df = pd.merge(df, geo_avg, left_on='customer_zip_code_prefix', right_on='geolocation_zip_code_prefix', how='left')
df = df.rename(columns={'geolocation_lat': 'customer_lat', 'geolocation_lng': 'customer_lng'})
df = df.drop(columns='geolocation_zip_code_prefix')

# %%
df = pd.merge(df, geo_avg, left_on='seller_zip_code_prefix', right_on='geolocation_zip_code_prefix', how='left')
df = df.rename(columns={'geolocation_lat': 'seller_lat', 'geolocation_lng': 'seller_lng'})
df = df.drop(columns='geolocation_zip_code_prefix')

# %%
df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])
df['order_delivered_customer_date'] = pd.to_datetime(df['order_delivered_customer_date'])

# %%
df['delivery_days'] = (df['order_delivered_customer_date'] - df['order_purchase_timestamp']).dt.total_seconds() / 86400

# %%
from geopy.distance import geodesic

def calculate_geodesic(row):
    if pd.isna(row['customer_lat']) or pd.isna(row['seller_lat']) or pd.isna(row['customer_lng']) or pd.isna(row['seller_lng']):
        return np.nan
    cust_loc = (row['customer_lat'], row['customer_lng'])
    sell_loc = (row['seller_lat'], row['seller_lng'])
    return geodesic(cust_loc, sell_loc).kilometers

df['dist_km'] = df.apply(calculate_geodesic, axis=1)

# %% Feature Engineering
df['same_state'] = np.where(df['customer_state'] == df['seller_state'], 1, 0)
df['vol'] = df['product_length_cm'] * df['product_height_cm'] * df['product_width_cm']

# %%
bad = (
    ((df['order_status'] == 'delivered') & df['order_delivered_customer_date'].isna()) |
    ((df['order_status'] == 'canceled') & df['order_delivered_customer_date'].notna())
)
df = df[~bad]

# %%
df = df.dropna(subset=['product_category_name', 'seller_id', 'product_id'], how='all')
df = df.drop(columns=['product_category_name', 'product_length_cm','product_height_cm', 'product_width_cm'])
df = df.rename(columns={'product_category_name_english_x': 'product_category_name_english'})
df = df.dropna(subset= 'product_category_name_english')

# %%
all_sellers = sorted(df['seller_id'].unique())
seller_map = {old: f'seller {i}' for i, old in enumerate(all_sellers, start=1)}
df['seller_id_new'] = df['seller_id'].map(seller_map)

# %%
cols = ['product_category_name_english', 'product_weight_g', 'order_id',
        'seller_id', 'product_id', 'customer_id', 'order_purchase_timestamp']
df['quantity'] = df.groupby(cols)['order_id'].transform('size')
df = df.drop_duplicates(subset=cols, keep='first')
df['quantity'] = df['quantity'].fillna(1).astype(int)

# %% Drop nulls & filter extreme outliers (حتى 30 يوم لضمان دقة عالية)
df = df.dropna(subset=['delivery_days'])
df = df[(df['delivery_days'] > 0) & (df['delivery_days'] <= 30)]

df.to_csv('olist_clean.csv', index=False)

# %% functions needed
def evaluate(actual, pred):
    return mean_absolute_error(actual, pred), r2_score(actual, pred)

# %%
df["month"] = df["order_purchase_timestamp"].dt.month
df["dow"]   = df["order_purchase_timestamp"].dt.dayofweek
df["day"] = df["order_purchase_timestamp"].dt.day
df["hour"]  = df["order_purchase_timestamp"].dt.hour

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
features = [
    'customer_zip_code_prefix', 'freight_value', 'seller_state', 'customer_state', 
    'seller_zip_code_prefix', 'product_weight_g', 'dist_km', 'same_state', 
    'vol', 'product_category_name_english', 'quantity', 'month', 'dow', 'day', 
    'hour', 'is_weekend', 'weight_to_dist', 'log_dist', 'log_weight', 'log_vol',
    'weight_x_dist', 'log_weight_dist', 'estimated_work_days'
]

X = df[features]
y = df['delivery_days']

X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)

# %% Ultimate Stacking Model (3 Power Models + Ridge Final Estimator)
categorical_cols = ['customer_state', 'seller_state', 'product_category_name_english']
numerical_cols = [f for f in features if f not in categorical_cols]

preprocess = ColumnTransformer(
    [('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan), categorical_cols)],
    remainder='passthrough'
)

# دمج 3 نماذج عملاقة لتقليل الخطأ تماماً
estimators = [
    ('hgb', HistGradientBoostingRegressor(
        loss='absolute_error',
        l2_regularization=3.0, max_depth=12, min_samples_leaf=25,  
        learning_rate=0.04, max_iter=350, max_features=0.85,
        random_state=42
    )),
    ('rf', RandomForestRegressor(
        n_estimators=100, max_depth=16, min_samples_split=5, 
        random_state=42, n_jobs=-1
    )),
    ('et', ExtraTreesRegressor(
        n_estimators=120, max_depth=16, min_samples_split=5, 
        random_state=42, n_jobs=-1
    ))
]

base_stack = Pipeline([
    ('prep', preprocess),
    ('stacking', StackingRegressor(
        estimators=estimators,
        final_estimator=Ridge(alpha=1.0),
        n_jobs=-1
    ))
])

model = TransformedTargetRegressor(
    regressor=base_stack,
    func=np.log1p,
    inverse_func=np.expm1
)

# %% training and predicting
model.fit(X_train, y_train)

y_pred_train = model.predict(X_train)
print("Train Error (MAE, R2):", evaluate(y_train, y_pred_train))

y_pred_val = model.predict(X_val)
print("Validation Error (MAE, R2):", evaluate(y_val, y_pred_val)) 

# %% saving the model
joblib.dump(model, 'delivery_model.joblib')

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
    vol=('vol', 'median'),
).reset_index().dropna(subset=['seller_lat', 'seller_lng'])
offers.to_csv('app_offers.csv', index=False)