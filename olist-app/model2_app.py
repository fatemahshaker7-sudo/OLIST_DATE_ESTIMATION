import datetime
import joblib
import numpy as np
import pandas as pd
import streamlit as st


def sine_km(lat1, lng1, lat2, lng2):
    lat1, lng1, lat2, lng2 = map(np.radians, [lat1, lng1, lat2, lng2])
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lng2 - lng1) / 2) ** 2
    return 2 * 6371 * np.arcsin(np.sqrt(a))


@st.cache_resource
def load_model():
    return joblib.load('delivery_model.joblib')


@st.cache_data
def load_data():
    df = pd.read_csv('olist_clean.csv')

    # location of every customer zip code
    customers = df.drop_duplicates('customer_zip_code_prefix').set_index('customer_zip_code_prefix')
    customers = customers[['customer_state', 'customer_lat', 'customer_lng']].dropna()

    # one row for each seller in each category
    customers = pd.read_csv('app_customers.csv', index_col=0)
    offers = pd.read_csv('app_offers.csv')
    return customers, offers


st.title("📦 Find The Fastest Seller")
# loading the model and the data
model = load_model()
customers, offers = load_data()
# inputs (the user should select from select boxes)
category = st.selectbox("Product category", sorted(offers['product_category_name_english'].unique()))
zip_code = st.selectbox("Your zip code", sorted(customers.index))
quantity = st.number_input("Quantity", min_value=1, max_value=50, value=1)
# look for the zip code chosen in the csv and take the datetime of now
cust = customers.loc[zip_code]
now = datetime.datetime.now()

# sellers of this category + the features the model needs
t = offers[offers['product_category_name_english'] == category].copy()
t['customer_state'] = cust['customer_state']
t['customer_zip_code_prefix'] = zip_code
t['dist_km'] = sine_km(cust['customer_lat'], cust['customer_lng'], t['seller_lat'], t['seller_lng'])
t['same_state'] = (t['seller_state'] == cust['customer_state']).astype(int)
t['quantity'] = quantity
t['month'] = now.month
t['dow'] = now.weekday()
t['day'] = now.day
t['hour'] = now.hour
t['days'] = model.predict(t[list(model.feature_names_in_)]).round().astype(int)

# price range like "20 - 500"
low = t['price_min'].round().astype(int).astype(str)
high = t['price_max'].round().astype(int).astype(str)
t['price'] = np.where(low == high, low, low + ' - ' + high)

# what the user sees
show = t.sort_values('days')[['seller_id_new', 'price', 'products', 'freight_value', 'days']]
show.columns = ['Seller', 'Price range', 'Products', 'Shipping cost', 'Delivery (days)']
show = show.round(2).reset_index(drop=True)
# showing the subbheader and the sellers table
st.subheader(f"{len(show)} sellers for {category}")
table = st.dataframe(show, hide_index=True, on_select="rerun", selection_mode="single-row")

# which row did the user click
selected = table.selection.rows
seller = show.iloc[selected[0]] if selected else None

# contact button only if the seller has more than one product
if seller is not None and seller['Products'] > 1:
    with st.popover(f"💬 Contact {seller['Seller']}"):
        message = st.text_area("Your message")
        if st.button("Send"):
            if message.strip():
                st.success("Your message has been sent")
            else:
                st.warning("Write a message first")

if st.button("Order now"):
    if seller is None:
        st.warning("Select a seller first")
    else:
        st.success("Your order has been placed")
