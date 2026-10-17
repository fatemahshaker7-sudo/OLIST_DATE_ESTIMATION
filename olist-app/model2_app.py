import datetime
import random
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st

BASE = Path(__file__).parent
STEPS = ["Location", "Shop", "Sellers", "Checkout", "Confirmation"]

# functions needed
def distance_km(lat1, lng1, lat2, lng2):
    """Distance in km between two points on Earth (haversine formula)."""
    lat1, lng1, lat2, lng2 = map(np.radians, [lat1, lng1, lat2, lng2])
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lng2 - lng1) / 2) ** 2
    return 2 * 6371 * np.arcsin(np.sqrt(a))

@st.cache_resource
def load_model():
    return joblib.load(BASE / 'delivery_model.joblib')

@st.cache_data
def load_data():
    customers = pd.read_csv(BASE / 'app_customers.csv', index_col=0)
    offers = pd.read_csv(BASE / 'app_offers.csv')
    return customers, offers

def predict_delivery_days(sellers, zip_code, quantity):
    """Ask the model how many days each seller needs to deliver to this zip code."""
    customer = customers.loc[zip_code]
    now = datetime.datetime.now()

    # we need this for the model to work (one row per seller)
    features = pd.DataFrame({
        # information about the seller
        'seller_state': sellers['seller_state'],
        'seller_zip_code_prefix': sellers['seller_zip_code_prefix'],
        'product_category_name_english': sellers['product_category_name_english'],
        'product_weight_g': sellers['product_weight_g'],
        'vol': sellers['vol'],
        'freight_value': sellers['freight_value'],

        # information about the customer (the same for every seller)
        'customer_state': customer['customer_state'],
        'customer_zip_code_prefix': zip_code,

        # information about the trip between them
        'dist_km': distance_km(customer['customer_lat'], customer['customer_lng'],
                               sellers['seller_lat'], sellers['seller_lng']),
        'same_state': (sellers['seller_state'] == customer['customer_state']).astype(int),

        # information about the order
        'quantity': quantity,
        'month': now.month,
        'dow': now.weekday(),
        'day': now.day,
        'hour': now.hour,})

    features = features[list(model.feature_names_in_)]
    return model.predict(features).round().astype(int)

# this function reruns the page while saving the previous selections in streamlit memory
def go_to_step(number):
    st.session_state.step = number
    st.rerun()

def remembered_selectbox(label, options, key):
    """A selectbox that keeps its value when the user moves between steps, either to edit or for cart info"""
    saved = cart.get(key)
    index = options.index(saved) if saved in options else 0
    cart[key] = st.selectbox(label, options, index=index)

def nav_buttons(next_label, back_step=None, next_disabled=False):
    """Back / Next buttons. Returns True when Next is clicked. Draws the back button only if it is not the first page"""
    back_col, next_col = st.columns(2)
    if back_step is not None and back_col.button("← Back"):
        go_to_step(back_step)
    return next_col.button(next_label, type="primary", disabled=next_disabled)

# The five steps of the customer journey
def location_step():
    """Step 0: where should it go? State, then only the zip codes of that state."""
    remembered_selectbox("Your state", sorted(customers['customer_state'].unique()), 'state')
    zip_codes = sorted(customers[customers['customer_state'] == cart['state']].index)
    remembered_selectbox("Your zip code", zip_codes, 'zip_code')

    if nav_buttons("Continue →"):
        go_to_step(1)

def shop_step():
    """Step 1: what does the customer want to buy? displays category and quantity"""
    categories = sorted(offers['product_category_name_english'].unique())
    remembered_selectbox("Product category", categories, 'category')
    cart['quantity'] = int(st.number_input("Quantity", 1, 50, cart.get('quantity', 1)))

    if nav_buttons("See Sellers →", back_step=0):
        go_to_step(2)

def sellers_step():
    """Step 2: list the sellers with their estimated delivery time and let the customer pick one only"""
    sellers = offers[offers['product_category_name_english'] == cart['category']].copy()

    # estimated delivery for every seller, using the location chosen in step 0
    sellers['days'] = predict_delivery_days(sellers, cart['zip_code'], cart['quantity'])
    now = datetime.datetime.now()
    sellers['arrival'] = [(now + datetime.timedelta(days=int(d))).strftime("%a, %d %b") for d in sellers['days']]

    # fastest first, cheaper first when the days are equal
    sellers = sellers.sort_values(['days', 'price_min']).reset_index(drop=True)

    # price range like "20 - 500"
    low = sellers['price_min'].round().astype(int).astype(str)
    high = sellers['price_max'].round().astype(int).astype(str)
    sellers['price'] = np.where(low == high, low, low + ' - ' + high)

    st.subheader(f"{len(sellers)} sellers for {cart['category']}")
    table = sellers[['seller_id_new', 'price', 'products', 'freight_value', 'days', 'arrival']].round(2)
    table.columns = ['Seller', 'Price range', 'Products', 'Shipping cost', 'Delivery (days)', 'Arrives by']
    event = st.dataframe(table, hide_index=True, on_select="rerun", selection_mode="single-row")

    selected_rows = event.selection.rows
    seller = sellers.iloc[selected_rows[0]] if selected_rows else None

    # the contact button only appears for sellers with more than one product
    if seller is not None and seller['products'] > 1:
        with st.popover(f"💬 Contact {seller['seller_id_new']}"):
            message = st.text_area("Your message")
            if st.button("Send"):
                if message.strip():
                    st.success("Your message has been sent")
                else:
                    st.warning("Write a message first")

    if nav_buttons("Checkout →", back_step=1, next_disabled=seller is None):
        cart['seller'] = seller.to_dict()
        cart['days'] = int(seller['days'])
        cart['arrival'] = seller['arrival']
        go_to_step(3)
    if seller is None:
        st.caption("Select a seller to continue.")

def checkout_step():
    """Step 3: review the order and pay."""
    seller, quantity = cart['seller'], cart['quantity']

    st.subheader("Order summary")
    st.write(f"**{seller['seller_id_new']}** · {cart['category']} × {quantity}")
    st.write(f"Deliver to zip code {cart['zip_code']}, {cart['state']}")
    st.write(f"Estimated arrival: **{cart['arrival']}** (~{cart['days']} days)")
    st.write(f"Shipping: R$ {seller['freight_value']:.2f}")
    total = seller['price_min'] * quantity + seller['freight_value']
    st.markdown(f"### Total from R$ {total:,.2f}")
    st.radio("Payment method", ["Credit Card", "Debit Card", "Pay Pal"], horizontal=True)

    if nav_buttons("Place order", back_step=2):
        cart['order_id'] = random.randint(100000, 999999)
        go_to_step(4)

def confirmation_step():
    """Step 4: the order is placed."""
    seller = cart['seller']
    st.balloons()
    st.success(f"Your order #{cart['order_id']} has been placed")
    st.write(f"{cart['quantity']} item(s) from **{seller['seller_id_new']}** · "
             f"arrives by **{cart['arrival']}**")
    st.markdown("✅ Order placed → 🕒 Preparing → 🚚 Shipped → 📦 Delivered")
    if st.button("Start a new order", type="primary"):
        cart.clear()
        go_to_step(0)

# Main app: load everything, draw the progress bar, run the current step
model = load_model()
customers, offers = load_data()

st.session_state.setdefault('step', 0)           # the app starts from the first page
cart = st.session_state.setdefault('cart', {})   # everything the customer chose so far
step = st.session_state.step

st.title("📦 Find The Fastest Seller")
st.progress((step + 1) / len(STEPS))
st.caption(" › ".join(f"**{name}**" if i == step else name for i, name in enumerate(STEPS)))

[location_step, shop_step, sellers_step, checkout_step, confirmation_step][step]()
