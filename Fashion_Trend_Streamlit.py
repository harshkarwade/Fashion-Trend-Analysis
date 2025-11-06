import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.ensemble import RandomForestClassifier # Imported to show the model type

# Set Streamlit page configuration for a wide, attractive layout
st.set_page_config(
    page_title="Fashion Target Audience Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 0. Setup and Session State ---

# Define consistent colors for the tiers
TIER_COLORS = {
    'Tier 1': '#4F46E5', # Indigo (Premium)
    'Tier 2': '#10B981', # Emerald (Mid-Range)
    'Tier 3': '#F59E0B'  # Amber (Value/Budget)
}

# Initialize session state for ML prediction override
if 'predicted_tier_override' not in st.session_state:
    st.session_state['predicted_tier_override'] = None
if 'selected_tiers_state' not in st.session_state:
    st.session_state['selected_tiers_state'] = ['Tier 1', 'Tier 2', 'Tier 3']


# --- 1. Mock Data Generation Functions ---
@st.cache_data
def generate_sales_data():
    """Generates a synthetic dataset for fashion sales segmented by city tier, price, discount, and GENDER."""
    np.random.seed(42)  # for reproducibility
    tiers = ['Tier 1', 'Tier 2', 'Tier 3']
    genders = ['Men', 'Women', 'Unisex']
    categories = [
        {'name': 'Dresses', 'base_price': 85},
        {'name': 'T-Shirts', 'base_price': 25},
        {'name': 'Jeans', 'base_price': 60},
        {'name': 'Accessories', 'base_price': 30},
        {'name': 'Outerwear', 'base_price': 120}
    ]
    base_sales = {'Tier 1': 350000, 'Tier 2': 250000, 'Tier 3': 180000}
    
    tier_category_ratios = {
        'Tier 1': {'Dresses': 0.25, 'T-Shirts': 0.10, 'Jeans': 0.20, 'Accessories': 0.15, 'Outerwear': 0.30},
        'Tier 2': {'Dresses': 0.15, 'T-Shirts': 0.30, 'Jeans': 0.20, 'Accessories': 0.10, 'Outerwear': 0.25},
        'Tier 3': {'Dresses': 0.10, 'T-Shirts': 0.40, 'Jeans': 0.15, 'Accessories': 0.10, 'Outerwear': 0.25},
    }
    category_gender_map = {
        'Dresses': {'Women': 0.95, 'Men': 0.00, 'Unisex': 0.05},
        'T-Shirts': {'Women': 0.4, 'Men': 0.5, 'Unisex': 0.1},
        'Jeans': {'Women': 0.5, 'Men': 0.5, 'Unisex': 0.0},
        'Accessories': {'Women': 0.6, 'Men': 0.3, 'Unisex': 0.1},
        'Outerwear': {'Women': 0.5, 'Men': 0.4, 'Unisex': 0.1}
    }

    data = []
    for tier in tiers:
        for cat in categories:
            for gender in genders:
                sales_ratio_base = tier_category_ratios[tier][cat['name']] * category_gender_map[cat['name']][gender]
                if sales_ratio_base == 0: continue
                sales = base_sales[tier] * sales_ratio_base * (1 + (np.random.rand() - 0.5) * 0.2)
                price_adj = 1.0
                if gender == 'Women' and cat['name'] in ['Dresses', 'Outerwear']: price_adj = 1.1 
                if gender == 'Men' and cat['name'] in ['Jeans']: price_adj = 1.05 
                avg_price_unadjusted = cat['base_price'] * price_adj * (1 + (np.random.rand() - 0.5) * 0.1)
                count = sales / avg_price_unadjusted
                
                if tier == 'Tier 1': discount = np.random.uniform(0.05, 0.20)
                elif tier == 'Tier 2': discount = np.random.uniform(0.15, 0.30)
                else: discount = np.random.uniform(0.25, 0.45)

                if cat['base_price'] > 100: price_segment = 'Luxury'
                elif cat['base_price'] > 70: price_segment = 'Premium'
                elif cat['base_price'] > 40: price_segment = 'Mid-Range'
                else: price_segment = 'Low-End'
                
                data.append({
                    'Tier': tier, 'Category': cat['name'], 'Gender': gender, 'Sales': sales, 'Count': count,
                    'Avg_Price': avg_price_unadjusted, 'Discount_Pct': discount * 100, 'Price_Segment': price_segment
                })
    return pd.DataFrame(data)

@st.cache_data
def generate_city_tier_data():
    """Generates synthetic city tier data for geographic mapping (partial list for simplicity)."""
    city_data = [
        {'City': 'Ahmedabad', 'State': 'Gujarat', 'Tier': 'Tier 1', 'Lat': 23.0225, 'Lon': 72.5714},
        {'City': 'Bengaluru', 'State': 'Karnataka', 'Tier': 'Tier 1', 'Lat': 12.9716, 'Lon': 77.5946},
        {'City': 'Chennai', 'State': 'Tamil Nadu', 'Tier': 'Tier 1', 'Lat': 13.0827, 'Lon': 80.2707},
        {'City': 'Delhi', 'State': 'Delhi', 'Tier': 'Tier 1', 'Lat': 28.7041, 'Lon': 77.1025},
        {'City': 'Hyderabad', 'State': 'Telangana', 'Tier': 'Tier 1', 'Lat': 17.3850, 'Lon': 78.4867},
        {'City': 'Kolkata', 'State': 'West Bengal', 'Tier': 'Tier 1', 'Lat': 22.5726, 'Lon': 88.3639},
        {'City': 'Mumbai', 'State': 'Maharashtra', 'Tier': 'Tier 1', 'Lat': 19.0760, 'Lon': 72.8777},
        {'City': 'Pune', 'State': 'Maharashtra', 'Tier': 'Tier 1', 'Lat': 18.5204, 'Lon': 73.8567},
        # Tier 2 (Partial list)
        {'City': 'Agra', 'State': 'Uttar Pradesh', 'Tier': 'Tier 2', 'Lat': 27.1767, 'Lon': 78.0081},
        {'City': 'Amritsar', 'State': 'Punjab', 'Tier': 'Tier 2', 'Lat': 31.6340, 'Lon': 74.8723},
        {'City': 'Bhopal', 'State': 'Madhya Pradesh', 'Tier': 'Tier 2', 'Lat': 23.2599, 'Lon': 77.4126},
        # Tier 3 (Partial list)
        {'City': 'Bikaner', 'State': 'Rajasthan', 'Tier': 'Tier 3', 'Lat': 28.0229, 'Lon': 73.3119},
        {'City': 'Cuttack', 'State': 'Odisha', 'Tier': 'Tier 3', 'Lat': 20.4625, 'Lon': 85.8828},
        {'City': 'Gandhinagar', 'State': 'Gujarat', 'Tier': 'Tier 3', 'Lat': 23.2639, 'Lon': 72.6412},
    ]
    return pd.DataFrame(city_data)

# Load the data
df_sales = generate_sales_data()
df_cities = generate_city_tier_data()

# --- 2. ML Prediction Logic (Simulated Random Forest Classifier) ---
def mock_ml_predict_tier(discount_price, original_price, discount_pct):
    """
    Simulates the prediction based on the K-Means clustering outcomes (price & discount being key features).
    """
    if original_price >= 150 and discount_pct <= 20:
        return 'Tier 1' # Premium/High-End, Low Discount
    elif discount_pct >= 35 and discount_price <= 40:
        return 'Tier 3' # Low-End/Value, High Discount
    else:
        return 'Tier 2' # Mid-Range

# --- 3. Dashboard Title and Description ---
st.title("🛍️ Fashion Market Segmentation & Strategic Dashboard")
st.markdown("Analyze **Sales Value**, **Pricing Strategy**, and **Discount Effectiveness** across **City Tiers**, **Product Categories**, and **Gender**.")
st.markdown("---")

# --- 4. Sidebar (Filters & ML Prediction Tool) ---
st.sidebar.header("🎯 Target Audience Filters")

# Function to handle tier multi-select changes
def tier_selection_callback():
    # If the user manually changes the filter, clear the prediction override
    st.session_state['predicted_tier_override'] = None
    st.session_state['selected_tiers_state'] = st.session_state['tier_multiselect']

# Dashboard Filters
selected_tiers = st.sidebar.multiselect(
    "1. Select City Tiers:",
    options=['Tier 1', 'Tier 2', 'Tier 3'],
    default=st.session_state['selected_tiers_state'],
    key='tier_multiselect',
    on_change=tier_selection_callback
)

# Use the state value for the dashboard filtering logic
if st.session_state['predicted_tier_override']:
    # If ML prediction happened, override the selected_tiers variable
    selected_tiers = [st.session_state['predicted_tier_override']]
    
all_categories = ['All Categories'] + df_sales['Category'].unique().tolist()
category_filter = st.sidebar.selectbox(
    "2. Filter by Product Category:",
    options=all_categories
)
all_genders = ['All Genders'] + df_sales['Gender'].unique().tolist()
gender_filter = st.sidebar.selectbox(
    "3. Filter by Gender:",
    options=all_genders
)

# --- ML PREDICTION SECTION ---
st.sidebar.markdown("---")
st.sidebar.header("🤖 ML Model: Target Audience Predictor")
st.sidebar.markdown("**Use the Model** to instantly predict the optimal Tier for a new product.")

with st.sidebar.form("ml_prediction_form"):
    original_price_input = st.number_input("Original Price ($)", min_value=10, value=75, step=5)
    discount_pct_input = st.slider("Discount Percentage (%)", min_value=0, max_value=80, value=15, step=1)
    
    discount_price_calc = original_price_input * (1 - discount_pct_input / 100)
    st.markdown(f"**Calculated Discounted Price: ${discount_price_calc:,.2f}**")
    
    submitted = st.form_submit_button("Predict Audience Tier & Analyze")

if submitted:
    predicted_tier = mock_ml_predict_tier(discount_price_calc, original_price_input, discount_pct_input)
    tier_color = TIER_COLORS.get(predicted_tier, '#374151')
    
    # --- Action: Set Session State to override filters ---
    st.session_state['predicted_tier_override'] = predicted_tier
    st.session_state['selected_tiers_state'] = [predicted_tier]

    # Display prediction result
    st.sidebar.markdown(f"""
        <div style='background-color: {tier_color}1A; padding: 10px; border-radius: 8px; border-left: 4px solid {tier_color}; margin-top: 15px;'>
        <h5 style='color: {tier_color}; margin: 0;'>Predicted Target Tier:</h5>
        <h2 style='color: {tier_color}; margin: 5px 0 0;'>{predicted_tier}</h2>
        </div>
        <p style='font-size: 12px; margin-top: 5px;'>**Dashboard updated** to analyze {predicted_tier} sales profile.</p>
    """, unsafe_allow_html=True)
    
    # Rerun the app to apply the state change
    st.rerun()

# --- 5. Data Filtering and KPI Calculation (Dynamic Filtering) ---
# Filter data using the 'selected_tiers' variable (which is overridden if a prediction was made)
df_selected_tier = df_sales[df_sales['Tier'].isin(selected_tiers)].copy()
if category_filter != 'All Categories':
    df_selected_tier = df_selected_tier[df_selected_tier['Category'] == category_filter].copy()
if gender_filter != 'All Genders':
    df_selected_tier = df_selected_tier[df_selected_tier['Gender'] == gender_filter].copy()

total_sales = df_selected_tier['Sales'].sum() if not df_selected_tier.empty else 0
total_count = df_selected_tier['Count'].sum() if total_sales > 0 else 0
avg_price_weighted = total_sales / total_count if total_count > 0 else 0
weighted_avg_discount = (df_selected_tier['Discount_Pct'] * df_selected_tier['Sales']).sum() / total_sales if total_sales > 0 else 0


# --- 6. KPI Metrics (Top Row) ---
tier_label = ', '.join(selected_tiers) if len(selected_tiers) < 3 else "All Tiers"
filter_title = f"{tier_label}"
if category_filter != 'All Categories': filter_title += f" ({category_filter})"
if gender_filter != 'All Genders': filter_title += f" for {gender_filter}"

st.header(f"Key Metrics for {filter_title}")
col1, col2, col3, col4 = st.columns(4)

with col1: st.metric(label="Total Sales Value", value=f"${total_sales:,.0f}", help="Total potential revenue for this segment.")
with col2: st.metric(label="Weighted Avg. Price Point", value=f"${avg_price_weighted:,.2f}", help="The typical price point for items sold.")
with col3: st.metric(label="Avg. Discount Rate", value=f"{weighted_avg_discount:,.1f}%", help="The sales-weighted average discount applied.")
with col4:
    if category_filter == 'All Categories':
        dominant_segment = df_selected_tier.groupby('Price_Segment')['Sales'].sum().idxmax() if not df_selected_tier.empty else 'N/A'
        st.metric(label="Dominant Price Segment", value=dominant_segment, help="The price segment generating the most sales.")
    else: st.metric(label="Selected Segment", value=f"{category_filter} / {gender_filter}", help="The category and gender currently being analyzed.")

st.markdown("---")

# --- 7. Dynamic Geographic Visualization (Map) ---
st.header("🇮🇳 Market Distribution: City Tier Map")

# Filter the city data based on the dynamic 'selected_tiers'
df_filtered_cities = df_cities[df_cities['Tier'].isin(selected_tiers)].copy()

fig_map = px.scatter_mapbox(
    df_filtered_cities, lat="Lat", lon="Lon", hover_name="City",
    hover_data={"State": True, "Tier": True, "Lat": False, "Lon": False},
    color="Tier", size_max=25, zoom=4.2, center={"lat": 22.0, "lon": 78},
    title=f"Geographic Focus: {', '.join(selected_tiers)} Cities", color_discrete_map=TIER_COLORS,
)

fig_map.update_layout(
    mapbox=dict(style="open-street-map", center={"lat": 22.0, "lon": 78}, zoom=4.2),
    margin={"r":0,"t":40,"l":0,"b":0}
)
st.plotly_chart(fig_map, use_container_width=True)

st.markdown("---")

# --- 8. Core Visualizations (Strategic Pillars) ---
st.header("Detailed Performance Analysis (Strategic Pillars)")
chart_col1, chart_col2 = st.columns(2)
chart_col3, chart_col4 = st.columns(2)

if df_selected_tier.empty:
    st.error("No sales data available for the selected combination of Tiers, Category, and Gender.")

else:
    # Chart 1: Price Segmentation Analysis
    with chart_col1:
        st.subheader(f"1. Price Segmentation Analysis in {filter_title}")
        df_price_seg = df_selected_tier.groupby('Price_Segment')['Sales'].sum().reset_index()
        segment_order = ['Low-End', 'Mid-Range', 'Premium', 'Luxury']
        df_price_seg['Price_Segment'] = pd.Categorical(df_price_seg['Price_Segment'], categories=segment_order, ordered=True)
        df_price_seg = df_price_seg.sort_values('Price_Segment')
        fig_seg = px.bar(df_price_seg, x='Price_Segment', y='Sales', color='Price_Segment', title="Sales by Price Category", labels={'Sales': 'Total Sales ($)'}, color_discrete_sequence=px.colors.qualitative.Bold)
        fig_seg.update_traces(texttemplate='%{y:$.2s}', textposition='outside')
        fig_seg.update_layout(xaxis_title="", yaxis_title="")
        st.plotly_chart(fig_seg, use_container_width=True)

    # Chart 2: Discount Strategy Analysis
    with chart_col2:
        st.subheader(f"2. Discount Strategy Analysis in {filter_title}")
        bins = [0, 10, 20, 30, 40, 50, 100]
        labels = ['0-10%', '10-20%', '20-30%', '30-40%', '40-50%', '50%+']
        df_temp = df_selected_tier.copy()
        df_temp['Discount_Group'] = pd.cut(df_temp['Discount_Pct'], bins=bins, labels=labels, right=False)
        df_discount_agg = df_temp.groupby('Discount_Group', observed=True)['Sales'].sum().reset_index()
        fig_disc = px.bar(df_discount_agg, x='Discount_Group', y='Sales', color='Discount_Group', title="Sales Volume by Discount Range", labels={'Sales': 'Total Sales ($)'}, color_discrete_sequence=px.colors.sequential.Cividis_r)
        fig_disc.update_traces(texttemplate='%{y:$.2s}', textposition='outside')
        fig_disc.update_layout(xaxis_title="", yaxis_title="")
        st.plotly_chart(fig_disc, use_container_width=True)

    # Chart 3: Gender/Tier Split
    with chart_col3:
        if len(selected_tiers) > 1:
            st.subheader(f"3. Focus Trend: Sales Split by Tier")
            df_tier_split = df_selected_tier.groupby('Tier')['Sales'].sum().reset_index()
            fig_tier_split = px.pie(df_tier_split, values='Sales', names='Tier', title="Tier Sales Contribution", hole=.5, color_discrete_map=TIER_COLORS)
            fig_tier_split.update_traces(textposition='inside', textinfo='percent+label', marker=dict(line=dict(color='#FFFFFF', width=1)))
            st.plotly_chart(fig_tier_split, use_container_width=True)
        else:
            st.subheader(f"3. Focus Trend: Sales Split by Gender")
            df_gender_mix = df_selected_tier.groupby('Gender')['Sales'].sum().reset_index()
            fig_gender_bar = px.bar(df_gender_mix, x='Gender', y='Sales', color='Gender', title=f"Gender Revenue Distribution in {selected_tiers[0]}", labels={'Sales': 'Total Sales ($)'}, color_discrete_sequence=px.colors.qualitative.Pastel)
            fig_gender_bar.update_traces(texttemplate='%{y:$.2s}', textposition='outside')
            fig_gender_bar.update_layout(xaxis_title="", yaxis_title="")
            st.plotly_chart(fig_gender_bar, use_container_width=True)

    # Chart 4: Strategic Pricing Comparison
    with chart_col4:
        if category_filter == 'All Categories':
            st.subheader(f"4. Brand Strategy Profile: Product Mix")
            df_category_mix = df_selected_tier.groupby('Category')['Sales'].sum().reset_index()
            fig_pie = px.pie(df_category_mix, values='Sales', names='Category', title=f"Category Revenue Distribution in {filter_title}", hole=.5, color_discrete_sequence=px.colors.sequential.Plasma_r)
            fig_pie.update_traces(textposition='inside', textinfo='percent+label', marker=dict(line=dict(color='#FFFFFF', width=1)))
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.subheader(f"4. Brand Strategy Profile: Strategic Pricing Comparison")
            df_price_comp = df_sales[(df_sales['Category'] == category_filter) & (df_sales['Tier'].isin(selected_tiers))].copy()
            if gender_filter != 'All Genders':
                df_price_comp = df_price_comp[df_price_comp['Gender'] == gender_filter].copy()
            df_price_agg = df_price_comp.groupby('Tier').agg(TotalSales=('Sales', 'sum'), TotalCount=('Count', 'sum')).reset_index()
            df_price_agg['Avg_Price'] = df_price_agg['TotalSales'] / df_price_agg['TotalCount']
            fig_price_comp = px.bar(df_price_agg, x='Tier', y='Avg_Price', color='Tier', title=f"Avg. Price for {category_filter} by City Tier", labels={'Avg_Price': 'Average Price ($)'}, color_discrete_map=TIER_COLORS)
            fig_price_comp.update_layout(showlegend=False, yaxis_tickprefix="$")
            fig_price_comp.update_traces(texttemplate='$%{y:,.2f}', textposition='outside')
            st.plotly_chart(fig_price_comp, use_container_width=True)


# --- 9. Product Deep Dive Analysis ---
st.markdown("---")
st.header("🛒 Product Deep Dive & Portfolio Analysis")

deep_col1, deep_col2 = st.columns(2)

if df_selected_tier.empty:
    st.error("No sales data available for the selected combination of Tiers, Category, and Gender.")

else:
    # Chart 5: Treemap for Overall Product Share
    with deep_col1:
        st.subheader("Product Revenue Share by Tier & Category")
        df_treemap = df_selected_tier.groupby(['Tier', 'Category'])['Sales'].sum().reset_index()
        fig_treemap = px.treemap(df_treemap, path=[px.Constant("All Selected Segments"), 'Tier', 'Category'], values='Sales', color='Tier', color_discrete_map={'(?)': '#262730', 'Tier 1': TIER_COLORS['Tier 1'], 'Tier 2': TIER_COLORS['Tier 2'], 'Tier 3': TIER_COLORS['Tier 3']}, title="Revenue Contribution Breakdown (Tier > Category)", hover_data={'Sales': ':.2s'})
        fig_treemap.update_layout(margin=dict(t=50, l=25, r=25, b=25))
        st.plotly_chart(fig_treemap, use_container_width=True)

    # Chart 6: Portfolio Efficiency (Sales vs. Count)
    with deep_col2:
        st.subheader("Portfolio Efficiency: Sales Value vs. Volume")
        df_scatter = df_selected_tier.groupby('Category').agg(TotalSales=('Sales', 'sum'), TotalCount=('Count', 'sum'), AvgPrice=('Avg_Price', 'mean')).reset_index()
        fig_scatter = px.scatter(df_scatter, x='TotalCount', y='TotalSales', size='AvgPrice', color='Category', hover_name='Category', title="Product Positioning (Sales Volume vs. Revenue)", labels={'TotalCount': 'Total Units Sold (Volume)', 'TotalSales': 'Total Revenue ($)', 'Avg_Price': 'Avg. Price'},)
        fig_scatter.add_vline(x=df_scatter['TotalCount'].mean(), line_width=1, line_dash="dash", line_color="gray")
        fig_scatter.add_hline(y=df_scatter['TotalSales'].mean(), line_width=1, line_dash="dash", line_color="gray")
        fig_scatter.update_layout(xaxis_title="Volume (Units Sold)", yaxis_title="Revenue ($)", legend_title="Category")
        st.plotly_chart(fig_scatter, use_container_width=True)

st.caption("Sales and City Tier data are synthetic for demonstration purposes.")
