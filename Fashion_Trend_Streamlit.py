import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

# Set Streamlit page configuration for a wide, attractive layout
st.set_page_config(
    page_title="Fashion Target Audience Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define consistent colors for the tiers
TIER_COLORS = {
    'Tier 1': '#4F46E5', # Indigo
    'Tier 2': '#10B981', # Emerald
    'Tier 3': '#F59E0B'  # Amber
}

# --- 1. Mock Data Generation Functions ---
# Generate synthetic sales data
@st.cache_data
def generate_sales_data():
    """Generates a synthetic dataset for fashion sales segmented by city tier."""
    np.random.seed(42)  # for reproducibility
    tiers = ['Tier 1', 'Tier 2', 'Tier 3']
    categories = [
        {'name': 'Dresses', 'T1_ratio': 0.25, 'T2_ratio': 0.15, 'T3_ratio': 0.10, 'price': 85},
        {'name': 'T-Shirts', 'T1_ratio': 0.10, 'T2_ratio': 0.30, 'T3_ratio': 0.40, 'price': 25},
        {'name': 'Jeans', 'T1_ratio': 0.20, 'T2_ratio': 0.20, 'T3_ratio': 0.15, 'price': 60},
        {'name': 'Accessories', 'T1_ratio': 0.15, 'T2_ratio': 0.10, 'T3_ratio': 0.10, 'price': 30},
        {'name': 'Outerwear', 'T1_ratio': 0.30, 'T2_ratio': 0.25, 'T3_ratio': 0.25, 'price': 120}
    ]
    base_sales = {
        'Tier 1': 350000,
        'Tier 2': 250000,
        'Tier 3': 180000
    }

    data = []
    for tier in tiers:
        tier_num = tier.split(' ')[1] 
        for cat in categories:
            sales_ratio = cat[f'T{tier_num}_ratio']
            sales = base_sales[tier] * sales_ratio * (1 + (np.random.rand() - 0.5) * 0.2)
            avg_price = cat['price'] * (1 + (np.random.rand() - 0.5) * 0.1)
            count = sales / avg_price 
            
            data.append({
                'Tier': tier,
                'Category': cat['name'],
                'Sales': sales,
                'Count': count,
                'Avg_Price': avg_price
            })

    return pd.DataFrame(data)

# Generate mock geographic data for India
@st.cache_data
def generate_city_tier_data():
    """Generates synthetic city tier data for mapping with comprehensive list."""
    city_data = [
        # --- Tier 1 Cities ---
        {'City': 'Ahmedabad', 'State': 'Gujarat', 'Tier': 'Tier 1', 'Lat': 23.0225, 'Lon': 72.5714},
        {'City': 'Bengaluru', 'State': 'Karnataka', 'Tier': 'Tier 1', 'Lat': 12.9716, 'Lon': 77.5946},
        {'City': 'Chennai', 'State': 'Tamil Nadu', 'Tier': 'Tier 1', 'Lat': 13.0827, 'Lon': 80.2707},
        {'City': 'Delhi', 'State': 'Delhi', 'Tier': 'Tier 1', 'Lat': 28.7041, 'Lon': 77.1025},
        {'City': 'Hyderabad', 'State': 'Telangana', 'Tier': 'Tier 1', 'Lat': 17.3850, 'Lon': 78.4867},
        {'City': 'Kolkata', 'State': 'West Bengal', 'Tier': 'Tier 1', 'Lat': 22.5726, 'Lon': 88.3639},
        {'City': 'Mumbai', 'State': 'Maharashtra', 'Tier': 'Tier 1', 'Lat': 19.0760, 'Lon': 72.8777},
        {'City': 'Pune', 'State': 'Maharashtra', 'Tier': 'Tier 1', 'Lat': 18.5204, 'Lon': 73.8567},
        
        # --- Tier 2 Cities ---
        {'City': 'Agra', 'State': 'Uttar Pradesh', 'Tier': 'Tier 2', 'Lat': 27.1767, 'Lon': 78.0081},
        {'City': 'Amritsar', 'State': 'Punjab', 'Tier': 'Tier 2', 'Lat': 31.6340, 'Lon': 74.8723},
        {'City': 'Bhopal', 'State': 'Madhya Pradesh', 'Tier': 'Tier 2', 'Lat': 23.2599, 'Lon': 77.4126},
        {'City': 'Bhubaneswar', 'State': 'Odisha', 'Tier': 'Tier 2', 'Lat': 20.2961, 'Lon': 85.8245},
        {'City': 'Chandigarh', 'State': 'Chandigarh', 'Tier': 'Tier 2', 'Lat': 30.7333, 'Lon': 76.7794},
        {'City': 'Coimbatore', 'State': 'Tamil Nadu', 'Tier': 'Tier 2', 'Lat': 11.0168, 'Lon': 76.9558},
        {'City': 'Dehradun', 'State': 'Uttarakhand', 'Tier': 'Tier 2', 'Lat': 30.3165, 'Lon': 78.0322},
        {'City': 'Faridabad', 'State': 'Haryana', 'Tier': 'Tier 2', 'Lat': 28.4089, 'Lon': 77.3178},
        {'City': 'Ghaziabad', 'State': 'Uttar Pradesh', 'Tier': 'Tier 2', 'Lat': 28.6692, 'Lon': 77.4538},
        {'City': 'Indore', 'State': 'Madhya Pradesh', 'Tier': 'Tier 2', 'Lat': 22.7196, 'Lon': 75.8577},
        {'City': 'Jaipur', 'State': 'Rajasthan', 'Tier': 'Tier 2', 'Lat': 26.9124, 'Lon': 75.7873},
        {'City': 'Kanpur', 'State': 'Uttar Pradesh', 'Tier': 'Tier 2', 'Lat': 26.4499, 'Lon': 80.3319},
        {'City': 'Kochi', 'State': 'Kerala', 'Tier': 'Tier 2', 'Lat': 9.9312, 'Lon': 76.2673},
        {'City': 'Lucknow', 'State': 'Uttar Pradesh', 'Tier': 'Tier 2', 'Lat': 26.8467, 'Lon': 80.9462},
        {'City': 'Nagpur', 'State': 'Maharashtra', 'Tier': 'Tier 2', 'Lat': 21.1458, 'Lon': 79.0882},
        {'City': 'Nashik', 'State': 'Maharashtra', 'Tier': 'Tier 2', 'Lat': 19.9975, 'Lon': 73.7898},
        {'City': 'Patna', 'State': 'Bihar', 'Tier': 'Tier 2', 'Lat': 25.5941, 'Lon': 85.1376},
        {'City': 'Surat', 'State': 'Gujarat', 'Tier': 'Tier 2', 'Lat': 21.1702, 'Lon': 72.8311},
        {'City': 'Vadodara', 'State': 'Gujarat', 'Tier': 'Tier 2', 'Lat': 22.3072, 'Lon': 73.1812},
        {'City': 'Visakhapatnam', 'State': 'Andhra Pradesh', 'Tier': 'Tier 2', 'Lat': 17.6868, 'Lon': 83.2185},
        
        # --- Tier 3 Cities ---
        {'City': 'Bikaner', 'State': 'Rajasthan', 'Tier': 'Tier 3', 'Lat': 28.0229, 'Lon': 73.3119},
        {'City': 'Cuttack', 'State': 'Odisha', 'Tier': 'Tier 3', 'Lat': 20.4625, 'Lon': 85.8828},
        {'City': 'Gandhinagar', 'State': 'Gujarat', 'Tier': 'Tier 3', 'Lat': 23.2639, 'Lon': 72.6412},
        {'City': 'Hosur', 'State': 'Tamil Nadu', 'Tier': 'Tier 3', 'Lat': 12.7483, 'Lon': 77.8208},
        {'City': 'Jhansi', 'State': 'Uttar Pradesh', 'Tier': 'Tier 3', 'Lat': 25.4484, 'Lon': 78.5685},
        {'City': 'Junagadh', 'State': 'Gujarat', 'Tier': 'Tier 3', 'Lat': 21.5222, 'Lon': 70.4579},
        {'City': 'Madurai', 'State': 'Tamil Nadu', 'Tier': 'Tier 3', 'Lat': 9.9252, 'Lon': 78.1198},
        {'City': 'Meerut', 'State': 'Uttar Pradesh', 'Tier': 'Tier 3', 'Lat': 28.9845, 'Lon': 77.7064},
        {'City': 'Mysuru', 'State': 'Karnataka', 'Tier': 'Tier 3', 'Lat': 12.2958, 'Lon': 76.6394},
        {'City': 'Rajkot', 'State': 'Gujarat', 'Tier': 'Tier 3', 'Lat': 22.3039, 'Lon': 70.8037},
        {'City': 'Rajahmundry', 'State': 'Andhra Pradesh', 'Tier': 'Tier 3', 'Lat': 17.0000, 'Lon': 81.7833},
        {'City': 'Roorkee', 'State': 'Uttarakhand', 'Tier': 'Tier 3', 'Lat': 29.8753, 'Lon': 77.8976},
        {'City': 'Rohtak', 'State': 'Haryana', 'Tier': 'Tier 3', 'Lat': 28.8955, 'Lon': 76.6066},
        {'City': 'Salem', 'State': 'Tamil Nadu', 'Tier': 'Tier 3', 'Lat': 11.6643, 'Lon': 78.1460},
        {'City': 'Shimla', 'State': 'Himachal Pradesh', 'Tier': 'Tier 3', 'Lat': 31.1048, 'Lon': 77.1734},
        {'City': 'Udaipur', 'State': 'Rajasthan', 'Tier': 'Tier 3', 'Lat': 24.5854, 'Lon': 73.7125},
        {'City': 'Vijayawada', 'State': 'Andhra Pradesh', 'Tier': 'Tier 3', 'Lat': 16.5062, 'Lon': 80.6480},
    ]
    return pd.DataFrame(city_data)

# Load the data
df_sales = generate_sales_data()
df_cities = generate_city_tier_data()

# --- 2. Dashboard Title and Description ---
st.title("🛍️ Fashion Market Segmentation & Strategy Dashboard")
st.markdown("Analyze **Sales Value**, **Product Mix**, and **Geographic Distribution** across **City Tiers** to strategically choose your market and pricing strategy.")
st.markdown("---")

# --- 3. Sidebar (Tier Selector) ---
st.sidebar.header("🎯 Target Audience Filter")
selected_tier = st.sidebar.selectbox(
    "Select a city tier for focused analysis:",
    options=['Tier 1', 'Tier 2', 'Tier 3'],
    index=0
)

# --- 4. Data Filtering and KPI Calculation ---

# Aggregate Total Tier Sales (for Chart 1: Sales Contribution)
df_tier_agg = df_sales.groupby('Tier').agg(
    TotalSales=('Sales', 'sum'),
    TotalCount=('Count', 'sum')
).reset_index()
df_tier_agg['AvgPrice'] = df_tier_agg['TotalSales'] / df_tier_agg['TotalCount']


# Filter data for the user's selected tier (for KPIs and Chart 2: Product Mix)
df_selected_tier = df_sales[df_sales['Tier'] == selected_tier]

# Calculate Key Performance Indicators (KPIs)
total_sales = df_selected_tier['Sales'].sum()
avg_price_weighted = df_selected_tier['Sales'].sum() / df_selected_tier['Count'].sum()
top_category = df_selected_tier.groupby('Category')['Sales'].sum().idxmax()


# --- 5. KPI Metrics (Top Row) ---
st.header(f"Key Metrics for {selected_tier}")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="Total Sales Value",
        value=f"${total_sales:,.0f}",
        help="Total potential revenue from this market segment."
    )

with col2:
    st.metric(
        label="Weighted Avg. Price Point",
        value=f"${avg_price_weighted:,.2f}",
        help="Indicates the typical price point for items sold in this tier. Crucial for pricing strategy."
    )

with col3:
    st.metric(
        label="Most Popular Category (By Sales)",
        value=top_category,
        help="The product category that generates the most revenue."
    )

st.markdown("---")

# --- 6. Dynamic Client Explanation/Takeaway ---
st.header("Client Presentation Summary")
st.subheader(f"Strategy Focus: **{selected_tier}**")

def get_client_insight(tier):
    """Generates a dynamic explanation for client presentation."""
    
    # Get the list of cities for the selected tier
    city_list = df_cities[df_cities['Tier'] == tier]['City'].tolist()
    
    if tier == 'Tier 1':
        return f"""
        <div style='background-color: #e0f2fe; padding: 15px; border-radius: 10px; border-left: 5px solid {TIER_COLORS['Tier 1']};'>
        <h4 style='color: {TIER_COLORS['Tier 1']};'>High-Value Market Strategy (Tier 1)</h4>
        <p>This segment represents the **highest value market** with significant purchasing power, supporting premium pricing. Our focus should be on **high-margin goods** like <b>Outerwear</b> and <b>Dresses</b>. The strategy here is quality, brand visibility, and high average transaction value. **Target cities are: {', '.join(city_list)}.**</p>
        </div>
        """
    elif tier == 'Tier 2':
        return f"""
        <div style='background-color: #ecfdf5; padding: 15px; border-radius: 10px; border-left: 5px solid {TIER_COLORS['Tier 2']};'>
        <h4 style='color: {TIER_COLORS['Tier 2']};'>Balanced Growth Strategy (Tier 2)</h4>
        <p>Tier 2 offers a strong balance of volume and value. While the average price is moderate, there is significant growth potential across categories like <b>Jeans</b> and <b>Outerwear</b>. We recommend a **mixed pricing strategy** focusing on perceived value and promotions to build loyalty. **Target cities are: {', '.join(city_list)}.**</p>
        </div>
        """
    else:
        return f"""
        <div style='background-color: #fffbeb; padding: 15px; border-radius: 10px; border-left: 5px solid {TIER_COLORS['Tier 3']};'>
        <h4 style='color: {TIER_COLORS['Tier 3']};'>Volume & Accessibility Strategy (Tier 3)</h4>
        <p>This segment is **highly price-sensitive** but offers the largest potential for volume sales. The dominant product is <b>T-Shirts</b>. Our strategy must be **cost-leadership**, focusing on essential, affordable apparel and optimizing logistics for wider reach. **Target cities are: {', '.join(city_list)}.**</p>
        </div>
        """

st.markdown(get_client_insight(selected_tier), unsafe_allow_html=True)
st.markdown("---")

# --- 7. Geographic Visualization (Map) ---
st.header("🇮🇳 Market Distribution: City Tier Map")

# Filter the city data based on the selected tier
df_filtered_cities = df_cities[df_cities['Tier'] == selected_tier].copy()

# Map chart using Plotly Express
fig_map = px.scatter_mapbox(
    df_filtered_cities, # <-- USES ONLY THE FILTERED DATA
    lat="Lat",
    lon="Lon",
    hover_name="City",
    hover_data={"State": True, "Tier": True, "Lat": False, "Lon": False},
    color="Tier",
    size_max=25,
    zoom=4.2,  # Tighter zoom to focus only on India
    center={"lat": 22.0, "lon": 78}, # Adjusted center for India
    title=f"Geographic Focus: {selected_tier} Cities",
    color_discrete_map=TIER_COLORS,
)

# Set map style and boundary limits for a tighter India view
fig_map.update_layout(
    mapbox=dict(
        style="open-street-map",
        # Use the same zoom and center for the mapbox config to maintain focus
        center={"lat": 22.0, "lon": 78},
        zoom=4.2
    ),
    margin={"r":0,"t":40,"l":0,"b":0}
)

st.plotly_chart(fig_map, use_container_width=True)

st.markdown("---")

# --- 8. Core Visualizations (Bottom Row) ---
chart_col1, chart_col2 = st.columns(2)

# Chart 1: Sales Contribution by City Tier (Bar Chart)
with chart_col1:
    st.subheader("Market Potential: Sales Contribution by Tier")
    
    fig_bar = px.bar(
        df_tier_agg,
        x='Tier',
        y='TotalSales',
        color='Tier',
        title="Total Market Value Breakdown",
        labels={'TotalSales': 'Total Sales Value ($)', 'Tier': 'City Tier'},
        color_discrete_map=TIER_COLORS
    )
    for tier in TIER_COLORS.keys():
        if tier == selected_tier:
            fig_bar.update_traces(marker_line_color='black', marker_line_width=3, selector=dict(name=tier))

    fig_bar.update_layout(showlegend=False, xaxis={'categoryorder':'total descending'})
    fig_bar.update_traces(texttemplate='%{y:$.2s}', textposition='outside')
    st.plotly_chart(fig_bar, use_container_width=True)

# Chart 2: Product Mix for Selected Tier (Donut Chart)
with chart_col2:
    st.subheader(f"Product Mix: {selected_tier}")
    
    df_category_mix = df_selected_tier.groupby('Category')['Sales'].sum().reset_index()

    fig_pie = px.pie(
        df_category_mix,
        values='Sales',
        names='Category',
        title=f"Category Revenue Distribution in {selected_tier}",
        hole=.5, 
        color_discrete_sequence=px.colors.sequential.Plasma_r
    )
    fig_pie.update_traces(textposition='inside', textinfo='percent+label', marker=dict(line=dict(color='#FFFFFF', width=1)))
    st.plotly_chart(fig_pie, use_container_width=True)


st.caption("Sales and City Tier data are synthetic for demonstration purposes.")
