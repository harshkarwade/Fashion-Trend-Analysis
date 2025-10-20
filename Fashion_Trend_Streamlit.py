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
    'Tier 1': '#4F46E5', # Indigo (High Value)
    'Tier 2': '#10B981', # Emerald (Balanced Growth)
    'Tier 3': '#F59E0B'  # Amber (Volume/Affordable)
}

# --- 1. Mock Data Generation Functions ---
# Generate synthetic sales data
@st.cache_data
def generate_sales_data():
    """Generates a synthetic dataset for fashion sales segmented by city tier, price, and discount."""
    np.random.seed(42)  # for reproducibility
    tiers = ['Tier 1', 'Tier 2', 'Tier 3']
    categories = [
        {'name': 'Dresses', 'T1_ratio': 0.25, 'T2_ratio': 0.15, 'T3_ratio': 0.10, 'base_price': 85},
        {'name': 'T-Shirts', 'T1_ratio': 0.10, 'T2_ratio': 0.30, 'T3_ratio': 0.40, 'base_price': 25},
        {'name': 'Jeans', 'T1_ratio': 0.20, 'T2_ratio': 0.20, 'T3_ratio': 0.15, 'base_price': 60},
        {'name': 'Accessories', 'T1_ratio': 0.15, 'T2_ratio': 0.10, 'T3_ratio': 0.10, 'base_price': 30},
        {'name': 'Outerwear', 'T1_ratio': 0.30, 'T2_ratio': 0.25, 'T3_ratio': 0.25, 'base_price': 120}
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
            
            # --- Sales & Price Simulation ---
            sales = base_sales[tier] * sales_ratio * (1 + (np.random.rand() - 0.5) * 0.2)
            avg_price_unadjusted = cat['base_price'] * (1 + (np.random.rand() - 0.5) * 0.1)
            count = sales / avg_price_unadjusted # Rough count calculation
            
            # --- Discount Simulation (Tier dependent) ---
            if tier == 'Tier 1':
                discount = np.random.uniform(0.05, 0.20) # Lower discount
            elif tier == 'Tier 2':
                discount = np.random.uniform(0.15, 0.30) # Moderate discount
            else: # Tier 3
                discount = np.random.uniform(0.25, 0.45) # Higher discount

            # --- Price Segmentation ---
            if cat['base_price'] > 100:
                price_segment = 'Luxury'
            elif cat['base_price'] > 70:
                price_segment = 'Premium'
            elif cat['base_price'] > 40:
                price_segment = 'Mid-Range'
            else:
                price_segment = 'Low-End'
            
            data.append({
                'Tier': tier,
                'Category': cat['name'],
                'Sales': sales,
                'Count': count,
                'Avg_Price': avg_price_unadjusted,
                'Discount_Pct': discount * 100,
                'Price_Segment': price_segment
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
        {'City': 'Nagpur', 'State': 'Maharashtra', 'Tier': 'Tier 2', 'Lat': 21.1458, 'Lon': 79.0882}, # Nagpur is correctly included here
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
st.title("🛍️ Fashion Market Segmentation & Strategic Dashboard")
st.markdown("Analyze **Sales Value**, **Pricing Strategy**, and **Discount Effectiveness** across **City Tiers** and **Product Categories**.")
st.markdown("---")

# --- 3. Sidebar (Filters) ---
st.sidebar.header("🎯 Target Audience Filters")
selected_tier = st.sidebar.selectbox(
    "1. Select a City Tier:",
    options=['Tier 1', 'Tier 2', 'Tier 3'],
    index=0
)

# New Category Filter
all_categories = ['All Categories'] + df_sales['Category'].unique().tolist()
category_filter = st.sidebar.selectbox(
    "2. Filter by Product Category:",
    options=all_categories
)

# --- 4. Data Filtering and KPI Calculation ---

# Aggregate Total Tier Sales (for Chart 1: Sales Contribution - NOT filtered by Category)
df_tier_agg = df_sales.groupby('Tier').agg(
    TotalSales=('Sales', 'sum'),
    TotalCount=('Count', 'sum')
).reset_index()
df_tier_agg['AvgPrice'] = df_tier_agg['TotalSales'] / df_tier_agg['TotalCount']


# Filter data based on selected tier AND category
df_selected_tier = df_sales[df_sales['Tier'] == selected_tier].copy()

if category_filter != 'All Categories':
    df_selected_tier = df_selected_tier[df_selected_tier['Category'] == category_filter].copy()

# Calculate Key Performance Indicators (KPIs)
total_sales = df_selected_tier['Sales'].sum() if not df_selected_tier.empty else 0
total_count = df_selected_tier['Count'].sum() if not df_selected_tier.empty else 0
avg_price_weighted = total_sales / total_count if total_count > 0 else 0
avg_discount = df_selected_tier['Discount_Pct'].mean() if not df_selected_tier.empty else 0
# Weighted Average Discount
weighted_avg_discount = (df_selected_tier['Discount_Pct'] * df_selected_tier['Sales']).sum() / total_sales if total_sales > 0 else 0


# --- 5. KPI Metrics (Top Row) ---
filter_title = f"{selected_tier}"
if category_filter != 'All Categories':
    filter_title += f" ({category_filter} Focus)"

st.header(f"Key Metrics for {filter_title}")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Total Sales Value",
        value=f"${total_sales:,.0f}",
        help="Total potential revenue for this segment and category filter."
    )

with col2:
    st.metric(
        label="Weighted Avg. Price Point",
        value=f"${avg_price_weighted:,.2f}",
        help="The typical price point for items sold in this filtered segment."
    )

with col3:
    st.metric(
        label="Avg. Discount Rate",
        value=f"{weighted_avg_discount:,.1f}%",
        help="The sales-weighted average discount applied in this segment."
    )

with col4:
    if category_filter == 'All Categories':
        # Find the dominant price segment
        dominant_segment = df_selected_tier.groupby('Price_Segment')['Sales'].sum().idxmax() if not df_selected_tier.empty else 'N/A'
        st.metric(
            label="Dominant Price Segment",
            value=dominant_segment,
            help="The price segment (Luxury, Premium, Mid-Range, Low-End) generating the most sales."
        )
    else:
         st.metric(
            label="Selected Category",
            value=category_filter,
            help="The category currently being analyzed."
        )

st.markdown("---")

# --- 6. Dynamic Client Explanation/Takeaway ---
st.header("Client Presentation Summary")
st.subheader(f"Strategic Focus: **{selected_tier}**")

def get_client_insight(tier, category, discount):
    """Generates a dynamic explanation for client presentation."""
    city_list = df_cities[df_cities['Tier'] == tier]['City'].tolist()
    
    cat_text = f"focusing only on **{category}**" if category != 'All Categories' else "covering all product categories"
    discount_text = f"The average discount is **{discount:,.1f}%**"
    
    if tier == 'Tier 1':
        title = "Premium Market Strategy (Tier 1)"
        insight = f"This segment supports premium pricing (Avg. Price: ${avg_price_weighted:,.2f}). {cat_text.capitalize()}, the strategy must emphasize brand value and exclusivity. {discount_text}, suggesting lower price sensitivity. **Target cities are: {', '.join(city_list)}.**"
        color = TIER_COLORS['Tier 1']
    elif tier == 'Tier 2':
        title = "Balanced Growth Strategy (Tier 2)"
        insight = f"Tier 2 offers a strong balance of volume and value. {cat_text.capitalize()}, a mixed pricing approach is best, focusing on perceived value and promotions. {discount_text}, indicating customers respond well to moderate deals. **Target cities are: {', '.join(city_list)}.**"
        color = TIER_COLORS['Tier 2']
    else:
        title = "Volume & Accessibility Strategy (Tier 3)"
        insight = f"This segment is highly price-sensitive, driven by volume sales. {cat_text.capitalize()}, the strategy must be cost-leadership and affordability. {discount_text}, confirming the need for aggressive pricing to capture market share. **Target cities are: {', '.join(city_list)}.**"
        color = TIER_COLORS['Tier 3']

    return f"""
        <div style='background-color: {color}1A; padding: 15px; border-radius: 10px; border-left: 5px solid {color};'>
        <h4 style='color: {color};'>{title}</h4>
        <p>{insight}</p>
        </div>
    """

st.markdown(get_client_insight(selected_tier, category_filter, weighted_avg_discount), unsafe_allow_html=True)
st.markdown("---")

# --- 7. Geographic Visualization (Map) ---
st.header("🇮🇳 Market Distribution: City Tier Map")

# Filter the city data based on the selected tier
df_filtered_cities = df_cities[df_cities['Tier'] == selected_tier].copy()

# Map chart using Plotly Express
fig_map = px.scatter_mapbox(
    df_filtered_cities, 
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
        center={"lat": 22.0, "lon": 78},
        zoom=4.2
    ),
    margin={"r":0,"t":40,"l":0,"b":0}
)

st.plotly_chart(fig_map, use_container_width=True)

st.markdown("---")

# --- 8. Core Visualizations (Bottom Row - Dynamic) ---
st.header("Detailed Performance Analysis")
chart_col1, chart_col2 = st.columns(2)
chart_col3, chart_col4 = st.columns(2)


# Chart 1: Price Segment Breakdown (Low/Mid/Premium/Luxury)
with chart_col1:
    st.subheader(f"Price Segment Breakdown in {filter_title}")
    if df_selected_tier.empty:
        st.warning("No data for this combination.")
    else:
        df_price_seg = df_selected_tier.groupby('Price_Segment')['Sales'].sum().reset_index()
        # Define order for price segments
        segment_order = ['Low-End', 'Mid-Range', 'Premium', 'Luxury']
        df_price_seg['Price_Segment'] = pd.Categorical(df_price_seg['Price_Segment'], categories=segment_order, ordered=True)
        df_price_seg = df_price_seg.sort_values('Price_Segment')

        fig_seg = px.bar(
            df_price_seg,
            x='Price_Segment',
            y='Sales',
            color='Price_Segment',
            title="Sales by Price Category",
            labels={'Sales': 'Total Sales ($)', 'Price_Segment': 'Price Segment'},
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        fig_seg.update_traces(texttemplate='%{y:$.2s}', textposition='outside')
        fig_seg.update_layout(xaxis_title="", yaxis_title="")
        st.plotly_chart(fig_seg, use_container_width=True)

# Chart 2: Discount Rate Distribution (Price Sensitivity)
with chart_col2:
    st.subheader(f"Discount Rate Distribution in {filter_title}")
    if df_selected_tier.empty:
        st.warning("No data for this combination.")
    else:
        # Create bins for discount percentage
        bins = [0, 10, 20, 30, 40, 50, 100]
        labels = ['0-10%', '10-20%', '20-30%', '30-40%', '40-50%', '50%+']
        
        # We use a copy for safer assignment
        df_temp = df_selected_tier.copy()
        df_temp['Discount_Group'] = pd.cut(df_temp['Discount_Pct'], bins=bins, labels=labels, right=False)
        
        df_discount_agg = df_temp.groupby('Discount_Group', observed=True)['Sales'].sum().reset_index()
        
        fig_disc = px.bar(
            df_discount_agg,
            x='Discount_Group',
            y='Sales',
            color='Discount_Group',
            title="Sales Volume by Discount Range",
            labels={'Sales': 'Total Sales ($)', 'Discount_Group': 'Discount Range'},
            color_discrete_sequence=px.colors.sequential.Cividis_r # Shows higher sales in darker color
        )
        fig_disc.update_traces(texttemplate='%{y:$.2s}', textposition='outside')
        fig_disc.update_layout(xaxis_title="", yaxis_title="")
        st.plotly_chart(fig_disc, use_container_width=True)

# Chart 3: Sales Contribution by City Tier (Bar Chart - Remains overall context)
with chart_col3:
    st.subheader("Market Potential: Overall Sales Contribution")
    
    # This chart remains unfiltered by Category to show total tier potential
    fig_bar = px.bar(
        df_tier_agg,
        x='Tier',
        y='TotalSales',
        color='Tier',
        title="Total Market Value Breakdown (All Categories)",
        labels={'TotalSales': 'Total Sales Value ($)', 'Tier': 'City Tier'},
        color_discrete_map=TIER_COLORS
    )
    for tier in TIER_COLORS.keys():
        if tier == selected_tier:
            # Highlight the selected tier for context
            fig_bar.update_traces(marker_line_color='black', marker_line_width=3, selector=dict(name=tier))

    fig_bar.update_layout(showlegend=False, xaxis={'categoryorder':'total descending'})
    fig_bar.update_traces(texttemplate='%{y:$.2s}', textposition='outside')
    st.plotly_chart(fig_bar, use_container_width=True)

# Chart 4: Product Mix (or Single Category View) for Selected Tier
with chart_col4:
    if category_filter == 'All Categories':
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
    else:
        st.subheader(f"Category Sales vs. Avg Price: {category_filter}")
        # When a single category is selected, show a small table of its data
        df_cat_data = df_selected_tier[['Category', 'Avg_Price', 'Sales', 'Discount_Pct']].head(1)
        st.dataframe(
            df_cat_data.rename(columns={'Avg_Price': 'Avg. Price', 'Discount_Pct': 'Avg. Discount %'}).set_index('Category'),
            column_config={
                "Avg. Price": st.column_config.NumberColumn(format="$%.2f"),
                "Sales": st.column_config.NumberColumn(format="$%d"),
                "Avg. Discount %": st.column_config.NumberColumn(format="%.1f%%"),
            },
            hide_index=False,
            use_container_width=True
        )
        st.info(f"The total sales for **{category_filter}** in **{selected_tier}** is **${total_sales:,.0f}** at a weighted average price of **${avg_price_weighted:,.2f}**.")


st.caption("Sales and City Tier data are synthetic for demonstration purposes.")
