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

# --- 1. Mock Data Generation Function ---
# This function creates synthetic data to simulate sales and product preferences 
# across different city tiers, ensuring the app runs without external files.
@st.cache_data
def generate_mock_data():
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
    # Assume Tier 1 has the highest total market value
    base_sales = {
        'Tier 1': 350000,
        'Tier 2': 250000,
        'Tier 3': 180000
    }

    data = []
    for tier in tiers:
        # Extract the tier number (e.g., '1' from 'Tier 1')
        tier_num = tier.split(' ')[1] 
        for cat in categories:
            sales_ratio = cat[f'T{tier_num}_ratio']
            
            # Sales with slight randomness
            sales = base_sales[tier] * sales_ratio * (1 + (np.random.rand() - 0.5) * 0.2)
            
            # Average price with slight randomness
            avg_price = cat['price'] * (1 + (np.random.rand() - 0.5) * 0.1)
            
            # Calculate unit count
            count = sales / avg_price 
            
            data.append({
                'Tier': tier,
                'Category': cat['name'],
                'Sales': sales,
                'Count': count,
                'Avg_Price': avg_price
            })

    df = pd.DataFrame(data)
    return df

# Load the data
df = generate_mock_data()

# --- 2. Dashboard Title and Description ---
st.title("🛍️ Fashion Target Audience Segmentation")
st.markdown("Analyze **Sales Value**, **Product Mix**, and **Average Price Points** across **City Tiers** to strategically choose your market.")
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
df_tier_agg = df.groupby('Tier').agg(
    TotalSales=('Sales', 'sum'),
    TotalCount=('Count', 'sum')
).reset_index()
# Calculate overall average price for all tiers
df_tier_agg['AvgPrice'] = df_tier_agg['TotalSales'] / df_tier_agg['TotalCount']


# Filter data for the user's selected tier (for KPIs and Chart 2: Product Mix)
df_selected_tier = df[df['Tier'] == selected_tier]

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
    # Use the weighted average price
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

# --- 6. Visualizations (Bottom Row) ---
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
        # Customize colors for better visual separation
        color_discrete_map={
            'Tier 1': '#4F46E5',  # Indigo
            'Tier 2': '#10B981',  # Emerald
            'Tier 3': '#F59E0B'   # Amber
        }
    )
    # Highlight the selected tier by adding a border
    for tier, color in zip(['Tier 1', 'Tier 2', 'Tier 3'], ['#4F46E5', '#10B981', '#F59E0B']):
        if tier == selected_tier:
            fig_bar.update_traces(marker_line_color='black', marker_line_width=3, selector=dict(name=tier))


    fig_bar.update_layout(showlegend=False, xaxis={'categoryorder':'total descending'})
    fig_bar.update_traces(texttemplate='%{y:$.2s}', textposition='outside')
    st.plotly_chart(fig_bar, use_container_width=True)

# Chart 2: Product Mix for Selected Tier (Donut Chart)
with chart_col2:
    st.subheader(f"Product Mix: {selected_tier}")
    
    # Aggregate category sales for the pie chart
    df_category_mix = df_selected_tier.groupby('Category')['Sales'].sum().reset_index()

    fig_pie = px.pie(
        df_category_mix,
        values='Sales',
        names='Category',
        title=f"Category Revenue Distribution in {selected_tier}",
        hole=.5, # Creates a donut chart
        color_discrete_sequence=px.colors.sequential.Plasma_r # Use a professional color sequence
    )
    fig_pie.update_traces(textposition='inside', textinfo='percent+label', marker=dict(line=dict(color='#FFFFFF', width=1)))
    st.plotly_chart(fig_pie, use_container_width=True)

# --- 7. Strategic Insights Section ---
st.markdown("## 💡 Strategic Insights")
if selected_tier == 'Tier 1':
    st.info("Tier 1 cities show the highest overall sales value and support the highest average price points. Focus on **Outerwear** and **Dresses** for maximum revenue.")
elif selected_tier == 'Tier 2':
    st.info("Tier 2 cities offer a strong, balanced market. **Jeans** and **Outerwear** are strong performers. A mid-range price strategy is generally effective here.")
else:
    st.info("Tier 3 cities are highly price-sensitive but offer a large volume opportunity (T-Shirts). Focus on **T-Shirts** and other essential, lower-priced apparel to capture this market.")

st.markdown("---")
st.caption("Data is synthetic and based on estimated ratios for demonstration purposes.")
