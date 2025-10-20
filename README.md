# Fashion Trend Analysis
This interactive geo-strategic analysis segments the market by Tier 1, 2, and 3 Indian cities to define the optimal pricing, product mix, and discount strategy for each target audience segment.
The uploaded Jupyter Notebook provides the foundational product analysis, which is then leveraged by the Streamlit application to perform advanced, strategic target audience segmentation across key business dimensions:

1. Foundational Analysis (From Jupyter Notebook)
Initial Data Review: Confirmed data types and structure using df.info() and df.describe(), ensuring the dataset is clean and ready for analysis.

Product Trend Identification: Visualized the Top 10 Most Common Product Types (e.g., Crew-Neck T-shirts, Straight Kurtas) using a bar plot. This establishes the baseline of what sells most frequently in the market by volume.

2. Strategic Segmentation and Business Profiling (From Streamlit Dashboard Logic)
The following areas represent the strategic analysis layers implemented in the final dashboard, moving beyond simple trend identification to actionable marketing and pricing strategy:

Price Segmentation Analysis
Goal: To understand the distribution of sales across different value buckets and determine where the majority of revenue is generated within a specific city tier or product category.

Methodology: Products are segmented into distinct price tiers: Low-End, Mid-Range, Premium, and Luxury (as calculated in the mock data).

Key Insight: The dashboard uses a dedicated bar chart to show the sales contribution of each segment. This is crucial for determining if a Tier 1 audience primarily buys Luxury Outerwear or if a Tier 3 audience drives volume in the Low-End T-shirt market.

Discount Strategy Analysis
Goal: To measure customer price sensitivity and identify the promotional sweet spot that maximizes sales volume without excessively cutting profit margins.

Methodology: Sales volume is mapped against predefined Discount Rate ranges (e.g., 0-10%, 10-20%, etc.).

Key Insight: The discount distribution chart reveals which discount bracket (e.g., 20-30%) drives the most consumer spending in the selected city tier. This informs promotional calendar planning for events like holiday sales or clearance.

Geographic and Tier-Level Strategy
Goal: To visualize and communicate the market potential based on geographical concentration and affluence.

Methodology: Cities are categorized into Tier 1, Tier 2, and Tier 3 based on predefined criteria, and a dynamic map of India highlights only the cities relevant to the selected segment.

Key Insight: The map provides a clear geographic focus for logistics, physical store expansion, or targeted digital marketing campaigns. The accompanying text summary articulates the strategic mandate for the chosen tier (e.g., Premium Strategy for Tier 1 vs. Volume Strategy for Tier 3).

Brand Level Strategy Profile (Inferred from Product/Price Analysis)
Goal: To define the required brand positioning based on the segment's consumption profile.

Methodology: This profile is built by combining the Avg. Price Point and Dominant Price Segment KPIs.

Key Insight: If a segment shows a high average price and dominant sales in the 'Premium' segment, the brand strategy for that tier must emphasize quality, exclusivity, and brand storytelling. Conversely, a 'Low-End' dominant segment requires a brand strategy focused on value, accessibility, and high rotation of inventory.
