import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px
import json
import folium
from streamlit_folium import st_folium

st.set_page_config(page_title="SPAR Expansion Zurich", layout="wide")

st.markdown("""
<h1 style='color:#005A9C;'>🧠 ZüriScope</h1>
<h3 style='font-weight:normal;'>Strategy meets Insight for Smarter SPAR Expansion</h3>
<p><strong>Presented by Aiyham, Juan, Marco</strong></p>
""", unsafe_allow_html=True)


st.markdown("<style>h1, h2, h3 { color: #005A9C; }</style>", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    forecast_df = pd.read_csv("KTZH_area_forecast.csv")
    supermarkets_df = pd.read_csv("all_supermarkets_with_size_and_rating.csv")
    return forecast_df, supermarkets_df

forecast_df, supermarkets_df = load_data()
area = pd.read_csv('Area_size.csv')

# Clean column names to fix invisible characters
area.columns = area.columns.str.strip().str.replace('\xa0', ' ')



# --- Population processing (2023 vs 2050)
filtered_df = forecast_df[forecast_df['year'].isin([2023, 2050])]
population_summary = filtered_df.groupby(['district', 'data type', 'year'])['number'].sum().reset_index()
population_pivot = population_summary.pivot_table(index=['district', 'data type'], columns='year', values='number', fill_value=0).reset_index()
population_pivot.columns.name = None
population_pivot.columns = ['district', 'data_type', 'pop_2023', 'pop_2050']

actual_data = population_pivot[population_pivot['data_type'] == 'Pop_Actual']
forecast_data = population_pivot[population_pivot['data_type'] == 'Pop_Forecast']

population_by_district = actual_data[['district', 'pop_2023']].merge(
    forecast_data[['district', 'pop_2050']], on='district', how='outer'
).fillna(0)

population_by_district['growth_rate'] = (
    (population_by_district['pop_2050'] - population_by_district['pop_2023']) / 
    population_by_district['pop_2023']
) * 100

population_by_district = population_by_district.rename(columns={
    'pop_2023': 'Pop_Actual',
    'pop_2050': 'Pop_Forecast'
})

# Merge actual and forecast data
if len(actual_data) > 0 and len(forecast_data) > 0:
    # Use 2023 as baseline (actual), 2030 and 2045 as forecasts
    population_analysis = actual_data[['district', 'pop_2023']].merge(
        forecast_data[['district', 'pop_2050']], 
        on='district', 
        how='outer'
    ).fillna(0)
    
    
    population_analysis['growth_rate_2050'] = (
        (population_analysis['pop_2050'] - population_analysis['pop_2023']) /
        population_analysis['pop_2023']
    ) * 100
    
else:
    # Alternative approach: use the filtered data directly by year
    population_2023 = filtered_df[filtered_df['year'] == 2023].groupby('district')['number'].sum()
    population_2050 = filtered_df[filtered_df['year'] == 2050].groupby('district')['number'].sum()
    
    # Create analysis DataFrame
    population_analysis = pd.DataFrame({
        'district': population_2023.index,
        'pop_2023': population_2023.values,
        'pop_2050': population_2050.reindex(population_2023.index, fill_value=0).values
    })
    
    # Calculate growth rates  
    population_analysis['growth_rate_2050'] = (
        (population_analysis['pop_2050'] - population_analysis['pop_2023']) /
        population_analysis['pop_2023']
    ) * 100

# Calculate the density of the district and absolute growth population
population_analysis['growth_abs'] = population_analysis['pop_2050']- population_analysis['pop_2023']
population_analysis = population_analysis.merge(area, on='district', how='left')
# Calculate and sort by correct column name
population_analysis['density_growth'] = population_analysis['growth_abs'] / population_analysis['Area (Area km²)']
population_analysis = population_analysis.sort_values(by='density_growth', ascending=False)

# Count total supermarkets per district
total_supermarkets = supermarkets_df.groupby('district').size().reset_index(name='total_supermarkets')

# Count SPAR supermarkets per district
spar_supermarkets = supermarkets_df[supermarkets_df['name'].str.contains('spar', case=False)]
spar_counts = spar_supermarkets.groupby('district').size().reset_index(name='spar_supermarkets')

# Merge both counts together
supermarket_coverage = pd.merge(total_supermarkets, spar_counts, on='district', how='left')

# --- Capacity calculations
size_weights = {'Small': 1, 'Medium': 2, 'Large': 3, 'Unknown': 1}
supermarkets_df['size_weight'] = supermarkets_df['size_category'].map(size_weights)

spar_df = supermarkets_df[supermarkets_df['name'].str.contains('spar', case=False)].copy()
spar_df['size_weight'] = spar_df['size_category'].map(size_weights)

spar_capacity = spar_df.groupby('district')['size_weight'].sum().reset_index(name='spar_capacity')
total_capacity = supermarkets_df.groupby('district')['size_weight'].sum().reset_index(name='total_supermarket_capacity')

capacity_df = pd.merge(total_capacity, spar_capacity, on='district', how='left')
capacity_df['spar_capacity'] = capacity_df['spar_capacity'].fillna(0)
capacity_df['spar_capacity_share'] = capacity_df['spar_capacity'] / capacity_df['total_supermarket_capacity']

master_df = pd.merge(population_by_district, capacity_df, on='district', how='left')
master_df['people_per_spar_unit'] = master_df['Pop_Forecast'] / master_df['spar_capacity'].replace(0, pd.NA)
master_df['people_per_spar_unit'] = master_df['people_per_spar_unit'].fillna(1e6)

# Supermarket counts
supermarket_counts = supermarkets_df.groupby('district').size().reset_index(name='total_supermarkets')
master_df = pd.merge(master_df, supermarket_counts, on='district', how='left')
master_df['total_supermarkets'] = master_df['total_supermarkets'].fillna(0)
master_df['people_per_supermarket'] = master_df['Pop_Forecast'] / master_df['total_supermarkets'].replace(0, pd.NA)

# --- Sidebar Filters
st.sidebar.header("🔎 Filter Options")
selected_districts = st.sidebar.multiselect(
    "Select Districts",
    options=master_df['district'].unique(),
    default=master_df['district'].unique()
)

all_brands = sorted(supermarkets_df['type'].dropna().unique())
selected_brands = st.sidebar.multiselect("Select Brands", options=all_brands, default=all_brands)
available_sizes = [s for s in supermarkets_df['size_category'].dropna().unique() if s != "Unknown"]
selected_sizes = st.sidebar.multiselect("Select Store Sizes", options=available_sizes, default=available_sizes)
min_rating = st.sidebar.slider("Minimum Rating", 0.0, 5.0, 0.0, 0.1)



# --- Apply filters
filtered_master_df = master_df[master_df['district'].isin(selected_districts)]
filtered_supermarkets_df = supermarkets_df[
    (supermarkets_df['district'].isin(selected_districts)) &
    (supermarkets_df['size_category'].isin(selected_sizes)) &
    (supermarkets_df['rating'] >= min_rating) &
    (supermarkets_df['type'].isin(selected_brands))
]

# --- Scoring
scoring_df = filtered_master_df[['district', 'Pop_Forecast', 'growth_rate', 'spar_capacity_share', 'people_per_spar_unit']].copy()
scoring_df['people_per_spar_unit'] = scoring_df['people_per_spar_unit'].fillna(1e6)

scaler = MinMaxScaler()
scoring_df[['Pop_Forecast', 'growth_rate']] = scaler.fit_transform(scoring_df[['Pop_Forecast', 'growth_rate']])
scoring_df['spar_capacity_share'] = 1 - scaler.fit_transform(scoring_df[['spar_capacity_share']])
scoring_df['people_per_spar_unit'] = scaler.fit_transform(scoring_df[['people_per_spar_unit']])


# Define opportunity_df here using your cleaned and merged data
opportunity_df = pd.merge(population_analysis, supermarket_coverage, on='district', how='left')
# You must ensure these columns exist: 'district', 'pop_2050', 'density_growth', 'spar_coverage_rate', 'people_per_supermarket'
opportunity_df['spar_coverage_rate'] = opportunity_df['spar_supermarkets'] / opportunity_df['total_supermarkets']
opportunity_df['people_per_spar'] = opportunity_df['pop_2050'] / opportunity_df['spar_supermarkets'].replace(0, pd.NA)
opportunity_df['people_per_supermarket']=opportunity_df['pop_2050'] / opportunity_df['total_supermarkets']
opportunity_df['growth_rate'] = opportunity_df['growth_rate_2050']  # use 2045 growth for ranking
best_opportunities = opportunity_df.sort_values(by=['density_growth', 'people_per_supermarket'], ascending=[False, True])

#  Sample structure — replace with your actual DataFrame if different


# Normalize and score
scoring_df = opportunity_df[['district', 'pop_2050', 'density_growth', 'spar_coverage_rate', 'people_per_supermarket']].copy()

scaler = MinMaxScaler()
scoring_df[['pop_2050', 'density_growth']] = scaler.fit_transform(scoring_df[['pop_2050', 'density_growth']])
scoring_df[['spar_coverage_rate', 'people_per_supermarket']] = 1 - scaler.fit_transform(scoring_df[['spar_coverage_rate', 'people_per_supermarket']])

scoring_df['score'] = (
    scoring_df['pop_2050'] * 0.3 +
    scoring_df['density_growth'] * 0.3 +
    scoring_df['spar_coverage_rate'] * 0.2 +
    scoring_df['people_per_supermarket'] * 0.2
)

top_ranked = scoring_df.sort_values(by='score', ascending=False)



# --- Tabs
tab_intro, tab1, tab2, tab3, tab4 = st.tabs([
    "📢 Introduction",
    "🏪 Available Spaces in Kreis 4",
    "📊 Opportunity Scoring",
    "🛒 Supermarket Summary",
    "📈 Population Summary"
])

with tab_intro:
    st.header("📍 SPAR Expansion Strategy in Canton Zurich")
    st.markdown("""
    Welcome to the final presentation of our **SPAR Expansion Dashboard**.
    
    This tool supports SPAR’s growth by combining:
    - 📊 Forecasted population trends (2023–2050)
    - 🏪 Supermarket capacity and competition
    - 📈 District-level scoring based on multiple weighted criteria

    > Use the filters on the left to explore strategic insights and see our top district recommendation.
    """)

with tab3:
    st.subheader("Supermarket Overview")

    total = len(filtered_supermarkets_df)
    total_spar = filtered_supermarkets_df['name'].str.contains("spar", case=False).sum()
    by_brand = filtered_supermarkets_df['type'].value_counts()
    by_size = filtered_supermarkets_df['size_category'].value_counts()
    avg_ratings = filtered_supermarkets_df.groupby('type')['rating'].mean()


    st.metric("Total Supermarkets", f"{total}")
    st.metric("SPAR Supermarkets", f"{total_spar}")

    st.write("**Supermarkets by Brand (with Avg. Rating):**")
    cols = st.columns(len(by_brand))

    for i, brand in enumerate(by_brand.index):
        count = by_brand[brand]
        rating = avg_ratings.get(brand, None)

        with cols[i]:
            st.metric(label=brand, value=f"{count}")
            st.markdown(f"⭐ {rating:.2f}" if pd.notna(rating) else "⭐ N/A")

    st.write("**Supermarkets by Size Category:**")
    cols_size = st.columns(len(by_size))

    for i, size in enumerate(by_size.index):
        count = by_size[size]
        with cols_size[i]:
            st.metric(label=size, value=f"{count}")


    st.caption(f"Showing {len(filtered_supermarkets_df)} supermarkets based on selected filters.")

    st.markdown("### 🗺️ Supermarket Locations in Selected Districts (Icon View)")

    # Side-by-side layout: map | legend
    col_map, col_legend = st.columns([5, 1])

    with col_map:
        super_map = folium.Map(location=[47.37, 8.55], zoom_start=11)

        icon_config = {
            'migros':  ('shopping-cart', 'orange'),
            'coop':    ('shopping-cart', 'purple'),
            'lidl':    ('shopping-cart', 'lightblue'),
            'aldi':    ('shopping-cart', 'darkblue'),
            'spar':    ('shopping-cart', 'green'),
            'denner':  ('shopping-cart', 'red')
        }

        for _, row in filtered_supermarkets_df.iterrows():
            brand = row['type'].strip().lower()
            icon_name, icon_color = icon_config.get(brand, ('question-sign', 'gray'))

            popup = (
                f"<b>{row['name']}</b><br>"
                f"{brand.capitalize()} ({row['size_category']})<br>"
                f"⭐ {row['rating']:.1f}"
            )

            folium.Marker(
                location=[row['lat'], row['lng']],
                popup=popup,
                icon=folium.Icon(color=icon_color, icon=icon_name, prefix='fa')
            ).add_to(super_map)

        st_folium(super_map, width=1400, height=600)

    with col_legend:
        st.markdown("#### 🧭 Legend")
        for brand, (icon, legend_color) in icon_config.items():
            brand_display = brand.capitalize()
            st.markdown(
                f"""
                <div style='margin-bottom:8px; display:flex; align-items:center;'>
                    <div style='width:16px; height:16px; background:{legend_color}; border-radius:50%; margin-right:8px;'></div>
                    <span style='font-size:14px'>{brand_display}</span>
                </div>
                """,
                unsafe_allow_html=True
            )


with tab4:
    st.subheader("📈 Population Summary")

    # Split layout: left for metrics, right for chart
    col1, col2 = st.columns([1, 2])

    with col1:
        unique_districts = population_by_district['district'].nunique()
        total_pop_2023 = population_by_district['Pop_Actual'].sum()
        total_pop_2050 = population_by_district['Pop_Forecast'].sum()
        growth_pct = (total_pop_2050 - total_pop_2023) / total_pop_2023 * 100

        st.metric("Districts Covered", unique_districts)
        st.metric("Population Years", "2023 - 2050")
        st.metric("Total Actual Population", f"{int(total_pop_2023):,}")
        st.metric("Total Forecast Population", f"{int(total_pop_2050):,}")
        st.metric("Projected Growth", f"{growth_pct:.2f}%")

    with col2:
        forecast_only = forecast_df[
            (forecast_df['data type'] == 'Pop_Forecast') & 
            (forecast_df['year'].between(2023, 2050))
        ]
        pop_trend = forecast_only.groupby(['year', 'district'])['number'].sum().reset_index()

        fig = px.line(
            pop_trend,
            x='year',
            y='number',
            color='district',
            title='Forecasted Population Trend by District',
            labels={'number': 'Population', 'year': 'Year'}
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("📊 Summary for All Districts")

    cols = st.columns(3)
    for i, row in population_by_district.iterrows():
        col = cols[i % 3]
        district = row['district']
        actual = row['Pop_Actual']
        forecast = row['Pop_Forecast']
        growth = row['growth_rate']

        with col:
            st.markdown(f"### {district}")
            st.metric("Years", "2023 - 2050")
            st.metric("Actual", f"{int(actual):,}")
            st.metric("Forecast", f"{int(forecast):,}")
            st.metric("Growth", f"{growth:.2f}%")
            st.markdown("<br><br>", unsafe_allow_html=True)


with tab2:
    st.subheader("🏆 Top 5 Recommended Districts for SPAR")

    st.markdown("""
    To identify the best district for SPAR expansion, we calculated a composite **score** based on four strategic factors:

    | Factor                     | Why it Matters                           | Scoring Direction     |
    |---------------------------|------------------------------------------|-----------------------|
    | **Forecast Population**   | Indicates future demand                  | 🔼 Higher is better   |
    | **Density Growth**        | Measures how fast demand is growing      | 🔼 Higher is better   |
    | **SPAR Coverage Rate**    | Shows how much SPAR is already present   | 🔽 Lower is better    |
    | **People per Supermarket**| Indicates how competitive the market is  | 🔼 Higher is better    |

    The final score combines these using custom weights:
    `30% Pop + 30% Growth + 20% SPAR Coverage + 20% Market Pressure`
    """)

    st.markdown("### 🥇 Top Ranked Districts")
    st.dataframe(top_ranked.head(12).style.format({
        'pop_2050': '{:,.0f}',
        'density_growth': '{:.2f}',
        'spar_coverage_rate': '{:.2%}',
        'people_per_supermarket': '{:,.1f}',
        'score': '{:.3f}'
    }), use_container_width=True)

    top_district = top_ranked.iloc[0]['district']
    st.success(f"🎯 **Recommendation: Open the next SPAR in {top_district}** — it ranked highest based on all opportunity factors.")

    # Optional bar chart
    st.markdown("### 📊 Opportunity Score by District")
    fig = px.bar(
        top_ranked.head(12),
        x='district',
        y='score',
        text='score',
        labels={'score': 'Opportunity Score'},
        title='Top Districts by Score'
    )
    fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
    fig.update_layout(yaxis_title="Score", xaxis_title="District", height=500)
    st.plotly_chart(fig, use_container_width=True)

     # --- Sub-district (Kreis) Analysis: Focus on Zürich
    st.markdown("## 🏙️ Best Kreis in Zürich for SPAR Expansion")

    st.markdown("""
    To refine our expansion strategy within the city of Zürich, we performed a more granular analysis at the **Kreis** level (sub-districts).

    This scoring uses:
    - **🏙️ Density** (proxy for demand concentration)
    - **🏪 SPAR Coverage Rate**
    - **👥 People per Supermarket**

    | Factor                     | Why it Matters                             | Scoring Direction |
    |---------------------------|---------------------------------------------|-------------------|
    | **Density**               | More dense = more concentrated demand      | 🔼 Higher is better |
    | **SPAR Coverage Rate**    | Lower = more room for SPAR                 | 🔽 Lower is better |
    | **People per Supermarket**| Higher = more pressure on current supply   | 🔼 Higher is better |

    We applied custom weights to these indicators:
    `50% Density + 20% SPAR Coverage + 30% Market Pressure`
    """)

    # Load and prepare data
    kreis_df = pd.read_csv("Kreis_Density.csv")
    kreis_df.rename(columns={'Kreis': 'kreis'}, inplace=True)

    supermarkets_df = pd.read_csv("supermarkets_with_kreis.csv")
    supermarkets_df.rename(columns={'bezeichnung': 'kreis', 'name_left': 'name'}, inplace=True)

    # Supermarket counts
    total_supermarkets = supermarkets_df.groupby('kreis').size().reset_index(name='total_supermarkets')
    spar_df = supermarkets_df[supermarkets_df['name'].str.contains('spar', case=False)]
    spar_counts = spar_df.groupby('kreis').size().reset_index(name='spar_supermarkets')

    # Merge and fill
    supermarket_coverage = pd.merge(total_supermarkets, spar_counts, on='kreis', how='left')
    supermarket_coverage['spar_supermarkets'] = supermarket_coverage['spar_supermarkets'].fillna(0).astype(int)

    # Merge with kreis data
    opportunity_df = pd.merge(kreis_df, supermarket_coverage, on='kreis', how='left')
    opportunity_df['spar_coverage_rate'] = opportunity_df['spar_supermarkets'] / opportunity_df['total_supermarkets']
    opportunity_df['people_per_spar'] = opportunity_df['pop_2024'] / opportunity_df['spar_supermarkets'].replace(0, pd.NA)
    opportunity_df['people_per_supermarket'] = opportunity_df['pop_2024'] / opportunity_df['total_supermarkets']

    # Scoring
    from sklearn.preprocessing import MinMaxScaler
    scoring_df = opportunity_df[['kreis', 'pop_2024', 'Density', 'spar_coverage_rate', 'people_per_supermarket']].copy()

    scaler = MinMaxScaler()
    scoring_df['Density_score'] = scaler.fit_transform(scoring_df[['Density']])
    scoring_df[['spar_coverage_score', 'people_per_supermarket_score']] = 1 - scaler.fit_transform(
        scoring_df[['spar_coverage_rate', 'people_per_supermarket']]
    )

    scoring_df['score'] = (
        scoring_df['Density_score'] * 0.5 +
        scoring_df['spar_coverage_score'] * 0.2 +
        scoring_df['people_per_supermarket_score'] * 0.3
    )

    # Show top ranked Kreise
    top_kreise = scoring_df.sort_values(by='score', ascending=False)
    st.markdown("### 🔝 Top Ranked Kreise in Zürich")
    st.dataframe(top_kreise.head(10).style.format({
        'pop_2024': '{:,.0f}',
        'Density_score': '{:.2f}',
        'spar_coverage_score': '{:.2f}',
        'people_per_supermarket_score': '{:.2f}',
        'score': '{:.3f}'
    }), use_container_width=True)

    # Highlight top recommendation
    best_kreis = top_kreise.iloc[0]['kreis']
    st.success(f"🏆 **Recommendation: Open the next SPAR in _{best_kreis}_** — it has the highest score based on demand density, market gap, and current competition.")
    
    # Load your GeoJSON for kreise (adapt path if needed)
    with open("stzh.adm_stadtkreise_a.json", "r") as f:
        kreise = json.load(f)
        top_kreise['geo_id'] = top_kreise['kreis'].str.extract(r'(\d+)')[0].astype(str)

        # Choropleth map for kreis scores
        st.markdown("### 🗺️ Weighted Score by Kreis (Map View)")

        fig = px.choropleth_mapbox(
            top_kreise,
            geojson=kreise,
            locations="geo_id",
            featureidkey="properties.name",
            color="score",
            hover_data={
                "kreis": True,
                "score": ':.2f',
                "Density": ':,.0f',
                "people_per_supermarket":':,.0f',
                "spar_coverage_rate":':.0%',
            },
            center={"lat": 47.38, "lon": 8.54},
            mapbox_style="carto-positron",
            zoom=10.8,
            opacity=0.7,
            width=900,
            height=600,
            labels={
                "kreis": "Kreis",
                "score": "Weighted score",
                "Density": "Population per km²",
                "people_per_supermarket": "Population per supermarket",
                "spar_coverage_rate": "number of SPAR out of total supermarkets"
            },
            title="<b>Weighted score per Kreis</b>",
            color_continuous_scale="Blues"
        )

        fig.update_layout(
            margin={"r": 0, "t": 35, "l": 0, "b": 0},
            font_family="Balto",
            font_color="black",
            hoverlabel=dict(bgcolor="white", font_size=12, font_family="Balto", font_color="black"),
            title=dict(font_size=20, x=0.38, y=0.96, xanchor='center', yanchor='bottom')
        )

        st.plotly_chart(fig, use_container_width=True)

with tab1:
    st.header("🏪 Retail Space Opportunities in Kreis 4")

    st.markdown("""
    After identifying **Kreis 4** as the most promising sub-district for SPAR expansion, we researched available commercial rental spaces in the area. Below are three shortlisted properties found online:
    """)

    property_list = [
        {
            "name": "Location 1 – 275 m², Ground floor",
            "url": "https://en.comparis.ch/immobilien/marktplatz/details/show/34349760",
            "address": "Hohlstrasse 188, 8004 Zürich",
            "note": "Larger unit suitable for full SPAR supermarket with loading access.",
            "image": "location_1.png"
        },
        {
            "name": "Location 2 – 130 m², Ground floor",
            "url": "https://en.comparis.ch/immobilien/marktplatz/details/show/35021188",
            "address": "Zeughausstrasse 3, 8004 Zürich",
            "note": "Great visibility, surrounded by residential buildings.",
            "image": "location_2.png"
        },
        {
            "name": "Location 3 – 250 m², Ground floor ",
            "url": "https://en.comparis.ch/immobilien/marktplatz/details/show/34477585",
            "address": "Ernastrasse 22, 8004 Zürich",
            "note": "Ideal for a medium-large SPAR format. High foot traffic area.",
            "image": "location_3.png"
        }
    ]

    for p in property_list:
        st.subheader(p["name"])
        st.markdown(f"📍 **Address:** {p['address']}")
        st.markdown(f"🔗 [View Listing]({p['url']})")
        st.markdown(f"📝 _{p['note']}_")
        st.image(p["image"], use_column_width=10)
        st.markdown("---")

