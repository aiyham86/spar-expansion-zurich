# spar_expansion_app.py

import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px
import json
import folium
from streamlit_folium import st_folium

st.set_page_config(page_title="SPAR Expansion Dashboard", layout="wide")

# Title
st.title("🛒 SPAR Expansion Opportunity Dashboard")

# Load data
@st.cache_data
def load_data():
    forecast_df = pd.read_csv("KTZH_area_forecast.csv")
    supermarkets_df = pd.read_csv("all_supermarkets_with_size_and_rating.csv")
    return forecast_df, supermarkets_df

forecast_df, supermarkets_df = load_data()

# --- Step 1: Fix Population using 2023 actual and 2050 forecast only ---
filtered_df = forecast_df[forecast_df['year'].isin([2023, 2050])]

# Group and pivot population data
population_summary = filtered_df.groupby(['district', 'data type', 'year'])['number'].sum().reset_index()
population_pivot = population_summary.pivot_table(
    index=['district', 'data type'],
    columns='year',
    values='number',
    fill_value=0
).reset_index()
population_pivot.columns.name = None
population_pivot.columns = ['district', 'data_type', 'pop_2023', 'pop_2050']

# Separate actual and forecast, then merge
actual_data = population_pivot[population_pivot['data_type'] == 'Pop_Actual'].copy()
forecast_data = population_pivot[population_pivot['data_type'] == 'Pop_Forecast'].copy()

population_by_district = actual_data[['district', 'pop_2023']].merge(
    forecast_data[['district', 'pop_2050']], on='district', how='outer'
).fillna(0)

population_by_district['growth_rate'] = (
    (population_by_district['pop_2050'] - population_by_district['pop_2023']) /
    population_by_district['pop_2023']
) * 100

# Rename columns to match old logic
population_by_district = population_by_district.rename(columns={
    'pop_2023': 'Pop_Actual',
    'pop_2050': 'Pop_Forecast'
})

# --- Step 2: Map Size Weights and Calculate Capacities ---
size_weights = {'Small': 1, 'Medium': 2, 'Large': 3, 'Unknown': 1}
supermarkets_df['size_weight'] = supermarkets_df['size_category'].map(size_weights)

# SPAR capacity
spar_df = supermarkets_df[supermarkets_df['name'].str.contains('spar', case=False)].copy()
spar_df['size_weight'] = spar_df['size_category'].map(size_weights)
spar_capacity = spar_df.groupby('district')['size_weight'].sum().reset_index(name='spar_capacity')

# Total supermarket capacity
total_capacity = supermarkets_df.groupby('district')['size_weight'].sum().reset_index(name='total_supermarket_capacity')

# Merge capacity data
capacity_df = pd.merge(total_capacity, spar_capacity, on='district', how='left')
capacity_df['spar_capacity'] = capacity_df['spar_capacity'].fillna(0)
capacity_df['spar_capacity_share'] = capacity_df['spar_capacity'] / capacity_df['total_supermarket_capacity']

# --- Step 3: Merge all into master_df ---
master_df = pd.merge(population_by_district, capacity_df, on='district', how='left')
master_df['people_per_spar_unit'] = master_df['Pop_Forecast'] / master_df['spar_capacity'].replace(0, pd.NA)
master_df['people_per_spar_unit'] = master_df['people_per_spar_unit'].fillna(1e6)

# --- Add total supermarket counts to master_df ---
supermarket_counts = supermarkets_df.groupby('district').size().reset_index(name='total_supermarkets')
master_df = pd.merge(master_df, supermarket_counts, on='district', how='left')
master_df['total_supermarkets'] = master_df['total_supermarkets'].fillna(0)
master_df['people_per_supermarket'] = master_df['Pop_Forecast'] / master_df['total_supermarkets'].replace(0, pd.NA)


# --- Sidebar Filters ---
st.sidebar.header("\U0001F50D Filter Options")
st.sidebar.markdown("---")


# Multi-select for districts
st.sidebar.subheader("Districts ")
selected_districts = st.sidebar.multiselect(
    "Select districts to include in the score",
    options=master_df['district'].unique(),
    default=master_df['district'].unique()
)

# Filter by size
st.sidebar.subheader("Size ")
available_sizes = supermarkets_df['size_category'].dropna().unique().tolist()
available_sizes = [size for size in available_sizes if size != "Unknown"]

selected_sizes = st.sidebar.multiselect(
    "Select supermarket sizes",
    options=available_sizes,
    default=available_sizes
)

# Filter by minimum rating
st.sidebar.subheader("Rate")
min_rating = st.sidebar.slider("Minimum average rating", 0.0, 5.0, 0.0, 0.1)

st.sidebar.subheader("⚖️ Scoring Weights")
weight_pop = st.sidebar.slider("Population Forecast Weight", 0.0, 1.0, 0.3, 0.05)
weight_growth = st.sidebar.slider("Growth Rate Weight", 0.0, 1.0, 0.3, 0.05)
weight_share = st.sidebar.slider("SPAR Capacity Share (Lower = Better)", 0.0, 1.0, 0.2, 0.05)
weight_people = st.sidebar.slider("People per SPAR Unit", 0.0, 1.0, 0.2, 0.05)

# Normalize total weight to 1.0
total_weight = weight_pop + weight_growth + weight_share + weight_people
weight_pop /= total_weight
weight_growth /= total_weight
weight_share /= total_weight
weight_people /= total_weight


# Filter data
filtered_master_df = master_df[master_df['district'].isin(selected_districts)]
filtered_supermarkets_df = supermarkets_df[
    (supermarkets_df['district'].isin(selected_districts)) &
    (supermarkets_df['size_category'].isin(selected_sizes)) &
    (supermarkets_df['rating'] >= min_rating)
]



# --- Scoring ---
scoring_df = filtered_master_df[['district', 'Pop_Forecast', 'growth_rate', 'spar_capacity_share', 'people_per_spar_unit']].copy()
scoring_df['people_per_spar_unit'] = scoring_df['people_per_spar_unit'].fillna(1e6)

# Normalize values
scaler = MinMaxScaler()
scoring_df[['Pop_Forecast', 'growth_rate']] = scaler.fit_transform(scoring_df[['Pop_Forecast', 'growth_rate']])
scoring_df['spar_capacity_share'] = 1 - scaler.fit_transform(scoring_df[['spar_capacity_share']])
scoring_df['people_per_spar_unit'] = scaler.fit_transform(scoring_df[['people_per_spar_unit']])

# Calculate weighted score
scoring_df['score'] = (
    scoring_df['Pop_Forecast'] * weight_pop +
    scoring_df['growth_rate'] * weight_growth +
    scoring_df['spar_capacity_share'] * weight_share +
    scoring_df['people_per_spar_unit'] * weight_people
)

# Top 5 districts
top_5 = scoring_df.sort_values(by='score', ascending=False).head(5)

# --- TABS FOR INSIGHTS ---
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Supermarket Summary",
    "📈 Population Summary",
    "🏆 SPAR Expansion Rankings",
    "📉 Supply vs. Demand"
])


with tab1:
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

    # Map: Supermarket Locations
    st.markdown("### 🗺️ Supermarket Locations in Selected Districts")

    # Color map for supermarket brands
    brand_colors = {
        "Spar": "red",
        "Migros": "orange",
        "Coop": "green",
        "Denner": "purple",
        "Aldi": "blue",
        "Lidl": "cadetblue"
    }

    # Size map
    size_radius = {"Small": 4, "Medium": 7, "Large": 10}

    # Create map centered on Zurich
    super_map = folium.Map(location=[47.37, 8.55], zoom_start=11)

    # Plot each supermarket
    for _, row in filtered_supermarkets_df.iterrows():
        brand = row['type'].capitalize()
        color = brand_colors.get(brand, "gray")
        size = size_radius.get(row['size_category'], 5)

        popup = f"<b>{row['name']}</b><br>{brand} ({row['size_category']})<br>⭐ {row['rating']:.1f}"

        folium.CircleMarker(
            location=[row['lat'], row['lng']],
            radius=size,
            popup=popup,
            color=color,
            fill=True,
            fill_opacity=0.7
        ).add_to(super_map)

    # Show the map
    st_folium(super_map, width=800, height=500)

with tab2:
    st.subheader("📈 Population Summary")

    # Split the layout: left for metrics, right for chart
    col1, col2 = st.columns([1, 2])  # Wider right column

    with col1:
        unique_districts = forecast_df['district'].nunique()
        years = forecast_df['year'].unique()
        total_pop = population_by_district['Pop_Actual'].sum()
        forecast_pop = population_by_district['Pop_Forecast'].sum()
        growth_pct = (forecast_pop - total_pop) / total_pop * 100

        st.metric("Districts Covered", unique_districts)
        st.metric("Population Years", f"{years.min()} - {years.max()}")
        st.metric("Total Actual Population", f"{int(total_pop):,}")
        st.metric("Total Forecast Population", f"{int(forecast_pop):,}")
        st.metric("Projected Growth", f"{growth_pct:.2f}%")

    with col2:
        forecast_only = forecast_df[forecast_df['data type'] == 'Pop_Forecast']
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

    districts = population_by_district['district'].unique()
    cols = st.columns(3)

    for i, district in enumerate(districts):
        col = cols[i % 3]
        data = forecast_df[forecast_df['district'] == district]
        actual = data[data['data type'] == 'Pop_Actual']['number'].sum()
        forecast = data[data['data type'] == 'Pop_Forecast']['number'].sum()
        growth = (forecast - actual) / actual * 100
        years = data['year'].unique()

        with col:
            st.markdown(f"### {district}")
            st.metric("Years", f"{years.min()} - {years.max()}")
            st.metric("Actual", f"{int(actual):,}")
            st.metric("Forecast", f"{int(forecast):,}")
            st.metric("Growth", f"{growth:.2f}%")
            st.markdown("<br><br>", unsafe_allow_html=True)  # Add vertical space

    

with tab3:
    st.subheader("\U0001F3C6 Top 5 Districts for SPAR Expansion")
    st.dataframe(top_5.style.format({
        'Pop_Forecast': '{:.2f}',
        'growth_rate': '{:.2f}%',
        'spar_capacity_share': '{:.2%}',
        'people_per_spar_unit': '{:,.0f}',
        'score': '{:.3f}'
    }), use_container_width=True)

with tab4:
    st.subheader("📉 Supply vs. Demand by District")

    # Calculate metrics
    master_df['people_per_market'] = master_df['Pop_Forecast'] / master_df['total_supermarket_capacity'].replace(0, pd.NA)
    master_df['people_per_spar_unit'] = master_df['Pop_Forecast'] / master_df['spar_capacity'].replace(0, pd.NA)

    summary_df = master_df[['district', 'Pop_Forecast', 'total_supermarket_capacity', 'spar_capacity',
                            'people_per_market', 'people_per_spar_unit']].copy()

    # Format nicely
    st.dataframe(summary_df.style.format({
        'Pop_Forecast': '{:,.0f}',
        'total_supermarket_capacity': '{:,.0f}',
        'spar_capacity': '{:,.0f}',
        'people_per_market': '{:,.0f}',
        'people_per_spar_unit': '{:,.0f}'
    }), use_container_width=True)

    # Optional chart: People per SPAR Unit
    st.markdown("### 👥 People per SPAR Unit (Higher = Underserved)")
    fig = px.bar(
        summary_df.sort_values(by='people_per_spar_unit', ascending=False),
        x='district',
        y='people_per_spar_unit',
        labels={'people_per_spar_unit': 'People per SPAR Unit'},
        text='people_per_spar_unit'
    )
    fig.update_traces(texttemplate='%{text:.0f}', textposition='outside')
    fig.update_layout(yaxis_title="People per SPAR", xaxis_title="District", height=500)
    st.plotly_chart(fig, use_container_width=True)




