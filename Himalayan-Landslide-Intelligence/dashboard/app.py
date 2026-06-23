import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib

log_model = joblib.load("logistic_regression_model.pkl")    # loading the Lr model
rf_model = joblib.load("random_forest_model.pkl")    # loading the rf model

st.set_page_config(   # setting the headline in the link section
    page_title="Himalayan Landslide Intelligence System",
    page_icon="⛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

df = pd.read_csv("himalayan_landslides.csv") 

country_map = {
    0: "India",
    1: "Nepal"
}

# actual page contents
st.markdown("""    
# ⛰️ Himalayan Landslide Intelligence System

### Machine Learning Model trained for disaster analytics and severity prediction platform

Analyze landslide patterns, visualize geographic risk zones, and predict disaster severity using Machine Learning.
""")

# filters in the sidebar thet dictate the entire dashboard
st.sidebar.header("🔍 Dashboard Filters")
filtered_df = df.copy()
filtered_df["country_display"] = filtered_df["country_name"].map(country_map)

selected_country = st.sidebar.selectbox(
    "Select Country",
    ["All"] + list(filtered_df["country_display"].unique())
)

selected_season = st.sidebar.selectbox(
    "Select Season",
    ["All"] + list(filtered_df["season"].unique())
)

# applying filters to the entire dashboard
if selected_country != "All":
    filtered_df = filtered_df[
        filtered_df["country_display"] == selected_country
    ]

if selected_season != "All":
    filtered_df = filtered_df[
        filtered_df["season"] == selected_season
    ]

# top aggregations for the page
col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Landslides",
    len(filtered_df)
)

col2.metric(
    "Countries",
    filtered_df["country_display"].nunique()
)

col3.metric(
    "Average Fatalities",
    round(filtered_df["fatality_count"].mean(), 2)
)

col4.metric(
    "Most Common Trigger",
    filtered_df["landslide_trigger"].mode()[0]
)

st.subheader("🌍 Geographic Distribution of Landslides")

# main geographical map 
fig_map = px.scatter_mapbox(
    filtered_df,
    lat="latitude",
    lon="longitude",
    color="severity",
    color_discrete_map={
        "low": "rgba(135, 206, 235, 0.5)",     # skyblue
        "medium": "rgba(255, 255, 0, 0.5)",     # yellow
        "high": "rgba(255, 0, 0, 0.5)"          # red
    },
    hover_name="country_display",
    hover_data=[
        "landslide_trigger",
        "fatality_count"
    ],
    zoom=3,
    height=600
)

fig_map.update_traces(
    marker=dict(size=7)
)

fig_map.update_layout(
    mapbox_style="carto-positron",
    mapbox=dict(
        center=dict(
            lat=28,
            lon=84
        ),
        zoom=3.5
    ),
    margin=dict(
        l=0,
        r=0,
        t=0,
        b=0
    ),
    height=700,
    legend=dict(
        title="Severity",
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    )
)

fig_map.update_traces(
    marker=dict(
        opacity=0.75,
        sizemin=5
    )
)

st.plotly_chart(
    fig_map,
    use_container_width=True
)

# graphs on the main page 
col1, col2 = st.columns(2)

# graphs
with col1:
    country_counts = filtered_df[
        "country_display"
    ].value_counts()

    fig_country = px.bar(
        x=country_counts.index,
        y=country_counts.values,
        color=country_counts.index,
        title="Country-wise Events",
        labels={
            "x": "Country",
            "y": "Landslide Count"
            }
    )

    st.plotly_chart(
        fig_country,
        use_container_width=True
    )

with col2:
    severity_counts = filtered_df[
        "severity"
    ].value_counts()

    fig_severity = px.pie(
        values=severity_counts.values,
        names=severity_counts.index,
        title="Severity Distribution"
    )

    st.plotly_chart(
        fig_severity,
        use_container_width=True
    )

trigger_counts = filtered_df[
    "landslide_trigger"
].value_counts()

fig_trigger = px.bar(
    x=trigger_counts.index,
    y=trigger_counts.values,
    color=trigger_counts.values,
    title="Landslide Trigger Analysis",
    labels={
            "x": "Trigger",
            "y": "Landslide Count"
            }
)

st.plotly_chart(
    fig_trigger,
    use_container_width=True
)

st.header("🤖 Machine Learning Model Performance")
col1, col2 = st.columns(2)

col1.metric(
    "Random Forest Accuracy",
    "65%"
)

col2.metric(
    "Logistic Regression Accuracy",
    "71%"
)

# two tabs for EDA and model prediction
tab1, tab2 = st.tabs([
    "📊 EDA",
    "🤖 Prediction System"
])

# EDA tab
with tab1:
    st.markdown("""
    ## 📊 Exploratory Data Analysis

    Explore regional landslide trends, seasonal patterns, triggers, and severity distribution across the Himalayan region.
    """)

    # Severity Distribution across countries
    country_severity = filtered_df.groupby(
        ["country_display", "severity"]
    ).size().reset_index(name="count")

    fig_country_severity = px.bar(
        country_severity,
        x="country_display",
        y="count",
        color="severity",
        barmode="group",
        title="Severity Distribution Across Countries",
        labels={
            "count": "Severity",
            "country_display": "Country"
            }
    )
    st.plotly_chart(fig_country_severity, use_container_width=True)
    st.divider()

    # fatality spread vs Triggers
    fig_trigger_fatality = px.box(
        filtered_df,
        x="landslide_trigger",
        y="fatality_count",
        color="severity",
        title="Fatality Spread Across Different Triggers",
        labels={
            "landslide_trigger": "Landslide Trigger",
            "fatality_count": "Fatality Count"
            }
    )
    st.plotly_chart(fig_trigger_fatality, use_container_width=True)
    st.divider()

    # Season vs Severity
    season_severity = filtered_df.groupby(
        ["season", "severity"]
    ).size().reset_index(name="count")

    fig_season_severity = px.bar(
        season_severity,
        x="season",
        y="count",
        color="severity",
        barmode="stack",
        title="Seasonal Pattern of Landslide Severity",
        labels={
            "count": "Severity",
            "season": "Season"
            }
    )
    st.plotly_chart(fig_season_severity, use_container_width=True)
    st.divider()

    # Trend chart for month vs landslide count
    month_map = {
        1: "Jan",
        2: "Feb",
        3: "Mar",
        4: "Apr",
        5: "May",
        6: "Jun",
        7: "Jul",
        8: "Aug",
        9: "Sep",
        10: "Oct",
        11: "Nov",
        12: "Dec"
    }

    monthly_trend = filtered_df.groupby(
        "month"
    ).size().reset_index(name="count")
    
    monthly_trend["month_name"] = monthly_trend["month"].map(month_map)

    fig_monthly = px.line(
        monthly_trend,
        x="month_name",
        y="count",
        markers=True,
        title="Monthly Landslide Trend",
        labels={
            "month_name": "Month",
            "count": "Number of Landslides"
        }
    )
    st.plotly_chart(fig_monthly, use_container_width=True)
    st.divider()
 
    # Key Insights
    st.markdown("### 🧠 Key Insights")

    st.info("""
    • Indian Himalayan region have significantly more landslides, although Nepal has a higher ratio of severe landslides.\n
    • Rainfall, closely followed by snowfall/melting snow have the highest fatality count.\n
    • Monsoon has the highest landslides, despite the severity being unbalanced.
    """)

# Model Prediction
with tab2:
    st.subheader("🤖 Landslide Severity Prediction System")
    st.markdown("""
    Predict landslide severity using trained Machine Learning models.
    """)
    st.divider()

    region_coordinates = {
        "Jammu and Kashmir": (34.08, 74.79),
        "Himachal Pradesh": (31.10, 77.17),
        "Uttaranchal": (30.31, 78.03),
        "Sikkim": (27.53, 88.51),
        "Arunachal Pradesh": (28.21, 94.72),
        "Meghalaya": (25.57, 91.88),
        "Assam": (26.20, 92.93),
        "Mizoram": (23.16, 92.94),
        "Nagaland": (26.15, 94.56),
        "Manipur": (24.66, 93.90),
        "Tripura": (23.94, 91.99),   
        "Gandaki": (28.37, 84.43),
        "Seti": (29.27, 80.94),
        "Bheri": (28.53, 81.78),
        "Narayani": (27.68, 84.43),
        "Dhawalagiri": (28.70, 83.50),
        "Rapti": (28.32, 82.52),
        "Bagmati": (27.70, 85.30),
        "Sagarmatha": (27.70, 86.72),
        "Janakpur": (26.72, 85.92),
        "Lumbini": (27.47, 83.27),
        "Koshi": (26.68, 87.27),
        "Karnali": (29.38, 82.18),
        "Mechi": (26.87, 88.08),
        "Mahakali": (29.84, 80.54)
        }

    # user inputs
    col1, col2 = st.columns(2)

    with col1:
        country = st.selectbox(
            "🌍 Country",
            ["India", "Nepal"]
        )
        season = st.selectbox(
            "🌦️ Season",
            ["monsoon", "non-monsoon"]
        )
        trigger = st.selectbox(
            "🌧️ Landslide Trigger",
            [
                "rainfall",
                "snowfall_snowmelt",
                "construction",
                "unknown",
                "earthquake",
                "tropical_cyclone",
                "no_apparent_trigger",
                "flooding",
                "mining",
                "other"
            ]
        )
        landslide_size = st.selectbox(
            "📏 Landslide Size",
            ["small", "medium", "large", "very_large"]
        )

    with col2:
        admin_division = st.selectbox(
            "🗺️ Administrative Division",
            [
                'Jammu and Kashmir',
                'Himachal Pradesh',
                'Uttaranchal',
                'Sikkim',
                'Arunachal Pradesh',
                'Meghalaya',
                'Assam',
                'Mizoram',
                'Nagaland',
                'Manipur',
                'Tripura',
                'Gandaki',
                'Seti',
                'Bheri',
                'Narayani',
                'Dhawalagiri',
                'Rapti',
                'Bagmati',
                'Sagarmatha',
                'Janakpur',
                'Lumbini',
                'Koshi',
                'Karnali',
                'Mechi',
                'Mahakali'
            ]
        )
        latitude, longitude = region_coordinates[admin_division]
        st.caption(
           f"📍 Coordinates used: Latitude {latitude}, Longitude {longitude}"
           ) 
        month = st.selectbox(
            "📅 Month",
            list(range(1, 13))
        )
        landslide_setting = st.selectbox(
            "⛰️ Landslide Setting",
            [
                'unknown',
                'infrastructure',
                'human_modified',
                'natural',
                'disturbed_nature',
                'other'
            ]
        )
    st.divider()
   
    # choose a model
    selected_model = st.selectbox(
        "🧠 Select Prediction Model",
        ["Random Forest", "Logistic Regression"]
    )
    st.divider()

    if st.button("🚀 Predict Severity"):
        country_map = {
            "India": 0,
            "Nepal": 1
        }
        size_map = {
            "small": 1,
            "medium": 2,
            "large": 3,
            "very_large": 4
        }
        input_data = pd.DataFrame({
            "country_name": [country_map[country]],
            "season": [season],
            "landslide_trigger": [trigger],
            "month": [month],
            "landslide_setting": [landslide_setting],
            "location_accuracy_km": [5.0],
            "latitude": [latitude],
            "longitude": [longitude],
            "gazetteer_distance": [10.0],
            "year": [2024],
            "landslide_category": ["landslide"],
            "landslide_size": [size_map[landslide_size]],
            "admin_division_name": [admin_division]
        })   

        # model prediction
        if selected_model == "Random Forest":
            prediction = rf_model.predict(input_data)[0]
            probability = rf_model.predict_proba(input_data).max()
        else:
            prediction = log_model.predict(input_data)[0]
            probability = log_model.predict_proba(input_data).max()
        predicted_severity = prediction.capitalize()
        confidence = round(probability * 100, 2)
        st.divider()

        # display results
        st.subheader("📊 Prediction Result")
        if predicted_severity == "Low":
            st.success(
                f"🟢 Predicted Severity: {predicted_severity}"
            )
        elif predicted_severity == "Medium":
            st.warning(
                f"🟡 Predicted Severity: {predicted_severity}"
            )
        else:
            st.error(
                f"🔴 Predicted Severity: {predicted_severity}"
            )
        st.metric(
            "Prediction Confidence",
            f"{confidence}%"
        )
        st.progress(float(probability))

        # prediction summary
        st.info(f"""
        ### 📌 Prediction Summary
        - **Country:** {country}
        - **Season:** {season}
        - **Trigger:** {trigger}
        - **Month:** {month}
        - **Landslide Setting:** {landslide_setting}
        - **Model Used:** {selected_model}
        """)

        # risk message
        if predicted_severity == "High":
            st.error("""
            ⚠️ High-risk landslide conditions detected.
            
            Immediate monitoring and disaster preparedness are recommended.
            """)
        elif predicted_severity == "Medium":
            st.warning("""
            ⚠️ Moderate landslide risk detected.
            
            Environmental monitoring is advised.
            """)
        else:
            st.success("""
            ✅ Low landslide severity predicted.
            
            Current environmental conditions indicate relatively lower risk.
            """)

