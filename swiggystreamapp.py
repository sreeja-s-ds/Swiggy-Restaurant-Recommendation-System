import streamlit as st
import pandas as pd
import numpy as np
import joblib

@st.cache_data
def load_resources():
    df = pd.read_csv("cleaned_data.csv")

    df['cost'] = df['cost'].astype(str).str.replace('₹', '', regex=False).str.replace(',', '', regex=False).str.strip()
    df['cost'] = pd.to_numeric(df['cost'], errors='coerce').fillna(0).astype(int)
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce').fillna(0.0)

    le_city = joblib.load("le_city.joblib")
    le_cuisine = joblib.load("le_cuisine.joblib")
    kmeans = joblib.load("kmeans_model_fixed.joblib")

    df['city_enc'] = le_city.transform(df['city'])
    df['cuisine_enc'] = le_cuisine.transform(df['cuisine'])

    return df, le_city, le_cuisine, kmeans

df, le_city, le_cuisine, kmeans_model = load_resources()

st.sidebar.title("🍴 Swiggy's Restaurant Recommender")
page = st.sidebar.selectbox("Navigation", ["Home", "Recommendations"])

if page == "Home":
    st.title("🍽️ Swiggy's Restaurant Recommendation App")
    st.write("Use the sidebar to find restaurant recommendations based on your preferences.")

if page == "Recommendations":
    st.title("🔍 Find Your Ideal Restaurant🍽️")

    selected_city = st.selectbox("📍City", sorted(df['city'].unique()))
    available_cuisines = sorted(df[df['city'] == selected_city]['cuisine'].unique())
    selected_cuisine = st.selectbox("🍴Cuisine", available_cuisines)

    with st.expander(f"🍱 Cuisines available in {selected_city}"):
        st.markdown(", ".join(available_cuisines))

    min_rating = st.slider("⭐Minimum Rating", 0.0, 5.0, 3.5)
    max_cost = st.slider("💰Maximum Cost (₹)", 100, 2000, 500)
    sort_by = st.selectbox("Sort By", ["None", "Rating: High to Low", "Cost: Low to High", "Cost: High to Low"])

    city_filter = df['city'] == selected_city
    cuisine_filter = df['cuisine'] == selected_cuisine
    rating_filter = df['rating'] >= min_rating
    cost_filter = df['cost'] <= max_cost

    full_filter = city_filter & cuisine_filter & rating_filter & cost_filter
    filtered_df = df[full_filter]

    if not filtered_df.empty:
        if sort_by == "Rating: High to Low":
            filtered_df = filtered_df.sort_values(by="rating", ascending=False)
        elif sort_by == "Cost: Low to High":
            filtered_df = filtered_df.sort_values(by="cost", ascending=True)
        elif sort_by == "Cost: High to Low":
            filtered_df = filtered_df.sort_values(by="cost", ascending=False)

        st.success(f"✅ Found {len(filtered_df)} matching restaurant(s).")
        st.subheader("🍽️ Matching Restaurants")
        for _, row in filtered_df.iterrows():
            st.markdown(f"### {row['name']}")
            st.write(f"📍 City: {row['city']}")
            st.write(f"🍴 Cuisine: {row['cuisine']}")
            st.write(f"⭐ Rating: {row['rating']} ({row.get('rating_count', 'Too Few Ratings')} reviews)")
            st.write(f"💰 Cost: ₹{row['cost']}")
            st.markdown("---")
    else:
        st.warning("⚠️ No exact match found with all filters.")
        partial_filter = city_filter & cuisine_filter
        filtered_df = df[partial_filter]
        if not filtered_df.empty:
            st.info(f"🔍 Found {len(filtered_df)} restaurants matching only city and cuisine.")
        else:
            st.error("No restaurants found with the selected filters.")
            st.info(f"🍴 Available cuisines in {selected_city}:")
            for cuisine in available_cuisines:
                st.markdown(f"- {cuisine}")
            st.stop()

    ref_row = filtered_df.iloc[0]
    try:
        city_code = le_city.transform([ref_row['city']])[0]
        cuisine_code = le_cuisine.transform([ref_row['cuisine']])[0]
        feature_vector = np.array([[city_code, cuisine_code, ref_row['rating'], ref_row['cost']]])

        cluster_label = kmeans_model.predict(feature_vector)[0]
        df['cluster'] = kmeans_model.labels_

        same_cluster = df[(df['cluster'] == cluster_label) & (df['city'] == selected_city)]
        recommendations = same_cluster[same_cluster.index != ref_row.name].head(5)

        if not recommendations.empty:
            st.subheader("🍽️ Other Recommended Restaurants")
            for _, row in recommendations.iterrows():
                st.markdown(f"### {row['name']}")
                st.write(f"📍 City: {row['city']}")
                st.write(f"🍴 Cuisine: {row['cuisine']}")
                st.write(f"⭐ Rating: {row['rating']} ({row.get('rating_count', 'Too Few Ratings')} reviews)")
                st.write(f"💰 Cost: ₹{row['cost']}")
                st.markdown("---")
        else:
            st.info("No other similar restaurants found in the same cluster for the selected city.")

    except Exception as e:
        st.error(f"Error generating recommendations: {e}")

    with st.expander("📊 Filter Match Counts"):
        st.write("City match count:", city_filter.sum())
        st.write("Cuisine match count:", cuisine_filter.sum())
        st.write("Rating match count:", rating_filter.sum())
        st.write("Cost match count:", cost_filter.sum())
        st.write("Total exact matches:", full_filter.sum())

    if st.button("🌢️ Surprise Me with a Random Pick"):
        random_pick = None

        if not filtered_df.empty:
            random_pick = filtered_df.sample(1).iloc[0]
        else:
            partial_df = df[(df['city'] == selected_city) & (df['cuisine'] == selected_cuisine)]
            if not partial_df.empty:
                random_pick = partial_df.sample(1).iloc[0]
            else:
                city_df = df[df['city'] == selected_city]
                if not city_df.empty:
                    random_pick = city_df.sample(1).iloc[0]

        if random_pick is not None:
            st.markdown("### 🎉 Here's a random restaurant based on your selection:")
            st.markdown(f"### {random_pick['name']}")
            st.write(f"📍 City: {random_pick['city']}")
            st.write(f"🍴 Cuisine: {random_pick['cuisine']}")
            st.write(f"⭐ Rating: {random_pick['rating']} ({random_pick.get('rating_count', 'Too Few Ratings')} reviews)")
            st.write(f"💰 Cost: ₹{random_pick['cost']}")

            tags = []
            if random_pick['rating'] >= 4.5:
                tags.append("🌟 Top Rated")
            if random_pick['cost'] <= 300:
                tags.append("💸 Budget Friendly")
            if tags:
                st.markdown("**Tags:** " + ", ".join(tags))
            st.markdown("---")
        else:
            st.warning("⚠️ No restaurants available for your selection to suggest randomly.")