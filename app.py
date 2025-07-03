import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Set page config
st.set_page_config(page_title="Customer Segmentation", layout="wide")

# Stylish title
st.markdown(st.markdown("<h2 style='text-align: center;'>🛍️ Customer Segmentation using <span style='color:#F63366;'>K-Means</span></h2>", unsafe_allow_html=True))
st.markdown("### Identify groups of similar customers based on any two numeric features (e.g., Annual Income and Spending Score).")

# Sidebar file upload
st.sidebar.title("🔧 Controls")
uploaded_file = st.sidebar.file_uploader("📂 Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.markdown("---")
    tab1, tab2, tab3 = st.tabs(["📋 Raw Data", "📊 Clustered Data", "📈 Cluster Profiles"])

    with tab1:
        st.dataframe(df)

    # Feature selection
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    features = st.sidebar.multiselect("🧮 Select 2 features for clustering", numeric_cols, default=numeric_cols[:2])

    if len(features) == 2:
        X = df[features]

        with st.spinner("🔄 Clustering in progress..."):

            # Standardize
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Elbow Method
            st.sidebar.markdown("## 📉 Elbow Method")
            show_elbow = st.sidebar.checkbox("Show Elbow Plot", value=True)

            if show_elbow:
                wcss = []
                for i in range(1, 11):
                    kmeans = KMeans(n_clusters=i, random_state=42)
                    kmeans.fit(X_scaled)
                    wcss.append(kmeans.inertia_)

                fig1, ax1 = plt.subplots(figsize=(6, 4))
                ax1.plot(range(1, 11), wcss, marker='o')
                ax1.set_title('Elbow Method')
                ax1.set_xlabel('Number of Clusters')
                ax1.set_ylabel('WCSS')
                st.pyplot(fig1)

            # Number of clusters
            k = st.sidebar.slider("🎯 Select number of clusters", 2, 10, 5)

            # Apply KMeans
            kmeans = KMeans(n_clusters=k, random_state=42)
            df['Cluster'] = kmeans.fit_predict(X_scaled)

            # Clustered Data
            with tab2:
                st.dataframe(df)

            # Cluster Profile
            profile = df.groupby('Cluster')[features].mean().round(2)
            with tab3:
                st.dataframe(profile)

            # Visualization
            st.subheader("📍 Cluster Visualization")
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            sns.scatterplot(
                x=features[0],
                y=features[1],
                hue='Cluster',
                data=df,
                palette='Set1',
                s=100,
                alpha=0.8,
                ax=ax2
            )
            plt.title('Customer Segments')
            st.pyplot(fig2)

            # Download
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Download Clustered Data", csv, "Clustered_Customers.csv", "text/csv")

    else:
        st.warning("⚠️ Please select exactly 2 numeric features for clustering.")

else:
    st.markdown("<h4 style='color:gray;'>📂 Please upload a CSV file using the sidebar to begin...</h4>", unsafe_allow_html=True)
