import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Set page config
st.set_page_config(page_title="Customer Segmentation", layout="wide")

# Title and description
st.title("🛍️ Customer Segmentation using K-Means")
st.markdown("Identify groups of similar customers based on **Annual Income** and **Spending Score**.")

# Sidebar for inputs
st.sidebar.title("🔧 Controls")
uploaded_file = st.sidebar.file_uploader("📂 Upload your Mall_Customers.csv file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    # Load default sample data from GitHub
    sample_url = "https://raw.githubusercontent.com/skadesplaire01/CodeClauseInternship_K-Means-Customer-Segmentation-app/main/Mall_Customers.csv"
    df = pd.read_csv(sample_url)
    st.info("📌 Sample data loaded from GitHub (Mall_Customers.csv)")


    with st.expander("📋 Raw Dataset Preview"):
        st.dataframe(df)

    # Select features
    X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Elbow Method
    st.sidebar.markdown("## Elbow Method")
    show_elbow = st.sidebar.checkbox("Show Elbow Plot", value=True)

    if show_elbow:
        wcss = []
        for i in range(1, 11):
            kmeans = KMeans(n_clusters=i, random_state=42)
            kmeans.fit(X_scaled)
            wcss.append(kmeans.inertia_)

        fig1, ax1 = plt.subplots()
        ax1.plot(range(1, 11), wcss, marker='o')
        ax1.set_title('Elbow Method')
        ax1.set_xlabel('Number of Clusters')
        ax1.set_ylabel('WCSS')
        st.pyplot(fig1)

    # Choose number of clusters
    k = st.sidebar.slider("🎯 Select number of clusters", 2, 10, 5)

    # KMeans
    kmeans = KMeans(n_clusters=k, random_state=42)
    df['Cluster'] = kmeans.fit_predict(X_scaled)

    # Clustered Data
    with st.expander("📊 Clustered Data"):
        st.dataframe(df)

    # Cluster Profile
    st.subheader("📈 Cluster Profiles")
    profile = df.groupby('Cluster')[['Annual Income (k$)', 'Spending Score (1-100)']].mean().round(2)
    st.dataframe(profile)

    # Scatter Plot
    st.subheader("📍 Cluster Visualization")
    fig2, ax2 = plt.subplots()
    sns.scatterplot(
        x='Annual Income (k$)',
        y='Spending Score (1-100)',
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
