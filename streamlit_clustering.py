import pandas as pd
import streamlit as st
import plotly.express as px
import plotly

st.title("Clustering : Analyse Streamlit")

data = st.file_uploader("Upload a Dataset", type=["csv", "txt"])


if data is not None:
    df = pd.read_csv(data)
    List_columns_without_labels = [w for w in df.columns if "label" not in w.lower()]
    List_columns_labels = [w for w in df.columns if "label" in w.lower()]

    if st.checkbox("Show Distribution by cluster"):
        st_columns_without_labels = st.selectbox("X", List_columns_without_labels)

        if st_columns_without_labels is not None:
            List_columns_labels = [w for w in df.columns if "label" in w.lower()]
            st_columns_labels = st.selectbox("Color", List_columns_labels)
            if st_columns_without_labels is not None and st_columns_labels is not None:
                fig = px.histogram(df, x=f"{st_columns_without_labels}", color=f"{st_columns_labels}", marginal="box")
                st.plotly_chart(fig)



    if st.checkbox("Show scatter plot"):
        st_columns_labels = st.selectbox("Color", List_columns_labels)
        str_col_1 = st.selectbox("X", List_columns_without_labels)
        str_col_2 = st.selectbox("Y", List_columns_without_labels)

        if str_col_1 is not None and str_col_2 is not None and st_columns_labels is not None:
            fig = px.scatter(df, x=str_col_1, y=str_col_2, color=st_columns_labels)
            st.plotly_chart(fig)



    if st.checkbox("Show nombre de clusters"):
        st_columns_labels = st.selectbox("Color", List_columns_labels)

        if st_columns_labels is not None:
            df_barplot = pd.DataFrame(df[st_columns_labels].value_counts())
            df_barplot.reset_index(inplace=True)
            df_barplot.columns = ['cluster', 'count']
            fig = px.bar(df_barplot, x='cluster', y='count')
            st.plotly_chart(fig)


    if st.checkbox("Show parralel plot"):
        st_columns_labels = st.selectbox("Color", List_columns_labels)
        st_columns_coordinates = st.multiselect("Dimension", List_columns_without_labels)

        if st_columns_labels is not None and len(st_columns_coordinates) < 10 and len(st_columns_coordinates) >= 3:
            X_parralel_means = df.groupby(by='kmeans_labels').mean()
            X_parralel_means.reset_index(inplace=True)
            fig = px.parallel_coordinates(X_parralel_means, color=st_columns_labels,
                                        dimensions=st_columns_coordinates,
                                        color_continuous_scale=px.colors.diverging.Tealrose,
                                        color_continuous_midpoint=2)
            st.plotly_chart(fig)