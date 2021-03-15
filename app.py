import pandas as pd
import streamlit as st
import plotly.express as px
import plotly
import os
import lightgbm as lgb
import os 
from joblib import load

st.title("Clustering : Analyse Streamlit")

data = st.file_uploader("Upload a Dataset", type=["csv", "txt"])

st.sidebar.title("Menu")
add_selectbox = st.sidebar.selectbox(
    'Analyse',
    ('Global', 'Numerique')
)



if data is not None:
    df = pd.read_csv(data)
    List_columns_without_labels = [w for w in df.columns if "label" not in w.lower()]
    List_columns_labels = [w for w in df.columns if "label" in w.lower()]

    if add_selectbox == "Global":
        if st.checkbox("Show Distribution by cluster"):
            st_columns_without_labels = st.selectbox("X", List_columns_without_labels)

            if st_columns_without_labels is not None:
                List_columns_labels = [w for w in df.columns if "label" in w.lower()]
                st_columns_labels = st.selectbox("Color", List_columns_labels)
                if st_columns_without_labels is not None and st_columns_labels is not None:
                    fig = px.histogram(df, x=st_columns_without_labels, color=st_columns_labels, marginal="box")
                    st.plotly_chart(fig)



        if st.checkbox("Show scatter plot"):
            st_columns_labels = st.selectbox("Color", List_columns_labels)
            str_col_1 = st.selectbox("X", List_columns_without_labels)
            str_col_2 = st.selectbox("Y", List_columns_without_labels)

            if str_col_1 is not None and str_col_2 is not None and st_columns_labels is not None:
                fig = px.scatter(df, x=str_col_1, y=str_col_2, color=st_columns_labels)
                st.plotly_chart(fig)



        if st.checkbox("Show nombre de clusters"):
            st_columns_labels = st.selectbox("Colonne", List_columns_labels)

            if st_columns_labels is not None:
                df_barplot = pd.DataFrame(df[st_columns_labels].value_counts())
                df_barplot.reset_index(inplace=True)
                df_barplot.columns = ['cluster', 'count']
                fig = px.bar(df_barplot, x='cluster', y='count')
                st.plotly_chart(fig)


        if st.checkbox("Show parralel plot"):
            st_columns_labels = st.selectbox("Colonne labels parralel plot", List_columns_labels)
            st_columns_coordinates = st.multiselect("Dimension", List_columns_without_labels)

            if st_columns_labels is not None and len(st_columns_coordinates) < 10 and len(st_columns_coordinates) >= 3:
                X_parralel_means = df.groupby(by=st_columns_labels).mean()
                X_parralel_means.reset_index(inplace=True)
                fig = px.parallel_coordinates(X_parralel_means, color=st_columns_labels,
                                            dimensions=st_columns_coordinates,
                                            color_continuous_scale=px.colors.diverging.Tealrose,
                                            color_continuous_midpoint=2)
                st.plotly_chart(fig)

        if st.checkbox("Show feature importance plot"):
            path_to_model = os.path.realpath('model')
            list_model = os.listdir(path_to_model)
            model = st.selectbox("Choix du model", list_model)
            model_selected = os.path.join(path_to_model, model)
            clf = load(model_selected) 
            pd_feature_importance = pd.DataFrame()
            pd_feature_importance['Features_name'] = clf.feature_name_
            pd_feature_importance['Features_importance'] = clf.feature_importances_
            pd_feature_importance.sort_values(by='Features_importance', ascending=True, inplace=True)
            level = st.slider("Nombre de variable :", 2, 15)
            if level is not None and  level is not None:
                fig = px.bar(pd_feature_importance.iloc[-level:], y="Features_name", x="Features_importance", orientation='h')
                st.plotly_chart(fig)

        if st.checkbox("Show categorical plot"):
            List_columns_categorical = list(df.select_dtypes(exclude=[float, int]).columns)
            st_columns_labels = st.selectbox("Colonne labels categorical plot", List_columns_labels)
            st_columns_categorical = st.selectbox("Colonne categorical plot", List_columns_categorical)
            X_bar = pd.pivot_table(df, index=st_columns_labels, columns=st_columns_categorical, aggfunc="count", fill_value=0)['CA']
            X_bar.reset_index(inplace=True)
            X_bar = X_bar.melt(st_columns_labels)
            fig = px.bar(X_bar, x=st_columns_labels, y="value", color=st_columns_categorical)
            st.plotly_chart(fig)
    
    
    if add_selectbox == "Numerique":
        
        if st.checkbox("Show sum numerical plot"):
            List_columns_numerical = df.select_dtypes(include=[float, int]).columns
            st_columns_labels = st.selectbox("Colonne labels sum numerical plot", List_columns_labels)
            st_columns = st.selectbox("Colonne sum numerical plot", List_columns_numerical)
            st.header("Sum")
            X_bar = df.groupby(by=st_columns_labels).sum()[[st_columns]].reset_index()
            fig = px.bar(X_bar, x=st_columns_labels, y=st_columns)
            st.plotly_chart(fig)

        
            st.header("Mean")
            List_columns_numerical = df.select_dtypes(include=[float, int]).columns
            X_bar = df.groupby(by=st_columns_labels).mean()[[st_columns]].reset_index()
            fig = px.bar(X_bar, x=st_columns_labels, y=st_columns)
            st.plotly_chart(fig)

        
            st.header("Std")
            List_columns_numerical = df.select_dtypes(include=[float, int]).columns
            X_bar = df.groupby(by=st_columns_labels).std()[[st_columns]].reset_index()
            fig = px.bar(X_bar, x=st_columns_labels, y=st_columns)
            st.plotly_chart(fig)

