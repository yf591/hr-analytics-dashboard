import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.data.loader import load_hr_data
import plotly.express as px
import plotly.graph_objects as go

def show():
    """
    トップページ - 人材分析概要を表示する関数
    """
    st.title("HR Analytics ダッシュボード")
    st.markdown("### 人材分析概要")
    
    # データロード
    df = load_hr_data()
    
    # フィルタリングオプション
    st.sidebar.header("フィルター設定")
    
    # 部署フィルター
    departments = ["All"] + sorted(df["Department"].unique().tolist())
    selected_dept = st.sidebar.selectbox("部署を選択", departments)
    
    # 年齢層フィルター
    age_bins = [18, 25, 35, 45, 55, 65]
    age_labels = ['18-24', '25-34', '35-44', '45-54', '55+']
    df['AgeGroup'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels, right=False)
    age_groups = ["All"] + sorted(df["AgeGroup"].unique().tolist())
    selected_age = st.sidebar.selectbox("年齢層を選択", age_groups)
    
    # 性別フィルター
    genders = ["All"] + sorted(df["Gender"].unique().tolist())
    selected_gender = st.sidebar.selectbox("性別を選択", genders)
    
    # データのフィルタリング
    filtered_df = df.copy()
    if selected_dept != "All":
        filtered_df = filtered_df[filtered_df["Department"] == selected_dept]
    if selected_age != "All":
        filtered_df = filtered_df[filtered_df["AgeGroup"] == selected_age]
    if selected_gender != "All":
        filtered_df = filtered_df[filtered_df["Gender"] == selected_gender]
    
    # KPIセクション
    st.header("主要KPI")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        attrition_rate = filtered_df['Attrition'].value_counts(normalize=True).get('Yes', 0) * 100
        overall_rate = df['Attrition'].value_counts(normalize=True).get('Yes', 0) * 100
        delta = attrition_rate - overall_rate
        st.metric("離職率", f"{attrition_rate:.1f}%", delta=f"{delta:.1f}%")
    
    with col2:
        avg_satisfaction = filtered_df['JobSatisfaction'].mean()
        overall_satisfaction = df['JobSatisfaction'].mean()
        delta = avg_satisfaction - overall_satisfaction
        st.metric("平均職務満足度", f"{avg_satisfaction:.2f}/4", delta=f"{delta:.2f}")
    
    with col3:
        avg_tenure = filtered_df['YearsAtCompany'].mean()
        overall_tenure = df['YearsAtCompany'].mean()
        delta = avg_tenure - overall_tenure
        st.metric("平均勤続年数", f"{avg_tenure:.1f}年", delta=f"{delta:.1f}年")
    
    with col4:
        high_risk = len(filtered_df[(filtered_df['Attrition'] == 'Yes') & 
                                (filtered_df['PerformanceRating'] >= 3)])
        st.metric("高パフォーマンス離職者数", f"{high_risk}人")
    
    st.markdown("---")
    
    # データの可視化セクション
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("部門別従業員構成")
        dept_counts = filtered_df['Department'].value_counts().reset_index()
        dept_counts.columns = ['Department', 'Count']
        
        fig = px.pie(dept_counts, values='Count', names='Department', 
                     title='部門別従業員構成',
                     color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("役職レベル分布")
        job_level_counts = filtered_df['JobLevel'].value_counts().sort_index().reset_index()
        job_level_counts.columns = ['JobLevel', 'Count']
        
        fig = px.bar(job_level_counts, x='JobLevel', y='Count',
                    title='役職レベル分布',
                    color='Count',
                    color_continuous_scale='Viridis')
        st.plotly_chart(fig, use_container_width=True)
    
    # 離職率分析
    st.subheader("離職分析")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("部門別離職率")
        dept_attrition = filtered_df.groupby('Department')['Attrition'].apply(
            lambda x: (x == 'Yes').mean() * 100
        ).reset_index()
        dept_attrition.columns = ['Department', 'Attrition Rate (%)']
        
        fig = px.bar(dept_attrition, x='Department', y='Attrition Rate (%)',
                    title='部門別離職率',
                    color='Attrition Rate (%)',
                    color_continuous_scale='Reds')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("年齢層別離職率")
        age_attrition = filtered_df.groupby('AgeGroup')['Attrition'].apply(
            lambda x: (x == 'Yes').mean() * 100
        ).reset_index()
        age_attrition.columns = ['AgeGroup', 'Attrition Rate (%)']
        
        fig = px.line(age_attrition, x='AgeGroup', y='Attrition Rate (%)',
                     title='年齢層別離職率',
                     markers=True,
                     line_shape='spline')
        st.plotly_chart(fig, use_container_width=True)
    
    # 満足度と勤続年数の関係
    st.subheader("満足度と勤続年数の関係")
    
    # 満足度カテゴリのマッピング
    satisfaction_mapping = {1: '低', 2: '中低', 3: '中高', 4: '高'}
    filtered_df['JobSatisfactionCategory'] = filtered_df['JobSatisfaction'].map(satisfaction_mapping)
    
    # YearsAtCompanyを範囲にまとめる - 最初からグループ化して文字列に変換
    bins = [0, 2, 5, 10, 15, 20, 40]
    labels = ['0-2年', '3-5年', '6-10年', '11-15年', '16-20年', '21年以上']
    filtered_df['YearGroups'] = pd.cut(filtered_df['YearsAtCompany'], bins=bins, labels=labels)
    
    # ヒートマップデータの準備 - 文字列カラムを使用
    heatmap_data = filtered_df.pivot_table(
        index='JobSatisfactionCategory',
        columns='YearGroups',
        values='EmployeeNumber',  # EmployeeCountの代わりにEmployeeNumberを使用
        aggfunc='count'
    ).fillna(0)
    
    # デバッグ情報（問題が解決しない場合のみ表示）
    # st.write("ヒートマップデータの列型:", [type(c) for c in heatmap_data.columns])
    # st.dataframe(heatmap_data)
    
    fig = px.imshow(heatmap_data,
                   labels=dict(x="勤続年数", y="満足度", color="従業員数"),
                   color_continuous_scale='Viridis')
    st.plotly_chart(fig, use_container_width=True)
    
    # アラートセクション - 離職リスクの高い従業員グループの特定
    st.header("リスクアラート")
    
    # 離職リスクの高いグループの特定（例：低満足度 + 長時間労働 + 昇進なし）
    risk_group = filtered_df[
        (filtered_df['JobSatisfaction'] <= 2) &
        (filtered_df['WorkLifeBalance'] <= 2) &
        (filtered_df['YearsSinceLastPromotion'] >= 5)
    ]
    
    risk_percentage = len(risk_group) / len(filtered_df) * 100 if len(filtered_df) > 0 else 0
    
    st.metric("離職リスクの高い従業員の割合", f"{risk_percentage:.1f}%")
    
    if len(risk_group) > 0:
        st.subheader("リスク要因分析")
        
        risk_factors = [
            ("低い職務満足度", len(filtered_df[filtered_df['JobSatisfaction'] <= 2])),
            ("低いワークライフバランス", len(filtered_df[filtered_df['WorkLifeBalance'] <= 2])),
            ("長期間昇進なし (5年以上)", len(filtered_df[filtered_df['YearsSinceLastPromotion'] >= 5])),
            ("過剰な残業", len(filtered_df[filtered_df['OverTime'] == 'Yes'])),
            ("低い環境満足度", len(filtered_df[filtered_df['EnvironmentSatisfaction'] <= 2]))
        ]
        
        risk_df = pd.DataFrame(risk_factors, columns=['リスク要因', '該当者数'])
        risk_df['該当割合 (%)'] = risk_df['該当者数'] / len(filtered_df) * 100
        
        fig = px.bar(risk_df, x='リスク要因', y='該当割合 (%)',
                    title='離職リスク要因分析',
                    color='該当割合 (%)',
                    color_continuous_scale='Reds')
        st.plotly_chart(fig, use_container_width=True)