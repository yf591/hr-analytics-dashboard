import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from src.data.loader import load_hr_data
from scipy import stats

def show():
    """
    給与・報酬分析ページを表示する関数
    """
    st.title("給与・報酬分析")
    st.write("従業員の給与データ分析と公平性評価")
    
    # データロード
    df = load_hr_data()
    
    # 給与関連の項目を抽出
    salary_cols = ['MonthlyIncome', 'MonthlyRate', 'DailyRate', 'HourlyRate', 'PercentSalaryHike']
    
    # タブで分析を整理
    tab1, tab2, tab3 = st.tabs(["📊 給与分布", "🔍 給与と業績の関係", "⚖️ 給与の公平性"])
    
    with tab1:
        st.header("給与分布分析")
        
        # 給与の統計情報
        st.subheader("給与の基本統計")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            avg_income = df['MonthlyIncome'].mean()
            st.metric("平均月収", f"${avg_income:,.2f}")
        
        with col2:
            median_income = df['MonthlyIncome'].median()
            st.metric("中央値月収", f"${median_income:,.2f}")
        
        with col3:
            income_range = df['MonthlyIncome'].max() - df['MonthlyIncome'].min()
            st.metric("給与範囲", f"${income_range:,.2f}")
        
        # 給与ヒストグラム
        st.subheader("給与分布")
        
        salary_measure = st.selectbox(
            "給与指標を選択",
            options=salary_cols,
            format_func=lambda x: {
                'MonthlyIncome': '月収',
                'MonthlyRate': '月給レート',
                'DailyRate': '日給レート',
                'HourlyRate': '時給レート',
                'PercentSalaryHike': '昇給率'
            }.get(x, x)
        )
        
        group_by = st.selectbox(
            "グループ分け",
            options=['なし', 'JobLevel', 'JobRole', 'Department', 'Gender'],
            format_func=lambda x: {
                'なし': 'なし',
                'JobLevel': '役職レベル',
                'JobRole': '職種',
                'Department': '部門',
                'Gender': '性別'
            }.get(x, x)
        )
        
        if group_by == 'なし':
            fig = px.histogram(df, x=salary_measure,
                              nbins=30,
                              title=f"{salary_measure}の分布",
                              labels={salary_measure: "金額"},
                              color_discrete_sequence=['#66b3ff'])
        else:
            fig = px.histogram(df, x=salary_measure,
                              color=group_by,
                              nbins=30,
                              barmode='overlay',
                              title=f"{salary_measure}の分布（{group_by}別）",
                              labels={salary_measure: "金額"})
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 箱ひげ図（部門・職種別の給与分布）
        st.subheader("職種・等級別の給与箱ひげ図")
        
        box_group = st.selectbox(
            "グループ分け因子",
            options=['JobRole', 'JobLevel', 'Department', 'Education'],
            format_func=lambda x: {
                'JobRole': '職種',
                'JobLevel': '役職レベル',
                'Department': '部門',
                'Education': '教育レベル'
            }.get(x, x)
        )
        
        fig = px.box(df, x=box_group, y='MonthlyIncome',
                    title=f"{box_group}別の月収分布",
                    color=box_group)
        
        # 職種が多い場合は横向きに表示
        if box_group == 'JobRole':
            fig.update_layout(xaxis_tickangle=-45)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 部門内の職種別給与分布（多変量分析）
        if st.checkbox("部門内の職種別給与分布を表示"):
            fig = px.box(df, x='JobRole', y='MonthlyIncome',
                        facet_col='Department',
                        title="部門・職種別の月収分布",
                        color='JobRole')
            
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.header("給与と業績の関係")
        
        # 給与と業績評価の散布図
        st.subheader("給与と業績評価の関係")
        
        fig = px.scatter(df, x='PerformanceRating', y='MonthlyIncome',
                        color='JobLevel',
                        size='YearsAtCompany',
                        hover_data=['JobRole', 'Department'],
                        title="業績評価と月収の関係")
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 給与と勤続年数の散布図
        st.subheader("給与と勤続年数の関係")
        
        color_var = st.selectbox(
            "色分け要素",
            options=['JobLevel', 'PerformanceRating', 'Department', 'Gender'],
            format_func=lambda x: {
                'JobLevel': '役職レベル',
                'PerformanceRating': '業績評価',
                'Department': '部門',
                'Gender': '性別'
            }.get(x, x)
        )
        
        fig = px.scatter(df, x='YearsAtCompany', y='MonthlyIncome',
                        color=color_var,
                        trendline='ols',
                        title="勤続年数と月収の関係")
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 昇給率と業績の関係
        st.subheader("昇給率と業績の関係")
        
        perf_hike = df.groupby('PerformanceRating')['PercentSalaryHike'].mean().reset_index()
        
        fig = px.bar(perf_hike, x='PerformanceRating', y='PercentSalaryHike',
                    title="業績評価別の平均昇給率",
                    color='PercentSalaryHike',
                    color_continuous_scale='Viridis',
                    labels={'PercentSalaryHike': '平均昇給率 (%)', 'PerformanceRating': '業績評価'})
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 満足度と給与の関係
        st.subheader("満足度と給与の関係")
        
        satisfaction_cols = ['JobSatisfaction', 'EnvironmentSatisfaction', 'WorkLifeBalance']
        selected_satisfaction = st.selectbox(
            "満足度指標を選択",
            options=satisfaction_cols,
            format_func=lambda x: {
                'JobSatisfaction': '職務満足度',
                'EnvironmentSatisfaction': '環境満足度', 
                'WorkLifeBalance': 'ワークライフバランス'
            }.get(x, x)
        )
        
        satisfaction_income = df.groupby(selected_satisfaction)['MonthlyIncome'].mean().reset_index()
        
        fig = px.bar(satisfaction_income, x=selected_satisfaction, y='MonthlyIncome',
                    title=f"{selected_satisfaction}と平均月収の関係",
                    color='MonthlyIncome',
                    color_continuous_scale='Viridis',
                    labels={'MonthlyIncome': '平均月収'})
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.header("給与の公平性分析")
        
        # 性別による給与差の分析
        st.subheader("性別による給与分析")
        
        gender_income = df.groupby(['Gender', 'JobLevel'])['MonthlyIncome'].mean().reset_index()
        
        fig = px.bar(gender_income, x='JobLevel', y='MonthlyIncome',
                    color='Gender',
                    barmode='group',
                    title="役職レベル・性別ごとの平均月収",
                    labels={'MonthlyIncome': '平均月収', 'JobLevel': '役職レベル'})
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 統計的検定
        st.subheader("統計的検定による給与差の分析")
        
        test_var = st.selectbox(
            "分析する変数",
            options=['Gender', 'Education', 'MaritalStatus', 'OverTime'],
            format_func=lambda x: {
                'Gender': '性別',
                'Education': '教育レベル',
                'MaritalStatus': '婚姻状況',
                'OverTime': '残業の有無'
            }.get(x, x)
        )
        
        # グループごとの給与データを取得
        groups = df.groupby(test_var)['MonthlyIncome'].apply(list).to_dict()
        
        # 統計量の計算
        stats_df = df.groupby(test_var)['MonthlyIncome'].agg(['count', 'mean', 'std']).reset_index()
        stats_df.columns = [test_var, 'サンプル数', '平均月収', '標準偏差']
        
        st.dataframe(stats_df)
        
        # 統計的検定の実施
        if len(groups) == 2:  # 2グループの場合はt検定
            group_values = list(groups.values())
            t_stat, p_val = stats.ttest_ind(group_values[0], group_values[1], equal_var=False)
            test_name = "Welchのt検定"
        else:  # 多グループの場合はANOVA
            group_values = list(groups.values())
            f_stat, p_val = stats.f_oneway(*group_values)
            test_name = "一元配置分散分析（ANOVA）"
        
        # 結果の表示
        st.write(f"**{test_name}の結果:**")
        
        if test_name == "Welchのt検定":
            st.write(f"t統計量: {t_stat:.4f}")
        else:
            st.write(f"F統計量: {f_stat:.4f}")
        
        st.write(f"p値: {p_val:.4f}")
        
        alpha = 0.05
        if p_val < alpha:
            st.error(f"p値が{alpha}未満であり、グループ間に統計的に有意な給与差が存在します。")
        else:
            st.success(f"p値が{alpha}以上であり、グループ間に統計的に有意な給与差は検出されませんでした。")
        
        # 給与格差のヒートマップ
        st.subheader("多変量による給与格差分析")
        
        row_var = st.selectbox(
            "行変数",
            options=['JobLevel', 'Department', 'JobRole', 'Education'],
            index=0,
            format_func=lambda x: {
                'JobLevel': '役職レベル',
                'Department': '部門',
                'JobRole': '職種',
                'Education': '教育レベル'
            }.get(x, x)
        )
        
        col_var = st.selectbox(
            "列変数",
            options=['Gender', 'MaritalStatus', 'OverTime', 'BusinessTravel'],
            index=0,
            format_func=lambda x: {
                'Gender': '性別',
                'MaritalStatus': '婚姻状況',
                'OverTime': '残業の有無',
                'BusinessTravel': '出張頻度'
            }.get(x, x)
        )
        
        # ピボットテーブルの作成
        pivot_df = df.pivot_table(
            index=row_var,
            columns=col_var,
            values='MonthlyIncome',
            aggfunc='mean'
        )
        
        # 差分データの計算（同じ行変数内での差）
        if len(pivot_df.columns) > 1:
            pivot_diff = pd.DataFrame()
            for i in range(len(pivot_df.columns)):
                for j in range(i+1, len(pivot_df.columns)):
                    col_name = f"{pivot_df.columns[i]}-{pivot_df.columns[j]}"
                    pivot_diff[col_name] = pivot_df.iloc[:, i] - pivot_df.iloc[:, j]
            
            # ヒートマップの表示
            st.write(f"**{row_var}ごとの{col_var}間の月収差（絶対値）**")
            
            fig = px.imshow(pivot_diff.abs(),
                           title=f"{row_var}ごとの{col_var}間の月収差",
                           color_continuous_scale='RdBu_r',
                           text_auto='.0f')
            
            st.plotly_chart(fig, use_container_width=True)
        
        # 元のピボットテーブルも表示
        st.write(f"**{row_var}と{col_var}ごとの平均月収**")
        
        fig = px.imshow(pivot_df,
                       title=f"{row_var}と{col_var}ごとの平均月収",
                       color_continuous_scale='Viridis',
                       text_auto='.0f')
        
        st.plotly_chart(fig, use_container_width=True)
    
    # フッター
    st.markdown("---")
    st.info("このページでは、給与・報酬の分析と公平性の評価を提供しています。"
            "給与体系の見直しやキャリア開発計画の策定にご活用ください。")