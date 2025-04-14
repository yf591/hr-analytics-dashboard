import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from src.data.loader import load_hr_data
from scipy import stats as scipy_stats  # statsモジュールの名前変更

def show():
    """
    従業員エンゲージメント分析ページを表示する関数
    """
    st.title("従業員エンゲージメント分析")
    st.write("従業員の満足度、エンゲージメント、職場環境に関する分析")
    
    # データロード
    df = load_hr_data()
    
    # 満足度関連の項目
    satisfaction_cols = [
        'JobSatisfaction', 'EnvironmentSatisfaction',
        'WorkLifeBalance', 'RelationshipSatisfaction'
    ]
    
    # 日本語の列名マッピング
    col_name_map = {
        'JobSatisfaction': '職務満足度',
        'EnvironmentSatisfaction': '環境満足度',
        'WorkLifeBalance': 'ワークライフバランス',
        'RelationshipSatisfaction': '人間関係満足度'
    }
    
    # 満足度レベルの対応表
    satisfaction_level_map = {1: '低', 2: '中低', 3: '中高', 4: '高'}
    
    # エンゲージメント全体の概要
    st.header("エンゲージメント概要")
    
    # 各満足度の平均値を表示
    cols = st.columns(len(satisfaction_cols))
    
    for i, col_name in enumerate(satisfaction_cols):
        with cols[i]:
            avg_score = df[col_name].mean()
            st.metric(col_name_map[col_name], f"{avg_score:.2f}/4")
    
    # タブで分析内容を整理
    tab1, tab2, tab3 = st.tabs(["📊 満足度分析", "🔍 エンゲージメント要因", "📋 アクションプラン"])
    
    with tab1:
        st.header("満足度分析")
        
        # 満足度の分布
        st.subheader("満足度指標の分布")
        
        selected_satisfaction = st.selectbox(
            "満足度指標を選択",
            options=satisfaction_cols,
            format_func=lambda x: col_name_map.get(x, x)
        )
        
        # 選択された満足度指標のレベル分布
        df[f'{selected_satisfaction}_Level'] = df[selected_satisfaction].map(satisfaction_level_map)
        
        sat_counts = df[f'{selected_satisfaction}_Level'].value_counts().reset_index()
        sat_counts.columns = ['Level', 'Count']
        
        fig = px.pie(sat_counts, values='Count', names='Level',
                    title=f"{col_name_map[selected_satisfaction]}の分布",
                    color_discrete_sequence=px.colors.sequential.Viridis)
        st.plotly_chart(fig, use_container_width=True)
        
        # 満足度クロス分析
        st.subheader("満足度間の相関関係")
        
        # 相関マトリックスの計算
        corr_matrix = df[satisfaction_cols].corr()
        
        # 日本語の列名に変換
        corr_matrix_jp = corr_matrix.copy()
        corr_matrix_jp.index = [col_name_map[col] for col in corr_matrix.index]
        corr_matrix_jp.columns = [col_name_map[col] for col in corr_matrix.columns]
        
        fig = px.imshow(corr_matrix_jp,
                       color_continuous_scale='RdBu_r',
                       zmin=-1, zmax=1,
                       text_auto='.2f')
        
        fig.update_layout(title="満足度指標間の相関関係")
        st.plotly_chart(fig, use_container_width=True)
        
        # グループ別の満足度分析
        st.subheader("グループ別の満足度分析")
        
        group_by = st.selectbox(
            "グループ分け",
            options=['Department', 'JobRole', 'JobLevel', 'Gender', 'MaritalStatus', 'AgeGroup'],
            format_func=lambda x: {
                'Department': '部門',
                'JobRole': '職種',
                'JobLevel': '役職レベル',
                'Gender': '性別',
                'MaritalStatus': '婚姻状況',
                'AgeGroup': '年齢層'
            }.get(x, x)
        )
        
        # 年齢グループの作成（存在しない場合）
        if group_by == 'AgeGroup' and 'AgeGroup' not in df.columns:
            age_bins = [18, 30, 40, 50, 60, 70]
            age_labels = ['18-29歳', '30-39歳', '40-49歳', '50-59歳', '60歳以上']
            df['AgeGroup'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels, right=False)
        
        sat_measure = st.selectbox(
            "満足度指標",
            options=satisfaction_cols,
            format_func=lambda x: col_name_map.get(x, x),
            key="group_satisfaction"
        )
        
        # グループ別の満足度平均値
        group_sat = df.groupby(group_by)[sat_measure].mean().reset_index()
        group_sat = group_sat.sort_values(sat_measure, ascending=False)
        
        fig = px.bar(group_sat, x=group_by, y=sat_measure,
                    title=f"{group_by}別の{col_name_map[sat_measure]}",
                    color=sat_measure,
                    color_continuous_scale='Viridis',
                    text_auto='.2f')
        
        if group_by == 'JobRole':
            fig.update_layout(xaxis_tickangle=-45)
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.header("エンゲージメント要因分析")
        
        # エンゲージメント要因の相関分析
        st.subheader("エンゲージメント要因の相関分析")
        
        # エンゲージメントに影響する可能性のある要因
        engagement_factors = [
            'YearsAtCompany', 'YearsSinceLastPromotion', 'YearsWithCurrManager',
            'TrainingTimesLastYear', 'DistanceFromHome', 'NumCompaniesWorked',
            'PercentSalaryHike', 'TotalWorkingYears', 'MonthlyIncome'
        ]
        
        # 要因のマッピング
        factor_map = {
            'YearsAtCompany': '勤続年数',
            'YearsSinceLastPromotion': '前回昇進からの年数',
            'YearsWithCurrManager': '現在の上司の下での年数',
            'TrainingTimesLastYear': '昨年の研修回数',
            'DistanceFromHome': '自宅からの距離',
            'NumCompaniesWorked': '過去の勤務企業数',
            'PercentSalaryHike': '昇給率',
            'TotalWorkingYears': '総労働年数',
            'MonthlyIncome': '月収'
        }
        
        # 選択した要因と満足度指標間の相関分析
        selected_factors = st.multiselect(
            "分析する要因を選択",
            options=engagement_factors,
            default=engagement_factors[:3],
            format_func=lambda x: factor_map.get(x, x)
        )
        
        selected_satisfaction_corr = st.selectbox(
            "満足度指標",
            options=satisfaction_cols,
            format_func=lambda x: col_name_map.get(x, x),
            key="correlation_satisfaction"
        )
        
        if selected_factors:
            # 散布図マトリックスの表示
            if len(selected_factors) <= 3:
                col_set = selected_factors + [selected_satisfaction_corr]
                
                fig = px.scatter_matrix(
                    df, dimensions=col_set,
                    color='Department',
                    labels={k: factor_map.get(k, k) for k in col_set},
                    title=f"要因と{col_name_map[selected_satisfaction_corr]}の関係"
                )
                
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)
            
            # 相関係数の計算と表示
            corr_data = []
            
            for factor in selected_factors:
                corr, p_value = scipy_stats.pearsonr(df[factor], df[selected_satisfaction_corr])
                corr_data.append({
                    'Factor': factor_map.get(factor, factor),
                    'Correlation': corr,
                    'P-Value': p_value,
                    'Significance': '有意' if p_value < 0.05 else '有意でない'
                })
            
            corr_df = pd.DataFrame(corr_data)
            corr_df = corr_df.sort_values('Correlation', ascending=False)
            
            # 相関係数の可視化
            fig = px.bar(corr_df, x='Factor', y='Correlation',
                        title=f"各要因と{col_name_map[selected_satisfaction_corr]}の相関係数",
                        color='Significance',
                        color_discrete_map={'有意': '#1E88E5', '有意でない': '#D81B60'},
                        text_auto='.3f')
            
            st.plotly_chart(fig, use_container_width=True)
            
            # 詳細な相関データを表示
            st.subheader("詳細な相関分析")
            st.dataframe(corr_df)
        
        # 残業と満足度の関係
        st.subheader("残業と満足度の関係")
        
        # 残業有無による満足度の違い
        overtime_sat = df.groupby('OverTime')[satisfaction_cols].mean().reset_index()
        overtime_sat_melted = overtime_sat.melt(id_vars=['OverTime'], 
                                              value_vars=satisfaction_cols,
                                              var_name='SatisfactionType', 
                                              value_name='Score')
        
        # 日本語の満足度タイプに変換
        overtime_sat_melted['SatisfactionType'] = overtime_sat_melted['SatisfactionType'].map(col_name_map)
        
        fig = px.bar(overtime_sat_melted, x='SatisfactionType', y='Score', color='OverTime',
                    barmode='group',
                    title="残業の有無による満足度の比較",
                    labels={'Score': '平均スコア', 'OverTime': '残業'},
                    text_auto='.2f')
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 残業と各満足度の関係についてt検定
        st.subheader("残業の影響度分析（統計的検定）")
        
        # t検定の結果を格納するデータフレーム
        ttest_results = []
        
        for col in satisfaction_cols:
            # 残業あり・なしのグループに分ける
            yes_group = df[df['OverTime'] == 'Yes'][col]
            no_group = df[df['OverTime'] == 'No'][col]
            
            # t検定の実施
            t_stat, p_val = scipy_stats.ttest_ind(yes_group, no_group, equal_var=False)
            
            # 結果の保存
            ttest_results.append({
                'SatisfactionType': col_name_map[col],
                'T-Statistic': t_stat,
                'P-Value': p_val,
                'Significance': '有意' if p_val < 0.05 else '有意でない',
                'Effect': '残業なしの方が高い' if t_stat < 0 else '残業ありの方が高い'
            })
        
        ttest_df = pd.DataFrame(ttest_results)
        
        # 結果の可視化
        fig = px.bar(ttest_df, x='SatisfactionType', y='T-Statistic',
                    color='Significance',
                    color_discrete_map={'有意': '#1E88E5', '有意でない': '#D81B60'},
                    title="残業が満足度に与える影響（t統計量）",
                    text_auto='.3f')
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 詳細な検定結果を表示
        st.dataframe(ttest_df)
    
    with tab3:
        st.header("エンゲージメント向上のためのアクションプラン")
        
        # 現在のエンゲージメント状況を評価
        avg_satisfaction = df[satisfaction_cols].mean().mean()
        
        # 満足度が低いグループを特定
        low_engagement = df[df[satisfaction_cols].mean(axis=1) < 2.5]
        low_engagement_pct = len(low_engagement) / len(df) * 100
        
        st.info(f"""
        **エンゲージメント概要:**
        - 全体平均満足度: {avg_satisfaction:.2f}/4
        - 低エンゲージメント従業員の割合: {low_engagement_pct:.1f}%
        - 最も満足度が高い部門: {df.groupby('Department')[satisfaction_cols].mean().mean(axis=1).idxmax()}
        - 最も満足度が低い部門: {df.groupby('Department')[satisfaction_cols].mean().mean(axis=1).idxmin()}
        """)
        
        # 低エンゲージメントの従業員特性を分析
        st.subheader("低エンゲージメント従業員の特性")
        
        if len(low_engagement) > 0:
            # 部門別の低エンゲージメント割合
            dept_low_engagement = low_engagement['Department'].value_counts()
            dept_total = df['Department'].value_counts()
            dept_pct = (dept_low_engagement / dept_total * 100).reset_index()
            dept_pct.columns = ['Department', 'LowEngagementPercentage']
            
            fig = px.bar(dept_pct, x='Department', y='LowEngagementPercentage',
                        title="部門別の低エンゲージメント従業員の割合",
                        color='LowEngagementPercentage',
                        color_continuous_scale='Reds',
                        text_auto='.1f')
            
            # 問題のある表示方法を修正
            fig.update_traces(
                text=[f"{val:.1f}%" for val in dept_pct['LowEngagementPercentage']],
                textposition='outside'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # 低エンゲージメント従業員の特徴
            st.write("**低エンゲージメント従業員の特徴:**")
            
            # カテゴリ特性
            cat_cols = ['OverTime', 'MaritalStatus', 'JobRole', 'BusinessTravel', 'Gender']
            col1, col2 = st.columns(2)
            
            for i, col in enumerate(cat_cols):
                with col1 if i % 2 == 0 else col2:
                    low_eng_cat = low_engagement[col].value_counts(normalize=True) * 100
                    all_cat = df[col].value_counts(normalize=True) * 100
                    cat_diff = pd.DataFrame({
                        'Category': low_eng_cat.index,
                        'LowEngagement': low_eng_cat.values,
                        'AllEmployees': [all_cat.get(cat, 0) for cat in low_eng_cat.index]
                    })
                    
                    cat_diff['Difference'] = cat_diff['LowEngagement'] - cat_diff['AllEmployees']
                    cat_diff = cat_diff.sort_values('Difference', ascending=False)
                    
                    # 最も顕著な特徴のみ表示
                    top_cat = cat_diff.iloc[0]
                    st.write(f"- **{col}**: {top_cat['Category']} " + 
                            f"({top_cat['LowEngagement']:.1f}% vs 全体{top_cat['AllEmployees']:.1f}%)")
            
            # 数値特性
            num_cols = ['YearsAtCompany', 'Age', 'MonthlyIncome', 'DistanceFromHome', 'WorkLifeBalance']
            num_diff = {}
            
            for col in num_cols:
                low_eng_mean = low_engagement[col].mean()
                all_mean = df[col].mean()
                diff_pct = (low_eng_mean - all_mean) / all_mean * 100
                num_diff[col] = {
                    'LowEngagement': low_eng_mean,
                    'AllEmployees': all_mean,
                    'DiffPercent': diff_pct
                }
            
            # 最も顕著な数値特性を表示
            sorted_num_diff = sorted(num_diff.items(), key=lambda x: abs(x[1]['DiffPercent']), reverse=True)
            
            for col, stats in sorted_num_diff[:3]:
                direction = "高い" if stats['DiffPercent'] > 0 else "低い"
                st.write(f"- **{col}**: 平均{stats['LowEngagement']:.1f} " + 
                        f"(全体平均より{abs(stats['DiffPercent']):.1f}%{direction})")
        
        # アクションプランの提案
        st.subheader("エンゲージメント向上のためのアクションプラン")
        
        action_plans = [
            {
                "Target": "残業が多い部門",
                "Issue": "残業とワークライフバランスの悪化がエンゲージメントに影響",
                "Actions": [
                    "作業効率化のためのプロセス見直し",
                    "業務量の適正化と人員配置の見直し",
                    "フレックスタイム制度の導入検討",
                    "時間外労働に関する意識改革"
                ],
                "KPIs": ["残業時間の削減率", "ワークライフバランス満足度"]
            },
            {
                "Target": "昇進間隔が長い従業員",
                "Issue": "キャリア停滞感によるモチベーション低下",
                "Actions": [
                    "キャリアパスの明確化と共有",
                    "定期的なキャリア面談の実施",
                    "キャリア開発プログラムの充実",
                    "公正な評価制度の確立"
                ],
                "KPIs": ["昇進率の改善", "キャリア満足度"]
            },
            {
                "Target": "研修機会が少ない従業員",
                "Issue": "スキル成長機会の不足による職務満足度低下",
                "Actions": [
                    "個別育成計画の策定",
                    "オンライン学習プラットフォームの導入",
                    "部門間のジョブローテーション促進",
                    "メンタリングプログラムの構築"
                ],
                "KPIs": ["研修参加率", "スキル習得度", "職務満足度の向上"]
            }
        ]
        
        # アクションプランの表示
        for i, plan in enumerate(action_plans):
            expander = st.expander(f"アクションプラン {i+1}: {plan['Target']}")
            with expander:
                st.write(f"**課題:** {plan['Issue']}")
                st.write("**推奨アクション:**")
                for action in plan['Actions']:
                    st.write(f"- {action}")
                st.write("**評価指標 (KPIs):**")
                for kpi in plan['KPIs']:
                    st.write(f"- {kpi}")
    
    # フッター
    st.markdown("---")
    st.info("このページでは、従業員のエンゲージメント、満足度要因、および改善のためのアクションプランを提供しています。"
            "組織文化の向上と従業員定着率の改善にご活用ください。")