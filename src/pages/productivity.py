import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from src.data.loader import load_hr_data
from scipy import stats as scipy_stats  # 名前の衝突を避けるため変更
# レスポンシブ対応と PDF 出力用のユーティリティをインポート
from src.utils.layout_utils import (
    display_optimized_chart,
    create_responsive_columns,
    add_page_break,
    format_dataframe_for_display
)

def show():
    """
    労働生産性/ワークスタイル分析ページを表示する関数
    """
    st.title("労働生産性/ワークスタイル分析")
    st.write("従業員の生産性パターン、ワークスタイルと業績の関係分析")
    
    # データロード
    df = load_hr_data()
    
    # 生産性に関連する項目
    productivity_cols = [
        'PerformanceRating', 'JobInvolvement', 'WorkLifeBalance',
        'BusinessTravel', 'OverTime', 'DistanceFromHome'
    ]
    
    # 労働スタイルの概要
    st.header("労働スタイルの概要")
    
    # レスポンシブ対応のカラムレイアウト
    col1, col2, col3 = create_responsive_columns([1, 1, 1])
    
    with col1:
        overtime_pct = (df['OverTime'] == 'Yes').mean() * 100
        st.metric("残業をしている従業員の割合", f"{overtime_pct:.1f}%")
    
    with col2:
        travel_freq = df['BusinessTravel'].value_counts(normalize=True)['Travel_Frequently'] * 100
        st.metric("頻繁に出張する従業員の割合", f"{travel_freq:.1f}%")
    
    with col3:
        wlb_score = df['WorkLifeBalance'].mean()
        st.metric("平均ワークライフバランススコア", f"{wlb_score:.2f}/4")
    
    # タブで分析内容を整理
    tab1, tab2, tab3 = st.tabs(["🕒 ワークスタイル分析", "📊 生産性要因", "💡 最適化シミュレーション"])
    
    with tab1:
        st.header("ワークスタイル分析")
        
        # 残業の状況
        st.subheader("残業状況の分析")
        
        # 部門別の残業状況
        overtime_by_dept = df.groupby('Department')['OverTime'].apply(
            lambda x: (x == 'Yes').mean() * 100
        ).reset_index()
        overtime_by_dept.columns = ['Department', 'OvertimePercentage']
        overtime_by_dept = overtime_by_dept.sort_values('OvertimePercentage', ascending=False)
        
        fig = px.bar(overtime_by_dept, x='Department', y='OvertimePercentage',
                    title="部門別の残業率",
                    color='OvertimePercentage',
                    color_continuous_scale='Reds',
                    text_auto='.1f')
        
        # テキスト表示を修正
        fig.update_traces(
            text=[f"{val:.1f}%" for val in overtime_by_dept['OvertimePercentage']],
            textposition='outside'
        )
        
        # 最適化した図を表示
        display_optimized_chart(fig)
        
        # 残業と勤務年数・年齢の関係
        st.subheader("残業と勤続年数・年齢の関係")
        
        # レスポンシブ対応のカラムレイアウト
        col1, col2 = create_responsive_columns()
        
        with col1:
            # 勤続年数と残業の関係
            tenure_overtime = df.groupby('YearsAtCompany')['OverTime'].apply(
                lambda x: (x == 'Yes').mean() * 100
            ).reset_index()
            tenure_overtime.columns = ['YearsAtCompany', 'OvertimePercentage']
            
            fig = px.line(tenure_overtime, x='YearsAtCompany', y='OvertimePercentage',
                         title="勤続年数と残業率の関係",
                         markers=True,
                         labels={'OvertimePercentage': '残業率 (%)', 'YearsAtCompany': '勤続年数'})
            
            # 最適化した図を表示
            display_optimized_chart(fig)
        
        with col2:
            # 年齢と残業の関係
            # 年齢をグループ化
            df['AgeGroup'] = pd.cut(df['Age'], bins=[18, 25, 35, 45, 55, 65],
                                   labels=['18-25', '26-35', '36-45', '46-55', '56-65'])
            
            age_overtime = df.groupby('AgeGroup')['OverTime'].apply(
                lambda x: (x == 'Yes').mean() * 100
            ).reset_index()
            age_overtime.columns = ['AgeGroup', 'OvertimePercentage']
            
            fig = px.bar(age_overtime, x='AgeGroup', y='OvertimePercentage',
                        title="年齢層と残業率の関係",
                        color='OvertimePercentage',
                        color_continuous_scale='Reds',
                        text_auto='.1f')
            
            # テキスト表示を修正
            fig.update_traces(
                text=[f"{val:.1f}%" for val in age_overtime['OvertimePercentage']],
                textposition='outside'
            )
            
            # 最適化した図を表示
            display_optimized_chart(fig)
        
        # PDF出力時のページ区切り
        add_page_break()
        
        # 出張頻度分析
        st.subheader("出張頻度の分析")
        
        # 部門・職種別の出張頻度
        group_by = st.selectbox(
            "グループ分け",
            options=['Department', 'JobRole', 'JobLevel', 'MaritalStatus'],
            format_func=lambda x: {
                'Department': '部門',
                'JobRole': '職種',
                'JobLevel': '役職レベル',
                'MaritalStatus': '婚姻状況'
            }.get(x, x)
        )
        
        # 出張頻度の数値化
        travel_map = {'Non-Travel': 0, 'Travel_Rarely': 1, 'Travel_Frequently': 2}
        df['TravelFrequencyScore'] = df['BusinessTravel'].map(travel_map)
        
        # グループ別の出張頻度スコア
        travel_by_group = df.groupby(group_by)['TravelFrequencyScore'].mean().reset_index()
        travel_by_group = travel_by_group.sort_values('TravelFrequencyScore', ascending=False)
        
        fig = px.bar(travel_by_group, x=group_by, y='TravelFrequencyScore',
                    title=f"{group_by}別の出張頻度スコア (0=なし, 1=少ない, 2=頻繁)",
                    color='TravelFrequencyScore',
                    color_continuous_scale='Blues',
                    text_auto='.2f')
        
        if group_by == 'JobRole':
            fig.update_layout(xaxis_tickangle=-45)
        
        # 最適化した図を表示
        display_optimized_chart(fig)
        
        # 出張頻度と残業の関係
        st.subheader("出張頻度と残業の関係")
        
        travel_overtime = df.groupby('BusinessTravel')['OverTime'].apply(
            lambda x: (x == 'Yes').mean() * 100
        ).reset_index()
        travel_overtime.columns = ['BusinessTravel', 'OvertimePercentage']
        
        fig = px.bar(travel_overtime, x='BusinessTravel', y='OvertimePercentage',
                    title="出張頻度と残業率の関係",
                    color='OvertimePercentage',
                    color_continuous_scale='Reds',
                    text_auto='.1f')
        
        # テキスト表示を修正
        fig.update_traces(
            text=[f"{val:.1f}%" for val in travel_overtime['OvertimePercentage']],
            textposition='outside'
        )
        
        # 最適化した図を表示
        display_optimized_chart(fig)
    
    with tab2:
        st.header("生産性要因分析")
        
        # PDF出力時のページ区切り
        add_page_break()
        
        # ワークライフバランスと業績の関係
        st.subheader("ワークライフバランスと業績の関係")
        
        wlb_perf = df.groupby('WorkLifeBalance')['PerformanceRating'].mean().reset_index()
        
        fig = px.bar(wlb_perf, x='WorkLifeBalance', y='PerformanceRating',
                    title="ワークライフバランスと平均業績評価の関係",
                    color='PerformanceRating',
                    color_continuous_scale='Viridis',
                    text_auto='.2f')
        
        # 最適化した図を表示
        display_optimized_chart(fig)
        
        # 残業と業績の関係
        st.subheader("残業と業績の関係")
        
        overtime_perf = df.groupby('OverTime')['PerformanceRating'].mean().reset_index()
        
        fig = px.bar(overtime_perf, x='OverTime', y='PerformanceRating',
                    title="残業の有無と平均業績評価の関係",
                    color='PerformanceRating',
                    color_continuous_scale='Viridis',
                    text_auto='.2f')
        
        # 最適化した図を表示
        display_optimized_chart(fig)
        
        # 職務満足度と業績の関係
        st.subheader("職務満足度と業績の関係")
        
        # 複数の満足度指標と業績の関係を分析
        satisfaction_cols = ['JobSatisfaction', 'EnvironmentSatisfaction', 'WorkLifeBalance', 'RelationshipSatisfaction']
        
        sat_perf_data = []
        
        for col in satisfaction_cols:
            for level in range(1, 5):
                perf_avg = df[df[col] == level]['PerformanceRating'].mean()
                sat_perf_data.append({
                    'SatisfactionType': col,
                    'SatisfactionLevel': level,
                    'AveragePerformance': perf_avg
                })
        
        sat_perf_df = pd.DataFrame(sat_perf_data)
        
        # 日本語の満足度タイプ名
        sat_type_map = {
            'JobSatisfaction': '職務満足度',
            'EnvironmentSatisfaction': '環境満足度',
            'WorkLifeBalance': 'ワークライフバランス',
            'RelationshipSatisfaction': '人間関係満足度'
        }
        
        sat_perf_df['SatisfactionTypeJP'] = sat_perf_df['SatisfactionType'].map(sat_type_map)
        
        fig = px.line(sat_perf_df, x='SatisfactionLevel', y='AveragePerformance',
                     color='SatisfactionTypeJP',
                     title="各種満足度レベルと平均業績評価の関係",
                     markers=True,
                     labels={'SatisfactionLevel': '満足度レベル', 'AveragePerformance': '平均業績評価'})
        
        # 最適化した図を表示
        display_optimized_chart(fig)
        
        # PDF出力時のページ区切り
        add_page_break()
        
        # 通勤距離と業績/満足度の関係
        st.subheader("通勤距離と業績/満足度の関係")
        
        # 通勤距離のグループ化
        df['CommuteDistanceGroup'] = pd.cut(df['DistanceFromHome'], 
                                           bins=[0, 5, 10, 20, 30, 100],
                                           labels=['0-5km', '6-10km', '11-20km', '21-30km', '30km超'])
        
        # 通勤距離グループと各種指標の関係
        metrics = {
            'PerformanceRating': '業績評価',
            'JobSatisfaction': '職務満足度',
            'WorkLifeBalance': 'ワークライフバランス'
        }
        
        selected_metric = st.selectbox(
            "分析する指標",
            options=list(metrics.keys()),
            format_func=lambda x: metrics.get(x, x)
        )
        
        commute_metric = df.groupby('CommuteDistanceGroup')[selected_metric].mean().reset_index()
        
        fig = px.bar(commute_metric, x='CommuteDistanceGroup', y=selected_metric,
                    title=f"通勤距離と{metrics[selected_metric]}の関係",
                    color=selected_metric,
                    color_continuous_scale='Viridis',
                    text_auto='.2f')
        
        # 最適化した図を表示
        display_optimized_chart(fig)
        
        # 残業率と通勤距離の関係
        commute_overtime = df.groupby('CommuteDistanceGroup')['OverTime'].apply(
            lambda x: (x == 'Yes').mean() * 100
        ).reset_index()
        commute_overtime.columns = ['CommuteDistanceGroup', 'OvertimePercentage']
        
        fig = px.bar(commute_overtime, x='CommuteDistanceGroup', y='OvertimePercentage',
                    title="通勤距離と残業率の関係",
                    color='OvertimePercentage',
                    color_continuous_scale='Reds',
                    text_auto='.1f')
        
        # テキスト表示を修正
        fig.update_traces(
            text=[f"{val:.1f}%" for val in commute_overtime['OvertimePercentage']],
            textposition='outside'
        )
        
        # 最適化した図を表示
        display_optimized_chart(fig)
    
    with tab3:
        st.header("生産性最適化シミュレーション")
        st.info("注: このセクションはシミュレーションデータに基づいています。実際の意思決定には追加データが必要です。")
        
        # PDF出力時のページ区切り
        add_page_break()
        
        # 現在の生産性指標の計算
        current_overtime_rate = (df['OverTime'] == 'Yes').mean() * 100
        current_perf_rating = df['PerformanceRating'].mean()
        current_wlb_score = df['WorkLifeBalance'].mean()
        
        # 残業削減シミュレーション
        st.subheader("残業削減シミュレーション")
        
        # 残業削減率の入力
        overtime_reduction = st.slider(
            "残業削減目標 (%)",
            min_value=0,
            max_value=100,
            value=30,
            step=10
        )
        
        # シミュレーション（仮定に基づく）
        # 残業削減によるワークライフバランスの向上
        wlb_improvement = overtime_reduction / 100 * 0.5  # 残業30%減 → WLB 15%向上と仮定
        
        # 残業削減による短期的な業績への影響
        short_term_perf_impact = -overtime_reduction / 100 * 0.2  # 残業30%減 → 業績 6%減と仮定
        
        # 長期的な業績への影響（WLB向上による）
        long_term_perf_impact = wlb_improvement * 0.3  # WLB 15%向上 → 業績 4.5%向上と仮定
        
        # シミュレーション結果の表示
        # レスポンシブ対応のカラムレイアウト
        col1, col2 = create_responsive_columns()
        
        with col1:
            st.subheader("短期的影響（3ヶ月）")
            
            st.metric(
                "残業率",
                f"{current_overtime_rate * (1 - overtime_reduction/100):.1f}%",
                f"{-overtime_reduction:.1f}%",
                delta_color="inverse"
            )
            
            st.metric(
                "ワークライフバランススコア",
                f"{current_wlb_score * (1 + wlb_improvement):.2f}",
                f"{wlb_improvement*100:.1f}%"
            )
            
            st.metric(
                "業績評価",
                f"{current_perf_rating * (1 + short_term_perf_impact):.2f}",
                f"{short_term_perf_impact*100:.1f}%"
            )
        
        with col2:
            st.subheader("長期的影響（1年）")
            
            st.metric(
                "残業率",
                f"{current_overtime_rate * (1 - overtime_reduction/100):.1f}%",
                f"{-overtime_reduction:.1f}%",
                delta_color="inverse"
            )
            
            st.metric(
                "ワークライフバランススコア",
                f"{current_wlb_score * (1 + wlb_improvement * 1.2):.2f}",
                f"{wlb_improvement*120:.1f}%"
            )
            
            st.metric(
                "業績評価",
                f"{current_perf_rating * (1 + short_term_perf_impact + long_term_perf_impact):.2f}",
                f"{(short_term_perf_impact + long_term_perf_impact)*100:.1f}%"
            )
        
        # PDF出力時のページ区切り
        add_page_break()
        
        # リモートワーク導入シミュレーション
        st.subheader("リモートワーク導入シミュレーション")
        
        # リモートワークの導入率
        remote_work_pct = st.slider(
            "リモートワーク導入率 (%)",
            min_value=0,
            max_value=100,
            value=40,
            step=10
        )
        
        # リモート対象の選定基準
        remote_target = st.multiselect(
            "リモートワーク優先対象",
            options=[
                "長距離通勤者（20km以上）",
                "残業が多い従業員",
                "ワークライフバランス低スコア従業員",
                "出張が少ない従業員"
            ],
            default=["長距離通勤者（20km以上）", "ワークライフバランス低スコア従業員"]
        )
        
        # 長距離通勤者の比率
        long_commute_pct = (df['DistanceFromHome'] > 20).mean() * 100
        
        # 残業が多い従業員の比率
        high_overtime_pct = (df['OverTime'] == 'Yes').mean() * 100
        
        # WLBが低い従業員の比率
        low_wlb_pct = (df['WorkLifeBalance'] <= 2).mean() * 100
        
        # 出張が少ない従業員の比率
        low_travel_pct = ((df['BusinessTravel'] == 'Non-Travel') | 
                          (df['BusinessTravel'] == 'Travel_Rarely')).mean() * 100
        
        # 対象となる従業員の割合（重複考慮なしの単純計算）
        target_employee_pct = 0
        
        if "長距離通勤者（20km以上）" in remote_target:
            target_employee_pct += long_commute_pct
        if "残業が多い従業員" in remote_target:
            target_employee_pct += high_overtime_pct
        if "ワークライフバランス低スコア従業員" in remote_target:
            target_employee_pct += low_wlb_pct
        if "出張が少ない従業員" in remote_target:
            target_employee_pct += low_travel_pct
        
        # 重複を考慮して調整（単純化のため）
        if len(remote_target) > 1:
            target_employee_pct = min(target_employee_pct * 0.7, 100)
        
        # 実際にリモートワークが適用される従業員の割合
        effective_remote_pct = min(remote_work_pct, target_employee_pct)
        
        # シミュレーション仮定
        # リモートワークによる通勤時間削減効果
        commute_time_saved = effective_remote_pct / 100 * 60  # 分/日
        
        # リモートワークによるWLB向上効果
        remote_wlb_improvement = effective_remote_pct / 100 * 0.3
        
        # リモートワークによる残業削減効果
        remote_overtime_reduction = effective_remote_pct / 100 * 0.2 * 100
        
        # リモートワークによる業績への影響（短期的には適応期間として若干低下、長期的には向上と仮定）
        remote_short_term_perf = -effective_remote_pct / 100 * 0.1
        remote_long_term_perf = effective_remote_pct / 100 * 0.15
        
        # シミュレーション結果の表示
        # レスポンシブ対応のカラムレイアウト
        col1, col2 = create_responsive_columns()
        
        with col1:
            st.subheader("短期的影響（3ヶ月）")
            
            st.metric(
                "1日あたりの通勤時間削減",
                f"{commute_time_saved:.1f}分",
                f"{commute_time_saved:.1f}分"
            )
            
            st.metric(
                "残業率",
                f"{current_overtime_rate - remote_overtime_reduction:.1f}%",
                f"{-remote_overtime_reduction:.1f}%",
                delta_color="inverse"
            )
            
            st.metric(
                "ワークライフバランススコア",
                f"{current_wlb_score * (1 + remote_wlb_improvement * 0.7):.2f}",
                f"{remote_wlb_improvement * 70:.1f}%"
            )
            
            st.metric(
                "業績評価",
                f"{current_perf_rating * (1 + remote_short_term_perf):.2f}",
                f"{remote_short_term_perf * 100:.1f}%"
            )
        
        with col2:
            st.subheader("長期的影響（1年）")
            
            st.metric(
                "1日あたりの通勤時間削減",
                f"{commute_time_saved:.1f}分",
                f"{commute_time_saved:.1f}分"
            )
            
            st.metric(
                "残業率",
                f"{current_overtime_rate - remote_overtime_reduction * 1.2:.1f}%",
                f"{-remote_overtime_reduction * 1.2:.1f}%",
                delta_color="inverse"
            )
            
            st.metric(
                "ワークライフバランススコア",
                f"{current_wlb_score * (1 + remote_wlb_improvement):.2f}",
                f"{remote_wlb_improvement * 100:.1f}%"
            )
            
            st.metric(
                "業績評価",
                f"{current_perf_rating * (1 + remote_long_term_perf):.2f}",
                f"{remote_long_term_perf * 100:.1f}%"
            )
        
        # PDF出力時のページ区切り
        add_page_break()
        
        # おすすめの労働環境改善策
        st.subheader("推奨労働環境改善策")
        
        # データから抽出した課題に基づく推奨策
        recommendations = []
        
        # 残業が多い部門向け推奨策
        high_overtime_depts = overtime_by_dept[overtime_by_dept['OvertimePercentage'] > 
                                              overtime_by_dept['OvertimePercentage'].mean()]
        
        if not high_overtime_depts.empty:
            recommendations.append({
                "Target": f"残業率の高い部門（{', '.join(high_overtime_depts['Department'].values)}）",
                "Issue": "残業率が全社平均より高く、ワークライフバランスが低下している可能性がある",
                "Actions": [
                    "業務プロセスの効率化と自動化",
                    "人員配置の最適化",
                    "業務の優先順位付けと不要業務の削減",
                    "管理職向けのタイムマネジメント研修"
                ]
            })
        
        # 通勤距離が長い従業員向け推奨策
        long_commute_pct = (df['DistanceFromHome'] > 15).mean() * 100
        
        if long_commute_pct > 20:
            recommendations.append({
                "Target": f"通勤距離が長い従業員（全体の{long_commute_pct:.1f}%）",
                "Issue": "長時間通勤による疲労とワークライフバランスの悪化",
                "Actions": [
                    "柔軟な勤務時間制度の導入",
                    "リモートワークの部分的導入",
                    "サテライトオフィスの検討",
                    "通勤手当の見直し"
                ]
            })
        
        # ワークライフバランスが低い職種向け推奨策
        wlb_by_role = df.groupby('JobRole')['WorkLifeBalance'].mean()
        low_wlb_roles = wlb_by_role[wlb_by_role < wlb_by_role.mean()].index.tolist()
        
        if low_wlb_roles:
            recommendations.append({
                "Target": f"ワークライフバランスが低い職種（{', '.join(low_wlb_roles[:2])}など）",
                "Issue": "特定職種における仕事と生活のバランス悪化",
                "Actions": [
                    "業務分担の見直し",
                    "職種特有のストレス要因の特定と軽減",
                    "休暇取得促進プログラムの導入",
                    "メンタルヘルスサポートの強化"
                ]
            })
        
        # 推奨策の表示（レスポンシブ対応）
        for i, rec in enumerate(recommendations):
            expander = st.expander(f"推奨策 {i+1}: {rec['Target']}")
            with expander:
                st.write(f"**課題:** {rec['Issue']}")
                st.write("**推奨アクション:**")
                for action in rec['Actions']:
                    st.write(f"- {action}")
    
    # フッター
    st.markdown("---")
    st.info("このページでは、労働生産性とワークスタイルに関する分析と最適化シミュレーションを提供しています。"
            "より効率的で健全な労働環境の構築にご活用ください。")