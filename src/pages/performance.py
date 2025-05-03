import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from src.data.loader import load_hr_data
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
# レスポンシブ対応と PDF 出力用のユーティリティをインポート
from src.utils.layout_utils import (
    display_optimized_chart,
    create_responsive_columns,
    add_page_break,
    format_dataframe_for_display
)

def show():
    """
    パフォーマンス分析ページを表示する関数
    """
    st.title("人材育成・パフォーマンス分析")
    st.write("従業員の業績評価、能力開発、キャリアパスに関する分析")
    
    # データロード
    df = load_hr_data()
    
    # 全体のパフォーマンス概要
    st.header("パフォーマンス概要")
    
    # レスポンシブ対応のカラムレイアウト
    col1, col2, col3 = create_responsive_columns([1, 1, 1])
    
    with col1:
        avg_perf = df['PerformanceRating'].mean()
        st.metric("平均業績評価", f"{avg_perf:.2f}/4")
    
    with col2:
        high_perf_pct = len(df[df['PerformanceRating'] >= 3]) / len(df) * 100
        st.metric("高業績者の割合", f"{high_perf_pct:.1f}%")
    
    with col3:
        avg_training = df['TrainingTimesLastYear'].mean()
        st.metric("年間平均研修回数", f"{avg_training:.1f}回")
    
    # タブで分析内容を整理
    tab1, tab2, tab3 = st.tabs(["📈 業績評価分析", "🔄 キャリア開発", "👥 タレントマッピング"])
    
    with tab1:
        st.header("業績評価分析")
        
        # 業績評価の分布
        st.subheader("業績評価の分布")
        
        # 業績評価の対応表（データによっては調整が必要）
        rating_map = {1: '低', 2: '中', 3: '高', 4: '最高'}
        df['PerformanceLevel'] = df['PerformanceRating'].map(rating_map)
        
        perf_counts = df['PerformanceLevel'].value_counts().reset_index()
        perf_counts.columns = ['PerformanceLevel', 'Count']
        
        # カスタム配色を設定（業績が低いから高いまで、直感的な色で表現）
        custom_colors = {'低': '#FF6666', '中': '#FFCC66', '高': '#66CC66', '最高': '#6666FF'}
        
        # 業績評価レベルの順序を明示的に設定
        perf_counts['PerformanceLevel'] = pd.Categorical(
            perf_counts['PerformanceLevel'], 
            categories=['低', '中', '高', '最高'], 
            ordered=True
        )
        perf_counts = perf_counts.sort_values('PerformanceLevel')
        
        fig = px.pie(perf_counts, values='Count', names='PerformanceLevel',
                    title="業績評価の分布",
                    color='PerformanceLevel',
                    color_discrete_map=custom_colors)
        
        # 凡例の位置を調整
        fig.update_layout(legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5
        ))
        
        # 最適化した図を表示
        display_optimized_chart(fig)
        
        # 部門・役職別の業績評価
        st.subheader("部門・役職別の業績評価")
        
        group_by = st.selectbox(
            "グループ分け",
            options=['Department', 'JobRole', 'JobLevel', 'Education', 'Gender'],
            format_func=lambda x: {
                'Department': '部門',
                'JobRole': '職種',
                'JobLevel': '役職レベル',
                'Education': '教育レベル',
                'Gender': '性別'
            }.get(x, x)
        )
        
        perf_by_group = df.groupby(group_by)['PerformanceRating'].mean().reset_index()
        perf_by_group = perf_by_group.sort_values('PerformanceRating', ascending=False)
        
        fig = px.bar(perf_by_group, x=group_by, y='PerformanceRating',
                    title=f"{group_by}別の平均業績評価",
                    color='PerformanceRating',
                    color_continuous_scale='Viridis',
                    text_auto='.2f')
        
        if group_by == 'JobRole':
            fig.update_layout(xaxis_tickangle=-45)
        
        # 最適化した図を表示
        display_optimized_chart(fig)
        
        # PDF出力時のページ区切りを挿入
        add_page_break()
        
        # 業績評価と他の要因の関係
        st.subheader("業績評価と他の要因の関係")
        
        factors = st.multiselect(
            "分析したい要因を選択してください",
            options=[
                'YearsAtCompany', 'Age', 'JobSatisfaction', 'WorkLifeBalance', 
                'TrainingTimesLastYear', 'YearsSinceLastPromotion', 'EnvironmentSatisfaction'
            ],
            default=['JobSatisfaction', 'TrainingTimesLastYear', 'WorkLifeBalance']
        )
        
        if factors:
            # モバイル対応のためにレスポンシブなカラムを使用
            cols = create_responsive_columns([1] * min(3, len(factors)))
            
            for i, factor in enumerate(factors):
                with cols[i % len(cols)]:
                    if df[factor].dtype in ['int64', 'float64'] and len(df[factor].unique()) <= 10:
                        # 少数の値しかない数値変数（例: 1-4のスコア）
                        factor_perf = df.groupby(factor)['PerformanceRating'].mean().reset_index()
                        
                        fig = px.bar(factor_perf, x=factor, y='PerformanceRating',
                                    title=f"{factor}と業績評価の関係",
                                    color='PerformanceRating',
                                    color_continuous_scale='Viridis',
                                    text_auto='.2f')
                        
                    elif df[factor].dtype in ['int64', 'float64']:
                        # 連続的な数値変数の場合は相関分析
                        corr = df[[factor, 'PerformanceRating']].corr().iloc[0, 1]
                        
                        fig = px.scatter(df, x=factor, y='PerformanceRating',
                                        title=f"{factor}と業績評価の関係 (相関: {corr:.2f})",
                                        color='PerformanceRating', 
                                        color_continuous_scale='Viridis',
                                        trendline="ols")
                    
                    # 最適化した図を表示
                    display_optimized_chart(fig)
    
    with tab2:
        st.header("キャリア開発分析")
        
        # PDF出力時のページ区切りを挿入
        add_page_break()
        
        # 昇進分析
        st.subheader("昇進分析")
        
        # 役職レベル別の昇進率
        promotion_by_level = df.groupby('JobLevel')['YearsSinceLastPromotion'].mean().reset_index()
        
        fig = px.bar(promotion_by_level, x='JobLevel', y='YearsSinceLastPromotion',
                    title="役職レベル別の平均昇進間隔",
                    color='YearsSinceLastPromotion',
                    color_continuous_scale='Blues',
                    text_auto='.1f')
        
        fig.update_traces(texttemplate='%{text}年', textposition='outside')
        # 最適化した図を表示
        display_optimized_chart(fig)
        
        # 研修と業績の関係
        st.subheader("研修と業績の関係")
        
        training_perf = df.groupby('TrainingTimesLastYear')['PerformanceRating'].mean().reset_index()
        
        fig = px.line(training_perf, x='TrainingTimesLastYear', y='PerformanceRating',
                     title="研修回数と平均業績評価の関係",
                     markers=True,
                     line_shape='spline')
        
        # 最適化した図を表示
        display_optimized_chart(fig)
        
        # キャリアパス分析（シミュレーション）
        st.subheader("キャリアパス分析（シミュレーション）")
        st.info("注: このセクションはデモ用のシミュレーションデータを使用しています。")
        
        # キャリアパスのシミュレーションデータを作成
        career_paths = {}
        
        # 部門・役職レベルごとの昇進経路を定義
        for dept in df['Department'].unique():
            # 各部門でのキャリアパスをシミュレーション
            career_paths[dept] = {}
            
            # 部門内の職種を取得
            dept_roles = df[df['Department'] == dept]['JobRole'].unique()
            
            # 職種ごとにキャリアパスを定義
            for role in dept_roles:
                # 例: 営業部門のキャリアパス
                if dept == 'Sales' and role == 'Sales Representative':
                    career_paths[dept][role] = [
                        'Sales Representative',
                        'Sales Executive',
                        'Sales Manager',
                        'Sales Director'
                    ]
                # 例: 研究開発部門のキャリアパス
                elif dept == 'Research & Development' and role == 'Laboratory Technician':
                    career_paths[dept][role] = [
                        'Laboratory Technician',
                        'Research Scientist',
                        'Manager R&D',
                        'Research Director'
                    ]
                # 例: 人事部門のキャリアパス
                elif dept == 'Human Resources' and role == 'Human Resources':
                    career_paths[dept][role] = [
                        'Human Resources',
                        'HR Manager',
                        'HR Director'
                    ]
        
        # 特定の役職のキャリアパスを可視化
        selected_dept = st.selectbox(
            "部門を選択",
            options=sorted(df['Department'].unique()),
            key="career_dept"
        )
        
        roles_in_dept = sorted(df[df['Department'] == selected_dept]['JobRole'].unique())
        
        selected_role = st.selectbox(
            "現在の職種を選択",
            options=roles_in_dept
        )
        
        if selected_dept in career_paths and selected_role in career_paths[selected_dept]:
            # キャリアパスが定義されている場合
            path = career_paths[selected_dept][selected_role]
            
            # キャリアパスの可視化
            current_pos = path.index(selected_role)
            
            # キャリアパスチャートの作成
            path_data = []
            for i, pos in enumerate(path):
                status = "現在" if pos == selected_role else "将来" if i > current_pos else "過去"
                path_data.append({
                    'Position': pos,
                    'Step': i,
                    'Status': status
                })
            
            path_df = pd.DataFrame(path_data)
            
            fig = px.scatter(path_df, x='Step', y=[0] * len(path_df), 
                             text='Position', color='Status',
                             color_discrete_map={'過去': 'gray', '現在': 'green', '将来': 'blue'},
                             title="キャリアパス")
            
            fig.update_traces(marker=dict(size=20))
            fig.update_layout(
                showlegend=True,
                yaxis=dict(visible=False, showticklabels=False, range=[-1, 1]),
                xaxis=dict(title='キャリアステージ')
            )
            
            # 最適化した図を表示
            display_optimized_chart(fig)
            
            # 次のポジションに求められるスキル（シミュレーション）
            if current_pos < len(path) - 1:
                next_pos = path[current_pos + 1]
                st.subheader(f"次のポジション ({next_pos}) に必要なスキル")
                
                # 次のポジションに必要なスキルをシミュレーション
                required_skills = {
                    'Sales Executive': ['アカウント管理', '交渉力', 'ソリューション提案', 'CRM活用'],
                    'Sales Manager': ['営業戦略策定', 'チームマネジメント', '売上予測', 'リーダーシップ'],
                    'Research Scientist': ['実験設計', 'データ分析', '論文執筆', 'プロジェクト管理'],
                    'Manager R&D': ['研究指導', 'プロジェクト管理', '予算管理', 'チームマネジメント'],
                    'HR Manager': ['人事制度設計', '労務管理', 'タレントマネジメント', '組織開発']
                }
                
                if next_pos in required_skills:
                    for skill in required_skills[next_pos]:
                        st.write(f"- {skill}")
                else:
                    st.write("- リーダーシップスキル")
                    st.write("- プロジェクト管理能力")
                    st.write("- コミュニケーション能力")
                    st.write("- 専門知識の向上")
        else:
            st.write("選択した職種のキャリアパスデータがありません。")
    
    with tab3:
        st.header("タレントマッピング")
        
        # PDF出力時のページ区切りを挿入
        add_page_break()
        
        # シンプルなタレントマッピング分析
        st.subheader("業績とポテンシャルのタレントマップ")
        
        # ポテンシャルの指標を生成（シミュレーション）
        np.random.seed(42)
        
        # ポテンシャルスコアの生成（業績評価と相関を持たせつつもばらつきを加える）
        df['PotentialScore'] = (
            df['PerformanceRating'] * 0.6 +
            df['JobSatisfaction'] * 0.2 +
            df['TrainingTimesLastYear'] * 0.1 +
            np.random.normal(0, 0.3, size=len(df))
        )
        
        # スコアを1-5の範囲に正規化
        df['PotentialScore'] = (df['PotentialScore'] - df['PotentialScore'].min()) / (df['PotentialScore'].max() - df['PotentialScore'].min()) * 4 + 1
        
        # 業績とポテンシャルのマッピング
        fig = px.scatter(df, x='PerformanceRating', y='PotentialScore',
                        color='JobLevel',
                        size='YearsAtCompany',
                        hover_data=['EmployeeNumber', 'JobRole', 'Department'],
                        title="業績とポテンシャルのタレントマップ")
        
        # 9-boxグリッドのための線を追加
        fig.add_hline(y=2.33, line_dash="dash", line_color="gray")
        fig.add_hline(y=3.67, line_dash="dash", line_color="gray")
        fig.add_vline(x=2.33, line_dash="dash", line_color="gray")
        fig.add_vline(x=3.67, line_dash="dash", line_color="gray")
        
        # 各領域にラベルを追加
        annotations = [
            dict(x=1.5, y=1.5, text="要育成", showarrow=False, font=dict(size=12, color="red")),
            dict(x=1.5, y=3, text="要観察", showarrow=False, font=dict(size=12, color="orange")),
            dict(x=1.5, y=4.5, text="潜在能力有", showarrow=False, font=dict(size=12, color="blue")),
            dict(x=3, y=1.5, text="安定貢献", showarrow=False, font=dict(size=12, color="orange")),
            dict(x=3, y=3, text="中核人材", showarrow=False, font=dict(size=12, color="black")),
            dict(x=3, y=4.5, text="将来有望", showarrow=False, font=dict(size=12, color="green")),
            dict(x=4.5, y=1.5, text="専門家", showarrow=False, font=dict(size=12, color="blue")),
            dict(x=4.5, y=3, text="高業績者", showarrow=False, font=dict(size=12, color="green")),
            dict(x=4.5, y=4.5, text="ハイポテンシャル", showarrow=False, font=dict(size=12, color="purple"))
        ]
        
        for annotation in annotations:
            fig.add_annotation(annotation)
        
        fig.update_layout(
            xaxis=dict(
                title="業績評価",
                range=[1, 5],
                tickvals=[1, 2, 3, 4, 5]
            ),
            yaxis=dict(
                title="ポテンシャル評価",
                range=[1, 5],
                tickvals=[1, 2, 3, 4, 5]
            )
        )
        
        # 最適化した図を表示
        display_optimized_chart(fig)
        
        # 部門/職種別のタレント分布
        st.subheader("部門/職種別のタレント分布")
        
        # タレントカテゴリの定義（9-box）
        def get_talent_category(row):
            perf = row['PerformanceRating']
            potential = row['PotentialScore']
            
            if perf < 2.33:
                if potential < 2.33:
                    return "要育成"
                elif potential < 3.67:
                    return "要観察"
                else:
                    return "潜在能力有"
            elif perf < 3.67:
                if potential < 2.33:
                    return "安定貢献"
                elif potential < 3.67:
                    return "中核人材"
                else:
                    return "将来有望"
            else:
                if potential < 2.33:
                    return "専門家"
                elif potential < 3.67:
                    return "高業績者"
                else:
                    return "ハイポテンシャル"
        
        df['TalentCategory'] = df.apply(get_talent_category, axis=1)
        
        # グループ選択
        group_field = st.selectbox(
            "グループ分け",
            options=['Department', 'JobRole'],
            format_func=lambda x: {'Department': '部門', 'JobRole': '職種'}.get(x, x)
        )
        
        # 選択したグループ別のタレントカテゴリ分布
        talent_dist = df.groupby([group_field, 'TalentCategory']).size().reset_index()
        talent_dist.columns = [group_field, 'TalentCategory', 'Count']
        
        # ピボットテーブルの作成
        talent_pivot = talent_dist.pivot_table(
            index=group_field,
            columns='TalentCategory',
            values='Count',
            fill_value=0
        )
        
        # ヒートマップの作成
        fig = px.imshow(
            talent_pivot,
            color_continuous_scale='Viridis',
            aspect="auto",
            text_auto=True,
            title=f"{group_field}別のタレントカテゴリ分布"
        )
        
        # 最適化した図を表示
        display_optimized_chart(fig)
    
    # フッター
    st.markdown("---")
    st.info("このページでは、従業員の業績評価、能力開発、キャリアパスに関する分析を提供しています。"
            "人材育成プログラムやサクセッションプランの策定にご活用ください。")