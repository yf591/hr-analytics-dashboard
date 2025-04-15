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
    
    col1, col2, col3 = st.columns(3)
    
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
        
        st.plotly_chart(fig, use_container_width=True)
        
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
        
        st.plotly_chart(fig, use_container_width=True)
        
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
            cols = st.columns(min(3, len(factors)))
            
            for i, factor in enumerate(factors):
                with cols[i % len(cols)]:
                    if df[factor].dtype in ['int64', 'float64'] and len(df[factor].unique()) > 5:
                        # 連続変数の場合は散布図
                        fig = px.scatter(df, x=factor, y='PerformanceRating',
                                        color='PerformanceRating',
                                        color_continuous_scale='Viridis',
                                        trendline='ols',
                                        title=f"{factor}と業績評価の関係")
                    else:
                        # カテゴリ変数またはレベル変数の場合は箱ひげ図
                        fig = px.box(df, x=factor, y='PerformanceRating',
                                    color=factor,
                                    title=f"{factor}と業績評価の関係")
                    
                    st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.header("キャリア開発分析")
        
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
        st.plotly_chart(fig, use_container_width=True)
        
        # 研修と業績の関係
        st.subheader("研修と業績の関係")
        
        training_perf = df.groupby('TrainingTimesLastYear')['PerformanceRating'].mean().reset_index()
        
        fig = px.line(training_perf, x='TrainingTimesLastYear', y='PerformanceRating',
                     title="研修回数と平均業績評価の関係",
                     markers=True,
                     line_shape='spline')
        
        st.plotly_chart(fig, use_container_width=True)
        
        # キャリアパス分析（シミュレーション）
        st.subheader("キャリアパス分析（シミュレーション）")
        st.info("注: このセクションはデモ用のシミュレーションデータを使用しています。")
        
        # キャリアパスのシミュレーションデータを作成
        career_paths = {}
        
        # 部門・役職レベルごとの昇進経路を定義
        for dept in df['Department'].unique():
            career_paths[dept] = {}
            
            # 各ジョブロールの昇進経路を定義
            roles = df[df['Department'] == dept]['JobRole'].unique()
            
            for role in roles:
                # 役職レベルに応じたキャリアパスを定義
                level_path = []
                
                # 現在の役職
                current_role = {
                    'role': role,
                    'level': 1,
                    'years_required': 0,
                    'skills_required': [],
                    'promotion_rate': 0
                }
                
                level_path.append(current_role)
                
                # 次のレベルへの昇進をシミュレーション
                for level in range(2, 6):
                    # 同じ部門内の可能性のある次の役職
                    next_roles = df[(df['Department'] == dept) & 
                                   (df['JobLevel'] >= level)]['JobRole'].unique()
                    
                    if len(next_roles) == 0:
                        break
                    
                    next_role = np.random.choice(next_roles)
                    
                    # 現在の部門内での昇進か、他部門への異動かを決定
                    is_promotion = np.random.choice([True, False], p=[0.8, 0.2])
                    next_dept = dept if is_promotion else np.random.choice(
                        [d for d in df['Department'].unique() if d != dept]
                    )
                    
                    # 平均昇進年数（レベルが上がるほど長くなる傾向）
                    years_required = level + np.random.randint(1, 3)
                    
                    # スキル要件のシミュレーション
                    skills = [
                        'リーダーシップ', '専門知識', 'コミュニケーション', '問題解決能力',
                        'チームマネジメント', '戦略的思考', '交渉力', 'プロジェクト管理'
                    ]
                    required_skills = np.random.choice(
                        skills, size=min(level+1, len(skills)), replace=False
                    ).tolist()
                    
                    # 昇進率（上位ほど狭き門に）
                    promotion_rate = max(5, 100 - (level * 15) - np.random.randint(0, 10))
                    
                    next_position = {
                        'role': next_role,
                        'department': next_dept,
                        'level': level,
                        'years_required': years_required,
                        'skills_required': required_skills,
                        'promotion_rate': promotion_rate
                    }
                    
                    level_path.append(next_position)
                
                career_paths[dept][role] = level_path
        
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
            path = career_paths[selected_dept][selected_role]
            
            # キャリアパスの可視化
            path_data = []
            
            for step in path:
                path_data.append({
                    'Level': step.get('level', 0),
                    'Role': step.get('role', ''),
                    'Department': step.get('department', selected_dept),
                    'YearsRequired': step.get('years_required', 0),
                    'PromotionRate': step.get('promotion_rate', 0)
                })
            
            path_df = pd.DataFrame(path_data)
            
            if not path_df.empty and len(path_df) > 1:
                # キャリアパスの可視化
                fig = go.Figure()
                
                for i, row in path_df.iterrows():
                    if i < len(path_df) - 1:
                        next_row = path_df.iloc[i+1]
                        
                        # ノード間の線を描画
                        fig.add_trace(go.Scatter(
                            x=[row['Level'], next_row['Level']],
                            y=[1, 1],
                            mode='lines',
                            line=dict(width=2, color='rgba(100, 100, 100, 0.5)'),
                            showlegend=False
                        ))
                
                # 各ポジションのノードを描画
                fig.add_trace(go.Scatter(
                    x=path_df['Level'],
                    y=[1] * len(path_df),
                    mode='markers+text',
                    marker=dict(
                        size=[30 + (10 * l) for l in path_df['Level']],
                        color=['rgba(55, 126, 184, 0.7)'] * len(path_df),
                        line=dict(color='rgba(0, 0, 0, 0.5)', width=1)
                    ),
                    text=path_df['Role'],
                    textposition='bottom center',
                    showlegend=False
                ))
                
                # レイアウト設定
                fig.update_layout(
                    title="キャリアパスの可視化",
                    xaxis=dict(
                        title="キャリアレベル",
                        showgrid=False,
                        zeroline=False,
                        showticklabels=True,
                        range=[0.5, max(path_df['Level']) + 0.5]
                    ),
                    yaxis=dict(
                        showgrid=False,
                        zeroline=False,
                        showticklabels=False,
                        range=[0.5, 1.5]
                    ),
                    height=300
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # 各ステップの詳細情報
                for i, step in enumerate(path[1:], 1):
                    expander = st.expander(f"キャリアステップ {i}: {step['role']} (レベル{step['level']})")
                    
                    with expander:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write(f"**部門:** {step.get('department', selected_dept)}")
                            st.write(f"**必要経験年数:** {step.get('years_required', 0)}年")
                            st.write(f"**昇進確率:** {step.get('promotion_rate', 0)}%")
                        
                        with col2:
                            st.write("**必要なスキル**")
                            for skill in step.get('skills_required', []):
                                st.write(f"- {skill}")
            else:
                st.warning(f"選択された職種 '{selected_role}' のキャリアパスデータが不足しています。")
        else:
            st.warning(f"選択された部門 '{selected_dept}' と職種 '{selected_role}' の組み合わせのキャリアパスデータがありません。")
    
    with tab3:
        st.header("タレントマッピング")
        
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
            # 低業績、低ポテンシャル
            dict(x=1.5, y=1.5, text="要育成", showarrow=False, font=dict(size=12, color="red")),
            # 低業績、中ポテンシャル
            dict(x=1.5, y=3, text="要観察", showarrow=False, font=dict(size=12, color="orange")),
            # 低業績、高ポテンシャル
            dict(x=1.5, y=4.5, text="潜在能力有", showarrow=False, font=dict(size=12, color="blue")),
            # 中業績、低ポテンシャル
            dict(x=3, y=1.5, text="安定貢献", showarrow=False, font=dict(size=12, color="orange")),
            # 中業績、中ポテンシャル
            dict(x=3, y=3, text="中核人材", showarrow=False, font=dict(size=12, color="black")),
            # 中業績、高ポテンシャル
            dict(x=3, y=4.5, text="将来有望", showarrow=False, font=dict(size=12, color="green")),
            # 高業績、低ポテンシャル
            dict(x=4.5, y=1.5, text="専門家", showarrow=False, font=dict(size=12, color="blue")),
            # 高業績、中ポテンシャル
            dict(x=4.5, y=3, text="高業績者", showarrow=False, font=dict(size=12, color="green")),
            # 高業績、高ポテンシャル
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
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 部門/職種別のタレント分布
        st.subheader("部門/職種別のタレント分布")
        
        # タレントカテゴリの定義（9-box）
        def get_talent_category(row):
            perf = row['PerformanceRating']
            pot = row['PotentialScore']
            
            if perf < 2.33:
                if pot < 2.33:
                    return "要育成"
                elif pot < 3.67:
                    return "要観察"
                else:
                    return "潜在能力有"
            elif perf < 3.67:
                if pot < 2.33:
                    return "安定貢献"
                elif pot < 3.67:
                    return "中核人材"
                else:
                    return "将来有望"
            else:
                if pot < 2.33:
                    return "専門家"
                elif pot < 3.67:
                    return "高業績者"
                else:
                    return "ハイポテンシャル"
        
        df['TalentCategory'] = df.apply(get_talent_category, axis=1)
        
        # グループ選択
        group_by_talent = st.selectbox(
            "グループ分け",
            options=['Department', 'JobRole', 'JobLevel', 'Gender'],
            key="talent_group",
            format_func=lambda x: {
                'Department': '部門',
                'JobRole': '職種',
                'JobLevel': '役職レベル',
                'Gender': '性別'
            }.get(x, x)
        )
        
        # タレントカテゴリと選択されたグループのクロス集計
        talent_cross = pd.crosstab(df[group_by_talent], df['TalentCategory'])
        
        # 百分率に変換（行方向）
        talent_cross_pct = talent_cross.div(talent_cross.sum(axis=1), axis=0) * 100
        
        # ヒートマップの表示
        fig = px.imshow(talent_cross_pct,
                       labels=dict(x="タレントカテゴリ", y=group_by_talent, color="割合 (%)"),
                       text_auto='.1f',
                       color_continuous_scale='Viridis')
        
        fig.update_layout(title=f"{group_by_talent}別のタレント分布 (%)")
        st.plotly_chart(fig, use_container_width=True)
        
        # K-means分析によるタレントクラスタリング
        st.subheader("タレントクラスタリング分析")
        
        n_clusters = st.slider("クラスタ数", min_value=2, max_value=6, value=4, step=1)
        
        # クラスタリングのための特徴量を選択
        features = [
            'PerformanceRating', 'PotentialScore', 'JobSatisfaction', 
            'TrainingTimesLastYear', 'YearsAtCompany', 'YearsSinceLastPromotion'
        ]
        
        # データの前処理
        scaler = StandardScaler()
        X = scaler.fit_transform(df[features])
        
        # K-meansクラスタリングの実行
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        df['Cluster'] = kmeans.fit_predict(X)
        
        # 次元削減なしで2つの主要な特徴量でプロット
        fig = px.scatter(df, x='PerformanceRating', y='PotentialScore',
                        color='Cluster',
                        title="タレントクラスタリング結果",
                        hover_data=['EmployeeNumber', 'JobRole', 'Department'])
        
        st.plotly_chart(fig, use_container_width=True)
        
        # クラスターの特徴を表示
        cluster_profiles = df.groupby('Cluster')[features].mean().reset_index()
        
        for i, row in cluster_profiles.iterrows():
            expander = st.expander(f"クラスター {int(row['Cluster'])+1} のプロファイル")
            with expander:
                # レーダーチャート用のデータ準備
                categories = ['業績評価', 'ポテンシャル', '職務満足度', '研修頻度', '勤続年数', '昇進間隔']
                values = [
                    row['PerformanceRating'],
                    row['PotentialScore'],
                    row['JobSatisfaction'],
                    row['TrainingTimesLastYear'],
                    row['YearsAtCompany'],
                    row['YearsSinceLastPromotion']
                ]
                
                # レーダーチャートの作成
                fig = go.Figure()
                
                fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=categories,
                    fill='toself',
                    name=f'クラスター {int(row["Cluster"])+1}'
                ))
                
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True
                        )
                    ),
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # クラスターのサイズ
                cluster_size = (df['Cluster'] == row['Cluster']).sum()
                st.write(f"**クラスターサイズ:** {cluster_size}人 ({cluster_size/len(df)*100:.1f}%)")
                
                # クラスター内の主要部門・役職
                dept_dist = df[df['Cluster'] == row['Cluster']]['Department'].value_counts().head(3)
                role_dist = df[df['Cluster'] == row['Cluster']]['JobRole'].value_counts().head(3)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**主要部門:**")
                    for dept, count in dept_dist.items():
                        st.write(f"- {dept}: {count}人 ({count/cluster_size*100:.1f}%)")
                
                with col2:
                    st.write("**主要職種:**")
                    for role, count in role_dist.items():
                        st.write(f"- {role}: {count}人 ({count/cluster_size*100:.1f}%)")
    
    # フッター
    st.markdown("---")
    st.info("このページでは、従業員の業績評価、能力開発、キャリアパスに関する分析を提供しています。"
            "人材育成プログラムやサクセッションプランの策定にご活用ください。")