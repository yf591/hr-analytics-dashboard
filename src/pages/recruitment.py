import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from src.data.loader import load_hr_data
# レスポンシブ対応と PDF 出力用のユーティリティをインポート
from src.utils.layout_utils import (
    display_optimized_chart,
    create_responsive_columns,
    add_page_break,
    format_dataframe_for_display
)

def show():
    """
    人材獲得分析ページを表示する関数
    """
    st.title("人材獲得分析")
    st.write("採用プロセスの分析と効率化提案")
    
    # データロード
    df = load_hr_data()
    
    # ダミーの採用データを生成（実際のプロジェクトでは実データを使用）
    # 実データがない場合、既存データから仮想的な採用データを生成
    
    # 過去10ヶ月分の採用データをシミュレーション
    np.random.seed(42)  # 再現性のため
    
    months = ["2024-07", "2024-08", "2024-09", "2024-10", "2024-11", "2024-12", 
              "2025-01", "2025-02", "2025-03", "2025-04"]
    
    recruitment_data = []
    
    departments = df['Department'].unique()
    jobroles = df['JobRole'].unique()
    
    for month in months:
        # 各部門・職種ごとの採用活動をシミュレーション
        for dept in departments:
            for role in jobroles:
                # 母集団の中で、この部門・職種の組み合わせが存在するか確認
                if len(df[(df['Department'] == dept) & (df['JobRole'] == role)]) > 0:
                    # 応募者数（ランダム）
                    applicants = np.random.randint(2, 30)
                    
                    # 一次面接通過者
                    first_interview = np.random.randint(1, max(2, int(applicants * 0.7)))
                    
                    # 二次面接通過者
                    second_interview = np.random.randint(1, max(2, int(first_interview * 0.7)))
                    
                    # オファー数
                    offers = np.random.randint(0, max(1, int(second_interview * 0.8)))
                    
                    # 内定承諾数
                    acceptances = np.random.randint(0, max(1, offers))
                    
                    # 採用コスト（応募者数と役職レベルに基づく）
                    avg_level = df[(df['Department'] == dept) & (df['JobRole'] == role)]['JobLevel'].mean()
                    cost_per_hire = int(200000 + avg_level * 50000 + applicants * 2000)
                    
                    # データ追加
                    recruitment_data.append({
                        'Month': month,
                        'Department': dept,
                        'JobRole': role,
                        'Applicants': applicants,
                        'FirstInterview': first_interview,
                        'SecondInterview': second_interview,
                        'Offers': offers,
                        'Acceptances': acceptances,
                        'CostPerHire': cost_per_hire,
                        'TimeToFill': np.random.randint(20, 90),  # 採用にかかった日数
                        'SourceChannel': np.random.choice(['リファラル', '求人サイト', 'SNS', '転職エージェント', '自社ウェブサイト'], 
                                                       p=[0.2, 0.3, 0.15, 0.25, 0.1])
                    })
    
    # データフレーム化
    recruitment_df = pd.DataFrame(recruitment_data)
    
    # 分析用タブの作成
    tab1, tab2, tab3 = st.tabs(["📊 採用ファネル分析", "💰 採用コスト分析", "🔍 採用ソース分析"])
    
    with tab1:
        st.header("採用ファネル分析")
        
        # 部門・職種フィルター
        col1, col2 = create_responsive_columns()
        
        with col1:
            dept_filter = st.multiselect(
                "部門",
                options=recruitment_df['Department'].unique(),
                default=recruitment_df['Department'].unique()
            )
        
        with col2:
            role_filter = st.multiselect(
                "職種",
                options=recruitment_df['JobRole'].unique(),
                default=recruitment_df['JobRole'].unique()
            )
        
        # フィルター適用
        filtered_df = recruitment_df[
            (recruitment_df['Department'].isin(dept_filter)) &
            (recruitment_df['JobRole'].isin(role_filter))
        ]
        
        # 期間ごとの採用ファネル
        st.subheader("期間ごとの採用ファネル")
        
        # 期間別集計
        funnel_by_month = filtered_df.groupby('Month').agg({
            'Applicants': 'sum',
            'FirstInterview': 'sum',
            'SecondInterview': 'sum',
            'Offers': 'sum',
            'Acceptances': 'sum'
        }).reset_index()
        
        # 転換率の計算
        funnel_by_month['FirstInterviewRate'] = funnel_by_month['FirstInterview'] / funnel_by_month['Applicants']
        funnel_by_month['SecondInterviewRate'] = funnel_by_month['SecondInterview'] / funnel_by_month['FirstInterview']
        funnel_by_month['OfferRate'] = funnel_by_month['Offers'] / funnel_by_month['SecondInterview']
        funnel_by_month['AcceptanceRate'] = funnel_by_month['Acceptances'] / funnel_by_month['Offers']
        funnel_by_month['OverallConversionRate'] = funnel_by_month['Acceptances'] / funnel_by_month['Applicants']
        
        # ヒートマップでの可視化
        conversion_metrics = ['FirstInterviewRate', 'SecondInterviewRate', 'OfferRate', 'AcceptanceRate', 'OverallConversionRate']
        conversion_labels = ['一次面接率', '二次面接率', 'オファー率', '内定承諾率', '全体転換率']
        
        # 転換率表示用のヒートマップ
        heat_data = funnel_by_month[['Month'] + conversion_metrics].set_index('Month')
        
        # ヒートマップ表示前に列名を日本語に変換
        heat_data.columns = conversion_labels
        
        fig = px.imshow(
            heat_data.T, 
            text_auto='.1%',
            aspect="auto",
            color_continuous_scale='RdYlGn',
            title="月別採用転換率ヒートマップ"
        )
        
        # 最適化した図を表示
        display_optimized_chart(fig)
        
        # 採用ファネルの可視化
        st.subheader("全期間の採用ファネル")
        
        # 全期間の集計
        total_funnel = filtered_df.sum()
        funnel_metrics = ['Applicants', 'FirstInterview', 'SecondInterview', 'Offers', 'Acceptances']
        funnel_values = total_funnel[funnel_metrics].values
        funnel_labels = ['応募者', '一次面接', '二次面接', 'オファー', '内定承諾']
        
        # ファネルチャート
        fig = go.Figure(go.Funnel(
            y=funnel_labels,
            x=funnel_values,
            textinfo="value+percent initial",
            marker=dict(color=["royalblue", "mediumslateblue", "slateblue", "darkslateblue", "midnightblue"])
        ))
        
        fig.update_layout(title_text="採用ファネル分析")
        
        # 最適化した図を表示
        display_optimized_chart(fig)
        
        # PDF出力時のページ区切り
        add_page_break()
        
        # 部門別の採用ファネル効率
        st.subheader("部門別の採用ファネル効率")
        
        dept_funnel = filtered_df.groupby('Department').agg({
            'Applicants': 'sum',
            'FirstInterview': 'sum',
            'SecondInterview': 'sum',
            'Offers': 'sum',
            'Acceptances': 'sum'
        }).reset_index()
        
        dept_funnel['OverallConversionRate'] = dept_funnel['Acceptances'] / dept_funnel['Applicants']
        dept_funnel = dept_funnel.sort_values('OverallConversionRate', ascending=False)
        
        fig = px.bar(
            dept_funnel,
            x='Department',
            y='OverallConversionRate',
            title="部門別の応募者→内定承諾の転換率",
            color='OverallConversionRate',
            color_continuous_scale='Viridis',
            text_auto='.1%'
        )
        
        # 最適化した図を表示
        display_optimized_chart(fig)
        
        # 職種別の採用難易度分析
        st.subheader("職種別の採用難易度")
        
        role_difficulty = filtered_df.groupby('JobRole').agg({
            'Applicants': 'sum',
            'Acceptances': 'sum',
            'TimeToFill': 'mean'
        }).reset_index()
        
        role_difficulty['ApplicantsPerHire'] = role_difficulty['Applicants'] / role_difficulty['Acceptances']
        role_difficulty = role_difficulty.sort_values('ApplicantsPerHire', ascending=False)
        
        fig = px.scatter(
            role_difficulty,
            x='ApplicantsPerHire',
            y='TimeToFill',
            color='Acceptances',
            size='Applicants',
            hover_name='JobRole',
            title="職種別の採用難易度マップ",
            labels={
                'ApplicantsPerHire': '1採用あたりの応募者数',
                'TimeToFill': '平均採用所要日数',
                'Acceptances': '採用数'
            }
        )
        
        # 最適化した図を表示
        display_optimized_chart(fig)
        
        # 採用プロセスボトルネック分析
        st.subheader("採用プロセスのボトルネック分析")
        
        # 全期間の転換率平均
        avg_conversion = {
            '応募→一次面接': (filtered_df['FirstInterview'].sum() / filtered_df['Applicants'].sum()),
            '一次→二次面接': (filtered_df['SecondInterview'].sum() / filtered_df['FirstInterview'].sum()),
            '二次面接→オファー': (filtered_df['Offers'].sum() / filtered_df['SecondInterview'].sum()),
            'オファー→内定承諾': (filtered_df['Acceptances'].sum() / filtered_df['Offers'].sum())
        }
        
        bottleneck_df = pd.DataFrame({
            'Stage': list(avg_conversion.keys()),
            'ConversionRate': list(avg_conversion.values())
        })
        
        bottleneck_df = bottleneck_df.sort_values('ConversionRate')
        
        fig = px.bar(
            bottleneck_df,
            x='Stage',
            y='ConversionRate',
            title="採用プロセスの段階別転換率",
            color='ConversionRate',
            color_continuous_scale='RdYlGn',
            text_auto='.1%'
        )
        
        # ボトルネックの識別
        bottleneck_stage = bottleneck_df.iloc[0]['Stage']
        bottleneck_rate = bottleneck_df.iloc[0]['ConversionRate']
        
        # 最適化した図を表示
        display_optimized_chart(fig)
        
        st.info(f"**主要なボトルネック**: {bottleneck_stage} (転換率: {bottleneck_rate:.1%})")
        
        # ボトルネックに基づく改善提案
        improvement_suggestions = {
            '応募→一次面接': [
                "採用要件の明確化と求人情報の改善",
                "応募者スクリーニングプロセスの見直し",
                "適切なキーワード・スキル検索の実施"
            ],
            '一次→二次面接': [
                "面接官のトレーニング強化",
                "一次面接の評価基準の見直し",
                "候補者への会社文化とビジョンの効果的な伝達"
            ],
            '二次面接→オファー': [
                "最終面接と評価プロセスの効率化",
                "内部承認プロセスの迅速化",
                "部門マネージャーと採用担当の連携強化"
            ],
            'オファー→内定承諾': [
                "オファーパッケージの競争力強化",
                "候補者とのコミュニケーション頻度の向上",
                "入社前フォローアッププログラムの導入"
            ]
        }
        
        st.write("**改善提案:**")
        for suggestion in improvement_suggestions[bottleneck_stage]:
            st.write(f"- {suggestion}")
    
    with tab2:
        st.header("採用コスト分析")
        
        # PDF出力時のページ区切り
        add_page_break()
        
        # 全体の採用コスト概要
        total_cost = filtered_df['CostPerHire'].sum()
        total_hires = filtered_df['Acceptances'].sum()
        avg_cost_per_hire = total_cost / total_hires if total_hires > 0 else 0
        
        # レスポンシブ対応のメトリック表示
        col1, col2, col3 = create_responsive_columns([1, 1, 1])
        
        with col1:
            st.metric("総採用コスト", f"¥{total_cost:,.0f}")
        
        with col2:
            st.metric("総採用数", f"{total_hires:.0f}人")
        
        with col3:
            st.metric("平均採用単価", f"¥{avg_cost_per_hire:,.0f}")
        
        # 月別の採用コストと採用数の推移
        st.subheader("月別の採用コストと採用数の推移")
        
        monthly_cost = filtered_df.groupby('Month').agg({
            'CostPerHire': 'sum',
            'Acceptances': 'sum'
        }).reset_index()
        
        # 採用コストと採用数の推移グラフ
        fig = px.bar(
            monthly_cost,
            x='Month',
            y='CostPerHire',
            title="月別の採用コストと採用数",
            labels={'CostPerHire': '採用コスト (円)', 'Month': '月'}
        )
        
        # 採用数の線グラフを追加
        fig.add_trace(
            go.Scatter(
                x=monthly_cost['Month'],
                y=monthly_cost['Acceptances'],
                mode='lines+markers',
                name='採用数',
                yaxis='y2'
            )
        )
        
        # 2軸グラフの設定
        fig.update_layout(
            yaxis=dict(
                title='採用コスト (円)',
                tickformat=",",  # 桁区切りのカンマ表示を追加
                exponentformat='none'  # 指数表記を使用しない
            ),
            yaxis2=dict(
                title='採用数',
                overlaying='y',
                side='right'
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # テキスト表示の修正
        for i in range(len(fig.data)):
            if hasattr(fig.data[i], 'type') and fig.data[i].type == 'bar':
                fig.data[i].text = ["¥{:,.0f}".format(val) for val in monthly_cost['CostPerHire']]
                fig.data[i].textposition = 'outside'
        
        # 最適化した図を表示
        display_optimized_chart(fig)
        
        # 部門別の採用コスト効率
        st.subheader("部門別の採用コスト効率")
        
        dept_cost = filtered_df.groupby('Department').agg({
            'CostPerHire': 'sum',
            'Acceptances': 'sum'
        }).reset_index()
        
        dept_cost['CostPerAcceptance'] = dept_cost['CostPerHire'] / dept_cost['Acceptances']
        dept_cost = dept_cost.sort_values('CostPerAcceptance')
        
        fig = px.bar(
            dept_cost,
            x='Department',
            y='CostPerAcceptance',
            title="部門別の採用単価",
            color='CostPerAcceptance',
            color_continuous_scale='Reds_r'
        )
        
        # y軸の範囲と書式を設定
        fig.update_layout(
            yaxis=dict(
                tickformat=",",  # 桁区切りのカンマ表示
                exponentformat='none',  # 指数表記を使用しない
                title="採用単価 (円)"  # y軸のタイトルを追加
            )
        )
        
        # テキスト表示の修正
        fig.update_traces(
            text=["¥{:,.0f}".format(val) for val in dept_cost['CostPerAcceptance']],
            textposition='outside'
        )
        
        # 最適化した図を表示
        display_optimized_chart(fig)
        
        # PDF出力時のページ区切り
        add_page_break()
        
        # 職種別の採用コスト効率
        st.subheader("職種別の採用コスト効率")
        
        role_cost = filtered_df.groupby('JobRole').agg({
            'CostPerHire': 'sum',
            'Acceptances': 'sum'
        }).reset_index()
        
        role_cost['CostPerAcceptance'] = role_cost['CostPerHire'] / role_cost['Acceptances']
        role_cost = role_cost.sort_values('CostPerAcceptance')
        
        fig = px.bar(
            role_cost,
            x='JobRole',
            y='CostPerAcceptance',
            title="職種別の採用単価",
            color='CostPerAcceptance',
            color_continuous_scale='Reds_r'
        )
        
        # y軸の範囲と書式を設定
        fig.update_layout(
            xaxis_tickangle=-45,
            yaxis=dict(
                range=[0, 20000000],  # y軸の範囲を0〜20Mに設定
                tickformat=",",  # 桁区切りのカンマ表示
                title="採用単価 (円)"  # y軸のタイトルを追加
            )
        )
        
        # テキスト表示の修正
        fig.update_traces(
            text=["¥{:,.0f}".format(val) for val in role_cost['CostPerAcceptance']],
            textposition='outside'
        )
        
        # 最適化した図を表示
        display_optimized_chart(fig)
        
        # 採用コスト最適化シミュレーション
        st.subheader("採用コスト最適化シミュレーション")
        
        # 最適化ターゲットの選択
        optimization_target = st.selectbox(
            "コスト最適化ターゲット",
            options=["高コスト職種の最適化", "採用チャネルの配分最適化", "採用プロセスの効率化"],
            index=0
        )
        
        if optimization_target == "高コスト職種の最適化":
            # 高コストの職種を特定
            high_cost_roles = role_cost.nlargest(3, 'CostPerAcceptance')
            
            st.write("**高コスト職種:**")
            for i, row in high_cost_roles.iterrows():
                st.write(f"- {row['JobRole']}: ¥{row['CostPerAcceptance']:,.0f} / 採用")
            
            # 最適化シミュレーション
            st.write("**コスト削減シミュレーション**")
            
            cost_reduction = st.slider(
                "高コスト職種のコスト削減目標 (%)",
                min_value=5,
                max_value=30,
                value=15,
                step=5
            )
            
            # コスト削減効果の計算
            high_cost_total = high_cost_roles['CostPerHire'].sum()
            savings = high_cost_total * (cost_reduction / 100)
            
            st.success(f"高コスト職種のコスト最適化により、約 ¥{savings:,.0f} の削減が見込まれます（{cost_reduction}%削減の場合）")
            
            # コスト削減の具体的な方法
            st.write("**コスト削減のための改善策:**")
            
            cost_savings_methods = [
                "リファラル採用プログラムの強化（従業員紹介の報酬を最適化）",
                "採用エージェント費用の交渉と複数エージェントの比較評価",
                "内部採用担当者のスキル向上とツール導入による外部依存度の削減",
                "採用ブランディング強化による直接応募の増加",
                "採用プロセスの効率化による時間短縮"
            ]
            
            for method in cost_savings_methods:
                st.write(f"- {method}")
        
        elif optimization_target == "採用チャネルの配分最適化":
            # チャネル別のコスト効率
            channel_cost = filtered_df.groupby('SourceChannel').agg({
                'CostPerHire': 'sum',
                'Acceptances': 'sum'
            }).reset_index()
            
            channel_cost['CostPerAcceptance'] = channel_cost['CostPerHire'] / channel_cost['Acceptances']
            channel_cost = channel_cost.sort_values('CostPerAcceptance')
            
            # 現在のチャネル配分
            current_allocation = filtered_df.groupby('SourceChannel').size() / len(filtered_df)
            
            # 最適な配分の提案（コスト効率に基づく）
            # 単純化のため、コスト効率の逆数に比例した配分を提案
            efficiency = 1 / channel_cost['CostPerAcceptance']
            proposed_allocation = efficiency / efficiency.sum()
            
            # チャネル配分比較
            allocation_df = pd.DataFrame({
                'SourceChannel': channel_cost['SourceChannel'],
                '現在の配分': [current_allocation.get(channel, 0) for channel in channel_cost['SourceChannel']],
                '推奨配分': proposed_allocation.values,
                'コスト効率': channel_cost['CostPerAcceptance'].values
            })
            
            # 可視化
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=allocation_df['SourceChannel'],
                y=allocation_df['現在の配分'],
                name='現在の配分',
                marker_color='royalblue'
            ))
            
            fig.add_trace(go.Bar(
                x=allocation_df['SourceChannel'],
                y=allocation_df['推奨配分'],
                name='推奨配分',
                marker_color='green'
            ))
            
            fig.update_layout(
                title="採用チャネル配分の最適化",
                barmode='group',
                yaxis=dict(title='配分率'),
                xaxis=dict(title='採用チャネル')
            )
            
            # 最適化した図を表示
            display_optimized_chart(fig)
            
            # コスト削減効果の試算
            current_cost = total_cost
            
            # 推奨配分を適用した場合の想定コストを計算
            theoretical_cost = 0
            for i, row in allocation_df.iterrows():
                channel_hire_need = total_hires * row['推奨配分']
                theoretical_cost += channel_hire_need * row['コスト効率']
            
            cost_diff = current_cost - theoretical_cost
            cost_diff_pct = (cost_diff / current_cost) * 100
            
            st.success(f"チャネル配分の最適化により、約 ¥{cost_diff:,.0f} ({cost_diff_pct:.1f}%) のコスト削減が見込まれます")
            
        elif optimization_target == "採用プロセスの効率化":
            # 採用プロセスの各段階の効率を分析
            process_time = {
                '応募者スクリーニング': 2,  # 日数（仮定）
                '一次面接調整': 3,
                '一次面接実施': 1,
                '二次面接調整': 4,
                '二次面接実施': 1,
                '内部評価・承認': 5,
                'オファー準備': 2,
                'オファー交渉': 3
            }
            
            process_df = pd.DataFrame({
                'Process': list(process_time.keys()),
                'Days': list(process_time.values())
            })
            
            # 現在の総日数
            current_days = sum(process_time.values())
            
            # 可視化
            fig = px.bar(
                process_df,
                x='Process',
                y='Days',
                title="採用プロセスの所要日数",
                color='Days',
                color_continuous_scale='Blues',
                text_auto='.0f'
            )
            
            # 最適化した図を表示
            display_optimized_chart(fig)
            
            # 効率化目標
            process_improvement = st.slider(
                "プロセス効率化による日数削減目標 (%)",
                min_value=10,
                max_value=40,
                value=20,
                step=5
            )
            
            # 効率化後の日数
            improved_days = current_days * (1 - process_improvement / 100)
            days_saved = current_days - improved_days
            
            st.success(f"プロセス効率化により、採用所要日数が {current_days:.0f}日 から {improved_days:.0f}日 に短縮可能です（{days_saved:.0f}日の短縮）")
            
            # 効率化による副次的効果
            st.write("**効率化による副次的効果:**")
            
            secondary_benefits = [
                f"候補者体験の向上による内定承諾率の上昇（推定 +{process_improvement/2:.0f}%）",
                f"採用担当者の工数削減による人件費の削減（推定 ¥{process_improvement*10000:.0f}/採用）",
                f"採用期間短縮による機会損失の削減（推定 ¥{days_saved*20000:.0f}/採用）",
                "採用プロセスの標準化による品質向上",
                "優秀な候補者の獲得確率の向上"
            ]
            
            for benefit in secondary_benefits:
                st.write(f"- {benefit}")
    
    with tab3:
        st.header("採用ソース分析")
        
        # PDF出力時のページ区切り
        add_page_break()
        
        # 採用ソース別の効率性
        st.subheader("採用ソース別の効率性")
        
        source_metrics = filtered_df.groupby('SourceChannel').agg({
            'Applicants': 'sum',
            'FirstInterview': 'sum',
            'Offers': 'sum',
            'Acceptances': 'sum',
            'CostPerHire': 'sum'
        }).reset_index()
        
        source_metrics['ApplicantToHire'] = source_metrics['Applicants'] / source_metrics['Acceptances']
        source_metrics['OfferAcceptanceRate'] = source_metrics['Acceptances'] / source_metrics['Offers']
        source_metrics['CostPerHire'] = source_metrics['CostPerHire'] / source_metrics['Acceptances']
        
        # ソース効率のレーダーチャート
        source_radar = source_metrics[['SourceChannel', 'ApplicantToHire', 'OfferAcceptanceRate', 'CostPerHire']]
        
        # スケーリングを行い、すべての指標で高いほど良くなるように変換
        max_cost = source_radar['CostPerHire'].max()
        source_radar['CostEfficiency'] = max_cost / source_radar['CostPerHire']
        
        max_app_to_hire = source_radar['ApplicantToHire'].max()
        source_radar['ApplicationEfficiency'] = max_app_to_hire / source_radar['ApplicantToHire']
        
        # レーダーチャート用のデータ準備
        fig = go.Figure()
        
        categories = ['応募効率', 'オファー承諾率', 'コスト効率']
        
        for i, source in enumerate(source_radar['SourceChannel']):
            fig.add_trace(go.Scatterpolar(
                r=[
                    source_radar.iloc[i]['ApplicationEfficiency'],
                    source_radar.iloc[i]['OfferAcceptanceRate'],
                    source_radar.iloc[i]['CostEfficiency']
                ],
                theta=categories,
                fill='toself',
                name=source
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            title="採用ソース効率性レーダーチャート"
        )
        
        # 最適化した図を表示
        display_optimized_chart(fig)
        
        # ソース別の主要指標表
        st.subheader("採用ソース別の主要指標")
        
        source_display = source_metrics[['SourceChannel', 'Applicants', 'Acceptances', 'OfferAcceptanceRate', 'CostPerHire']]
        source_display = source_display.rename(columns={
            'SourceChannel': '採用ソース',
            'Applicants': '応募者数',
            'Acceptances': '採用数',
            'OfferAcceptanceRate': 'オファー承諾率',
            'CostPerHire': '採用単価'
        })
        
        # 表示用に書式を整える
        source_display['オファー承諾率'] = source_display['オファー承諾率'].map('{:.1%}'.format)
        source_display['採用単価'] = source_display['採用単価'].map('¥{:,.0f}'.format)
        
        # 最適化したデータフレーム表示
        format_dataframe_for_display(source_display)
        
        # 職種別の効果的な採用ソース
        st.subheader("職種別の効果的な採用ソース")
        
        # 職種とソースのクロス集計
        role_source = filtered_df.groupby(['JobRole', 'SourceChannel']).agg({
            'Acceptances': 'sum'
        }).reset_index()
        
        # 職種ごとの合計採用数を計算
        role_totals = role_source.groupby('JobRole')['Acceptances'].sum().reset_index()
        role_totals = role_totals.rename(columns={'Acceptances': 'TotalAcceptances'})
        
        # 合計を結合
        role_source = pd.merge(role_source, role_totals, on='JobRole')
        
        # 比率を計算
        role_source['Percentage'] = role_source['Acceptances'] / role_source['TotalAcceptances']
        
        # 各職種で最も効果的なソースを特定
        best_sources = role_source.loc[role_source.groupby('JobRole')['Acceptances'].idxmax()]
        
        # ヒートマップ用のピボットテーブル
        pivot_role_source = role_source.pivot_table(
            index='JobRole',
            columns='SourceChannel',
            values='Percentage',
            aggfunc='sum'
        ).fillna(0)
        
        # ヒートマップの描画
        fig = px.imshow(
            pivot_role_source,
            text_auto='.0%',
            aspect="auto",
            color_continuous_scale='Blues',
            title="職種別の採用ソース効果ヒートマップ"
        )
        
        # 最適化した図を表示
        display_optimized_chart(fig)
        
        # PDF出力時のページ区切り
        add_page_break()
        
        # 提案：職種別の推奨採用ソース
        st.subheader("職種別の推奨採用戦略")
        
        selected_role = st.selectbox(
            "職種を選択",
            options=filtered_df['JobRole'].unique()
        )
        
        # 選択した職種のソース効果
        role_specific = role_source[role_source['JobRole'] == selected_role]
        role_specific = role_specific.sort_values('Acceptances', ascending=False)
        
        fig = px.pie(
            role_specific,
            names='SourceChannel',
            values='Acceptances',
            title=f"{selected_role}の採用ソース分布",
            hole=0.4
        )
        
        # 最適化した図を表示
        display_optimized_chart(fig)
        
        # 選択した職種に対する推奨戦略
        # 実際のプロジェクトでは、データに基づいた精緻な推奨が必要
        recommendations = {
            'Sales Representative': {
                'best_sources': ['リファラル', 'SNS'],
                'strategies': [
                    "営業チームのネットワークを活用したリファラル採用の強化",
                    "LinkedIn等のプロフェッショナルSNSでのターゲティング広告",
                    "業界イベントやセミナーでの採用活動",
                    "成功報酬型の採用エージェントの活用"
                ]
            },
            'Research Scientist': {
                'best_sources': ['転職エージェント', '自社ウェブサイト'],
                'strategies': [
                    "専門分野の学会・カンファレンスでのリクルーティング",
                    "大学や研究機関との連携",
                    "科学系オンラインコミュニティでの企業技術ブログ発信",
                    "研究開発環境のPRを強化した採用ページの作成"
                ]
            },
            'Human Resources': {
                'best_sources': ['求人サイト', 'リファラル'],
                'strategies': [
                    "HR専門の転職サイトへの求人掲載",
                    "HR系プロフェッショナルコミュニティへの参加",
                    "HR認定資格を持つ人材へのターゲティング",
                    "社内イベント情報のSNS発信によるブランディング強化"
                ]
            }
        }
        
        # 他の職種に対するデフォルト推奨
        default_recommendations = {
            'best_sources': ['求人サイト', 'リファラル'],
            'strategies': [
                "効果的な求人内容の最適化（職務内容、必要スキル、企業文化の明確な提示）",
                "従業員リファラルプログラムの報酬見直しと促進キャンペーン",
                "採用ブランディングの強化（企業文化や成長機会のアピール）",
                "ターゲット層が利用するオンラインコミュニティへの参加"
            ]
        }
        
        role_rec = recommendations.get(selected_role, default_recommendations)
        
        st.write(f"**{selected_role}に効果的な採用ソース:** {', '.join(role_rec['best_sources'])}")
        
        st.write("**推奨採用戦略:**")
        for strategy in role_rec['strategies']:
            st.write(f"- {strategy}")
        
        # 採用トレンド分析
        st.subheader("採用ソーストレンド分析")
        
        source_trend = filtered_df.groupby(['Month', 'SourceChannel']).agg({
            'Acceptances': 'sum'
        }).reset_index()
        
        fig = px.line(
            source_trend,
            x='Month',
            y='Acceptances',
            color='SourceChannel',
            title="月別の採用ソース効果推移",
            markers=True
        )
        
        # 最適化した図を表示
        display_optimized_chart(fig)
        
        # PDF出力時のページ区切り
        add_page_break()
        
        # 地域別・採用ソース分析（ダミーデータ）
        st.subheader("地域別の効果的な採用ソース")
        
        # 地域情報をランダムに生成
        np.random.seed(42)
        regions = ['東京', '大阪', '名古屋', '福岡', '札幌', '仙台']
        region_data = []
        
        for _, row in filtered_df.iterrows():
            region = np.random.choice(regions)
            region_data.append({
                'Month': row['Month'],
                'Department': row['Department'],
                'JobRole': row['JobRole'],
                'SourceChannel': row['SourceChannel'],
                'Region': region,
                'Acceptances': row['Acceptances']
            })
        
        region_df = pd.DataFrame(region_data)
        
        # 地域別の採用ソース効果
        region_source = region_df.groupby(['Region', 'SourceChannel']).agg({
            'Acceptances': 'sum'
        }).reset_index()
        
        # 地域ごとの合計採用数を計算
        region_totals = region_source.groupby('Region')['Acceptances'].sum().reset_index()
        region_totals = region_totals.rename(columns={'Acceptances': 'TotalAcceptances'})
        
        # 合計を結合
        region_source = pd.merge(region_source, region_totals, on='Region')
        
        # 比率を計算
        region_source['Percentage'] = region_source['Acceptances'] / region_source['TotalAcceptances']
        
        # ヒートマップ用のピボットテーブル
        pivot_region_source = region_source.pivot_table(
            index='Region',
            columns='SourceChannel',
            values='Percentage',
            aggfunc='sum'
        ).fillna(0)
        
        # ヒートマップの描画
        fig = px.imshow(
            pivot_region_source,
            text_auto='.0%',
            aspect="auto",
            color_continuous_scale='Blues',
            title="地域別の採用ソース効果ヒートマップ"
        )
        
        # 最適化した図を表示
        display_optimized_chart(fig)
    
    # フッター
    st.markdown("---")
    st.info("このページでは、採用プロセスの効率性、コスト、および効果的な採用ソースに関する分析を提供しています。"
            "より効果的な採用戦略の策定にご活用ください。")