import sys
import os

# プロジェクトルートをPythonパスに追加
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
# 以下は元のコードをそのまま使用
from src.data.loader import load_hr_data
from src.pages import attrition, recruitment, performance, compensation, engagement, productivity, simulation, home

# アプリケーションのタイトル設定
st.set_page_config(
    page_title="HR Analytics Dashboard",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# レスポンシブデザインとPDF出力用のCSS調整
st.markdown("""
<style>
    /* レスポンシブ対応のためのCSS */
    .reportview-container .main .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 2rem;
        margin: 0 auto;
    }
    
    /* テーブルのレスポンシブ対応 */
    .dataframe-container {
        overflow-x: auto !important;
    }
    
    /* グラフのサイズ制限（PDF出力用） */
    .stPlotlyChart, .stChart {
        width: 100%;
        max-height: 500px; /* PDF出力時にページをはみ出さないよう高さ制限 */
    }
    
    /* モバイル対応 */
    @media screen and (max-width: 640px) {
        .row-widget.stButton > button {
            width: 100%;
        }
        
        /* モバイルでのフォントサイズ調整 */
        h1 {
            font-size: 1.8rem !important;
        }
        h2 {
            font-size: 1.5rem !important;
        }
        h3 {
            font-size: 1.2rem !important;
        }
        
        /* モバイルでのグラフ高さ調整 */
        .stPlotlyChart, .stChart {
            max-height: 300px;
        }
    }
    
    /* PDF出力用の余白調整 */
    @media print {
        .main .block-container {
            padding: 0.5rem !important;
        }
        
        h1, h2, h3 {
            margin-top: 0.5rem !important;
            margin-bottom: 0.5rem !important;
        }
        
        /* PDF出力時のページ区切り考慮 */
        .pdf-page-break {
            page-break-after: always;
        }
    }
</style>
""", unsafe_allow_html=True)

# サイドバーにフィルター追加
st.sidebar.title("HR Analytics")
st.sidebar.header("Navigation")

# ページ選択
page_options = {
    "🏠 ホーム・人材分析概要": "home",
    "🚪 離職分析": "attrition",
    "🎯 人材獲得分析": "recruitment",
    "📈 人材育成・パフォーマンス分析": "performance",
    "💰 給与・報酬分析": "compensation",
    "🌟 従業員エンゲージメント": "engagement",
    "⏱️ 労働生産性/ワークスタイル分析": "productivity",
    "🔮 予測分析・シミュレーション": "simulation"
}

# PDF出力機能の追加
st.sidebar.markdown("---")
st.sidebar.subheader("レポート出力")
pdf_filename = st.sidebar.text_input("ファイル名", "hr_analytics_report")
if st.sidebar.button("PDF出力"):
    st.sidebar.info("PDF出力準備中... ブラウザの印刷機能を使用してください。")
    st.sidebar.markdown("""
    **PDF出力手順:**
    1. ブラウザのPDF印刷機能を使用（Ctrl+P または ⌘+P）
    2. 用紙サイズはA4縦向き推奨
    3. 余白を「最小」に設定
    4. 背景のグラフィックを有効化
    """)

selected_page = st.sidebar.radio("ページ選択", list(page_options.keys()))

# データをロード
df = load_hr_data()

# フッター情報
st.sidebar.markdown("---")
st.sidebar.info(
    "IBM HR Analytics Dataset\n\n"
    "This dashboard uses the IBM HR Analytics Employee Attrition & Performance dataset from Kaggle."
)

# ユーティリティ関数（ページ区切り用）
def add_page_break():
    """PDF出力時のページ区切りを挿入するヘルパー関数"""
    st.markdown('<div class="pdf-page-break"></div>', unsafe_allow_html=True)

# 選択されたページを表示
if selected_page == "🏠 ホーム・人材分析概要":
    if hasattr(home, 'show') and callable(home.show):
        home.show()
    else:
        # ホームページが実装されていない場合は基本情報を表示
        st.title("IBM HR Analytics Dashboard")
        st.markdown("### 従業員の離職と業績分析")

        # データの概要を表示
        st.subheader("データセット概要")
        st.write(f"総レコード数: {df.shape[0]}")
        st.write(f"特徴量数: {df.shape[1]}")

        # 基本的なデータプレビュー
        st.subheader("データプレビュー")
        st.dataframe(df.head())

        # 離職率の基本的な統計
        col1, col2, col3 = st.columns(3)
        with col1:
            attrition_rate = df['Attrition'].value_counts(normalize=True)['Yes'] * 100
            st.metric("全体離職率", f"{attrition_rate:.2f}%")
        with col2:
            avg_age = df['Age'].mean()
            st.metric("平均年齢", f"{avg_age:.1f}歳")
        with col3:
            avg_tenure = df['YearsAtCompany'].mean()
            st.metric("平均勤続年数", f"{avg_tenure:.1f}年")

        # 部門別離職率
        st.subheader("部門別離職率")
        dept_attrition = df.groupby('Department')['Attrition'].apply(
            lambda x: (x == 'Yes').mean() * 100
        ).reset_index()
        dept_attrition.columns = ['Department', 'Attrition Rate (%)']
        st.bar_chart(dept_attrition.set_index('Department'))

elif selected_page == "🚪 離職分析":
    attrition.show()
elif selected_page == "🎯 人材獲得分析":
    if hasattr(recruitment, 'show') and callable(recruitment.show):
        recruitment.show()
    else:
        st.title("人材獲得分析")
        st.info("このページは現在開発中です。近日公開予定です。")
elif selected_page == "📈 人材育成・パフォーマンス分析":
    if hasattr(performance, 'show') and callable(performance.show):
        performance.show()
    else:
        st.title("人材育成・パフォーマンス分析")
        st.info("このページは現在開発中です。近日公開予定です。")
elif selected_page == "💰 給与・報酬分析":
    if hasattr(compensation, 'show') and callable(compensation.show):
        compensation.show()
    else:
        st.title("給与・報酬分析")
        st.info("このページは現在開発中です。近日公開予定です。")
elif selected_page == "🌟 従業員エンゲージメント":
    if hasattr(engagement, 'show') and callable(engagement.show):
        engagement.show()
    else:
        st.title("従業員エンゲージメント")
        st.info("このページは現在開発中です。近日公開予定です。")
elif selected_page == "⏱️ 労働生産性/ワークスタイル分析":
    if hasattr(productivity, 'show') and callable(productivity.show):
        productivity.show()
    else:
        st.title("労働生産性/ワークスタイル分析")
        st.info("このページは現在開発中です。近日公開予定です。")
elif selected_page == "🔮 予測分析・シミュレーション":
    if hasattr(simulation, 'show') and callable(simulation.show):
        simulation.show()
    else:
        st.title("予測分析・シミュレーション")
        st.info("このページは現在開発中です。近日公開予定です。")
