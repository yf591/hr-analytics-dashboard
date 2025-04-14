import streamlit as st
import pandas as pd
import os

# 自作モジュールとページをインポート
from src.data.loader import load_hr_data
from src.pages import attrition, recruitment, performance, compensation, engagement, productivity, simulation, home

# アプリケーションのタイトル設定
st.set_page_config(
    page_title="HR Analytics Dashboard",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

selected_page = st.sidebar.radio("ページ選択", list(page_options.keys()))

# データをロード
df = load_hr_data()

# フッター情報
st.sidebar.markdown("---")
st.sidebar.info(
    "IBM HR Analytics Dataset\n\n"
    "This dashboard uses the IBM HR Analytics Employee Attrition & Performance dataset from Kaggle."
)

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
