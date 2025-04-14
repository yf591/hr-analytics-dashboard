import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from src.data.loader import load_hr_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
import lightgbm as lgb
from src.models.attrition import MODELS

def show():
    """
    離職分析ページを表示する関数
    """
    st.title("離職分析")
    st.write("従業員の離職要因と離職予測モデルの詳細分析")
    
    # データロード
    df = load_hr_data()
    
    # 離職率の概要
    st.header("離職概要")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # 離職数の内訳
        attrition_counts = df['Attrition'].value_counts()
        st.subheader("離職の内訳")
        fig = px.pie(values=attrition_counts.values, 
                     names=attrition_counts.index, 
                     title="在籍状況の内訳",
                     color_discrete_sequence=['#66b3ff', '#ff9999'],
                     hole=0.4)
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig)
    
    with col2:
        # 部門別の離職率
        st.subheader("部門別の離職率")
        dept_attrition = df.groupby('Department')['Attrition'].apply(
            lambda x: (x == 'Yes').mean() * 100
        ).reset_index()
        dept_attrition.columns = ['Department', 'Attrition Rate (%)']
        
        fig = px.bar(dept_attrition, x='Department', y='Attrition Rate (%)',
                    title='部門別離職率',
                    color='Attrition Rate (%)',
                    color_continuous_scale='Reds')
        st.plotly_chart(fig)
    
    # タブで詳細分析を整理
    tab1, tab2, tab3 = st.tabs(["📊 離職要因分析", "🔍 離職パターン", "🤖 離職予測モデル"])
    
    with tab1:
        st.header("離職要因分析")
        
        # 重要な要因を選択
        factors = st.multiselect(
            "分析したい要因を選択してください",
            options=[
                'Age', 'Gender', 'MaritalStatus', 'Education', 'JobLevel', 
                'JobRole', 'OverTime', 'BusinessTravel', 'YearsAtCompany', 
                'YearsSinceLastPromotion', 'WorkLifeBalance', 'JobSatisfaction'
            ],
            default=['JobSatisfaction', 'OverTime', 'YearsAtCompany', 'WorkLifeBalance']
        )
        
        if factors:
            cols = st.columns(min(3, len(factors)))
            
            for i, factor in enumerate(factors):
                with cols[i % len(cols)]:
                    st.subheader(f"{factor}と離職の関係")
                    
                    if df[factor].dtype == 'object':
                        # カテゴリ変数の場合
                        cat_attrition = df.groupby(factor)['Attrition'].apply(
                            lambda x: (x == 'Yes').mean() * 100
                        ).reset_index()
                        cat_attrition.columns = [factor, 'Attrition Rate (%)']
                        
                        fig = px.bar(cat_attrition, x=factor, y='Attrition Rate (%)',
                                    color='Attrition Rate (%)',
                                    color_continuous_scale='Reds')
                    else:
                        # 数値変数の場合
                        if len(df[factor].unique()) <= 10:
                            # 少数のユニークな値（例：評価スコア）
                            cat_attrition = df.groupby(factor)['Attrition'].apply(
                                lambda x: (x == 'Yes').mean() * 100
                            ).reset_index()
                            cat_attrition.columns = [factor, 'Attrition Rate (%)']
                            
                            fig = px.bar(cat_attrition, x=factor, y='Attrition Rate (%)',
                                        color='Attrition Rate (%)',
                                        color_continuous_scale='Reds')
                        else:
                            # 連続変数の場合、ヒストグラムを表示
                            fig = px.histogram(df, x=factor, color='Attrition',
                                              barmode='group',
                                              color_discrete_map={'Yes': '#ff9999', 'No': '#66b3ff'})
                    
                    st.plotly_chart(fig, use_container_width=True)
        
        # 相関分析（数値変数間）
        st.subheader("数値特徴量間の相関分析")
        
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        corr_matrix = df[numeric_cols].corr()
        
        fig = px.imshow(corr_matrix, 
                       color_continuous_scale='RdBu_r',
                       zmin=-1, zmax=1,
                       text_auto='.2f')
        
        # 相関ヒートマップのサイズを2倍に拡大
        fig.update_layout(
            width=1000,  # 幅を拡大（デフォルトより約2倍）
            height=900,  # 高さを拡大（デフォルトより約2倍）
            font=dict(size=14)  # フォントサイズも大きくする
        )
        
        # 数値のフォントサイズも調整
        fig.update_traces(
            textfont=dict(size=12)  # セル内の数値のフォントサイズを調整
        )
        
        st.plotly_chart(fig, use_container_width=False)  # use_container_widthをFalseに変更して指定サイズを維持
    
    with tab2:
        st.header("離職パターン分析")
        
        # 年齢と勤続年数による離職パターン
        st.subheader("年齢と勤続年数による離職パターン")
        
        fig = px.scatter(df, x='Age', y='YearsAtCompany', 
                        color='Attrition', 
                        color_discrete_map={'Yes': '#ff9999', 'No': '#66b3ff'},
                        size='MonthlyIncome',
                        hover_data=['JobRole', 'Department', 'JobSatisfaction'])
        st.plotly_chart(fig, use_container_width=True)
        
        # 満足度要因と離職の関係
        st.subheader("満足度要因と離職の関係")
        
        satisfaction_cols = ['JobSatisfaction', 'EnvironmentSatisfaction', 
                           'WorkLifeBalance', 'RelationshipSatisfaction']
        
        satisfaction_data = df.melt(
            id_vars=['EmployeeNumber', 'Attrition'],
            value_vars=satisfaction_cols,
            var_name='Satisfaction Type',
            value_name='Satisfaction Level'
        )
        
        fig = px.box(satisfaction_data, x='Satisfaction Type', y='Satisfaction Level', 
                    color='Attrition',
                    color_discrete_map={'Yes': '#ff9999', 'No': '#66b3ff'},
                    notched=True)
        st.plotly_chart(fig, use_container_width=True)
        
        # 離職コホート分析（勤続年数別）
        st.subheader("勤続年数帯別の離職率")
        
        df['TenureGroup'] = pd.cut(df['YearsAtCompany'], 
                                  bins=[0, 2, 5, 10, 15, 40],
                                  labels=['0-2年', '3-5年', '6-10年', '11-15年', '16年以上'])
        
        tenure_attrition = df.groupby('TenureGroup')['Attrition'].apply(
            lambda x: (x == 'Yes').mean() * 100
        ).reset_index()
        tenure_attrition.columns = ['勤続年数', '離職率 (%)']
        
        fig = px.line(tenure_attrition, x='勤続年数', y='離職率 (%)', 
                     markers=True, line_shape='spline',
                     color_discrete_sequence=['#ff9999'])
        fig.update_traces(marker_size=12)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.header("離職予測モデル")
        
        # モデル選択UI
        model_type = st.selectbox(
            "分析モデルを選択してください",
            options=list(MODELS.keys()),  # MODELSから選択肢を動的に生成
            index=0,  # デフォルトはランダムフォレスト
            help="異なる機械学習モデルを選択して精度を比較できます"
        )
        
        # モデル構築のためのデータ準備
        model_ready = st.checkbox("離職予測モデルを構築する", value=False)
        
        if model_ready:
            st.write(f"選択されたモデル: **{model_type}**")
            st.write("機械学習モデルを構築中です...")
            
            # 特徴量とターゲットの準備
            X = df.drop(['Attrition', 'EmployeeNumber', 'EmployeeCount', 'StandardHours'], axis=1)
            y = (df['Attrition'] == 'Yes').astype(int)
            
            # 特徴量の前処理
            categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
            numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
            
            # 前処理パイプラインを修正
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), numeric_cols),
                    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
                ],
                verbose_feature_names_out=False  # 特徴量名の衝突を防ぐ
            )
            
            # モデルパイプライン - 選択されたモデルを使用
            model = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('classifier', MODELS[model_type])  # 選択されたモデルを使用
            ])
            
            # トレーニングとテストの分割
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            
            # モデルのトレーニング
            with st.spinner(f"{model_type}モデルを学習中..."):
                model.fit(X_train, y_train)
            
            # 予測
            y_pred = model.predict(X_test)
            
            # モデル評価
            accuracy = accuracy_score(y_test, y_pred)
            
            st.success(f"{model_type}モデルの精度: {accuracy:.2f}")
            
            # 混同行列 - Plotlyを使用して日本語対応
            st.subheader("混同行列")
            
            cm = confusion_matrix(y_test, y_pred)
            
            # Plotlyで混同行列を作成（matplotlib/seabornの代わりに）
            labels = ['在籍', '離職']
            fig = px.imshow(
                cm,
                x=['予測: 在籍', '予測: 離職'],
                y=['実際: 在籍', '実際: 離職'],
                color_continuous_scale='Blues',
                text_auto=True
            )
            # 混同行列の大きさを調整 - 縦横ともに大きくする
            fig.update_layout(
                xaxis=dict(title='予測クラス'),
                yaxis=dict(title='実際のクラス'),
                width=600,  # 幅を拡大（デフォルトは約400px）
                height=600, # 高さを拡大（デフォルトは約400px）
                font=dict(size=16) # フォントサイズも大きくする
            )
            # 数値の表示サイズも大きくする
            fig.update_traces(
                text=cm,
                texttemplate='%{text}',
                textfont=dict(size=24)  # セル内の数値のフォントサイズを大きくする
            )
            st.plotly_chart(fig)
            
            # 重要な特徴量の抽出方法を修正
            try:
                st.subheader("特徴量の重要度")
                
                # モデルタイプに応じた特徴量重要度の抽出
                if hasattr(model['classifier'], 'feature_importances_'):
                    # ツリーベースのモデル（ランダムフォレスト、XGBoost、LightGBM、勾配ブースティングなど）
                    importances = model['classifier'].feature_importances_
                elif hasattr(model['classifier'], 'coef_'):
                    # 線形モデル（ロジスティック回帰など）
                    importances = np.abs(model['classifier'].coef_[0])
                else:
                    st.warning(f"このモデル（{model_type}）の特徴量重要度を抽出する方法がありません。")
                    importances = None
                
                if importances is not None:
                    # 変換された特徴量名を取得
                    if hasattr(model['preprocessor'], 'get_feature_names_out'):
                        feature_names = model['preprocessor'].get_feature_names_out()
                    else:
                        # 古いバージョンのscikit-learnに対応
                        feature_names = []
                        for name, trans, cols in model['preprocessor'].transformers_:
                            if name == 'cat' and hasattr(trans, 'get_feature_names_out'):
                                cat_features = trans.get_feature_names_out(cols)
                                feature_names.extend(cat_features)
                            else:
                                feature_names.extend(cols)
                    
                    # デバッグ情報
                    if len(importances) != len(feature_names):
                        st.write(f"特徴量名の数: {len(feature_names)}, 重要度の数: {len(importances)}")
                        # 長さが合わない場合は単純なインデックスを使用
                        feature_names = [f"特徴量 {i}" for i in range(len(importances))]
                    
                    # 特徴量の重要度をデータフレームにまとめる
                    feature_importance = pd.DataFrame({
                        'Feature': feature_names,
                        'Importance': importances
                    }).sort_values('Importance', ascending=False).head(15)
                    
                    fig = px.bar(
                        feature_importance, 
                        x='Importance', 
                        y='Feature',
                        orientation='h',
                        color='Importance',
                        color_continuous_scale='Viridis'
                    )
                    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                    st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"特徴量の重要度を表示できませんでした: {e}")
                
                # エラーログの詳細表示（デバッグ用）
                import traceback
                st.write(traceback.format_exc())
                
                # 可能であればフォールバック表示
                try:
                    if hasattr(model['classifier'], 'feature_importances_'):
                        feature_indices = np.argsort(model['classifier'].feature_importances_)[::-1][:15]
                        top_importances = model['classifier'].feature_importances_[feature_indices]
                        
                        fallback_df = pd.DataFrame({
                            'Feature': [f"特徴量 {i}" for i in feature_indices],
                            'Importance': top_importances
                        })
                        
                        fig = px.bar(
                            fallback_df, 
                            x='Importance', 
                            y='Feature',
                            orientation='h',
                            color='Importance',
                            color_continuous_scale='Viridis'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                except:
                    st.warning("特徴量重要度の表示に失敗しました。別のモデルを試してみてください。")
            
            # 離職リスクスコアの分布
            if hasattr(model['classifier'], 'predict_proba'):
                st.subheader("離職リスクスコアの分布")
                
                try:
                    # 元のX_testを使用してリスクスコアを計算
                    risk_scores = model.predict_proba(X_test)[:, 1]
                    
                    fig = px.histogram(risk_scores, nbins=20,
                                     labels={'value': '離職リスクスコア', 'count': '従業員数'},
                                     title=f'{model_type}による離職リスクスコアの分布',
                                     color_discrete_sequence=['#ff9999'])
                    fig.update_layout(bargap=0.1)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # リスクスコアのしきい値選択
                    threshold = st.slider("リスクスコアのしきい値", 0.0, 1.0, 0.5, 0.05)
                    high_risk_count = sum(risk_scores >= threshold)
                    
                    st.metric("高リスク従業員数", f"{high_risk_count}人", 
                             delta=f"{high_risk_count/len(risk_scores)*100:.1f}% of test set")
                
                except Exception as e:
                    st.error(f"離職リスクスコアの計算中にエラーが発生しました: {e}")
                    st.write("代替方法でリスクスコアを計算します...")
                    
                    # 特徴量の不一致を回避するための代替方法
                    # トレーニングデータの一部を使用してリスクスコアを表示
                    sample_idx = np.random.choice(len(X_train), min(100, len(X_train)), replace=False)
                    X_sample = X_train.iloc[sample_idx]
                    sample_scores = model.predict_proba(X_sample)[:, 1]
                    
                    fig = px.histogram(sample_scores, nbins=20,
                                    labels={'value': '離職リスクスコア (サンプル)', 'count': '従業員数'},
                                    title=f'{model_type}による離職リスクスコアの分布 (トレーニングデータからのサンプル)',
                                    color_discrete_sequence=['#ff9999'])
                    fig.update_layout(bargap=0.1)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.info("注意: 特徴量の不一致により、テストデータでのリスクスコア計算ができませんでした。"\
                           "上のグラフはトレーニングデータのサンプルに基づいています。")
            else:
                st.warning(f"選択されたモデル（{model_type}）は確率予測をサポートしていないため、リスクスコアを表示できません。")
    
    # フッター
    st.markdown("---")
    st.info("このページでは、従業員の離職要因の分析と離職予測モデルを提供しています。"
            "データに基づいた施策立案にご活用ください。")
