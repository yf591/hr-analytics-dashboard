import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from src.data.loader import load_hr_data
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, r2_score, mean_absolute_error
import xgboost as xgb
import lightgbm as lgb
from src.models.attrition import MODELS

def show():
    """
    予測分析・シミュレーションページを表示する関数
    """
    st.title("予測分析・シミュレーション")
    st.write("機械学習モデルによる将来予測と人事施策のシミュレーション")
    
    # データロード
    df = load_hr_data()
    
    # タブで分析内容を整理
    tab1, tab2, tab3 = st.tabs(["🔮 離職予測", "💰 給与予測", "👥 人員計画シミュレーション"])
    
    with tab1:
        st.header("離職予測モデル")
        
        # モデルの構築と評価
        st.subheader("離職予測モデルの性能")
        
        # 予測に使用する特徴量
        features = [
            'Age', 'MonthlyIncome', 'DistanceFromHome', 'OverTime',
            'JobSatisfaction', 'WorkLifeBalance', 'YearsAtCompany',
            'YearsSinceLastPromotion', 'JobLevel', 'MaritalStatus',
            'NumCompaniesWorked', 'TrainingTimesLastYear', 'BusinessTravel'
        ]
        
        # 目的変数 (Attrition)
        target = 'Attrition'
        
        # データの準備
        X = df[features].copy()
        y = df[target].copy()
        
        # カテゴリ変数とカテゴリでない変数の分類
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        # モデル構築のための前処理パイプライン
        categorical_transformer = OneHotEncoder(handle_unknown='ignore')
        numeric_transformer = StandardScaler()
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ]
        )
        
        # モデルパイプライン
        attrition_pipe = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
        ])
        
        # データの分割
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # モデルの学習
        with st.spinner('モデルを学習中...'):
            attrition_pipe.fit(X_train, y_train)
        
        # テストデータでの予測
        y_pred = attrition_pipe.predict(X_test)
        
        # モデルの評価
        accuracy = accuracy_score(y_test, y_pred)
        
        # 結果表示
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("モデル精度", f"{accuracy:.2%}")
            
            # 混同行列
            cm = confusion_matrix(y_test, y_pred)
            cm_df = pd.DataFrame(
                cm, 
                index=['実際: 在職', '実際: 離職'], 
                columns=['予測: 在職', '予測: 離職']
            )
            
            st.write("混同行列:")
            st.dataframe(cm_df)
        
        with col2:
            # 特徴量重要度
            feature_names = (
                numeric_features + 
                list(attrition_pipe.named_steps['preprocessor'].transformers_[1][1].get_feature_names_out(categorical_features))
            )
            
            feature_importance = pd.DataFrame(
                attrition_pipe.named_steps['classifier'].feature_importances_,
                index=feature_names,
                columns=['importance']
            ).sort_values('importance', ascending=False)
            
            # 上位10個の特徴量のみ表示
            top_features = feature_importance.head(10)
            
            fig = px.bar(
                top_features,
                y=top_features.index,
                x='importance',
                orientation='h',
                title="特徴量重要度（上位10）",
                labels={'importance': '重要度', 'index': '特徴量'}
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # インタラクティブな離職予測
        st.subheader("インタラクティブな離職リスク予測")
        st.write("従業員の特性を調整して、離職リスクの変化をシミュレーションできます。")
        
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.slider("年齢", min_value=18, max_value=60, value=35)
            job_level = st.selectbox("役職レベル", options=[1, 2, 3, 4, 5], index=1)
            monthly_income = st.slider("月収", min_value=1000, max_value=20000, value=5000, step=500)
            distance = st.slider("通勤距離(km)", min_value=1, max_value=30, value=10)
            job_satisfaction = st.selectbox("職務満足度", options=[1, 2, 3, 4], index=2)
            work_life_balance = st.selectbox("ワークライフバランス", options=[1, 2, 3, 4], index=2)
        
        with col2:
            years_at_company = st.slider("勤続年数", min_value=0, max_value=40, value=5)
            years_since_promotion = st.slider("前回昇進からの年数", min_value=0, max_value=15, value=2)
            num_companies = st.slider("過去の勤務企業数", min_value=0, max_value=9, value=2)
            training_times = st.slider("昨年の研修回数", min_value=0, max_value=6, value=2)
            overtime = st.selectbox("残業", options=["Yes", "No"], index=0)
            marital_status = st.selectbox("婚姻状況", options=["Single", "Married", "Divorced"], index=0)
            business_travel = st.selectbox("出張頻度", options=["Non-Travel", "Travel_Rarely", "Travel_Frequently"], index=1)
        
        # 予測用のデータフレーム作成
        prediction_data = pd.DataFrame({
            'Age': [age],
            'JobLevel': [job_level],
            'MonthlyIncome': [monthly_income],
            'DistanceFromHome': [distance],
            'JobSatisfaction': [job_satisfaction],
            'WorkLifeBalance': [work_life_balance],
            'YearsAtCompany': [years_at_company],
            'YearsSinceLastPromotion': [years_since_promotion],
            'NumCompaniesWorked': [num_companies],
            'TrainingTimesLastYear': [training_times],
            'OverTime': [overtime],
            'MaritalStatus': [marital_status],
            'BusinessTravel': [business_travel]
        })
        
        # 離職確率の予測
        probability = attrition_pipe.predict_proba(prediction_data)[0, 1]
        risk_level = "高" if probability > 0.7 else "中" if probability > 0.3 else "低"
        
        # 結果の表示
        st.subheader("離職リスク予測結果")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("離職確率", f"{probability:.1%}")
        
        with col2:
            st.metric("リスクレベル", risk_level)
        
        with col3:
            if risk_level == "高":
                st.error("早急な対応が必要です")
            elif risk_level == "中":
                st.warning("注意が必要です")
            else:
                st.success("リスクは低いです")
        
        # 「もし～だったら？」シミュレーション
        st.subheader("「もし～だったら？」シミュレーション")
        
        what_if_scenarios = st.multiselect(
            "シミュレーションしたい変更を選択してください",
            options=[
                "給与を20%アップ",
                "研修回数を2回増やす",
                "残業をなくす",
                "昇進させる",
                "ワークライフバランスを改善"
            ]
        )
        
        if what_if_scenarios:
            # 元のデータをコピー
            scenario_data = prediction_data.copy()
            
            for scenario in what_if_scenarios:
                if scenario == "給与を20%アップ":
                    scenario_data['MonthlyIncome'] = scenario_data['MonthlyIncome'] * 1.2
                elif scenario == "研修回数を2回増やす":
                    # .valuesを使って値を取得し、スカラー値として処理する
                    curr_training = scenario_data['TrainingTimesLastYear'].values[0]
                    scenario_data['TrainingTimesLastYear'] = min(6, curr_training + 2)
                elif scenario == "残業をなくす":
                    scenario_data['OverTime'] = "No"
                elif scenario == "昇進させる":
                    # .valuesを使って値を取得し、スカラー値として処理する
                    curr_level = scenario_data['JobLevel'].values[0]
                    scenario_data['JobLevel'] = min(5, curr_level + 1)
                    scenario_data['YearsSinceLastPromotion'] = 0
                elif scenario == "ワークライフバランスを改善":
                    # .valuesを使って値を取得し、スカラー値として処理する
                    curr_wlb = scenario_data['WorkLifeBalance'].values[0]
                    scenario_data['WorkLifeBalance'] = min(4, curr_wlb + 1)
            
            # 変更後の離職確率予測
            new_probability = attrition_pipe.predict_proba(scenario_data)[0, 1]
            probability_change = new_probability - probability
            
            # 結果表示
            st.write(f"選択した変更を適用した場合の離職確率: **{new_probability:.1%}**")
            st.write(f"離職確率の変化: **{probability_change:.1%}**")
            
            # ゲージチャートでのビジュアル比較
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=new_probability * 100,
                delta={"reference": probability * 100, "valueformat": ".1f"},
                gauge={"axis": {"range": [0, 100]},
                      "bar": {"color": "darkblue"},
                      "steps": [
                          {"range": [0, 30], "color": "green"},
                          {"range": [30, 70], "color": "yellow"},
                          {"range": [70, 100], "color": "red"}
                      ],
                      "threshold": {
                          "line": {"color": "red", "width": 4},
                          "thickness": 0.75,
                          "value": 70
                      }},
                title={"text": "離職確率 (%)"}
            ))
            
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.header("給与予測モデル")
        
        # 回帰モデルの辞書を定義
        REGRESSION_MODELS = {
            "ランダムフォレスト回帰": RandomForestRegressor(n_estimators=100, random_state=42),
            "線形回帰": LinearRegression(),
            "勾配ブースティング回帰": GradientBoostingRegressor(n_estimators=100, random_state=42),
            "XGBoost回帰": xgb.XGBRegressor(n_estimators=100, random_state=42),
            "LightGBM回帰": lgb.LGBMRegressor(n_estimators=100, random_state=42)
        }
        
        # モデル選択UI
        salary_model_type = st.selectbox(
            "給与予測モデルを選択してください",
            options=list(REGRESSION_MODELS.keys()),
            index=0,  # デフォルトはランダムフォレスト回帰
            help="異なる回帰モデルを選択して精度を比較できます"
        )
        
        # 給与予測モデルの構築
        st.subheader("給与予測モデルの性能")
        
        # 給与予測に使用する特徴量
        salary_features = [
            'JobLevel', 'Age', 'YearsAtCompany', 'TotalWorkingYears',
            'Department', 'JobRole', 'EducationField', 'Education',
            'PerformanceRating', 'Gender'
        ]
        
        # 目的変数
        salary_target = 'MonthlyIncome'
        
        # 欠損値のないレコードだけ使用
        salary_df = df[salary_features + [salary_target]].dropna()
        
        # データの分割
        X_salary = salary_df[salary_features]
        y_salary = salary_df[salary_target]
        
        # カテゴリ変数とカテゴリでない変数の分類
        salary_cat_features = X_salary.select_dtypes(include=['object', 'category']).columns.tolist()
        salary_num_features = X_salary.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        # モデル構築のための前処理パイプライン
        salary_preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), salary_num_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), salary_cat_features)
            ]
        )
        
        # モデルパイプライン - 選択されたモデルを使用
        salary_pipe = Pipeline(steps=[
            ('preprocessor', salary_preprocessor),
            ('regressor', REGRESSION_MODELS[salary_model_type])  # 選択されたモデルを使用
        ])
        
        # データの分割
        X_train_salary, X_test_salary, y_train_salary, y_test_salary = train_test_split(
            X_salary, y_salary, test_size=0.3, random_state=42
        )
        
        # モデルの学習
        with st.spinner(f'{salary_model_type}モデルを学習中...'):
            salary_pipe.fit(X_train_salary, y_train_salary)
        
        # テストデータでの予測
        y_pred_salary = salary_pipe.predict(X_test_salary)
        
        # モデルの評価（R^2スコアと平均絶対誤差）
        r2 = r2_score(y_test_salary, y_pred_salary)
        mae = mean_absolute_error(y_test_salary, y_pred_salary)
        
        # 結果表示
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"選択されたモデル: **{salary_model_type}**")
            st.metric("決定係数 (R²)", f"{r2:.2f}")
            st.metric("平均絶対誤差", f"${mae:.2f}")
        
        with col2:
            # 予測と実際の値の散布図
            comparison_df = pd.DataFrame({
                'Actual': y_test_salary,
                'Predicted': y_pred_salary
            })
            
            fig = px.scatter(
                comparison_df, x='Actual', y='Predicted',
                title="予測給与 vs 実際の給与",
                labels={'Actual': '実際の給与', 'Predicted': '予測給与'}
            )
            
            # 45度線（理想的な予測線）を追加
            fig.add_trace(
                go.Scatter(
                    x=[min(y_test_salary), max(y_test_salary)], 
                    y=[min(y_test_salary), max(y_test_salary)],
                    mode='lines',
                    line=dict(color='red', dash='dash'),
                    name='理想的な予測'
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # 特徴量重要度の表示（モデルがfeature_importances_属性を持つ場合）
        if hasattr(salary_pipe.named_steps['regressor'], 'feature_importances_'):
            st.subheader("給与への影響要因")
            
            # 特徴量名の取得
            if hasattr(salary_pipe.named_steps['preprocessor'], 'get_feature_names_out'):
                feature_names = salary_pipe.named_steps['preprocessor'].get_feature_names_out()
            else:
                # 古いバージョンのscikit-learnでは別の方法で取得
                feature_names = []
                for name, trans, cols in salary_pipe.named_steps['preprocessor'].transformers_:
                    if hasattr(trans, 'get_feature_names_out'):
                        names = trans.get_feature_names_out(cols)
                        feature_names.extend(names)
                    else:
                        feature_names.extend(cols)
            
            # 特徴量重要度の取得
            try:
                importances = salary_pipe.named_steps['regressor'].feature_importances_
                
                # 重要度データフレーム作成
                if len(importances) == len(feature_names):
                    feature_importance = pd.DataFrame({
                        'Feature': feature_names,
                        'Importance': importances
                    }).sort_values('Importance', ascending=False).head(15)
                    
                    fig = px.bar(
                        feature_importance, 
                        x='Importance', 
                        y='Feature',
                        orientation='h',
                        title="給与への影響要因（上位15）",
                        color='Importance',
                        color_continuous_scale='Viridis'
                    )
                    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("特徴量名と重要度の数が一致しないため、表示できません。")
            except Exception as e:
                st.warning(f"重要度の表示中にエラーが発生しました: {e}")
        elif hasattr(salary_pipe.named_steps['regressor'], 'coef_'):
            # 線形モデルの場合は係数を表示
            st.subheader("給与への影響要因（係数）")
            
            # 特徴量名の取得（上記と同様）
            if hasattr(salary_pipe.named_steps['preprocessor'], 'get_feature_names_out'):
                feature_names = salary_pipe.named_steps['preprocessor'].get_feature_names_out()
            else:
                feature_names = []
                for name, trans, cols in salary_pipe.named_steps['preprocessor'].transformers_:
                    if hasattr(trans, 'get_feature_names_out'):
                        names = trans.get_feature_names_out(cols)
                        feature_names.extend(names)
                    else:
                        feature_names.extend(cols)
            
            try:
                coefs = salary_pipe.named_steps['regressor'].coef_
                
                if len(coefs) == len(feature_names):
                    feature_coefs = pd.DataFrame({
                        'Feature': feature_names,
                        'Coefficient': coefs
                    }).sort_values('Coefficient', ascending=False).head(15)
                    
                    fig = px.bar(
                        feature_coefs, 
                        x='Coefficient', 
                        y='Feature',
                        orientation='h',
                        title="給与への影響要因（係数上位15）",
                        color='Coefficient',
                        color_continuous_scale='RdBu'
                    )
                    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("特徴量名と係数の数が一致しないため、表示できません。")
            except Exception as e:
                st.warning(f"係数の表示中にエラーが発生しました: {e}")
        
        # インタラクティブな給与予測
        st.subheader("給与シミュレーター")
        st.write("従業員の特性を入力して、予測される給与水準を確認できます。")
        
        col1, col2 = st.columns(2)
        
        with col1:
            sim_job_level = st.selectbox("役職レベル", options=[1, 2, 3, 4, 5], index=1, key="sim_job_level")
            sim_age = st.slider("年齢", min_value=18, max_value=60, value=35, key="sim_age")
            sim_years_company = st.slider("勤続年数", min_value=0, max_value=40, value=5, key="sim_years_company")
            sim_total_working = st.slider("総労働経験年数", min_value=0, max_value=40, value=10, key="sim_total_working")
            sim_performance = st.selectbox("業績評価", options=[1, 2, 3, 4], index=2, key="sim_performance")
        
        with col2:
            sim_department = st.selectbox("部門", options=df['Department'].unique(), key="sim_department")
            sim_job_role = st.selectbox("職種", options=df['JobRole'].unique(), key="sim_job_role")
            sim_education = st.selectbox("教育レベル", options=[1, 2, 3, 4, 5], index=2, key="sim_education")
            sim_education_field = st.selectbox("専門分野", options=df['EducationField'].unique(), key="sim_education_field")
            sim_gender = st.selectbox("性別", options=df['Gender'].unique(), key="sim_gender")
        
        # 予測用のデータフレーム作成
        salary_prediction_data = pd.DataFrame({
            'JobLevel': [sim_job_level],
            'Age': [sim_age],
            'YearsAtCompany': [sim_years_company],
            'TotalWorkingYears': [sim_total_working],
            'Department': [sim_department],
            'JobRole': [sim_job_role],
            'Education': [sim_education],
            'EducationField': [sim_education_field],
            'PerformanceRating': [sim_performance],
            'Gender': [sim_gender]
        })
        
        # 給与予測
        predicted_salary = salary_pipe.predict(salary_prediction_data)[0]
        
        # 同じ職種と役職の平均給与を取得
        peer_avg_salary = df[(df['JobRole'] == sim_job_role) & 
                            (df['JobLevel'] == sim_job_level)]['MonthlyIncome'].mean()
        
        # 結果の表示
        st.subheader("給与予測結果")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("予測される月収", f"${predicted_salary:,.2f}")
            st.metric("年収換算", f"${predicted_salary * 12:,.2f}")
        
        with col2:
            peer_comparison = predicted_salary - peer_avg_salary
            peer_pct = (peer_comparison / peer_avg_salary) * 100
            
            st.metric(
                "同じ職種・役職との比較",
                f"${predicted_salary - peer_avg_salary:,.2f}",
                f"{peer_pct:+.1f}%"
            )
            
            # 市場価値の評価
            if peer_pct > 10:
                st.success("市場平均より高い給与水準です")
            elif peer_pct < -10:
                st.warning("市場平均より低い給与水準です")
            else:
                st.info("市場平均に近い給与水準です")
        
        # 昇給シミュレーション
        st.subheader("昇給シミュレーション")
        
        sim_options = st.multiselect(
            "シミュレーションしたい変更を選択してください",
            options=[
                "1段階昇進",
                "業績評価を1ポイント上げる",
                "勤続年数が1年増える",
                "部門異動",
                "職種変更"
            ]
        )
        
        if sim_options:
            # 元のデータをコピー
            salary_scenario_data = salary_prediction_data.copy()
            scenario_description = []
            
            for option in sim_options:
                if option == "1段階昇進":
                    if salary_scenario_data['JobLevel'].values[0] < 5:
                        salary_scenario_data['JobLevel'] = salary_scenario_data['JobLevel'] + 1
                        scenario_description.append("役職レベルが上がる")
                
                elif option == "業績評価を1ポイント上げる":
                    if salary_scenario_data['PerformanceRating'].values[0] < 4:
                        salary_scenario_data['PerformanceRating'] = salary_scenario_data['PerformanceRating'] + 1
                        scenario_description.append("業績評価が向上する")
                
                elif option == "勤続年数が1年増える":
                    salary_scenario_data['YearsAtCompany'] = salary_scenario_data['YearsAtCompany'] + 1
                    salary_scenario_data['TotalWorkingYears'] = salary_scenario_data['TotalWorkingYears'] + 1
                    scenario_description.append("勤続年数が増える")
                
                elif option == "部門異動":
                    new_dept = st.selectbox(
                        "異動先の部門を選択",
                        options=[d for d in df['Department'].unique() if d != sim_department],
                        key="new_dept"
                    )
                    salary_scenario_data['Department'] = new_dept
                    scenario_description.append(f"{new_dept}部門に異動する")
                
                elif option == "職種変更":
                    new_role = st.selectbox(
                        "新しい職種を選択",
                        options=[r for r in df['JobRole'].unique() if r != sim_job_role],
                        key="new_role"
                    )
                    salary_scenario_data['JobRole'] = new_role
                    scenario_description.append(f"{new_role}に職種変更する")
            
            # 変更後の給与予測
            new_predicted_salary = salary_pipe.predict(salary_scenario_data)[0]
            salary_change = new_predicted_salary - predicted_salary
            salary_change_pct = (salary_change / predicted_salary) * 100
            
            # 結果表示
            st.write(f"**シナリオ:** {', '.join(scenario_description)}")
            st.write(f"**変更後の予測月収:** ${new_predicted_salary:,.2f}")
            st.write(f"**月収変化:** ${salary_change:+,.2f} ({salary_change_pct:+.1f}%)")
            
            # バーチャートでの比較
            compare_df = pd.DataFrame({
                'シナリオ': ['現在', '変更後'],
                '月収': [predicted_salary, new_predicted_salary]
            })
            
            fig = px.bar(
                compare_df, x='シナリオ', y='月収',
                color='シナリオ',
                text_auto='.2f',
                title="給与変化のシミュレーション"
            )
            
            # テキスト表示を修正
            fig.update_traces(
                text=[f"${val:,.2f}" for val in compare_df['月収']],
                textposition='outside'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.header("人員計画シミュレーション")
        st.info("注: このセクションはシミュレーションデータに基づいています。実際の意思決定には追加データが必要です。")
        
        # 部門選択
        selected_dept = st.selectbox(
            "部門を選択",
            options=df['Department'].unique()
        )
        
        # 選択した部門のデータ
        dept_df = df[df['Department'] == selected_dept]
        
        # 現在の部門構成の分析
        st.subheader("現在の部門構成")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 役職レベル別の人数
            level_counts = dept_df['JobLevel'].value_counts().sort_index()
            level_counts.index = level_counts.index.map(lambda x: f"レベル {x}")
            
            fig = px.pie(
                names=level_counts.index,
                values=level_counts.values,
                title=f"{selected_dept}部門の役職レベル構成",
                hole=0.4
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # 職種別の人数
            role_counts = dept_df['JobRole'].value_counts()
            
            fig = px.bar(
                x=role_counts.index,
                y=role_counts.values,
                title=f"{selected_dept}部門の職種構成",
                labels={'x': '職種', 'y': '人数'},
                color=role_counts.values,
                color_continuous_scale='Viridis'
            )
            
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
        
        # 将来の離職率予測
        st.subheader("離職率予測（1年後）")
        
        # 離職確率を計算
        X_dept = dept_df[features].copy()
        dept_attrition_probs = attrition_pipe.predict_proba(X_dept)[:, 1]
        
        # 職種別の平均離職確率
        role_attrition = {}
        for role in dept_df['JobRole'].unique():
            role_idx = dept_df['JobRole'] == role
            if sum(role_idx) > 0:  # 該当者が存在する場合
                role_attrition[role] = dept_attrition_probs[role_idx].mean()
        
        # 離職確率の可視化
        role_attrition_df = pd.DataFrame({
            'JobRole': list(role_attrition.keys()),
            'AttritionProbability': list(role_attrition.values())
        }).sort_values('AttritionProbability', ascending=False)
        
        fig = px.bar(
            role_attrition_df,
            x='JobRole',
            y='AttritionProbability',
            title=f"{selected_dept}部門の職種別離職確率",
            labels={'AttritionProbability': '平均離職確率', 'JobRole': '職種'},
            color='AttritionProbability',
            color_continuous_scale='Reds',
            text_auto='.1%'
        )
        
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
        
        # 人員計画シミュレーション
        st.subheader("人員計画シミュレーション")
        
        # 計画期間の選択
        planning_period = st.slider(
            "計画期間（年）",
            min_value=1,
            max_value=5,
            value=3
        )
        
        # 成長率の設定
        growth_rate = st.slider(
            "年間成長率 (%)",
            min_value=-10,
            max_value=30,
            value=5,
            step=5
        ) / 100
        
        # 離職率の調整
        attrition_adjustment = st.slider(
            "離職率の調整（現在の予測に対する倍率）",
            min_value=0.5,
            max_value=1.5,
            value=1.0,
            step=0.1
        )
        
        # 職種ごとのシミュレーション実施
        sim_results = []
        
        for role in dept_df['JobRole'].unique():
            # 現在の人数
            current_headcount = sum(dept_df['JobRole'] == role)
            
            # 予測離職率
            role_idx = dept_df['JobRole'] == role
            if sum(role_idx) > 0:
                predicted_attrition_rate = dept_attrition_probs[role_idx].mean() * attrition_adjustment
            else:
                predicted_attrition_rate = 0.1  # デフォルト値
            
            # 年ごとのシミュレーション
            for year in range(1, planning_period + 1):
                # 成長による必要人数の増加
                target_headcount = current_headcount * (1 + growth_rate) ** year
                
                # 離職による減少
                expected_attrition = current_headcount * predicted_attrition_rate * year
                
                # 必要採用数 = 目標人数 - (現在の人数 - 離職予測)
                hiring_need = target_headcount - (current_headcount - expected_attrition)
                
                sim_results.append({
                    'Year': year,
                    'JobRole': role,
                    'CurrentHeadcount': current_headcount,
                    'TargetHeadcount': target_headcount,
                    'ExpectedAttrition': expected_attrition,
                    'HiringNeed': hiring_need
                })
        
        # 結果のデータフレーム化
        sim_df = pd.DataFrame(sim_results)
        
        # 年別・職種別の採用必要数
        hiring_by_year = sim_df.pivot_table(
            index='Year',
            columns='JobRole',
            values='HiringNeed',
            aggfunc='sum'
        ).fillna(0)
        
        # 結果の可視化
        st.write("### 職種別・年次別の採用必要数")
        
        # 積み上げ棒グラフ
        hiring_long = sim_df.groupby(['Year', 'JobRole'])['HiringNeed'].sum().reset_index()
        
        fig = px.bar(
            hiring_long,
            x='Year',
            y='HiringNeed',
            color='JobRole',
            title="年次・職種別の必要採用人数",
            labels={'HiringNeed': '採用必要数', 'Year': '計画年'},
            barmode='stack'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 詳細な採用計画表
        st.write("### 詳細採用計画表")
        
        # 年別の合計採用数
        total_by_year = sim_df.groupby('Year')['HiringNeed'].sum().reset_index()
        total_by_year.columns = ['計画年', '合計採用必要数']
        total_by_year['合計採用必要数'] = total_by_year['合計採用必要数'].round().astype(int)
        
        st.dataframe(total_by_year)
        
        # 職種別の採用戦略提案
        high_attrition_roles = role_attrition_df[role_attrition_df['AttritionProbability'] > 0.15]['JobRole'].tolist()
        
        if high_attrition_roles:
            st.warning(f"以下の職種は離職リスクが高いため、採用と定着施策の両方に注力する必要があります：{', '.join(high_attrition_roles)}")
        
        # 採用コスト試算
        st.subheader("採用コスト試算")
        
        # 職種別採用コスト設定
        st.write("職種別の1人あたり採用コスト（円）を設定してください：")
        
        hiring_costs = {}
        col1, col2 = st.columns(2)
        
        roles = list(dept_df['JobRole'].unique())
        half = len(roles) // 2 + len(roles) % 2
        
        for i, role in enumerate(roles):
            with col1 if i < half else col2:
                default_cost = 500000 if "Manager" in role or "Director" in role else 300000
                hiring_costs[role] = st.number_input(
                    f"{role}",
                    min_value=100000,
                    max_value=2000000,
                    value=default_cost,
                    step=50000,
                    key=f"cost_{role}"
                )
        
        # コスト計算
        cost_data = []
        
        for _, row in sim_df.iterrows():
            hiring_cost = hiring_costs.get(row['JobRole'], 300000)
            total_cost = row['HiringNeed'] * hiring_cost
            
            cost_data.append({
                'Year': row['Year'],
                'JobRole': row['JobRole'],
                'HiringNeed': row['HiringNeed'],
                'HiringCost': hiring_cost,
                'TotalCost': total_cost
            })
        
        cost_df = pd.DataFrame(cost_data)
        
        # 年別の総採用コスト
        yearly_cost = cost_df.groupby('Year')['TotalCost'].sum().reset_index()
        
        fig = px.bar(
            yearly_cost,
            x='Year',
            y='TotalCost',
            title="年次別の採用予算",
            labels={'TotalCost': '採用コスト（円）', 'Year': '計画年'},
            color='TotalCost',
            color_continuous_scale='Blues',
            text_auto='.0f'
        )
        
        # テキスト表示を修正
        fig.update_traces(
            text=[f"¥{val:,.0f}" for val in yearly_cost['TotalCost']],
            textposition='outside'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # 職種別の採用コスト
        role_cost = cost_df.groupby('JobRole')['TotalCost'].sum().reset_index()
        role_cost = role_cost.sort_values('TotalCost', ascending=False)
        
        fig = px.pie(
            role_cost,
            names='JobRole',
            values='TotalCost',
            title="職種別の採用コスト（全期間）",
            hole=0.4
        )
        
        fig.update_traces(texttemplate='¥%{value:,.0f}')
        st.plotly_chart(fig, use_container_width=True)
    
    # フッター
    st.markdown("---")
    st.info("このページでは、機械学習モデルを活用した離職予測、給与予測、および人員計画シミュレーションを提供しています。"
            "より戦略的な人事意思決定の支援にご活用ください。")