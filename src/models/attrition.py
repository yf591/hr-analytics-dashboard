import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import streamlit as st
import shap
import matplotlib.pyplot as plt
import os
import joblib
import yaml
import xgboost as xgb
import lightgbm as lgb

# 設定ファイルのロード
def load_config():
    """
    設定ファイルを読み込む
    
    Returns:
    --------
    dict
        設定情報
    """
    config_path = 'config.yaml'
    if os.path.exists(config_path):
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    else:
        # デフォルト設定
        return {
            "models": {
                "attrition": {
                    "algorithms": ["LogisticRegression", "RandomForest", "GradientBoosting", "XGBoost", "LightGBM"],
                    "default": "RandomForest",
                    "hyperparameters": {
                        "XGBoost": {"n_estimators": 100, "max_depth": 6, "learning_rate": 0.1},
                        "LightGBM": {"n_estimators": 100, "max_depth": 6, "learning_rate": 0.1}
                    }
                }
            }
        }

# 設定を読み込む
config = load_config()

# モデル定義を辞書で管理
"""
1. **モデル定義（MODELS辞書）**:
   - 複数のモデル（ランダムフォレスト、勾配ブースティング、ロジスティック回帰、XGBoost、LightGBM）を定義
   - ユーザーが選択できるようにしています
"""
MODELS = {
    "ランダムフォレスト": RandomForestClassifier(
        n_estimators=100, max_depth=10, random_state=42
    ),
    "勾配ブースティング": GradientBoostingClassifier(
        n_estimators=100, max_depth=5, random_state=42
    ),
    "ロジスティック回帰": LogisticRegression(
        C=1.0, class_weight='balanced', random_state=42
    ),
    "XGBoost": xgb.XGBClassifier(
        n_estimators=config["models"]["attrition"]["hyperparameters"]["XGBoost"].get("n_estimators", 100),
        max_depth=config["models"]["attrition"]["hyperparameters"]["XGBoost"].get("max_depth", 6),
        learning_rate=config["models"]["attrition"]["hyperparameters"]["XGBoost"].get("learning_rate", 0.1),
        random_state=42,
        enable_categorical=True  # カテゴリカル変数を処理可能にする
    ),
    "LightGBM": lgb.LGBMClassifier(
        n_estimators=config["models"]["attrition"]["hyperparameters"]["LightGBM"].get("n_estimators", 100),
        max_depth=config["models"]["attrition"]["hyperparameters"]["LightGBM"].get("max_depth", 6),
        learning_rate=config["models"]["attrition"]["hyperparameters"]["LightGBM"].get("learning_rate", 0.1),
        random_state=42
    )
}


"""
2. **train_attrition_model関数**:
   - データセットを受け取り、選択されたモデルタイプで学習を行う
   - 離職予測に有用な特徴量を選択
   - データを訓練用とテスト用に分割
   - 特徴量を標準化し、選択されたモデルでパイプラインを構築
   - 学習したモデルを保存し、パイプライン、使用した特徴量、精度スコアを返す
"""
@st.cache_data
def train_attrition_model(df, model_type="ランダムフォレスト", test_size=0.3, random_state=42):
    """
    離職予測モデルを学習します
    
    Parameters:
    ----------
    df : pandas.DataFrame
        学習データ
    model_type : str
        モデルタイプ
    test_size : float
        テストデータの割合
    random_state : int
        乱数シード
        
    Returns:
    -------
    tuple
        (パイプライン, 特徴量名リスト, 精度スコア)
    """
    # 特徴量列
    feature_cols = [
        'Age', 'JobLevel', 'MonthlyIncome', 'TotalWorkingYears',
        'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion',
        'YearsWithCurrManager', 'DistanceFromHome', 'JobSatisfaction',
        'EnvironmentSatisfaction', 'WorkLifeBalance', 'OvertimeBinary',
        'StockOptionLevel', 'TrainingTimesLastYear', 'NumCompaniesWorked'
    ]
    
    # 利用可能な特徴量のみを使用
    available_features = [col for col in feature_cols if col in df.columns]
    
    # 特徴量と目標変数の準備
    X = df[available_features].copy()
    y = df['AttritionBinary'] if 'AttritionBinary' in df.columns else (df['Attrition'] == 'Yes').astype(int)
    
    # データ分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # モデル選択
    if model_type not in MODELS:
        model_type = "ランダムフォレスト"  # デフォルト
        
    # パイプライン構築（標準化 → モデル）
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', MODELS[model_type])
    ])
    
    # モデル学習
    pipeline.fit(X_train, y_train)
    
    # 精度評価
    score = pipeline.score(X_test, y_test)
    
    # モデルを保存
    os.makedirs('models', exist_ok=True)
    joblib.dump(pipeline, f'models/attrition_{model_type}.joblib')
    
    return pipeline, available_features, score


"""
3. **predict_attrition_risk関数**:
   - 従業員データセットを受け取り、離職リスクを予測
   - 既存のモデルファイルがあればロード、なければ学習
   - 各従業員の離職リスクスコア（0〜1の確率値）を計算
   - リスクスコアが追加されたデータフレームを返す
"""
@st.cache_data
def predict_attrition_risk(df, model_type="ランダムフォレスト"):
    """
    従業員の離職リスクを予測し、リスクスコアをデータフレームに追加
    
    Parameters:
    -----------
    df : pandas.DataFrame
        従業員データ
    model_type : str
        使用するモデルタイプ
        
    Returns:
    --------
    pandas.DataFrame
        離職リスクスコアを含むデータフレーム
    """
    # 特徴量列
    feature_cols = [
        'Age', 'JobLevel', 'MonthlyIncome', 'TotalWorkingYears',
        'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion',
        'YearsWithCurrManager', 'DistanceFromHome', 'JobSatisfaction',
        'EnvironmentSatisfaction', 'WorkLifeBalance', 'OvertimeBinary',
        'StockOptionLevel', 'TrainingTimesLastYear', 'NumCompaniesWorked'
    ]
    
    # 利用可能な特徴量のみを使用
    available_features = [col for col in feature_cols if col in df.columns]
    
    # モデルの存在確認、なければ学習
    model_path = f'models/attrition_{model_type}.joblib'
    if os.path.exists(model_path):
        pipeline = joblib.load(model_path)
    else:
        pipeline, _, _ = train_attrition_model(df, model_type)
    
    # 予測用データの準備
    X_predict = df[available_features].copy()
    
    # 予測
    risk_scores = pipeline.predict_proba(X_predict)[:, 1]
    
    # 結果を元のデータフレームにコピー
    result_df = df.copy()
    result_df['attrition_risk'] = risk_scores
    
    return result_df


"""
4. **get_attrition_factors関数**:
   - 学習されたモデルから離職に影響する主な要因を抽出
   - モデルタイプに応じて特徴量重要度を取得（ツリーベースモデルとロジスティック回帰で異なる方法）
   - 重要度順にソートして上位n個の要因を返す
"""
def get_attrition_factors(df, model_type="ランダムフォレスト", top_n=10):
    """
    離職に影響する主な要因を取得
    
    Parameters:
    -----------
    df : pandas.DataFrame
        従業員データ
    model_type : str
        使用するモデルタイプ
    top_n : int
        上位何個の要因を返すか
        
    Returns:
    --------
    pandas.DataFrame
        要因とその重要度を含むデータフレーム
    """
    # 特徴量列
    feature_cols = [
        'Age', 'JobLevel', 'MonthlyIncome', 'TotalWorkingYears',
        'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion',
        'YearsWithCurrManager', 'DistanceFromHome', 'JobSatisfaction',
        'EnvironmentSatisfaction', 'WorkLifeBalance', 'OvertimeBinary',
        'StockOptionLevel', 'TrainingTimesLastYear', 'NumCompaniesWorked'
    ]
    
    # 利用可能な特徴量のみを使用
    available_features = [col for col in feature_cols if col in df.columns]
    
    # モデルの存在確認、なければ学習
    model_path = f'models/attrition_{model_type}.joblib'
    if os.path.exists(model_path):
        pipeline = joblib.load(model_path)
    else:
        pipeline, available_features, _ = train_attrition_model(df, model_type)
    
    # モデル抽出
    model = pipeline.named_steps['model']
    
    # 特徴量の重要度を取得
    if hasattr(model, 'feature_importances_'):
        # ツリーベースモデルの場合
        importances = model.feature_importances_
        feature_importance = pd.DataFrame({
            'Feature': available_features,
            'Importance': importances
        })
    else:
        # ロジスティック回帰の場合
        importances = np.abs(model.coef_[0])
        feature_importance = pd.DataFrame({
            'Feature': available_features,
            'Importance': importances
        })
    
    # 重要度でソート
    feature_importance = feature_importance.sort_values('Importance', ascending=False)
    
    # 上位n個を返す
    return feature_importance.head(top_n)


"""
5. **plot_shap_values関数**:
   - SHAPライブラリを使用してモデルの予測を説明
   - 各特徴量がモデルの予測にどう影響するかを可視化
   - 計算コスト削減のためサンプリングオプションあり
   - SHAP要約プロットを含むmatplotlibの図を返す
"""
def plot_shap_values(df, model_type="ランダムフォレスト", sample_size=100):
    """
    SHAP値を使用してモデルの説明可能性を可視化
    
    Parameters:
    -----------
    df : pandas.DataFrame
        従業員データ
    model_type : str
        使用するモデルタイプ
    sample_size : int
        SHAP計算に使用するサンプル数
        
    Returns:
    --------
    matplotlib.figure.Figure
        SHAP要約プロット
    """
    # 特徴量列
    feature_cols = [
        'Age', 'JobLevel', 'MonthlyIncome', 'TotalWorkingYears',
        'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion',
        'YearsWithCurrManager', 'DistanceFromHome', 'JobSatisfaction',
        'EnvironmentSatisfaction', 'WorkLifeBalance', 'OvertimeBinary',
        'StockOptionLevel', 'TrainingTimesLastYear', 'NumCompaniesWorked'
    ]
    
    # 利用可能な特徴量のみを使用
    available_features = [col for col in feature_cols if col in df.columns]
    
    # モデルの存在確認、なければ学習
    model_path = f'models/attrition_{model_type}.joblib'
    if os.path.exists(model_path):
        pipeline = joblib.load(model_path)
    else:
        pipeline, available_features, _ = train_attrition_model(df, model_type)
    
    # サンプリング（計算コスト削減のため）
    if len(df) > sample_size:
        df_sample = df.sample(sample_size, random_state=42)
    else:
        df_sample = df
    
    # 特徴量データ
    X_sample = df_sample[available_features].copy()
    
    # スケーラーを使用して標準化
    X_scaled = pipeline.named_steps['scaler'].transform(X_sample)
    
    # モデル抽出
    model = pipeline.named_steps['model']
    
    # SHAP計算
    explainer = shap.Explainer(model, X_scaled)
    shap_values = explainer(X_scaled)
    
    # SHAP要約プロット
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, feature_names=available_features, show=False)
    plt.tight_layout()
    
    return fig

"""
上記で作成した関数を組み合わせることで、ダッシュボード上で離職予測モデルを学習・評価し、予測結果や要因分析を視覚化できる。

`loader.py` モジュールは以下の役割がある。
- データセットの読み込みと基本的な前処理を担当
- 特徴量エンジニアリングを行い、分析に役立つ派生変数を作成
- `@st.cache_data` デコレータでStreamlitアプリ内でのデータ読み込みを最適化
"""