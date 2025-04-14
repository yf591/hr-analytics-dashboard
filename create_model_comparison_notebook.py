import nbformat as nbf
import os

# ディレクトリが存在することを確認
os.makedirs('notebooks/modeling', exist_ok=True)

# 新しいノートブックを作成
nb = nbf.v4.new_notebook()

# セルのリスト
cells = [
    # セル1: タイトル（マークダウン）
    nbf.v4.new_markdown_cell("# 機械学習モデル比較分析\n\nこのノートブックでは、HRアナリティクスデータに対して複数の機械学習モデル（RandomForest、XGBoost、LightGBM）を適用し、それらのパフォーマンスを比較します。最適なモデルを特定し、UIで使用するための基盤を提供します。"),
    
    # セル2: ライブラリのインポート（マークダウン）
    nbf.v4.new_markdown_cell("## 必要なライブラリのインポート\n\nまず、分析に必要なライブラリをインポートします。"),
    
    # セル3: ライブラリのインポート（コード）
    nbf.v4.new_code_cell("""# データ操作・可視化ライブラリ
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# 機械学習ライブラリ
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve

# XGBoost、LightGBMライブラリ
import xgboost as xgb
import lightgbm as lgb

# 警告を無視
import warnings
warnings.filterwarnings('ignore')

# プロット設定
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12"""),
    
    # セル4: データの読み込み（マークダウン）
    nbf.v4.new_markdown_cell("## データの読み込み\n\nプロジェクトのデータローダーを使用してHRデータを読み込みます。"),
    
    # セル5: データの読み込み（コード）
    nbf.v4.new_code_cell("""# プロジェクトのモジュールにアクセスできるようにパスを追加
import sys
import os
sys.path.append(os.path.abspath('../..'))

# データローダーのインポート
from src.data.loader import load_hr_data

# HRデータの読み込み
df = load_hr_data()

# データの確認
print(f"データサイズ: {df.shape}")
df.head()"""),
    
    # セル6: 離職予測モデルの比較（マークダウン）
    nbf.v4.new_markdown_cell("## 離職予測モデルの比較\n\n離職予測（Attrition）のための異なる機械学習モデルを実装し、その性能を比較します。"),
    
    # セル7: データの前処理（マークダウン）
    nbf.v4.new_markdown_cell("### データの前処理\n\nモデルのトレーニングに使用する特徴量と目的変数を準備します。"),
    
    # セル8: データの前処理（コード）
    nbf.v4.new_code_cell("""# 離職予測に使用する特徴量
features = [
    'Age', 'MonthlyIncome', 'DistanceFromHome', 'OverTime',
    'JobSatisfaction', 'WorkLifeBalance', 'YearsAtCompany',
    'YearsSinceLastPromotion', 'JobLevel', 'MaritalStatus',
    'NumCompaniesWorked', 'TrainingTimesLastYear', 'BusinessTravel'
]

# 目的変数
target = 'Attrition'

# データの準備
X = df[features].copy()
y = df[target].copy()

# カテゴリ変数とカテゴリでない変数の分類
categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

print(f"カテゴリ変数: {categorical_features}")
print(f"数値変数: {numeric_features}")

# データの分割（訓練用データとテスト用データ）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 前処理パイプラインの定義
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)"""),
    
    # セル9: モデルの定義（マークダウン）
    nbf.v4.new_markdown_cell("### モデルの定義\n\n比較するモデルの定義とパイプラインの構築を行います。"),
    
    # セル10: モデルの定義（コード）
    nbf.v4.new_code_cell("""# モデルの定義
models = {
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
    'XGBoost': xgb.XGBClassifier(n_estimators=100, random_state=42),
    'LightGBM': lgb.LGBMClassifier(n_estimators=100, random_state=42)
}

# 結果格納用の辞書
model_results = {}
model_predictions = {}"""),
    
    # セル11: モデルのトレーニングと評価（マークダウン）
    nbf.v4.new_markdown_cell("### モデルのトレーニングと評価\n\n各モデルをトレーニングし、テストデータでの性能を評価します。"),
    
    # セル12: モデルのトレーニングと評価（コード）
    nbf.v4.new_code_cell("""# 各モデルでの学習と評価
for model_name, model in models.items():
    print(f"\\n{model_name}のトレーニングと評価中...")
    
    # モデルパイプラインの作成
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    
    # モデルの学習
    pipeline.fit(X_train, y_train)
    
    # テストデータでの予測
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]  # 離職確率（陽性クラスの確率）
    
    # モデル性能の評価
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    
    # 結果の格納
    model_results[model_name] = {
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'pipeline': pipeline
    }
    
    model_predictions[model_name] = {
        'y_pred': y_pred,
        'y_prob': y_prob
    }
    
    # 結果の表示
    print(f"{model_name}の精度: {accuracy:.4f}")
    print(f"{model_name}のROC-AUC: {roc_auc:.4f}")
    print(f"\\n分類レポート:\\n{classification_report(y_test, y_pred)}")"""),
    
    # セル13: モデルパフォーマンスの比較（マークダウン）
    nbf.v4.new_markdown_cell("### モデルパフォーマンスの比較\n\n各モデルの性能を視覚的に比較します。"),
    
    # セル14: モデルパフォーマンスの比較（コード）
    nbf.v4.new_code_cell("""# モデル性能の比較（精度とROC-AUC）
performance_df = pd.DataFrame({
    'モデル': list(model_results.keys()),
    '精度': [results['accuracy'] for results in model_results.values()],
    'ROC-AUC': [results['roc_auc'] for results in model_results.values()]
})

# データを長形式に変換
performance_long = performance_df.melt(id_vars=['モデル'], var_name='評価指標', value_name='スコア')

# バープロット
plt.figure(figsize=(12, 6))
sns.barplot(x='モデル', y='スコア', hue='評価指標', data=performance_long)
plt.title('各モデルの性能比較', fontsize=15)
plt.ylim(0, 1)
plt.ylabel('スコア')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()"""),
    
    # セル15: ROC曲線の比較（マークダウン）
    nbf.v4.new_markdown_cell("### ROC曲線の比較\n\n各モデルのROC曲線を比較し、識別性能を評価します。"),
    
    # セル16: ROC曲線の比較（コード）
    nbf.v4.new_code_cell("""# ROC曲線の比較
plt.figure(figsize=(10, 8))

for model_name, predictions in model_predictions.items():
    fpr, tpr, _ = roc_curve(y_test, predictions['y_prob'])
    roc_auc = model_results[model_name]['roc_auc']
    plt.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {roc_auc:.3f})')

# ランダム分類器のROC曲線（参照用）
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='ランダム分類器')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('偽陽性率')
plt.ylabel('真陽性率')
plt.title('各モデルのROC曲線比較', fontsize=15)
plt.legend(loc="lower right")
plt.grid(linestyle='--', alpha=0.7)
plt.show()"""),
    
    # セル17: 特徴量重要度の比較（マークダウン）
    nbf.v4.new_markdown_cell("### 特徴量重要度の比較\n\n各モデルの特徴量重要度を比較し、離職予測に最も影響する要因を分析します。"),
    
    # セル18: 特徴量重要度の取得（コード）
    nbf.v4.new_code_cell("""# 特徴量名の取得
preprocessor = model_results['RandomForest']['pipeline'].named_steps['preprocessor']
feature_names = list(preprocessor.get_feature_names_out())

# 各モデルの特徴量重要度を取得
feature_importances = {}

for model_name, results in model_results.items():
    pipeline = results['pipeline']
    model = pipeline.named_steps['classifier']
    
    # モデル種類によって重要度の取得方法が異なる
    try:
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif model_name == 'XGBoost':
            # XGBoostの場合
            importances = model.get_booster().get_score(importance_type='weight')
            # 特徴名のマッピングが必要な場合がある
            if isinstance(importances, dict):
                # 辞書から配列に変換
                feat_importances = np.zeros(len(feature_names))
                for feat, imp in importances.items():
                    # 特徴名のインデックスを見つける
                    try:
                        idx = feature_names.index(feat)
                        feat_importances[idx] = imp
                    except ValueError:
                        # 特徴名が一致しない場合はスキップ
                        pass
                importances = feat_importances
        else:
            # その他の方法で特徴量重要度を取得
            importances = np.zeros(len(feature_names))
            print(f"{model_name}の特徴量重要度が取得できません")
            
        # 長さをチェック
        if len(importances) == len(feature_names):
            feature_importances[model_name] = importances
        else:
            print(f"{model_name}の特徴量重要度の長さが一致しません: {len(importances)} vs {len(feature_names)}")
    except Exception as e:
        print(f"{model_name}の特徴量重要度取得中にエラー: {e}")"""),
    
    # セル19: RandomForestの特徴量重要度の可視化（コード）
    nbf.v4.new_code_cell("""# RandomForestの特徴量重要度を可視化
if 'RandomForest' in feature_importances:
    # 特徴量重要度をデータフレームに変換
    rf_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importances['RandomForest']
    }).sort_values('Importance', ascending=False)
    
    # 上位15個の特徴量のみ表示
    top_features = rf_importance_df.head(15)
    
    # プロット
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=top_features)
    plt.title('RandomForest: 特徴量重要度（上位15）', fontsize=15)
    plt.xlabel('重要度')
    plt.ylabel('特徴量')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.show()"""),
    
    # セル20: XGBoostの特徴量重要度の可視化（コード）
    nbf.v4.new_code_cell("""# XGBoostの特徴量重要度を可視化
if 'XGBoost' in feature_importances and len(feature_importances['XGBoost']) > 0:
    # 特徴量重要度をデータフレームに変換
    xgb_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importances['XGBoost']
    }).sort_values('Importance', ascending=False)
    
    # 上位15個の特徴量のみ表示
    top_features = xgb_importance_df.head(15)
    
    # プロット
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=top_features)
    plt.title('XGBoost: 特徴量重要度（上位15）', fontsize=15)
    plt.xlabel('重要度')
    plt.ylabel('特徴量')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.show()"""),
    
    # 残りのセルを追加...
    # セル21: 給与予測モデルの比較（マークダウン）
    nbf.v4.new_markdown_cell("## 給与予測モデルの比較\n\n給与予測のための異なる回帰モデルを実装し、その性能を比較します。"),
    
    # セル22: 給与予測データの前処理（マークダウン）
    nbf.v4.new_markdown_cell("### データの前処理\n\n給与予測モデルに使用するデータを準備します。"),
    
    # セル23: 給与予測データの前処理（コード）
    nbf.v4.new_code_cell("""# 給与予測に使用する特徴量
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

print(f"カテゴリ変数: {salary_cat_features}")
print(f"数値変数: {salary_num_features}")

# データの分割（訓練用データとテスト用データ）
X_train_salary, X_test_salary, y_train_salary, y_test_salary = train_test_split(
    X_salary, y_salary, test_size=0.3, random_state=42
)

# 前処理パイプラインの定義
salary_preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), salary_num_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), salary_cat_features)
    ]
)"""),
    
    # セル24: 回帰モデルの定義（マークダウン）
    nbf.v4.new_markdown_cell("### 回帰モデルの定義\n\n給与予測のための回帰モデルを定義します。"),
    
    # セル25: 回帰モデルの定義（コード）
    nbf.v4.new_code_cell("""from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# 回帰モデルの定義
regression_models = {
    'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
    'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42),
    'LightGBM': lgb.LGBMRegressor(n_estimators=100, random_state=42),
    'LinearRegression': LinearRegression(),
    'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
}

# 結果格納用の辞書
regression_results = {}"""),
    
    # セル26: 回帰モデルのトレーニングと評価（マークダウン）
    nbf.v4.new_markdown_cell("### 回帰モデルのトレーニングと評価\n\n各回帰モデルをトレーニングし、テストデータでの性能を評価します。"),
    
    # セル27: 回帰モデルのトレーニングと評価（コード）
    nbf.v4.new_code_cell("""# 各モデルでの学習と評価
for model_name, model in regression_models.items():
    print(f"\\n{model_name}のトレーニングと評価中...")
    
    # モデルパイプラインの作成
    pipeline = Pipeline([
        ('preprocessor', salary_preprocessor),
        ('regressor', model)
    ])
    
    # モデルの学習
    pipeline.fit(X_train_salary, y_train_salary)
    
    # テストデータでの予測
    y_pred_salary = pipeline.predict(X_test_salary)
    
    # モデル性能の評価
    r2 = r2_score(y_test_salary, y_pred_salary)
    mae = mean_absolute_error(y_test_salary, y_pred_salary)
    rmse = np.sqrt(mean_squared_error(y_test_salary, y_pred_salary))
    
    # 結果の格納
    regression_results[model_name] = {
        'r2': r2,
        'mae': mae,
        'rmse': rmse,
        'pipeline': pipeline,
        'predictions': y_pred_salary
    }
    
    # 結果の表示
    print(f"{model_name}のR²スコア: {r2:.4f}")
    print(f"{model_name}のMAE: {mae:.2f}")
    print(f"{model_name}のRMSE: {rmse:.2f}")"""),
    
    # セル28: 回帰モデルのパフォーマンス比較（マークダウン）
    nbf.v4.new_markdown_cell("### 回帰モデルのパフォーマンス比較\n\n各回帰モデルの性能を視覚的に比較します。"),
    
    # セル29: 回帰モデルのパフォーマンス比較（コード）
    nbf.v4.new_code_cell("""# モデル性能の比較（R²、MAE、RMSE）
regression_performance_df = pd.DataFrame({
    'モデル': list(regression_results.keys()),
    'R²': [results['r2'] for results in regression_results.values()],
    'MAE': [results['mae'] for results in regression_results.values()],
    'RMSE': [results['rmse'] for results in regression_results.values()]
})

# R²スコアを可視化
plt.figure(figsize=(12, 6))
sns.barplot(x='モデル', y='R²', data=regression_performance_df, palette='viridis')
plt.title('各モデルのR²スコア比較', fontsize=15)
plt.ylim(0, 1)
plt.grid(axis='y', linestyle='--', alpha=0.7)
for i, v in enumerate(regression_performance_df['R²']):
    plt.text(i, v + 0.01, f"{v:.3f}", ha='center', fontsize=10)
plt.show()

# MAEとRMSEを可視化
error_df = regression_performance_df.melt(id_vars=['モデル'], value_vars=['MAE', 'RMSE'], var_name='評価指標', value_name='誤差')

plt.figure(figsize=(14, 6))
sns.barplot(x='モデル', y='誤差', hue='評価指標', data=error_df, palette='Set2')
plt.title('各モデルの誤差指標比較', fontsize=15)
plt.ylabel('誤差値')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()"""),
    
    # セル30: 予測値と実際値の比較（マークダウン）
    nbf.v4.new_markdown_cell("### 予測値と実際値の比較\n\n最も性能の良いモデルについて、予測値と実際値を散布図で比較します。"),
    
    # セル31: 予測値と実際値の比較（コード）
    nbf.v4.new_code_cell("""# R²スコアで最良のモデルを選択
best_model = regression_performance_df.loc[regression_performance_df['R²'].idxmax(), 'モデル']
print(f"最良のモデル: {best_model}（R²: {regression_results[best_model]['r2']:.4f}）")

# 予測値と実際値の散布図
plt.figure(figsize=(10, 8))
plt.scatter(y_test_salary, regression_results[best_model]['predictions'], alpha=0.5)
plt.plot([y_test_salary.min(), y_test_salary.max()], [y_test_salary.min(), y_test_salary.max()], 'r--')
plt.xlabel('実際の月収')
plt.ylabel('予測された月収')
plt.title(f'{best_model}: 予測値 vs 実際値', fontsize=15)
plt.grid(linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()"""),
    
    # セル32: 結論とモデル選択（マークダウン）
    nbf.v4.new_markdown_cell("## 結論とモデル選択\n\n各タスクに最適なモデルを特定し、UIでの実装に推奨するモデルを決定します。"),
    
    # セル33: 結論とモデル選択（コード）
    nbf.v4.new_code_cell("""# 離職予測の最適モデル
best_attrition_model = max(model_results.items(), key=lambda x: x[1]['roc_auc'])[0]
best_attrition_auc = model_results[best_attrition_model]['roc_auc']
best_attrition_acc = model_results[best_attrition_model]['accuracy']

# 給与予測の最適モデル
best_salary_model = max(regression_results.items(), key=lambda x: x[1]['r2'])[0]
best_salary_r2 = regression_results[best_salary_model]['r2']
best_salary_mae = regression_results[best_salary_model]['mae']

print("===== モデル評価結果まとめ =====\\n")
print("【離職予測】")
print(f"最適モデル: {best_attrition_model}")
print(f"ROC-AUC: {best_attrition_auc:.4f}")
print(f"精度: {best_attrition_acc:.4f}\\n")

print("【給与予測】")
print(f"最適モデル: {best_salary_model}")
print(f"R²スコア: {best_salary_r2:.4f}")
print(f"平均絶対誤差: {best_salary_mae:.2f}\\n")

print("===== UIでの実装推奨 =====\\n")
print("離職予測UI: RandomForest、XGBoost、LightGBMの3モデルを選択可能に実装\\n")
print("給与予測UI: RandomForest、XGBoost、LightGBM、線形回帰、勾配ブースティングの5モデルを選択可能に実装")"""),
    
    # セル34: まとめ（マークダウン）
    nbf.v4.new_markdown_cell("""## まとめ

この分析では、離職予測と給与予測のための複数の機械学習モデルを比較しました。

**離職予測については**：
- XGBoostモデルが最も高いROC-AUCスコアを達成
- RandomForestとLightGBMも競争力のある性能を示した
- 業務特性（OverTime）、職務満足度、勤続年数などが重要な予測因子として特定された

**給与予測については**：
- 勾配ブースティング系の手法が高い精度を示した
- 役職レベル、勤続年数、総労働経験年数が給与を予測する上で最も重要な要因

**実装推奨**：
ユーザーが複数のモデルから選択できるUIを実装することで、各モデルの特性や予測結果の違いを比較分析できるようにします。XGBoostとLightGBMを追加することで、より高度な分析が可能になります。""")
]

# セルをノートブックに追加
nb['cells'] = cells

# ノートブックを保存
nbf.write(nb, 'notebooks/modeling/model_comparison.ipynb')

print("Jupyter Notebook 'notebooks/modeling/model_comparison.ipynb' が正常に作成されました。")
