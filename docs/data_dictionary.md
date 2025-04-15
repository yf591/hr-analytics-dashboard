# HR Analytics データ辞書

ここではIBM HR Analytics Employee Attrition & Performance データセットの各項目の定義、データ型、および値の範囲について説明します。

## データ概要

- **ソース**: IBM HR Analytics Employee Attrition & Performance データセット
- **レコード数**: 1,470
- **特徴量数**: 35
- **目的**: 従業員の離職（Attrition）予測と、給与（MonthlyIncome）予測

## 基本情報フィールド

| フィールド名 | データ型 | 説明 | 値の範囲・例 |
|------------|---------|------|------------|
| EmployeeNumber | 整数 | 従業員固有の識別番号 | 1-2000 |
| Age | 整数 | 従業員の年齢 | 18-60 |
| Gender | カテゴリカル | 従業員の性別 | 'Male', 'Female' |
| MaritalStatus | カテゴリカル | 婚姻状況 | 'Single', 'Married', 'Divorced' |

## 職務関連フィールド

| フィールド名 | データ型 | 説明 | 値の範囲・例 |
|------------|---------|------|------------|
| Department | カテゴリカル | 所属部門 | 'Sales', 'Research & Development', 'Human Resources' |
| JobRole | カテゴリカル | 職種 | 'Sales Executive', 'Research Scientist', 'Laboratory Technician', 'Manufacturing Director', 'Healthcare Representative', 'Manager', 'Sales Representative', 'Research Director', 'Human Resources' |
| JobLevel | 整数 | 職位レベル | 1-5 (1が最低、5が最高) |
| JobInvolvement | 整数 | 職務への関与度 | 1-4 (1が低、4が高) |
| JobSatisfaction | 整数 | 職務満足度 | 1-4 (1が低、4が高) |

## 報酬関連フィールド

| フィールド名 | データ型 | 説明 | 値の範囲・例 |
|------------|---------|------|------------|
| MonthlyIncome | 整数 | 月収（米ドル） | 1,000-20,000 |
| MonthlyRate | 整数 | 月間レート | 2,000-27,000 |
| DailyRate | 整数 | 日当 | 100-1,500 |
| HourlyRate | 整数 | 時給 | 30-100 |
| PercentSalaryHike | 整数 | 前回の昇給率（％） | 10-25 |
| StockOptionLevel | 整数 | ストックオプションレベル | 0-3 |

## キャリア関連フィールド

| フィールド名 | データ型 | 説明 | 値の範囲・例 |
|------------|---------|------|------------|
| YearsAtCompany | 整数 | 現在の会社での勤続年数 | 0-40 |
| YearsInCurrentRole | 整数 | 現在の役職での年数 | 0-18 |
| YearsSinceLastPromotion | 整数 | 前回の昇進からの経過年数 | 0-15 |
| YearsWithCurrManager | 整数 | 現在の上司の下での勤務年数 | 0-17 |
| TotalWorkingYears | 整数 | 総労働経験年数 | 0-40 |
| NumCompaniesWorked | 整数 | 過去に勤務した企業数 | 0-9 |

## 労働環境/条件フィールド

| フィールド名 | データ型 | 説明 | 値の範囲・例 |
|------------|---------|------|------------|
| OverTime | カテゴリカル | 残業の有無 | 'Yes', 'No' |
| BusinessTravel | カテゴリカル | 出張頻度 | 'Non-Travel', 'Travel_Rarely', 'Travel_Frequently' |
| DistanceFromHome | 整数 | 自宅からの距離（マイル） | 1-30 |
| WorkLifeBalance | 整数 | ワークライフバランス評価 | 1-4 (1が悪い、4が良い) |
| EnvironmentSatisfaction | 整数 | 職場環境満足度 | 1-4 (1が低、4が高) |
| RelationshipSatisfaction | 整数 | 職場での人間関係満足度 | 1-4 (1が低、4が高) |

## 教育/スキル関連フィールド

| フィールド名 | データ型 | 説明 | 値の範囲・例 |
|------------|---------|------|------------|
| Education | 整数 | 教育レベル | 1-5 (1=高校以下, 2=専門学校, 3=学士, 4=修士, 5=博士) |
| EducationField | カテゴリカル | 専攻分野 | 'Life Sciences', 'Medical', 'Marketing', 'Technical Degree', 'Human Resources', 'Other' |
| TrainingTimesLastYear | 整数 | 昨年の研修参加回数 | 0-6 |

## パフォーマンス関連フィールド

| フィールド名 | データ型 | 説明 | 値の範囲・例 |
|------------|---------|------|------------|
| PerformanceRating | 整数 | 業績評価スコア | 1-4 (1=Low, 2=Good, 3=Excellent, 4=Outstanding) |
| StandardHours | 整数 | 標準労働時間 | 通常80 (このデータセットでは全員同じ値) |

## 目標変数

| フィールド名 | データ型 | 説明 | 値の範囲・例 |
|------------|---------|------|------------|
| Attrition | カテゴリカル | 離職の有無 | 'Yes', 'No' (Yesが約16%) |

## 派生変数

以下は、分析のために追加で作成した派生変数です。

| フィールド名 | データ型 | 説明 | 派生元 |
|------------|---------|------|--------|
| AttritionBinary | 整数 | 離職の有無（バイナリ形式） | Attrition ('Yes'=1, 'No'=0) |
| AgeGroup | カテゴリカル | 年齢グループ | Age (5つのビンに分類) |
| TenureGroup | カテゴリカル | 勤続年数グループ | YearsAtCompany (5つのビンに分類) |
| IncomeLevel | カテゴリカル | 収入レベル（四分位数） | MonthlyIncome (4つのビンに分類) |
| OvertimeBinary | 整数 | 残業の有無（バイナリ形式） | OverTime ('Yes'=1, 'No'=0) |
| OverallSatisfaction | 浮動小数点 | 総合満足度スコア | JobSatisfaction, EnvironmentSatisfaction, WorkLifeBalance, RelationshipSatisfactionの平均 |

## 特記事項

- **欠損値**: このデータセットには欠損値は含まれていません。
- **不均衡データ**: 離職(Attrition)は全体の約16%であり、クラス不均衡があります。
- **相関関係**: 月収(MonthlyIncome)と職位レベル(JobLevel)には強い相関関係があります。
- **固定値**: 標準労働時間(StandardHours)はすべての従業員で同じ値(80)であるため、予測モデルでは使用しません。

---

## データ利用上の注意点

1. このデータはIBMが生成した架空のデータセットであり、実在の企業や従業員のデータではありません。
2. 予測モデル構築の際は、クラス不均衡に対処するための手法を検討する必要があります。
3. 一部の変数（例: PerformanceRating, StandardHours）は分布に偏りがあるため、予測力が限られる可能性があります。

---

## HR指標の意味とデータ活用ガイド

### 満足度指標の解釈

満足度指標（JobSatisfaction, EnvironmentSatisfaction, WorkLifeBalance, RelationshipSatisfaction）は1-4のスケールで測定されますが、これらの解釈には組織文化や業界の文脈が重要です：

| スコア | 一般的な解釈 | アクションの優先度 |
|-------|------------|-----------------|
| 1 (低) | 深刻な不満あり | 緊急対応が必要 |
| 2 (中低) | 一部不満あり | 短期的な改善が推奨 |
| 3 (中高) | 概ね満足 | 維持・強化が望ましい |
| 4 (高) | 非常に満足 | ベストプラクティスとして共有 |

### 主要業績指標（KPI）としての活用

このデータセットの指標は、以下のHR KPIの分析に活用できます：

1. **離職率予測**：Attrition変数を目標として、リスク要因を特定
2. **従業員エンゲージメント**：満足度指標の複合スコアから全体傾向を把握
3. **報酬最適化**：MonthlyIncomeと他変数の関係から適切な報酬設計を検討
4. **キャリア開発効果**：YearsSinceLastPromotionとパフォーマンス/満足度の関係を分析
5. **人材育成ROI**：TrainingTimesLastYearとパフォーマンス向上の相関を測定

### 部門別分析の視点

部門（Department）ごとに異なる特性があるため、分析時には以下の点に注意すると洞察が深まります：

- **Sales（営業）**：BusinessTravel、WorkLifeBalanceの影響が特に強い傾向
- **Research & Development（研究開発）**：EducationField、JobLevelの影響が顕著
- **Human Resources（人事）**：RelationshipSatisfaction、EnvironmentSatisfactionが重要

### データ傾向の時系列解釈

YearsAtCompany、YearsInCurrentRole、YearsSinceLastPromotionなどの時間関連変数は、従業員のライフサイクルを表しています：

- **入社1-2年目**：適応期（JobSatisfactionの変動が大きい）
- **3-5年目**：成長期（YearsSinceLastPromotionの影響が増大）
- **6-10年目**：安定期/転機（WorkLifeBalanceの重要性が高まる）
- **10年以上**：定着期（環境要因より内発的モチベーションの影響が強い）

### AI・機械学習活用のヒント

このデータを用いたモデリングでは、以下のアプローチが効果的です：

1. **特徴量エンジニアリング**：複数の満足度指標を組み合わせた合成変数の作成
2. **離職予兆スコア**：確率予測値を0-100のスコアに変換し早期警戒システムに活用
3. **フィードバックループ**：モデル予測と実際の離職データを継続的に比較・更新
4. **LLMとの連携**：モデル予測結果からLLMで具体的なアクションプランを生成

### インサイト抽出のための質問例

データ分析時に以下の質問を検討することで、より深い洞察を得られます：

1. 離職リスクが高い従業員に共通する特徴は何か？
2. どの満足度指標が離職に最も強い影響を与えているか？
3. 残業（OverTime）が満足度に与える影響は部門によって異なるか？
4. キャリア停滞（YearsSinceLastPromotion）と離職の関係性は役職レベルによって異なるか？
5. 従業員の総合満足度を高める最も効果的な要因は何か？

## カスタマイズのポイント

実際のプロジェクトに合わせて、以下の点をカスタマイズすると良いと思います。

1. **プロジェクト固有の前処理**:
   - 実際に作成した特徴量があれば、「派生変数」セクションに追加
   - データクリーニングで対処した問題があれば、「特記事項」に記載

2. **テクニカル情報**:
   - 変数の統計的特性（平均、中央値、最頻値など）
   - 欠損値の処理方法
   - 外れ値の特定と処理方法

3. **ビジネスコンテキスト**:
   - 各変数のビジネス上の意味や重要性
   - 特定の値の解釈（例: 満足度スコア「1」は何を意味するのか）
   - 業界標準や平均との比較

4. **具体例**:
   - 実際のデータレコードの例
   - 値の分布を示すヒストグラムや図（または図へのリンク）