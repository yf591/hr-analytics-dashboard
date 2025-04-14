import pandas as pd
import os

def load_hr_data():
    """
    IBM HR Analyticsデータセットを読み込む関数
    """
    # データセットファイルへのパス
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data')
    raw_dir = os.path.join(data_dir, 'raw')
    file_path = os.path.join(raw_dir, 'WA_Fn-UseC_-HR-Employee-Attrition.csv')
    
    # CSVファイルからデータを読み込む
    df = pd.read_csv(file_path)
    
    return df
