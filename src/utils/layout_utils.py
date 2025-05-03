import streamlit as st
import plotly.graph_objects as go
import pandas as pd

def create_plotly_config():
    """
    レスポンシブ対応とPDF出力を最適化するPlotlyグラフの設定を返す
    """
    return {
        'displayModeBar': False,  # モードバーを表示しない（PDFでスペース節約）
        'responsive': True,       # レスポンシブ対応を有効化
        'toImageButtonOptions': { # 画像出力設定
            'format': 'png',
            'filename': 'chart',
            'height': 500,
            'width': 700,
            'scale': 2          # 高解像度で出力
        }
    }

def create_plotly_layout_template():
    """
    PDFレポート出力に最適化されたPlotlyレイアウトテンプレートを返す
    """
    return {
        'font': {'size': 10},  # PDFで読みやすいフォントサイズ
        'margin': {'l': 40, 'r': 30, 't': 50, 'b': 40},  # 余白調整
        'colorway': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', 
                    '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
                    '#bcbd22', '#17becf'],  # 印刷時に見やすい配色
        'legend': {'orientation': 'h', 'y': -0.15},  # 凡例を下部に配置
        'height': 400,  # PDFページに収まる高さ
    }

def create_responsive_columns(ratios=None, gap="small"):
    """
    レスポンシブ対応のカラムを作成する
    
    Parameters:
    ----------
    ratios : list, optional
        各カラムの幅の比率。例: [1, 2, 1] は1:2:1の比率
    gap : str, optional
        カラム間のギャップ ("small", "medium", "large")
        
    Returns:
    -------
    list
        カラムのリスト
    """
    # デフォルト比率は等幅
    if ratios is None:
        ratios = [1, 1]
    
    # 合計を計算
    total = sum(ratios)
    
    # スケーリング係数
    scaled_ratios = [r/total for r in ratios]
    
    # Streamlitのcolumnsを作成
    return st.columns(scaled_ratios, gap=gap)

def add_page_break():
    """
    PDF出力時のページ区切りを挿入する
    """
    st.markdown('<div class="pdf-page-break"></div>', unsafe_allow_html=True)

def format_dataframe_for_display(df, max_rows=10):
    """
    データフレームをPDF出力とレスポンシブ表示に最適化して表示
    
    Parameters:
    ----------
    df : pandas.DataFrame
        表示するデータフレーム
    max_rows : int, optional
        表示する最大行数
    """
    # データフレームをレスポンシブ対応の表として表示
    st.markdown("""
    <style>
    .dataframe-container {
        overflow-x: auto;
        max-width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # max_rowsに制限してデータフレームを表示
    st.markdown('<div class="dataframe-container">', unsafe_allow_html=True)
    st.dataframe(df.head(max_rows), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

def optimize_chart_for_pdf(fig):
    """
    Plotlyグラフをレスポンシブ対応しPDF出力に最適化
    
    Parameters:
    ----------
    fig : plotly.graph_objects.Figure
        最適化するPlotlyの図
    
    Returns:
    -------
    plotly.graph_objects.Figure
        最適化された図
    """
    # レイアウトテンプレートを適用
    template = create_plotly_layout_template()
    
    # 現在のレイアウトを更新
    current_layout = fig.layout
    for key, value in template.items():
        if isinstance(value, dict):
            # 深いマージ
            if hasattr(current_layout, key):
                for sub_key, sub_value in value.items():
                    if hasattr(getattr(current_layout, key), sub_key):
                        setattr(getattr(current_layout, key), sub_key, sub_value)
            else:
                setattr(current_layout, key, value)
        else:
            # トップレベルの値を設定
            setattr(current_layout, key, value)
    
    # フォントサイズを調整
    fig.update_layout(
        title_font_size=14,
        xaxis_title_font_size=12,
        yaxis_title_font_size=12,
        legend_font_size=10
    )
    
    # トレースのテキストサイズを調整
    for trace in fig.data:
        if hasattr(trace, 'textfont'):
            trace.textfont.size = 9
    
    return fig

def display_optimized_chart(fig, use_container_width=True):
    """
    最適化したグラフを表示
    
    Parameters:
    ----------
    fig : plotly.graph_objects.Figure
        表示するPlotlyの図
    use_container_width : bool, optional
        コンテナ幅を使用するかどうか
    """
    # グラフを最適化
    optimized_fig = optimize_chart_for_pdf(fig)
    
    # 設定を適用してグラフを表示
    st.plotly_chart(
        optimized_fig,
        use_container_width=use_container_width,
        config=create_plotly_config()
    )

def check_mobile():
    """
    モバイルデバイスからのアクセスかチェック（近似値）
    実際のStreamlitではユーザーエージェントへのアクセスが制限されるため、
    簡易的なチェックに基づいています。
    
    Returns:
    -------
    bool
        モバイルデバイスの場合はTrue
    """
    # Streamlitではwindow幅が直接取得できないため、
    # CSSを使って近似的に判定する
    is_mobile = False
    
    # デバイス情報用のプレースホルダを作成
    device_info = st.empty()
    
    # JavaScriptでウィンドウ幅を検出して設定するコード
    device_info.markdown("""
    <script>
        if (window.innerWidth < 768) {
            document.documentElement.style.setProperty('--is-mobile', 'true');
        } else {
            document.documentElement.style.setProperty('--is-mobile', 'false');
        }
    </script>
    <div id="device-detector"></div>
    """, unsafe_allow_html=True)
    
    # この関数では実際のデバイス判定はできないので、CSSの可変グリッドに頼る
    return is_mobile