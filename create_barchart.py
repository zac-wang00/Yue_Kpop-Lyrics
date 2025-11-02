import pandas as pd
import numpy as np
import ast  # 用於將字串格式的列表安全地轉換回 Python 列表
import os
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from collections import Counter

# --- 【【使用者可自訂設定區】】 ---

# 1. 來源檔案 (必須是 'analyze_lyrics.py' 處理後的輸出檔)
INPUT_FILE = "lyrics_with_tokens_SM.csv" 

# 2. 要用來計算詞頻的欄位
#    ( 'korean_tokens', 'english_tokens', 或 'final_tokens' )
TOKEN_COLUMN = "final_tokens" # 建議使用這個，代表韓英總和

# 3. 輸出的圖片檔案名稱
OUTPUT_FILE = "combined_top20_barchart.png"

# 4. 長條圖顏色
CHART_COLOR = 'Steelblue'

# 5. 要顯示的詞彙數量 (Top N)
TOP_N = 20

# ---------------------------------

def find_font_path():
    """自動在常見路徑中尋找可用的韓文字型"""
    macos_paths = [
        '/System/Library/Fonts/Supplemental/AppleSDGothicNeo.ttc',
        '/System/Library/Fonts/Supplemental/AppleGothic.ttf',
        '/System/Library/Fonts/AppleGothic.ttf',
        '/Library/Fonts/NanumGothic.ttf' 
    ]
    windows_paths = [
        'C:/Windows/Fonts/malgun.ttf', 
        'C:/Windows/Fonts/gulim.ttc',
        'C:/Windows/Fonts/batang.ttc', 
        'C:/Windows/Fonts/msyh.ttc'
    ]
    
    if os.name == 'nt': # Windows
        paths_to_check = windows_paths
    elif os.name == 'posix': # macOS or Linux
        paths_to_check = macos_paths
    else:
        paths_to_check = []

    for path in paths_to_check:
        if os.path.exists(path):
            print(f"-> 成功自動找到韓文字型: {path}")
            return path 

    print("【【警告】】自動尋找字型失敗。")
    print("           長條圖上的韓文標籤 (korean_wordcloud.png) 可能會顯示為方塊。")
    return None

def set_matplotlib_font(font_path):
    """設定 Matplotlib 全域字型以顯示韓文"""
    if font_path:
        try:
            fm.fontManager.addfont(font_path)
            font_name = fm.FontProperties(fname=font_path).get_name()
            plt.rcParams['font.family'] = font_name
            plt.rcParams['axes.unicode_minus'] = False # 解決 CJK 字型導致的「負號」顯示問題
            print(f"-> Matplotlib 全域字型已設定為: {font_name}")
            return True
        except Exception as e:
            print(f"【警告】設定 Matplotlib 字型時發生錯誤: {e}。長條圖標籤可能顯示為方塊。")
            return False
    else:
        print("【警告】未找到韓文字型，長條圖標籤可能顯示為方塊。")
        return False

def safe_literal_eval(val):
    """
    安全地將 CSV 中的字串 "['a', 'b']" 轉換為 Python 列表 ['a', 'b']
    """
    if pd.notna(val) and isinstance(val, str):
        try:
            return ast.literal_eval(val)
        except:
            return []
    return [] # 確保 NaN 或非字串值也轉換成空列表

# (請確保在檔案頂部 import numpy as np)

def generate_barchart_from_file(input_file, token_column, output_file, top_n, color): # 'color' 參數在漸層版中不會被用到
    """
    (此腳本的核心功能)
    從處理好的 CSV 檔案讀取 tokens 並產生長條圖
    【【此版本整合了字體與漸層色建議】】
    """
    
    # 1. 載入資料 ( ... 此部分不變 ... )
    print(f"正在讀取 {input_file}...")
    if not os.path.exists(input_file):
        print(f"錯誤：找不到檔案 {input_file}。請先執行 'analyze_lyrics.py' 產生此檔案。")
        return
    try:
        df = pd.read_csv(input_file)
    except Exception as e:
        print(f"讀取檔案時發生錯誤: {e}")
        return

    # 2. 檢查欄位是否存在 ( ... 此部分不變 ... )
    if token_column not in df.columns:
        print(f"錯誤：CSV 檔案中找不到 '{token_column}' 欄位。")
        print(f"可用的欄位有: {df.columns.to_list()}")
        return

    # 3. 轉換並彙整所有詞彙 ( ... 此部分不變 ... )
    print(f"正在讀取並轉換 '{token_column}' 欄位...")
    df[token_column] = df[token_column].apply(safe_literal_eval)
    all_tokens_list = df[token_column].explode().dropna().tolist()

    if not all_tokens_list:
        print("錯誤：找不到任何詞彙。請檢查您的 CSV 檔案內容是否正確。")
        return
    
    # 4. 計算詞頻 ( ... 此部分不變 ... )
    word_counts = Counter(all_tokens_list)
    print(f"計算完成。共找到 {len(word_counts)} 個不重複的詞彙。")

    # 5. 取得 Top N 詞彙 ( ... 此部分不變 ... )
    top_n_words = word_counts.most_common(top_n)
    
    # 6. 分離詞彙和頻率 ( ... 此部分不變 ... )
    words, frequencies = zip(*top_n_words)
    
    # 7. 建立畫布
    plt.figure(figsize=(10, 12)) 
    
    # 8. 【【改善】】繪製水平長條圖 (使用漸層色)
    #    產生一個漸層色列表 (從 0.4=中藍 到 0.9=深藍)
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, top_n))
    
    plt.barh(words[::-1], frequencies[::-1], color=colors, edgecolor='grey', linewidth=0.5)
    
    # 9. 【【改善】】設定標題和標籤 (字體加大加粗)
    plt.xlabel('Frequency', fontsize=12, fontweight='bold')
    plt.ylabel('Words', fontsize=12, fontweight='bold')
    plt.title(f'Top {top_n} Word Frequency_SM', fontsize=16, fontweight='bold')
    
    # 10. 【【改善】】設定刻度字體
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=11)
    
    # 11. 自動調整邊距
    plt.tight_layout()
    
    # 12. 儲存圖片
    try:
        plt.savefig(output_file)
        print(f"\n--- 所有任務完成 ---")
        print(f"成功：長條圖已儲存至: {output_file}")
    except Exception as e:
        print(f"儲存長條圖 {output_file} 時發生錯誤: {e}")
    
    plt.close()

def main():
    """主執行函數"""
    print("--- 獨立長條圖產生腳本 ---")
    
    # 1. 自動尋找並設定字型
    print("--- 正在自動尋找韓文字型檔 ---")
    font_path = find_font_path()
    set_matplotlib_font(font_path)
    
    # 2. 執行核心功能
    generate_barchart_from_file(
        input_file=INPUT_FILE,
        token_column=TOKEN_COLUMN,
        output_file=OUTPUT_FILE,
        top_n=TOP_N,
        color=CHART_COLOR
    )

if __name__ == '__main__':
    main()