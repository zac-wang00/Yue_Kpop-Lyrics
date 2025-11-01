import pandas as pd
import re
import os

# --- 設定 ---
INPUT_FILE = 'lyrics_success_simple.csv'      # 您要讀取的原始檔案
OUTPUT_FILE = 'lyrics_success_simple_cleaned.csv' # 您要輸出的乾淨檔案
TARGET_COLUMN = 'lyrics'                          # 您要清理的欄位名稱
# -------------

def clean_lyrics(text):
    """
    清理歌詞，移除 [] 內的標籤並整理多餘的換行。
    """
    if pd.isna(text):
        return ""
    
    # 1. 刪除所有被 [] 包住的內容 (非貪婪模式)
    cleaned_text = re.sub(r'\[.*?\]', '', str(text))
    
    # 2. 清理因刪除標籤後產生的多餘空白行
    # 將兩個以上的連續換行(中間可能夾雜空白) 替換為 兩個換行
    cleaned_text = re.sub(r'(\n\s*){2,}', '\n\n', cleaned_text).strip()
    
    return cleaned_text

def main():
    """主執行函數"""
    print(f"--- 開始清理檔案 ---")
    
    # 檢查檔案是否存在
    if not os.path.exists(INPUT_FILE):
        print(f"錯誤：在當前資料夾中找不到檔案 '{INPUT_FILE}'")
        print("請確認此 .py 檔案與您的 csv 檔案放在同一個資料夾中。")
        return

    try:
        # 讀取 CSV 檔案
        print(f"正在讀取 '{INPUT_FILE}'...")
        df = pd.read_csv(INPUT_FILE)
        
        # 檢查欄位是否存在
        if TARGET_COLUMN not in df.columns:
            print(f"錯誤：在 CSV 中找不到 '{TARGET_COLUMN}' 欄位。")
            print(f"可用的欄位有: {df.columns.to_list()}")
            return
            
        # 套用清理函數
        print(f"正在清理 '{TARGET_COLUMN}' 欄位...")
        df[TARGET_COLUMN] = df[TARGET_COLUMN].apply(clean_lyrics)
        
        # 儲存為新檔案
        df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
        
        print(f"\n--- 清理完成！ ---")
        print(f"已成功儲存檔案: '{OUTPUT_FILE}'")

    except Exception as e:
        print(f"處理過程中發生錯誤: {e}")

if __name__ == '__main__':
    main()