import pandas as pd
import json
import re
import os
import sys
import nltk
from konlpy.tag import Okt
from nltk.tokenize import TweetTokenizer
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# --- 使用者設定區 ---
INPUT_FILE = 'lyrics_success_simple_cleaned.csv' # 您的原始檔案
STOPWORD_FILE = 'ko.json' # 您的韓文停用詞檔案
OUTPUT_CSV_FILE = 'lyrics_processed_with_tokens.csv' # 【【您的新需求：輸出的檔案名稱】】

# --- 程式碼主體 ---

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
    print("           韓文文字雲 (korean_wordcloud.png) 可能會顯示為方塊。")
    return None

def setup_nltk():
    """下載 NLTK 必要的停用詞和斷詞模型"""
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        print("正在下載 NLTK 英文停用詞庫...")
        nltk.download('stopwords')
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("正在下載 NLTK 英文斷詞模型...")
        nltk.download('punkt')

def expand_contractions(text):
    """將常見的口語化縮寫還原 (wanna -> want to)"""
    if pd.isna(text) or not text.strip():
        return ''
    text = str(text).lower()
    text = re.sub(r'\bwanna\b', 'want to', text)
    text = re.sub(r'\bgonna\b', 'going to', text)
    text = re.sub(r'\bgotta\b', 'get to', text)
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"you're", "you are", text)
    text = re.sub(r"don't", "do not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"it's", "it is", text)
    text = re.sub(r"that's", "that is", text)
    return text

def load_stopwords():
    """載入韓文和英文停用詞 (使用優化後的版本)"""
    
    # 1. 載入英文停用詞
    eng_stopwords = set(nltk.corpus.stopwords.words('english'))
    custom_eng_stopwords = {
          'oh', 'yeah', 'woah', 'woah-oh', 'hey', 'ooh', 'ah', 'la', 'nan', 'na',
   'na-na', 'ayy', 'let', 'get', 'go', 'gonna', 'wanna', 'like', 'feel', 'know',
   'im', 'youre', 'thats', 'dont', 'cant', 'couldnt', 'every', 'thing', 'one', 'two', 'three', 'four', 'five',
   'look', 'see', 'ba', 'da', 'mm', 'hm', 'du', 'eh', 'woo', 'ha', 'yah', 'ah', 'uh', 'yo', 'huh', 'bum', 'hmm',
   'ko', 'eh', 'woo', 'oo', 'ho', 'pa' , 'li', 'whoo', 'mmh', 'ge', 'ru', 'baam', 'ta'
}

    eng_stopwords.update(custom_eng_stopwords)
    print(f"載入 {len(eng_stopwords)} 個英文停用詞 (已整合 K-Pop 常用擬聲詞)。")

    # 2. 載入韓文停用詞
    if not os.path.exists(STOPWORD_FILE):
        print(f"警告：找不到韓文停用詞檔案 {STOPWORD_FILE}。將只使用內建停用詞。")
        kor_stopwords = set()
    else:
        with open(STOPWORD_FILE, 'r', encoding='utf-8') as f:
            kor_stopwords = set(json.load(f))
            
    # 【關鍵優化】加入詞幹停用詞
    custom_kor_stopwords = {
        '난', '넌', '내', '네', '걍', '넘', '막', '음', '아', '널', '위해', 
        '정말', '내가', '네가', '너의', '나의', '우리', '모두', '오늘', '지금',
        '처럼', '그냥', '수', '것', '이', '그', '저', '너', '나', '게', '날', '밤', '맘',
        'ㅋ', 'ㅎ', 'ㅠ', 'ㅜ',
        '하다', '있다', '없다', '같다', '되다', '않다', '싶다', '보다', '모르다',
        '버리다', '주다', '가다', '오다', '보이다', '아니다', '좋다', '싫다', '원하다',
        '듣다', '들다'
    }
    kor_stopwords.update(custom_kor_stopwords)
    print(f"載入 {len(kor_stopwords)} 個韓文停用詞 (已整合 {STOPWORD_FILE} 與高頻詞幹)。")
    
    return eng_stopwords, kor_stopwords

def separate_languages(text):
    """將字串分離為純韓文和純英文"""
    if pd.isna(text):
        return "", ""
    kor_pattern = re.compile(r'[가-힣]+')
    eng_pattern = re.compile(r'[a-zA-Z]+')
    kor_matches = kor_pattern.findall(str(text))
    eng_matches = eng_pattern.findall(str(text))
    return ' '.join(kor_matches), ' '.join(eng_matches)

def process_korean(text, okt, stopwords):
    """使用 KoNLPy (Okt) 進行詞幹提取 (stem=True)"""
    try:
        pos_tags = okt.pos(text, norm=True, stem=True)
    except Exception:
        return []
    tokens = []
    for word, pos in pos_tags:
        if pos in ['Noun', 'Verb', 'Adjective', 'Adverb']:
            if word not in stopwords and len(word) > 1: 
                tokens.append(word)
    return tokens

def process_english(text, tokenizer, stopwords):
    """使用 TweetTokenizer 和 縮寫還原"""
    text = expand_contractions(text)
    tokens = tokenizer.tokenize(text)
    processed_tokens = []
    for word in tokens:
        if word.isalpha(): 
            if word not in stopwords and len(word) > 1:
                processed_tokens.append(word)
    return processed_tokens

def generate_wordcloud(counts, font_path, filename):
    """根據詞頻字典產生文字雲並儲存"""
    if not counts:
        print(f"沒有足夠的詞彙來產生 {filename}。")
        return
    if 'korean' in filename or 'combined' in filename:
        if not font_path:
            print(f"【【錯誤】】無法產生 {filename}，因為未找到韓文字型。")
            return
    try:
        wc = WordCloud(
            font_path=font_path,
            width=1000,
            height=600,
            background_color='white',
            max_words=200
        ).generate_from_frequencies(counts)
        plt.figure(figsize=(12, 8))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        plt.savefig(filename)
        print(f"文字雲已儲存至: {filename}")
        plt.close() # 儲存後關閉圖形，節省記憶體
    except Exception as e:
        print(f"產生文字雲 {filename} 時發生錯誤: {e}")

def main():
    """主執行函數"""
    
    # 0. 準備 NLTK
    setup_nltk()

    # 1. 載入資料
    print(f"正在讀取 {INPUT_FILE}...")
    if not os.path.exists(INPUT_FILE):
        print(f"錯誤：找不到歌詞檔案 {INPUT_FILE}。")
        return
    df = pd.read_csv(INPUT_FILE)
    if 'lyrics' not in df.columns:
        print("錯誤：CSV 檔案中找不到 'lyrics' 欄位。")
        return
        
    # 2. 載入停用詞
    eng_stopwords, kor_stopwords = load_stopwords()

    # 3. 自動設定 JAVA_HOME
    conda_prefix = os.environ.get('CONDA_PREFIX')
    if conda_prefix and 'JAVA_HOME' not in os.environ:
        print(f"-> 偵測到 Conda 環境，正在手動設定 JAVA_HOME: {conda_prefix}")
        os.environ['JAVA_HOME'] = conda_prefix
            
    # 4. 初始化分析器
    try:
        print("正在初始化韓文斷詞器 (Okt)...")
        okt = Okt()
        print("正在初始化英文斷詞器 (TweetTokenizer)...")
        tweet_tokenizer = TweetTokenizer(
            preserve_case=False,
            reduce_len=True,
            strip_handles=True
        )
        print("-> 分析器初始化成功。")
    except Exception as e:
        print("\n【【【 錯誤：初始化 KoNLPy (Okt) 失敗 】】】")
        print(f"錯誤訊息: {e}")
        print("請確認您已執行: conda install -c conda-forge openjdk")
        print("並已【關閉並重新啟動】您的終端機。")
        sys.exit()

    # 5. 自動尋找字型
    print("\n--- 正在自動尋找韓文字型檔 ---")
    FONT_PATH = find_font_path()
    
    # 6. 處理所有歌詞
    print("\n--- 正在處理所有歌詞 (斷詞、停用詞移除) ---")
    
    # 用於詞頻和文字雲的「全域」列表
    all_kor_tokens = []
    all_eng_tokens = []
    
    # 【【新功能】】 用於儲存到 CSV 的「逐行」列表
    korean_lyrics_list = []
    english_lyrics_list = []
    korean_tokens_list = []
    english_tokens_list = []
    final_tokens_list = []
    
    for row in df.itertuples():
        lyric = row.lyrics
        
        # 1. 分離語言 (需求 2)
        kor_text, eng_text = separate_languages(lyric)
        korean_lyrics_list.append(kor_text)
        english_lyrics_list.append(eng_text)
        
        # 2. 處理韓文 (需求 3)
        kor_tokens = process_korean(kor_text, okt, kor_stopwords)
        korean_tokens_list.append(kor_tokens)
        all_kor_tokens.extend(kor_tokens) # 也加入到全域列表
        
        # 3. 處理英文 (需求 3)
        eng_tokens = process_english(eng_text, tweet_tokenizer, eng_stopwords)
        english_tokens_list.append(eng_tokens)
        all_eng_tokens.extend(eng_tokens) # 也加入到全域列表
        
        # 4. 合併 Token (需求 4)
        final_tokens_list.append(kor_tokens + eng_tokens)
        
    print("處理完成！")
    print(f"共找到 {len(all_kor_tokens)} 個有效韓文詞彙。")
    print(f"共找到 {len(all_eng_tokens)} 個有效英文詞彙。")

    # 7. 【【新功能】】 將處理結果儲存至新的 CSV 檔案 (需求 1)
    print("\n--- 正在將處理結果儲存至新 CSV 檔案 ---")
    
    # 將新列表作為新欄位加入 DataFrame
    df['korean_lyrics'] = korean_lyrics_list
    df['english_lyrics'] = english_lyrics_list
    df['korean_tokens'] = korean_tokens_list
    df['english_tokens'] = english_tokens_list
    df['final_tokens'] = final_tokens_list
    
    try:
        df.to_csv(OUTPUT_CSV_FILE, index=False, encoding='utf-8-sig')
        print(f"成功：已將包含 Token 的完整資料儲存至: {OUTPUT_CSV_FILE}")
    except Exception as e:
        print(f"儲存 CSV 檔案時發生錯誤: {e}")

    # 8. 詞頻分析 (原有功能)
    print("\n--- 韓文詞頻分析 (Top 100) ---")
    kor_counts = Counter(all_kor_tokens)
    for word, count in kor_counts.most_common(100):
        print(f"{word}: {count}")

    print("\n--- 英文詞頻分析 (Top 100) ---")
    eng_counts = Counter(all_eng_tokens)
    for word, count in eng_counts.most_common(100):
        print(f"{word}: {count}")
        
    print("\n(請檢視以上詞頻列表，將不想要的詞彙手動新增到 ko.json 或程式碼中的 custom_stopwords 列表中，然後重新執行，即可優化結果。)")

    # 9. 產生文字雲 (原有功能)
    print("\n--- 正在產生文字雲 ---")
    
    if FONT_PATH:
        generate_wordcloud(kor_counts, FONT_PATH, 'korean_wordcloud.png')
        generate_wordcloud(kor_counts + eng_counts, FONT_PATH, 'combined_wordcloud.png')
    else:
        print("【【錯誤】】未找到韓文字型檔，無法產生韓文及綜合文字雲。")
        
    generate_wordcloud(eng_counts, None, 'english_wordcloud.png')

    print("\n--- 所有任務完成 ---")

if __name__ == '__main__':
    main()