import pandas as pd
import json
import re
import os
import sys
import nltk
from konlpy.tag import Okt
import spacy  # 【【新功能】】 匯入 spaCy
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# --- 使用者設定區 ---
INPUT_FILE = 'lyrics_success_simple_cleaned.csv' # 您的原始檔案
STOPWORD_FILE = 'ko.json' # 您的韓文停用詞檔案
OUTPUT_CSV_FILE = 'lyrics_processed_with_tokens.csv' # 輸出的檔案名稱

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
    text = re.sub(r'\bgimme\b', 'give me', text)
    text = re.sub(r'\bkinda\b', 'kind of', text)
    text = re.sub(r'\bbabe\b', 'baby', text)
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"you're", "you are", text)
    text = re.sub(r"don't", "do not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"it's", "it is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r'\bbout\b', "about", text)
    text = re.sub(r'\bthinkin\b', "thinking", text)
    return text

def load_stopwords():
    """載入韓文和英文停用詞 (使用優化後的版本)"""
    
    # 1. 載入英文停用詞
    eng_stopwords = set(nltk.corpus.stopwords.words('english'))
    custom_eng_stopwords = {
        'ah', 'ayy', 'woo', 'eh', 'ta', 'du', 'ha', 'yo', 'mm', 'yah', 'huh', 'ay',
        'ooh', 'aah', 'na', 'oh', 'uh', 'la', 'da', 'ya', 'hey', 'hoo', 'woah',
        'wan', 'gon', 'cause', 'let', 'one', 'come', 'never', 'take', 'every', 
        'always', 'could', 'still', 'put', 'ever', 'two', 'three', 'please', 
        'everybody', 'yes', 'okay', 'something', 'nothing', 'everything',
        'im', 'youre', 'thats', 'dont', 'cant', 'couldnt', 'thing', 'four',
        'five', 'look', 'see', 'get', 'go', 'know', 'like','yeah','ba','nae','doo',
        'em', 'ho', 'whoa', 'nan', 'ra', 'pa', 'gee', 'de', 'ee', 'di', 'lo', 'ru', 'wow', 'hm', 'babe',
        'six', 'seven', 'eight', 'nine', 'ten', 'bam', 'bba', 'ou', 'ddu', 'ye', 'ok', 'would', 'boom', 
        'hmm', 'ohohohoh', 'wooo', 'hu', 'nah', 'whoo', 'woah-oh', 'na-na', 'gonna', 'wanna', 'feel', 
        'got', 'bum', 'ko', 'oo', 'li', 'mmh', 'ge', 'baam', 'ri', 'cha', 'mon', 'bu', 'xoxo','yuh',
        'ti', 'girle', 'el', 'mi', 'ni', 'rum', 'purr', 'wa', 'pam', 'tu', 'su', 'nal', 'bo', 'pang', 'thang',
        'fi', 'ol', 'te', 'ne', 'jigeum', 'mo', 'blo', 'montana', 'si', 'db', 'ding', 'com'
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
        "!", "\"", "$", "%", "&", "'", "(", ")", "*", "+", ",", "-", ".", "...", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
          ";", "<", "=", ">", "?", "@", "\\", "^", "_", "`", "|", "~", "·", "—", "——","'", "“", "”", "…", "、", "。", "〈", "〉",
            "《", "》","가","가까스로","가령","각","각각","각자","각종","갖고말하자면","같다","같이","개의치않고","거니와","거바","거의","것","것과 같이",
            "것들","게다가","게우다","겨우","견지에서","결과에 이르다","결국","결론을 낼 수 있다","겸사겸사","고려하면","고로","곧","공동으로","과","과연","관계가 있다",
            "관계없이","관련이 있다","관하여","관한","관해서는","구","구체적으로","구토하다","그","그들","그때","그래","그래도","그래서","그러나","그러니","그러니까","그러면",
            "그러므로","그러한즉","그런 까닭에","그런데","그런즉","그럼","그럼에도 불구하고","그렇게 함으로써","그렇지","그렇지 않다면","그렇지 않으면","그렇지만","그렇지않으면","그리고",
            "그리하여","그만이다","그에 따르는","그위에","그저","그중에서","그치지 않다","근거로","근거하여","기대여","기점으로","기준으로","기타","까닭으로","까악","까지","까지 미치다",
            "까지도","꽈당","끙끙","끼익","나","나머지는","남들","남짓","너","너희","너희들","네","넷","년","논하지 않다","놀라다","누가 알겠는가","누구","다른","다른 방면으로","다만",
            "다섯","다소","다수","다시 말하자면","다시말하면","다음","다음에","다음으로","단지","답다","당신","당장","대로 하다","대하면","대하여","대해 말하자면","대해서","댕그","더구나","더군다나",
            "더라도","더불어","더욱더","더욱이는","도달하다","도착하다","동시에","동안","된바에야","된이상","두번째로","둘","둥둥","뒤따라","뒤이어","든간에","들","등","등등","딩동","따라","따라서","따위",
            "따지지 않다","딱","때","때가 되어","때문에","또","또한","뚝뚝","라 해도","령","로","로 인하여","로부터","로써","륙","를","마음대로","마저","마저도","마치","막론하고","만 못하다","만약","만약에",
            "만은 아니다","만이 아니다","만일","만큼","말하자면","말할것도 없고","매","매번","메쓰겁다","몇","모","모두","무렵","무릎쓰고","무슨","무엇","무엇때문에","물론","및","바꾸어말하면","바꾸어말하자면","바꾸어서 말하면",
            "바꾸어서 한다면","바꿔 말하면","바로","바와같이","밖에 안된다","반대로","반대로 말하자면","반드시","버금","보는데서","보다더","보드득","본대로","봐","봐라","부류의 사람들","부터","불구하고","불문하고","붕붕","비걱거리다",
            "비교적","비길수 없다","비로소","비록","비슷하다","비추어 보아","비하면","뿐만 아니라","뿐만아니라","뿐이다","삐걱","삐걱거리다","사","삼","상대적으로 말하자면","생각한대로","설령","설마","설사","셋","소생","소인","솨","쉿",
            "습니까","습니다","시각","시간","시작하여","시초에","시키다","실로","심지어","아","아니","아니나다를가","아니라면","아니면","아니었다면","아래윗","아무거나","아무도","아야","아울러","아이","아이고","아이구","아이야","아이쿠","아하",
            "아홉","안 그러면","않기 위하여","않기 위해서","알 수 있다","알았어","앗","앞에서","앞의것","야","약간","양자","어","어기여차","어느","어느 년도","어느것","어느곳","어느때","어느쪽","어느해","어디","어때","어떠한","어떤","어떤것",
            "어떤것들","어떻게","어떻해","어이","어째서","어쨋든","어쩔수 없다","어찌","어찌됏든","어찌됏어","어찌하든지","어찌하여","언제","언젠가","얼마","얼마 안 되는 것","얼마간","얼마나","얼마든지","얼마만큼","얼마큼","엉엉","에","에 가서",
            "에 달려 있다","에 대해","에 있다","에 한하다","에게","에서","여","여기","여덟","여러분","여보시오","여부","여섯","여전히","여차","연관되다","연이서","영","영차","옆사람","예","예를 들면","예를 들자면","예컨대","예하면","오","오로지",
            "오르다","오자마자","오직","오호","오히려","와","와 같은 사람들","와르르","와아","왜","왜냐하면","외에도","요만큼","요만한 것","요만한걸","요컨대","우르르","우리","우리들","우선","우에 종합한것과같이","운운","월","위에서 서술한바와같이","위하여",
            "위해서","윙윙","육","으로","으로 인하여","으로서","으로써","을","응","응당","의","의거하여","의지하여","의해","의해되다","의해서","이","이 되다","이 때문에","이 밖에","이 외에","이 정도의","이것","이곳","이때","이라면","이래","이러이러하다","이러한",
            "이런","이럴정도로","이렇게 많은 것","이렇게되면","이렇게말하자면","이렇구나","이로 인하여","이르기까지","이리하여","이만큼","이번","이봐","이상","이어서","이었다","이와 같다","이와 같은","이와 반대로","이와같다면","이외에도","이용하여","이유만으로","이젠","이지만",
            "이쪽","이천구","이천육","이천칠","이천팔","인 듯하다","인젠","일","일것이다","일곱","일단","일때","일반적으로","일지라도","임에 틀림없다","입각하여","입장에서","잇따라","있다","자","자기","자기집","자마자","자신","잠깐","잠시","저","저것","저것만큼","저기","저쪽",
            "저희","전부","전자","전후","점에서 보아","정도에 이르다","제","제각기","제외하고","조금","조차","조차도","졸졸","좀","좋아","좍좍","주룩주룩","주저하지 않고","줄은 몰랏다","줄은모른다","중에서","중의하나","즈음하여","즉","즉시","지든지","지만","지말고","진짜로","쪽으로",
            "차라리","참","참나","첫번째로","쳇","총적으로","총적으로 말하면","총적으로 보면","칠","콸콸","쾅쾅","쿵","타다","타인","탕탕","토하다","통하여","툭","퉤","틈타","팍","팔","퍽","펄렁","하","하게될것이다","하게하다","하겠는가","하고 있다","하고있었다","하곤하였다","하구나",
            "하기 때문에","하기 위하여","하기는한데","하기만 하면","하기보다는","하기에","하나","하느니","하는 김에","하는 편이 낫다","하는것도","하는것만 못하다","하는것이 낫다","하는바","하더라도","하도다","하도록시키다","하도록하다","하든지","하려고하다","하마터면","하면 할수록","하면된다","하면서",
            "하물며","하여금","하여야","하자마자","하지 않는다면","하지 않도록","하지마","하지마라","하지만","하하","한 까닭에","한 이유는","한 후","한다면","한다면 몰라도","한데","한마디","한적이있다","한켠으로는","한항목","할 따름이다","할 생각이다","할 줄 안다","할 지경이다","할 힘이 있다","할때",
            "할만하다","할망정","할뿐","할수있다","할수있어","할줄알다","할지라도","할지언정","함께","해도된다","해도좋다","해봐요","해서는 안된다","해야한다","해요","했어요","향하다","향하여","향해서","허","허걱","허허","헉","헉헉","헐떡헐떡","형식으로 쓰여","혹시","혹은","혼자","훨씬","휘익","휴","흐흐","흥","힘입어","︿",
            "하다","없다","않다","날","보다","말","너무","내게","있어","않아","싶다","넡","수","돼다","헤","걸","내","난","난"
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
        # 只保留有意義的詞性：名詞、動詞、形容詞
        if pos in ['Noun', 'Verb', 'Adjective']:
            if word not in stopwords and len(word) > 1: 
                tokens.append(word)
    return tokens

# 【【【 函數已更新 】】】
def process_english(text, nlp, stopwords):
    """
    【新功能】使用 spaCy 進行斷詞、詞形還原 (Lemmatization) 和停用詞移除
    """
    # 1. 仍然先執行縮寫還原
    text = expand_contractions(text)
    
    # 2. 將文字交給 spaCy 處理
    doc = nlp(text)
    
    processed_tokens = []
    for token in doc:
        # 3. 取得詞形還原後的詞 (e.g., "nights" -> "night")
        #    並統一轉為小寫
        lemma = token.lemma_.lower()
        
        # 4. 處理代名詞 (spaCy 會將 'I', 'you' 還原成 '-PRON-')
        #    我們將其保留為原始文字
        if token.lemma_ == '-PRON-':
            lemma = token.text.lower()
            
        # 5. 只保留純字母的單字
        if lemma.isalpha(): 
            # 6. 移除停用詞和單字
            if lemma not in stopwords and len(lemma) > 1:
                processed_tokens.append(lemma)
                
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
        plt.close() 
    except Exception as e:
        print(f"產生文字雲 {filename} 時發生錯誤: {e}")

# 【【【 新功能 】】】
def load_spacy_model():
    """
    載入 spaCy 英文模型，如果找不到則提供安裝指示
    """
    model_name = "en_core_web_sm"
    try:
        # 嘗試載入模型
        nlp = spacy.load(model_name)
        print(f"-> 成功載入 SpaCy 英文模型 ({model_name})。")
        return nlp
    except OSError:
        # 如果模型不存在
        print("\n" + "="*50)
        print(f"【【 錯誤：找不到 SpaCy 英文模型 '{model_name}' 】】")
        print("     spaCy 需要一個語言模型來進行詞形還原。")
        print("\n--- 請在您的終端機中執行以下指令來下載模型: ---")
        print(f"     python -m spacy download {model_name}")
        print("="*50 + "\n")
        print("下載完成後，請重新執行此程式。")
        sys.exit() # 停止程式，讓使用者去下載

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
            
    # 4. 初始化分析器 (【【已更新】】)
    try:
        # 【新功能】 載入 spaCy 模型
        nlp_eng = load_spacy_model()
        
        print("正在初始化韓文斷詞器 (Okt)...")
        okt = Okt()

        # (已移除 TweetTokenizer)
        
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
    
    # 6. 處理所有歌詞 (【【已更新】】)
    print("\n--- 正在處理所有歌詞 (斷詞、停用詞移除、詞形還原) ---")
    
    all_kor_tokens = []
    all_eng_tokens = []
    korean_lyrics_list = []
    english_lyrics_list = []
    korean_tokens_list = []
    english_tokens_list = []
    final_tokens_list = []
    
    for row in df.itertuples():
        lyric = row.lyrics
        
        kor_text, eng_text = separate_languages(lyric)
        korean_lyrics_list.append(kor_text)
        english_lyrics_list.append(eng_text)
        
        kor_tokens = process_korean(kor_text, okt, kor_stopwords)
        korean_tokens_list.append(kor_tokens)
        all_kor_tokens.extend(kor_tokens)
        
        # 【【已更新】】
        # 將 nlp_eng 模型傳入，取代 TweetTokenizer
        eng_tokens = process_english(eng_text, nlp_eng, eng_stopwords)
        english_tokens_list.append(eng_tokens)
        all_eng_tokens.extend(eng_tokens)
        
        final_tokens_list.append(kor_tokens + eng_tokens)
        
    print("處理完成！")
    print(f"共找到 {len(all_kor_tokens)} 個有效韓文詞彙。")
    print(f"共找到 {len(all_eng_tokens)} 個有效英文詞彙。")

    # 7. 將處理結果儲存至新的 CSV 檔案
    print("\n--- 正在將處理結果儲存至新 CSV 檔案 ---")
    
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

    # 8. 詞頻分析
    print("\n--- 韓文詞頻分析 (Top 100) ---")
    kor_counts = Counter(all_kor_tokens)
    for word, count in kor_counts.most_common(100):
        print(f"{word}: {count}")

    print("\n--- 英文詞頻分析 (Top 100) ---")
    eng_counts = Counter(all_eng_tokens)
    for word, count in eng_counts.most_common(100):
        print(f"{word}: {count}")
        
    print("\n(請檢視以上詞頻列表，將不想要的詞彙手動新增到 ko.json 檔案中，然後重新執行，即可優化結果。)")

    # 9. 產生文字雲
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
