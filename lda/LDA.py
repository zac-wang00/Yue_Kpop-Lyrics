import pandas as pd
from gensim import corpora
from gensim.models import LdaModel
from matplotlib.font_manager import FontProperties
import re
import ast # 引入 Abstract Syntax Tree 模組
import numpy as np
from tqdm import tqdm

input_file = "data/lyrics_processed_with_tokens.csv"
#company = "SM"
df = pd.read_csv(input_file)
# ====================================================================
# ⚠️ 1. 資料讀取與準備 (請根據你的實際程式碼修改這部分)
# ====================================================================

# 假設你的 DataFrame 已經載入，並且已經完成了所有的預處理和合併步驟
# 例如： df = pd.read_csv('your_data.csv')
# 假設 df['final_tokens'] 欄位是 List of Strings 類型

# 確保 'final_tokens' 欄位中的每個字串都被安全地評估為 Python 列表
def convert_str_to_list(list_str):
    try:
        # ast.literal_eval 比 eval() 更安全，專門用於評估字串中的基本數據結構
        return ast.literal_eval(list_str)
    except (ValueError, TypeError):
        # 如果遇到 NaN 或無法評估的字串，返回空列表
        return []



# 應用轉換，這將是你的新最終詞彙欄位
df['final_tokens_restored'] = df['final_tokens'].apply(convert_str_to_list)
documents = df['final_tokens_restored'].tolist()

#df = df[df['label name'] == company]
#documents = df['final_tokens_restored'].tolist()
# --------------------------------------------------------------------
# 接下來的 LDA 流程，請使用這個新的還原欄位
# --------------------------------------------------------------------
# 1. 建立文檔集合 (List of Lists)

# 2. 移除空文檔（安全操作）
documents = [doc for doc in documents if doc]
# ====================================================================
# 2. 數據驗證與安全檢查 (避免 ValueError: cannot compute LDA over an empty collection)
# ====================================================================
total_docs = len(documents)
empty_docs_count = sum(1 for doc in documents if not doc)
total_tokens = sum(len(doc) for doc in documents)

print("\n--- 數據流失最終檢查 ---")
print(f"文檔總數 (歌曲數): {total_docs}")
print(f"空列表文檔數: {empty_docs_count}")
print(f"所有文檔中詞彙的總計數: {total_tokens}")

if total_tokens == 0:
    print("🚨 致命錯誤：所有文檔詞彙總計數為 0。請檢查 DataFrame 原始欄位。")
    exit()  # 停止執行

# --------------------------------------------------------------------
# 移除空文檔（如果空文檔數量不多，這樣可以避免它們干擾後續處理）
documents = [doc for doc in documents if doc]
# --------------------------------------------------------------------


# ====================================================================
# 3. 建立詞典 (Dictionary) 和語料庫 (Corpus)
# ====================================================================

print("\n開始建立詞典...")
# 使用所有文檔建立詞典
dictionary = corpora.Dictionary(documents)

# 詞彙過濾：使用最寬鬆的條件來避免丟失核心詞
dictionary.filter_extremes(
    no_below=10,  # 詞彙至少在 2 首歌中出現過
    #no_above=0.99,  # 詞彙只有在超過 99% 的歌中出現才移除
    keep_n=None
)

print(f"✅ 詞彙表大小 (過濾後): {len(dictionary)}")

# 建立 BoW 語料庫 (將詞彙轉換為 (ID, Count) 格式)
corpus = [dictionary.doc2bow(doc) for doc in documents]
print(f"✅ 語料庫文檔數: {len(corpus)}")

# ====================================================================
# 4. 訓練 LDA 模型 (LdaModel)
# ====================================================================

# ⚠️ 關鍵參數： num_topics (建議從 10 開始嘗試)
NUM_TOPICS = 6

print(f"\n開始訓練 {NUM_TOPICS} 個主題的 LDA 模型...")
# 稀疏 Alpha: 鼓勵每首歌只專注於少數主題
EXPERIMENTAL_ALPHA = 0.01

# 稀疏 Eta: 鼓勵每個主題只由少數關鍵詞組成 (0.1 是常見的稀疏值)
EXPERIMENTAL_ETA = 0.1

lda_model = LdaModel(
    corpus=corpus,
    id2word=dictionary,
    num_topics=NUM_TOPICS,
    random_state=42,  # 設定隨機種子，確保結果可重現
    chunksize=100,
    passes=20,  # 增加迭代次數以提高模型品質
    alpha=EXPERIMENTAL_ALPHA,
    eta=EXPERIMENTAL_ETA
)

print("✅ LDA 模型訓練完成。")

# ====================================================================
# 5. 結果解讀：檢視主題 (使用 CJK 字型確保韓文顯示)
# ====================================================================

# ⚠️ 請確保你的 FONT_PATH 指向一個支持韓文的字型，例如 Malgun Gothic
FONT_PATH = 'C:\\Windows\\Fonts\\malgun.ttf'
try:
    cjk_font = FontProperties(fname=FONT_PATH)
except:
    print("⚠️ 警告：無法載入韓文字型。終端機輸出可能會出現亂碼。")

print("\n--- LDA 主題模型結果 (Top 10 詞彙) ---")

for idx, topic in lda_model.print_topics(num_words=10):
    # 清理輸出格式：移除數字權重和小數點，只保留詞彙
    # 範例輸出: 0.050*"word" + 0.040*"word2"
    cleaned_topic = re.sub(r'\d\.\d{3}\*"', '', topic).replace('"', '').replace(' + ', ' / ')

    # 打印結果 (如果終端機支持，韓文會正常顯示)
    print(f"🌟 主題 #{idx + 1}：")
    print(f"   {cleaned_topic}\n")

## ------------------------------------------------------------------
## 輸出 1: 主題-詞彙概率 (phi)
## ------------------------------------------------------------------

print("==============================================")
print(f"✨ 主題-詞彙概率 (Top 15 詞彙，共 {NUM_TOPICS} 個主題)")
print("==============================================")

# 獲取每個主題的 Top 詞彙和權重
topics_and_probs = lda_model.show_topics(
    num_topics=NUM_TOPICS,
    num_words=15,  # 這裡我們提取 Top 15 詞彙，比你之前給的 Top 10 更詳細
    formatted=False
)

for topic_id, word_probs in topics_and_probs:
    print(f"\n🌟 主題 #{topic_id + 1}:")
    # 將詞彙及其概率格式化輸出
    output_str = ", ".join([f"{word} ({prob:.4f})" for word, prob in word_probs])
    print(output_str)

## ------------------------------------------------------------------
## 輸出 2: 文檔-主題概率 (theta) - 新增欄位
## ------------------------------------------------------------------

print("\n==============================================")
print("📄 文檔-主題概率 (新增主題分佈欄位)")
print("==============================================")

# 使用 lda_model.get_document_topics() 確保輸出是稀疏格式的 (topic_id, probability) 元組列表
# 必須傳入 corpus 參數作為輸入。
doc_topics = [
    lda_model.get_document_topics(
        doc,
        minimum_probability=0.0
    )
    for doc in corpus
]
# ----------------------------------------------------------------------

# 2. 格式化為 DataFrame 結構 (保持不變)
# 我們需要一個函數來將 (Topic_ID, Probability) 列表轉換為固定長度的概率列表
def format_topic_distribution(topic_list, num_topics):
    """將主題概率列表轉換為固定長度（1到N）的概率向量"""

    # 創建一個長度為 num_topics 的零向量
    prob_vector = np.zeros(num_topics)

    # 將主題列表中的概率填入對應的位置
    for topic_id, prob in topic_list:
        if topic_id < num_topics:
            prob_vector[topic_id] = prob

    return prob_vector.tolist()


# 3. 應用格式化函數
# 這裡應該可以成功執行，因為 doc_topics 已經是預期的 (id, prob) 列表
topic_distributions = [format_topic_distribution(doc, NUM_TOPICS) for doc in doc_topics]

# 4. 創建新的 DataFrame 欄位名稱 (保持不變)
topic_cols = [f'Topic_{i + 1}_Prob' for i in range(NUM_TOPICS)]

# 5. 將主題概率添加到你的原始 DataFrame (df) (保持不變)
df_topic_probs = pd.DataFrame(topic_distributions, columns=topic_cols)

# 確保 df 和 df_topic_probs 的長度一致
# ⚠️ 由於你的原始代碼中 df 的來源和 'corpus' 的處理沒有完全顯示，
# 這裡我們假設它們是同步的。
df = pd.concat([df.reset_index(drop=True), df_topic_probs], axis=1)

print(f"✅ 已成功為 DataFrame 新增 {NUM_TOPICS} 個主題概率欄位。")
print("\n--- 帶有主題概率的 DataFrame 前 5 行 ---")
print(df[topic_cols].head())
# ------------------------------------------------------------------
# 【修正：處理 NaN 值以避免 FutureWarning】
# ------------------------------------------------------------------

# 1. 定義主題概率欄位
topic_cols = [f'Topic_{i + 1}_Prob' for i in range(NUM_TOPICS)]

# 2. 【核心修正】在計算最大值之前，將所有 NaN 替換為 0.0
# 這能確保 idxmax 總能找到一個最大值 (即使它是 0.0)
df[topic_cols] = df[topic_cols].fillna(0.0)

# 3. 找出每行 (每首歌) 的最大概率值和主導主題
# 修正後的代碼將不再觸發 FutureWarning
df['Dominant_Topic_Prob'] = df[topic_cols].max(axis=1)  # 最大概率值
df['Dominant_Topic'] = df[topic_cols].idxmax(axis=1)  # 最大概率值所在欄位名稱 (例如 'Topic_2_Prob')

# 4. 清理欄位名稱 (移除'_Prob' 和 'Topic_'，並轉換為整數)
# 此步驟實現您想要的結果：欄位只保留 1 到 10 的數字
df['Dominant_Topic_index'] = (
    df['Dominant_Topic']
    .str.replace('_Prob', '') # 移除 '_Prob' -> 'Topic_X'
    .str.replace('Topic_', '') # 移除 'Topic_' -> 'X'
    .astype(int)              # 轉換為整數
)

df['Dominant_Topic'] = df['Dominant_Topic'].str.replace('_Prob', '') # 移除 '_Prob' -> 'Topic_X'

# ------------------------------------------------------------------
# 【修正/新增：找出 Top 3 主導主題及其索引】
# ------------------------------------------------------------------

# 1. 定義主題概率欄位
topic_cols = [f'Topic_{i + 1}_Prob' for i in range(NUM_TOPICS)]

# 2. 【核心修正】在計算最大值之前，將所有 NaN 替換為 0.0
df[topic_cols] = df[topic_cols].fillna(0.0)


# 3. 找出 Top 3 主題及其概率和索引
def get_top_n_topics(row, n=3):
    """從概率欄位中找出概率最高的前 N 個主題的名稱、值和索引。"""
    # 選擇所有主題概率欄位，並將結果排序
    sorted_probs = row[topic_cols].sort_values(ascending=False).head(n)

    results = {}
    for rank in range(n):
        topic_key = f'Top{rank + 1}'

        if rank < len(sorted_probs):
            # 獲取第 rank+1 名的主題欄位名稱 (e.g., 'Topic_X_Prob')
            topic_col_name = sorted_probs.index[rank]

            # 獲取主題概率值
            prob_value = sorted_probs.iloc[rank]

            # 提取純主題名稱 (e.g., 'Topic_X')
            topic_name = topic_col_name.replace('_Prob', '')

            # 提取主題索引 (e.g., X)
            topic_index = int(topic_name.replace('Topic_', ''))

            results[f'{topic_key}_Topic'] = topic_name
            results[f'{topic_key}_Prob'] = prob_value
            results[f'{topic_key}_Topic_Index'] = topic_index  # <-- 新增索引欄位
        else:
            # 如果主題數少於 N，則填入預設值
            results[f'{topic_key}_Topic'] = 'N/A'
            results[f'{topic_key}_Prob'] = 0.0
            results[f'{topic_key}_Topic_Index'] = 0  # <-- N/A 索引設為 0

    return pd.Series(results)


# 應用此函數到 DataFrame 的每一行
df_top_topics = df.apply(get_top_n_topics, axis=1)

# 將 Top 3 結果與原始 DataFrame 合併
# 注意：這裡我們使用 errors='ignore' 來安全地刪除舊的 Dominant 欄位
df = pd.concat([df.drop(columns=['Dominant_Topic', 'Dominant_Topic_Prob', 'Dominant_Topic_index'], errors='ignore'),
                df_top_topics], axis=1)

# 將 Top 1 視為 Dominant Topic (與舊欄位保持一致)
df['Dominant_Topic'] = df['Top1_Topic']
df['Dominant_Topic_Prob'] = df['Top1_Prob']
df['Dominant_Topic_index'] = df['Top1_Topic_Index']  # <-- 直接使用 Top1_Topic_Index


# ------------------------------------------------------------------
# ------------------------------------------------------------------
# 【新增替換：最代表性文檔檢視】
# ... (display_representative_documents 函式保持不變) ...

def display_representative_documents(df, num_topics, top_n=5):
    """
    對每個主題，找出概率最高的 Top N 首歌 (即最能代表該主題的文檔)。
    現在使用 Top1_Topic 欄位進行篩選。
    """
    print("\n==============================================")
    print("👑 最代表性文檔檢視 (Top 5 歌曲/文檔)")
    print("==============================================")

    for i in range(1, num_topics + 1):
        topic_name = f'Topic_{i}'

        # 篩選出以當前主題為主導主題的歌曲 (使用 Top1_Topic)
        topic_subset = df[df['Top1_Topic'] == topic_name]

        if topic_subset.empty:
            print(f"主題 #{i} ({topic_name})：沒有主導歌曲。")
            continue

        # 根據 Top1_Prob 降序排序，選出 Top N
        top_documents = topic_subset.sort_values(
            by='Top1_Prob', # 這裡使用 Top1_Prob
            ascending=False
        ).head(top_n)

        print(f"\n--- 主題 #{i} ({topic_name}) ---")

        # 打印 Top N 歌曲資訊
        for index, row in top_documents.iterrows():
            prob = row['Top1_Prob'] # 使用 Top1_Prob
            artist = row.get('recording_artist_credit', 'N/A')
            title = row.get('recording_title', 'N/A')

            print(f"[{prob:.4f}] {artist} - 《{title}》")

# 調用新的檢視函數
display_representative_documents(df, NUM_TOPICS, top_n=5)

# ------------------------------------------------------------------
# 最終打印 (可選，作為一個總結)
# ------------------------------------------------------------------
print("\n--- 歌曲主導主題歸類結果 (前 5 筆) ---")
# 確保使用 df 中的實際欄位名
print(df[['recording_artist_credit', 'recording_title', 'Dominant_Topic', 'Dominant_Topic_Prob']].head())
#import pyLDAvis.gensim_models
#import pyLDAvis

# 刪除 pyLDAvis.enable_notebook()，因為您在非 Notebook 環境下運行
# pyLDAvis.enable_notebook()

# 準備可視化數據
#data = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary)

# 保存為 HTML 文件
# 文件名將是 "5_topic_model.html" (如果 num_topics=5)
#pyLDAvis.save_html(data, f"./{NUM_TOPICS}_topic_model.html")

#print(f"✅ 可視化圖表已成功保存為：{NUM_TOPICS}_topic_model.html")
#print("請在您的瀏覽器中打開此文件來查看結果。")



from wordcloud import WordCloud
import matplotlib.pyplot as plt


def generate_wordcloud(counts, font_path, filename):
    """根據詞頻字典產生文字雲並儲存"""
    if not counts:
        print(f"沒有足夠的詞彙來產生 {filename}。")
        return

    # 設置 WordCloud 參數
    wc = WordCloud(
        font_path=font_path,  # 使用韓文字型
        width=1000,
        height=600,
        background_color='white',
        max_words=200,
        prefer_horizontal=0.9  # 盡量讓文字水平顯示
    ).generate_from_frequencies(counts)

    plt.figure(figsize=(12, 8))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title(filename.replace(".png", ""), fontsize=20)
    plt.savefig(filename)
    print(f"文字雲已儲存至: {filename}")
    plt.close()


# --------------------------------------------------------------------
# 執行文字雲生成
# --------------------------------------------------------------------

print("\n==============================================")
print("☁️ 主題詞彙文字雲生成")
print("==============================================")

# 獲取所有主題的 Top 詞彙和權重 (這裡使用你前面提取的數據)
# 這裡我們使用 Top 50 詞彙以獲得更豐富的文字雲
topics_and_probs = lda_model.show_topics(
    num_topics=NUM_TOPICS,
    num_words=100,  # 增加詞彙量以豐富視覺效果
    formatted=False
)

for topic_id, word_probs in topics_and_probs:
    # 將 word_probs (List of Tuples) 轉換為 WordCloud 需要的字典格式 {word: probability}
    word_freq_dict = dict(word_probs)

    filename = f"Topic_{topic_id + 1}_Wordcloud.png"

    # 調用文字雲生成函式
    generate_wordcloud(
        counts=word_freq_dict,
        font_path=FONT_PATH,  # 確保這裡使用你定義的 FONT_PATH
        filename=filename
    )


#output_file = f"LDA_topic{NUM_TOPICS}_{company}.csv"
output_file = f"LDA_topic{NUM_TOPICS}_kpop.csv"
df.to_csv(output_file, index=False, encoding='utf-8-sig')

import os

# 定義保存檔案路徑
MODEL_DIR = "lda_model_assets"
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

lda_model_path = os.path.join(MODEL_DIR, f"lda_kpop_{NUM_TOPICS}_topics.model")
dictionary_path = os.path.join(MODEL_DIR, f"lda_kpop_{NUM_TOPICS}_dictionary.dict")

# 1. 保存 LDA 模型 (使用 Gensim 內建的 save 方法)
lda_model.save(lda_model_path)

# 2. 保存詞典 (這是將新文檔轉換為 BoW 格式所必需的)
dictionary.save(dictionary_path)

print(f"\n✅ LDA 模型已保存至：{lda_model_path}")
print(f"✅ 詞典已保存至：{dictionary_path}")