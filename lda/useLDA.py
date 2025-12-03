import os
import gensim
import pandas as pd
import ast # 用於安全轉換字串列表
import numpy as np
from tqdm import tqdm # 用於顯示處理進度

# ====================================================================
# 1. 模型載入和配置
# ====================================================================

# 配置：請確認這些路徑和參數與您訓練時完全一致
BASE_DIR = 'C:\\Users\\zac\\PyCharmMiscProject'  # 使用雙反斜線
MODEL_DIR = os.path.join(BASE_DIR, "lda_model_assets")
NUM_TOPICS = 10

lda_model_path = os.path.join(MODEL_DIR, f"lda_kpop_{NUM_TOPICS}_topics.model")
dictionary_path = os.path.join(MODEL_DIR, f"lda_kpop_{NUM_TOPICS}_dictionary.dict")

try:
    # 載入模型和詞典
    loaded_lda_model = gensim.models.LdaModel.load(lda_model_path)
    loaded_dictionary = gensim.corpora.Dictionary.load(dictionary_path)
    print(f"✅ 成功載入 {NUM_TOPICS} 個主題的 LDA 模型和詞典。")

except FileNotFoundError:
    # 如果載入失敗，則退出程式
    print("🚨 致命錯誤：找不到模型或詞典檔案。請確認 BASE_DIR 設定是否正確。")
    exit()


# ====================================================================
# 2. 數據讀取和預處理
# ====================================================================

# 定義新數據檔案路徑
NEW_DATA_FILE = 'data/new_LDA.csv' # 使用原始字串或 os.path.join 處理 Windows 路徑

df_new = pd.read_csv(NEW_DATA_FILE)

# 定義安全轉換函數 (從您原來的代碼複製)
def convert_str_to_list(list_str):
    try:
        return ast.literal_eval(list_str)
    except (ValueError, TypeError):
        return []

# 應用轉換
df_new['final_tokens_restored'] = df_new['final_tokens'].apply(convert_str_to_list)

# 準備文檔列表並移除空文檔
documents_new = df_new['final_tokens_restored'].tolist()
documents_new_cleaned = [doc for doc in documents_new if doc]

# 由於我們只對非空文檔進行處理，我們需要一個索引來將結果重新映射回原始 df_new
non_empty_indices = [i for i, doc in enumerate(documents_new) if doc]

# 創建 BoW 語料庫
corpus_new = [loaded_dictionary.doc2bow(doc) for doc in tqdm(documents_new_cleaned, desc="轉換為 BoW 格式")]

print(f"✅ 讀取 {len(df_new)} 筆數據，其中 {len(corpus_new)} 筆有效文檔用於推斷。")

# ====================================================================
# 3. 主題推斷 (Inference)
# ====================================================================

# 運行主題推斷
print("\n開始進行主題推斷...")
doc_topics_inferred = [
    loaded_lda_model.get_document_topics(
        doc,
        minimum_probability=0.0
    )
    for doc in tqdm(corpus_new, desc="推斷主題概率")
]


# ====================================================================
# 4. 結果格式化和合併
# ====================================================================

# 將 (Topic_ID, Probability) 列表轉換為固定長度的概率向量 (從您原來的代碼複製)
def format_topic_distribution(topic_list, num_topics):
    prob_vector = np.zeros(num_topics)
    for topic_id, prob in topic_list:
        if topic_id < num_topics:
            prob_vector[topic_id] = prob
    return prob_vector.tolist()

# 應用格式化函數
topic_distributions = [format_topic_distribution(doc, NUM_TOPICS) for doc in doc_topics_inferred]

# 創建新的 DataFrame 欄位名稱
topic_cols = [f'Topic_{i + 1}_Prob' for i in range(NUM_TOPICS)]
df_topic_probs = pd.DataFrame(topic_distributions, columns=topic_cols)

# 創建一個新的完整 DataFrame 來保存推斷結果
# 首先建立一個空的 DataFrame，長度與原始 df_new 相同
df_results = df_new.copy()

# 初始化主題概率欄位為 0 (處理那些空文檔)
for col in topic_cols:
    df_results[col] = 0.0

# 將推斷的概率填入對應的非空行
# df_topic_probs 的行數 == documents_new_cleaned 的行數
df_results.loc[non_empty_indices, topic_cols] = df_topic_probs.values


# 5. 找出主導主題
df_results['Dominant_Topic_Prob'] = df_results[topic_cols].max(axis=1)
df_results['Dominant_Topic_index'] = df_results[topic_cols].idxmax(axis=1).str.replace('_Prob', '').str.replace('Topic_', '').astype(int)
df_results['Dominant_Topic'] = df_results['Dominant_Topic_index'].apply(lambda x: f'Topic_{x}')


# ====================================================================
# 5. 輸出結果
# ====================================================================

output_file_name = 'data_new_LDA_with_topics.csv'
df_results.to_csv(output_file_name, index=False, encoding='utf-8-sig')

print("\n--- 推斷結果範例 (前 5 筆) ---")
# 假設你的原始 data/new_LDA.csv 有 'title' 或 'song' 欄位
display_cols = ['Dominant_Topic', 'Dominant_Topic_Prob'] + topic_cols
print(df_results.head()[display_cols])

print(f"\n✨ 成功將主題結果輸出到檔案：{output_file_name}")