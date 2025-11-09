import matplotlib.pyplot as plt
from gensim.models.coherencemodel import CoherenceModel
import pandas as pd
from gensim import corpora
from gensim.models import LdaModel
import ast # 引入 Abstract Syntax Tree 模組
from tqdm import tqdm


def convert_str_to_list(list_str):
    try:
        # ast.literal_eval 比 eval() 更安全，專門用於評估字串中的基本數據結構
        return ast.literal_eval(list_str)
    except (ValueError, TypeError):
        # 如果遇到 NaN 或無法評估的字串，返回空列表
        return []

def compute_coherence_values(dictionary, corpus, texts, topic_range):
    """
    計算給定主題數量範圍下的 Coherence Score
    """
    coherence_values = []

    # 使用 tqdm 顯示進度條
    for num_topics in tqdm(topic_range, desc="計算 Coherence Score"):
        # 核心修正：將 passes 降至最低安全值
        # 降低迭代次數，以極大加速單次模型訓練
        MIN_PASSES = 20

        # 核心修正：使用較小的 chunksize 減少記憶體壓力
        MIN_CHUNKSIZE = 100
        # 訓練 LDA 模型 (使用與之前相同的基礎參數)
        lda_model = LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=num_topics,
            random_state=42,
            chunksize=MIN_CHUNKSIZE,
            passes=MIN_PASSES,
            alpha='auto'
        )

        # 計算 C_v Coherence Score
        coherence_model = CoherenceModel(
            model=lda_model,
            texts=texts,
            dictionary=dictionary,
            coherence='c_v'
            #topn=10
            #workers=1

        )
        coherence_values.append(coherence_model.get_coherence())

    return coherence_values



if __name__ == '__main__':
    # 資料讀取和預處理
    input_file = "merged_lyrics_with_labels.csv"
    df = pd.read_csv(input_file)
    # 應用轉換，這將是你的新最終詞彙欄位
    df['final_tokens_restored'] = df['final_tokens'].apply(convert_str_to_list)
    documents = df['final_tokens_restored'].tolist()

    # 移除空文檔（安全操作）
    documents = [doc for doc in documents if doc]
    # ====================================================================
    # 數據驗證與安全檢查 (避免 ValueError: cannot compute LDA over an empty collection)
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
    # 建立詞典 (Dictionary) 和語料庫 (Corpus)
    # ====================================================================

    print("\n開始建立詞典...")
    # 使用所有文檔建立詞典
    dictionary = corpora.Dictionary(documents)

    # 詞彙過濾：使用最寬鬆的條件來避免丟失核心詞
    dictionary.filter_extremes(
        no_below=2,  # 詞彙至少在 2 首歌中出現過
        no_above=0.99,  # 詞彙只有在超過 99% 的歌中出現才移除
        keep_n=None
    )

    print(f"✅ 詞彙表大小 (過濾後): {len(dictionary)}")

    # 建立 BoW 語料庫 (將詞彙轉換為 (ID, Count) 格式)
    corpus = [dictionary.doc2bow(doc) for doc in documents]
    print(f"✅ 語料庫文檔數: {len(corpus)}")

    # 3. 定義主題範圍
    min_topics = 2
    max_topics = 15
    step = 1
    topic_range = range(min_topics, max_topics + 1, step)

    # 4. 執行 Coherence Score 計算 (報錯的程式碼行現在被保護了)
    coherence_scores = compute_coherence_values(
        dictionary=dictionary,
        corpus=corpus,
        texts=documents,
        topic_range=topic_range
    )
    # --------------------------------------------------------------------
    # 繪圖和選擇最佳主題數量
    # --------------------------------------------------------------------

    # 尋找 Coherence Score 最高的點
    max_score = max(coherence_scores)
    optimal_topic_index = coherence_scores.index(max_score)
    optimal_num_topics = topic_range[optimal_topic_index]

    # 繪製圖表
    plt.figure(figsize=(10, 6))
    plt.plot(topic_range, coherence_scores, marker='o', linestyle='-', color='skyblue')

    # 標記最佳主題數量
    plt.scatter(optimal_num_topics, max_score, color='red', s=100,
                label=f'Best Number of Topics: {optimal_num_topics} (Score: {max_score:.4f})')
    plt.axvline(x=optimal_num_topics, color='r', linestyle='--', linewidth=0.8)

    # 設定圖表標籤
    plt.title("LDA Coherence Score", fontsize=16)
    plt.xlabel("Number of Topics", fontsize=12)
    plt.ylabel("Coherence Score ($C_v$)", fontsize=12)
    plt.xticks(topic_range)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()

    # 顯示結果
    print("\n--- 主題連貫性分數結果 ---")
    for num_topics, score in zip(topic_range, coherence_scores):
        print(f"主題數量 {num_topics}: Score = {score:.4f}")

    print(f"\n✨ 推薦的最佳主題數量是: {optimal_num_topics} (Coherence Score: {max_score:.4f})")

    plt.show()
