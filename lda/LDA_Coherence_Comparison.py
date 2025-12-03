import matplotlib.pyplot as plt
from gensim.models.coherencemodel import CoherenceModel
import pandas as pd
from gensim import corpora
from gensim.models import LdaModel
import ast  # 引入 Abstract Syntax Tree 模組
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
            # topn=10
            # workers=1

        )
        coherence_values.append(coherence_model.get_coherence())

    return coherence_values


# 【新增設定】定義要比較的公司列表
COMPANIES_TO_ANALYZE = ['HYBE', 'JYP', 'YG', 'SM']

if __name__ == '__main__':
    # 資料讀取和預處理
    input_file = "merged_lyrics_with_labels.csv"
    df = pd.read_csv(input_file)
    # 應用轉換，這將是你的新最終詞彙欄位
    df['final_tokens_restored'] = df['final_tokens'].apply(convert_str_to_list)
    documents = df['final_tokens_restored'].tolist()

    # --------------------------------------------------------------------
    # 初始化多公司分析結果字典
    # --------------------------------------------------------------------
    all_results = {}

    # 設置顏色和樣式
    plot_styles = {
        'HYBE': {'color': 'red', 'label': 'HYBE', 'marker': 'o'},
        'JYP': {'color': 'blue', 'label': 'JYP', 'marker': 's'},
        'YG': {'color': 'green', 'label': 'YG', 'marker': '^'},
        'SM': {'color': 'yellow', 'label': 'SM', 'marker': '*'},
        # 可以新增更多公司
    }

    # --------------------------------------------------------------------
    # 循環處理每個公司
    # --------------------------------------------------------------------
    for company_name in COMPANIES_TO_ANALYZE:
        print("\n" + "=" * 60)
        print(f"🔬 正在分析公司子集：{company_name}")
        print("=" * 60)

        # 篩選特定公司的文檔
        df_subset = df[df['label name'] == company_name]
        documents = df_subset['final_tokens_restored'].tolist()

        # 移除空文檔（安全操作）
        documents = [doc for doc in documents if doc]

        # 數據驗證
        total_docs = len(documents)
        total_tokens = sum(len(doc) for doc in documents)

        if total_docs < 20 or total_tokens == 0:
            print(f"🚨 警告：{company_name} 的數據量不足 ({total_docs} 首歌)。跳過分析。")
            continue

        # 建立詞典和語料庫
        dictionary = corpora.Dictionary(documents)
        dictionary.filter_extremes(no_below=2, no_above=0.99, keep_n=None)
        corpus = [dictionary.doc2bow(doc) for doc in documents]

        print(f"-> 文檔數: {len(corpus)}, 詞彙表大小: {len(dictionary)}")

        # 3. 定義主題範圍 (與總體分析保持一致)
        min_topics = 2
        max_topics = 15
        step = 1
        topic_range = range(min_topics, max_topics + 1, step)

        # 4. 執行 Coherence Score 計算
        coherence_scores = compute_coherence_values(
            dictionary=dictionary,
            corpus=corpus,
            texts=documents,  # 注意：texts 必須是分好詞的列表，即 documents
            topic_range=topic_range
        )

        # 儲存結果
        all_results[company_name] = coherence_scores

        # 輸出單公司結果
        max_score = max(coherence_scores)
        optimal_num_topics = topic_range[coherence_scores.index(max_score)]
        print(f"\n✨ {company_name} 最佳主題數是: {optimal_num_topics} (Score: {max_score:.4f})")

    # --------------------------------------------------------------------
    # 繪製多公司比較折線圖
    # --------------------------------------------------------------------

    print("\n--- 繪製 Coherence Score 比較圖 ---")

    plt.figure(figsize=(12, 7))

    for company_name, scores in all_results.items():
        style = plot_styles.get(company_name, {'color': 'black', 'label': company_name, 'marker': 'x'})

        plt.plot(topic_range, scores,
                 marker=style['marker'],
                 linestyle='-',
                 color=style['color'],
                 label=f"{style['label']} - Max Score: {max(scores):.4f}")

        # 標記每個公司的最高點
        max_score = max(scores)
        optimal_num_topics = topic_range[scores.index(max_score)]
        plt.scatter(optimal_num_topics, max_score,
                    color=style['color'],
                    s=100,
                    alpha=0.6)

    # 設定圖表標籤
    plt.title("LDA Coherence Score Comparison chart (different entertainment companies)", fontsize=16)
    plt.xlabel("Number of Topics", fontsize=12)
    plt.ylabel("Coherence Score ($C_v$)", fontsize=12)
    plt.xticks(topic_range)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(title="companies", loc='upper right')
    plt.tight_layout()
    plt.savefig('Company_Coherence_Comparison.png')
    print("✅ 比較圖已儲存為 Company_Coherence_Comparison.png")
    plt.show()