import pandas as pd
import ast
import numpy as np
import os
import re
import requests
from collections import Counter
from scipy.stats import entropy
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from lexical_diversity import lex_div as ld
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import pairwise_distances

# --- 使用者設定區 ---
INPUT_FILE = "merged_lyrics_with_labels.csv"
TOKEN_COLUMN = "final_tokens"
LABEL_COLUMN = "label name"
DATE_COLUMN = "release_date"
ARTIST_COLUMN = "recording_artist_credit"
TITLE_COLUMN = "recording_title"
CATEGORY_COLUMN = "singer_category"

OUTPUT_DIR = "final_analysis_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 1. 字型設定 ---
FONT_FILENAME = "NanumGothic.ttf"
FONT_URL = "https://github.com/google/fonts/raw/main/ofl/nanumgothic/NanumGothic-Regular.ttf"

def setup_font():
    print("正在設定字型 (NanumGothic)...")
    if not os.path.exists(FONT_FILENAME):
        print(f"下載 {FONT_FILENAME}...")
        try:
            r = requests.get(FONT_URL, allow_redirects=True)
            with open(FONT_FILENAME, "wb") as f:
                f.write(r.content)
            print("下載完成。")
        except Exception as e:
            print(f"下載失敗: {e}")
            return None

    try:
        fm.fontManager.addfont(FONT_FILENAME)
        font_prop = fm.FontProperties(fname=FONT_FILENAME)
        font_name = font_prop.get_name()
        
        plt.rcParams['font.family'] = font_name
        plt.rcParams['font.sans-serif'] = [font_name]
        plt.rcParams['axes.unicode_minus'] = False
        sns.set_theme(style="whitegrid", font=font_name, rc={"axes.unicode_minus": False})
        
        return font_name
    except Exception as e:
        print(f"字型設定錯誤: {e}")
        return None

# --- 輔助函數 ---

def safe_literal_eval(val):
    if pd.isna(val) or not isinstance(val, str): return []
    try: return ast.literal_eval(val)
    except: return []

def map_company_name(label_name):
    if pd.isna(label_name): return "Other"
    for co in ['JYP', 'YG', 'SM', 'HYBE']:
        if co in label_name: return co
    return "Other"

def clean_artist_name(name):
    if pd.isna(name): return "Unknown"
    if "TOMORROW X TOGETHER" in name.upper(): return "TXT"
    name = re.split(r'\s+(feat\.|with|x|&|,|\()\s+', name, flags=re.IGNORECASE)[0]
    return name.strip()

def calculate_ttr(token_list):
    if not token_list: return 0
    return len(set(token_list)) / len(token_list)

def calculate_entropy(token_list):
    if not token_list: return 0
    counts = Counter(token_list)
    probs = [c/len(token_list) for c in counts.values()]
    return entropy(probs, base=2)

def dummy_tokenizer(tokens):
    return tokens

# --- 離群值偵測 ---

def detect_and_save_outliers(df, group_col, metrics):
    print(f"\n--- 正在偵測 {group_col} 的離群值 ---")
    outlier_records = []
    groups = df[group_col].dropna().unique()
    
    for g in groups:
        group_df = df[df[group_col] == g]
        for m in metrics:
            Q1 = group_df[m].quantile(0.25)
            Q3 = group_df[m].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            
            outliers = group_df[(group_df[m] < lower) | (group_df[m] > upper)]
            
            for _, row in outliers.iterrows():
                outlier_records.append({
                    'Group_Type': group_col,
                    'Group_Name': g,
                    'Metric': m,
                    'Value': row[m],
                    'Artist': row['main_artist'],
                    'Title': row.get(TITLE_COLUMN, 'Unknown'),
                    'Year': row['year']
                })

    if outlier_records:
        outlier_df = pd.DataFrame(outlier_records)
        save_path = f"{OUTPUT_DIR}/outliers_{group_col}.csv"
        outlier_df.to_csv(save_path, index=False, encoding='utf-8-sig')
        print(f"--> 離群值已存至: {save_path}")

# --- 核心計算函數 ---

def print_top5_pairs(sim_matrix, metric_name):
    stacked = sim_matrix.stack()
    stacked = stacked[stacked.index.get_level_values(0) != stacked.index.get_level_values(1)]
    unique_pairs = {}
    for (idx1, idx2), val in stacked.items():
        pair = tuple(sorted((idx1, idx2)))
        if pair not in unique_pairs:
            unique_pairs[pair] = val
    sorted_pairs = sorted(unique_pairs.items(), key=lambda item: item[1], reverse=True)[:5]
    print(f"\n>>> Top 5 {metric_name} Pairs <<<")
    for (g1, g2), score in sorted_pairs:
        print(f"  {g1} <-> {g2} : {score:.4f}")
    print("-----------------------------------")

def calculate_group_jaccard(df, group_col, output_name, annot=True):
    print(f"計算 Jaccard Similarity ({group_col})...")
    if df[group_col].dtype.name == 'category':
        groups = df[group_col].cat.categories.tolist()
    else:
        if group_col == 'main_artist':
             groups = df[group_col].unique().tolist()
        else:
             groups = sorted(df[group_col].dropna().unique())
    group_vocabs = {}
    for g in groups:
        tokens = [t for sublist in df[df[group_col] == g]['tokens'] for t in sublist]
        group_vocabs[g] = set(tokens)
    sim_matrix = pd.DataFrame(index=groups, columns=groups, dtype=float)
    for g1 in groups:
        for g2 in groups:
            v1 = group_vocabs[g1]
            v2 = group_vocabs[g2]
            if not v1 or not v2: sim = 0
            else: sim = len(v1 & v2) / len(v1 | v2)
            sim_matrix.loc[g1, g2] = sim
    print_top5_pairs(sim_matrix, f"Jaccard ({group_col})")
    
    if group_col == 'year':
        plt.figure(figsize=(12, 10))
        sns.heatmap(sim_matrix, annot=False, cmap='Blues')
    elif group_col == 'main_artist':
        plt.figure(figsize=(14, 12))
        sns.heatmap(sim_matrix, annot=True, cmap='Blues', fmt='.2f')
    else:
        plt.figure(figsize=(10, 8))
        sns.heatmap(sim_matrix, annot=annot, cmap='Blues', fmt='.3f')
    plt.title(f'Jaccard Similarity by {group_col}')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{output_name}.png")
    plt.close()

def calculate_yue_similarity_heatmap(df, group_col, output_name, annot=True):
    print(f"計算 Yue Similarity Heatmap ({group_col})...")
    if df[group_col].dtype.name == 'category':
        groups = df[group_col].cat.categories.tolist()
    else:
        if group_col == 'main_artist':
             groups = df[group_col].unique().tolist()
        else:
             groups = sorted(df[group_col].dropna().unique())
    group_probs = {}
    for g in groups:
        tokens = [t for sublist in df[df[group_col] == g]['tokens'] for t in sublist]
        if not tokens:
            group_probs[g] = {}
            continue
        counts = Counter(tokens)
        total = len(tokens)
        probs = {k: v / total for k, v in counts.items()}
        group_probs[g] = probs
    sim_matrix = pd.DataFrame(index=groups, columns=groups, dtype=float)
    for g1 in groups:
        for g2 in groups:
            p1 = group_probs[g1]
            p2 = group_probs[g2]
            numerator = 0
            sum_p1_sq = sum(p ** 2 for p in p1.values())
            sum_p2_sq = sum(p ** 2 for p in p2.values())
            common_vocab = set(p1.keys()) & set(p2.keys())
            for word in common_vocab:
                numerator += p1[word] * p2[word]
            denominator = sum_p1_sq + sum_p2_sq - numerator
            if denominator == 0: sim = 0
            else: sim = numerator / denominator
            sim_matrix.loc[g1, g2] = sim
    print_top5_pairs(sim_matrix, f"Yue Index ({group_col})")
    if group_col == 'year':
        plt.figure(figsize=(12, 10))
        sns.heatmap(sim_matrix, annot=False, cmap='Purples')
    elif group_col == 'main_artist':
        plt.figure(figsize=(14, 12))
        sns.heatmap(sim_matrix, annot=False, cmap='Purples')
    else:
        plt.figure(figsize=(10, 8))
        sns.heatmap(sim_matrix, annot=annot, cmap='Purples', fmt='.3f')
    plt.title(f'Yue Similarity (Weighted) by {group_col}')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{output_name}.png")
    plt.close()

# --- 繪圖函數 (修正版 - 對數直方圖) ---

def plot_distribution(df, group_col, metric_col, title, filename, plot_type='box'):
    """
    繪製分佈圖 (修正版)
    - Hist: 使用 log_scale=True 來處理右偏分佈，取代原本的 xlim 限制
    - Box: 保留 MTLD 的 ylim 限制
    """
    plt.figure(figsize=(12, 7))
    
    if plot_type == 'box':
        sns.boxplot(data=df, x=group_col, y=metric_col, palette='pastel')
        plt.xticks(rotation=45, ha='right')
        
        # 僅針對 Box Plot 保留 MTLD 的 Y 軸截斷
        if metric_col == 'mtld':
            plt.ylim(0, 100)
            
    elif plot_type == 'hist':
        # 【修改】 使用 Log Scale 直方圖 (取代硬性截斷)
        
        # 為了避免 log(0) 錯誤，只選取大於 0 的數據
        plot_data = df[df[metric_col] > 0]
        
        sns.histplot(
            data=plot_data, 
            x=metric_col, 
            hue=group_col, 
            kde=True, 
            element="step", 
            common_norm=False,
            log_scale=True # 【關鍵】開啟對數尺度
        )
        plt.ylabel("Song Count (Freq)")
        plt.xlabel(f"{metric_col} (Log Scale)")
        
        # 移除原本的 xlim 限制，因為 Log Scale 會自動處理長尾
        
    plt.title(title)
    if plot_type == 'box':
        plt.xlabel(group_col)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{filename}")
    plt.close()

def plot_artist_kde_mtld(df, group_col, metric_col, title, filename):
    """
    Top 20 藝人的 KDE 圖也改用 Log Scale
    """
    plt.figure(figsize=(14, 8))
    
    # 確保資料大於 0
    plot_data = df[df[metric_col] > 0]
    
    sns.kdeplot(
        data=plot_data, 
        x=metric_col, 
        hue=group_col, 
        common_norm=False, 
        palette='tab20', 
        linewidth=2,
        log_scale=True # 【關鍵】開啟對數尺度
    )
    plt.title(title)
    plt.xlabel(f"{metric_col} (Log Scale)")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{filename}")
    plt.close()

def plot_bar(series, title, xlabel, ylabel, filename):
    plt.figure(figsize=(10, 6))
    series.plot(kind='bar', color='skyblue')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45)
    for i, v in enumerate(series):
        plt.text(i, v, f"{v:.4f}", ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{filename}")
    plt.close()

def calculate_yue_novelty_bar(df, group_col):
    print(f"計算 Yue Novelty (Bar Chart) ({group_col})...")
    groups = sorted(df[group_col].dropna().unique())
    results = {}
    for g in groups:
        sub_df = df[df[group_col] == g]
        token_lists = sub_df['tokens'].tolist()
        if len(token_lists) < 2:
            results[g] = 0; continue
        vectorizer = TfidfVectorizer(tokenizer=dummy_tokenizer, preprocessor=dummy_tokenizer, token_pattern=None, binary=True)
        try:
            tfidf_matrix = vectorizer.fit_transform(token_lists)
            jaccard_dist = pairwise_distances(tfidf_matrix.toarray(), metric='jaccard')
            jaccard_sim = 1 - jaccard_dist
            np.fill_diagonal(jaccard_sim, 0)
            max_sim = np.max(jaccard_sim, axis=1)
            results[g] = np.mean(1 - max_sim)
        except: results[g] = 0
    return pd.Series(results)

# --- 主程式 ---

def main():
    setup_font()
    
    print(f"正在讀取 {INPUT_FILE} ...")
    df = pd.read_csv(INPUT_FILE)
    
    print("資料清理中...")
    df['tokens'] = df[TOKEN_COLUMN].apply(safe_literal_eval)
    df['company'] = df[LABEL_COLUMN].apply(map_company_name)
    df = df[df['company'] != 'Other'] 
    
    df[DATE_COLUMN] = df[DATE_COLUMN].astype(str)
    df['year'] = df[DATE_COLUMN].str.extract(r'^(\d{4})').astype(float)
    df = df.dropna(subset=['year', 'tokens'])
    df['year'] = df['year'].astype(int)
    
    bins = [2000, 2005, 2010, 2015, 2020, 2025, 2030]
    labels = ['2000-2004', '2005-2009', '2010-2014', '2015-2019', '2020-2024', '2025+']
    df['period'] = pd.cut(df['year'], bins=bins, labels=labels, right=False)
    
    if ARTIST_COLUMN in df.columns:
        df['main_artist'] = df[ARTIST_COLUMN].apply(clean_artist_name)
    else:
        df['main_artist'] = "Unknown"
        
    if CATEGORY_COLUMN in df.columns:
        df['category'] = df[CATEGORY_COLUMN].astype(str).str.lower().str.strip()
        df = df[df['category'].isin(['solo', 'group'])]
    else:
        df['category'] = None

    print("計算 TTR, Entropy, MTLD...")
    df['length'] = df['tokens'].apply(len)
    df['ttr'] = df['tokens'].apply(calculate_ttr)
    df['entropy'] = df['tokens'].apply(calculate_entropy)
    df['mtld'] = df['tokens'].apply(lambda t: ld.mtld(t) if len(t) > 10 else 0)

    metrics = ['ttr', 'entropy', 'mtld']

    # --- 分析 1: 公司 (Company) ---
    print("\n--- [分析 1] 公司維度 ---")
    for m in metrics:
        plot_distribution(df, 'company', m, f'{m.upper()} Distribution by Company', f'company_boxplot_{m}.png', 'box')
        # Histogram 使用 Log Scale
        plot_distribution(df, 'company', m, f'{m.upper()} Distribution by Company (Log Scale)', f'company_hist_{m}.png', 'hist')
    
    detect_and_save_outliers(df, 'company', metrics)
    calculate_group_jaccard(df, 'company', 'company_jaccard_heatmap')
    calculate_yue_similarity_heatmap(df, 'company', 'company_yue_similarity_heatmap.png')
    yue_company_bar = calculate_yue_novelty_bar(df, 'company')
    plot_bar(yue_company_bar, 'Novelty (Yue Index) by Company', 'Company', 'Novelty', 'company_yue_novelty_bar.png')

    # --- 分析 2: 年代 (Year & Period) ---
    print("\n--- [分析 2] 年代維度 ---")
    yearly_avg = df.groupby(['year', 'company'])[metrics].mean().reset_index()
    for m in metrics:
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=yearly_avg, x='year', y=m, hue='company', marker='o')
        plt.title(f'Yearly Trend of {m.upper()}')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/year_trend_{m}.png")
        plt.close()
        
    calculate_group_jaccard(df, 'year', 'year_jaccard_heatmap')
    calculate_group_jaccard(df, 'period', 'period_jaccard_heatmap')
    calculate_yue_similarity_heatmap(df, 'year', 'year_yue_similarity_heatmap.png')
    calculate_yue_similarity_heatmap(df, 'period', 'period_yue_similarity_heatmap.png')
    yue_year_bar = calculate_yue_novelty_bar(df, 'year')
    plt.figure(figsize=(12, 6))
    yue_year_bar.plot(kind='line', marker='o', color='purple')
    plt.title('Song Novelty (Yue Index) Over Time')
    plt.ylabel('Novelty')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/year_yue_novelty_trend.png")
    plt.close()

    # --- 分析 3: 類別 (Solo vs Group) ---
    if df['category'].notna().any():
        print("\n--- [分析 3] Solo vs Group ---")
        for m in metrics:
            plot_distribution(df, 'category', m, f'{m.upper()} by Category', f'category_boxplot_{m}.png', 'box')
            # Histogram 使用 Log Scale
            plot_distribution(df, 'category', m, f'{m.upper()} by Category (Log Scale)', f'category_hist_{m}.png', 'hist')
            
        detect_and_save_outliers(df, 'category', metrics)
        calculate_group_jaccard(df, 'category', 'category_jaccard_heatmap')
        calculate_yue_similarity_heatmap(df, 'category', 'category_yue_similarity_heatmap.png')
        yue_cat_bar = calculate_yue_novelty_bar(df, 'category')
        plot_bar(yue_cat_bar, 'Novelty by Category', 'Category', 'Novelty', 'category_yue_novelty_bar.png')

    # --- 分析 4: 藝人 (Artist) ---
    print("\n--- [分析 4] 藝人維度 ---")
    top_artists = df['main_artist'].value_counts().head(20).index.tolist()
    df_top20 = df[df['main_artist'].isin(top_artists)].copy()
    df_top20['main_artist'] = pd.Categorical(df_top20['main_artist'], categories=top_artists, ordered=True)
    df_top20 = df_top20.sort_values('main_artist')
    
    for m in metrics:
        plot_distribution(
            df_top20, 'main_artist', m, 
            f'{m.upper()} Distribution (Top 20 Artists)', 
            f'artist_top20_boxplot_{m}.png', 
            'box'
        )
        
        # KDE 使用 Log Scale
        plot_artist_kde_mtld(
            df_top20, 'main_artist', m, 
            f'{m.upper()} Distribution Density (Top 20 Artists, Log Scale)', 
            f'artist_top20_kde_{m}.png'
        )
        
    artist_year_counts = df_top20.pivot_table(index='main_artist', columns='year', values='tokens', aggfunc='count', fill_value=0)
    plt.figure(figsize=(20, 10))
    sns.heatmap(artist_year_counts, cmap="YlGnBu", annot=True, fmt='d', cbar_kws={'label': 'Song Count'})
    plt.title('Top 20 Artists Activity (Song Count per Year)')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/artist_activity_heatmap.png")
    plt.close()
    
    calculate_group_jaccard(df_top20, 'main_artist', 'artist_top20_jaccard_heatmap')
    calculate_yue_similarity_heatmap(df_top20, 'main_artist', 'artist_top20_yue_similarity_heatmap', annot=False)

    print(f"\n所有分析完成！請查看 '{OUTPUT_DIR}' 資料夾。")

if __name__ == "__main__":
    main()