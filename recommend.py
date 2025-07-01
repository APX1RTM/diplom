import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
from nltk.corpus import stopwords
from pymystem3 import Mystem
import random
import os
from tqdm import tqdm
import pickle

mystem = Mystem()
stop_words = set(stopwords.words('russian'))

# 1. Загрузка и предобработка данных
def preprocess_data(file_path, cache_path='text_cache.pkl'):
    # Проверка кэша
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            df = pickle.load(f)
        print("Загружен кэшированный датасет")
        return df

    df = pd.read_csv(file_path)
    
    df = df.drop_duplicates(subset=['video_id']).reset_index(drop=True)
    df['tags'] = df['tags'].fillna(df['title'])
    df['text'] = df['title'] + ' ' + df['tags']
    
    def clean_text_batch(texts):
        lemmatized_texts = []
        batch_size = 1000
        for i in tqdm(range(0, len(texts), batch_size), desc="Лемматизация текстов"):
            batch = texts[i:i + batch_size]
            # Лемматизация каждой строки отдельно для точного соответствия
            for text in batch:
                words = mystem.lemmatize(text.lower())
                words = [word for word in words if word.isalnum() and word not in stop_words]
                lemmatized_texts.append(' '.join(words))
        return lemmatized_texts

    # Применяем пакетную обработку
    lemmatized_texts = clean_text_batch(df['text'].tolist())
    
    # Проверка длины
    if len(lemmatized_texts) != len(df):
        raise ValueError(f"Длина лемматизированных текстов ({len(lemmatized_texts)}) не совпадает с длиной df ({len(df)})")
    
    df['text'] = lemmatized_texts
    
    df = pd.get_dummies(df, columns=['categoryId'])
    
    df['norm_likes'] = df['likes'] / df['likes'].max()
    df['norm_views'] = df['view_count'] / df['view_count'].max()
    
    # Сохранение в кэш
    with open(cache_path, 'wb') as f:
        pickle.dump(df, f)
    
    return df

# 2. Генерация синтетических пользовательских данных
def generate_user_data(df, num_users=100, output_path='user_data.csv'):
    video_ids = df['video_id'].values
    user_data = []
    for user_id in range(num_users):
        num_interactions = random.randint(0, 100)  # Диапазон 0–100
        viewed_videos = random.sample(list(video_ids), num_interactions) if num_interactions > 0 else []
        for video_id in viewed_videos:
            rating = df[df['video_id'] == video_id]['norm_likes'].values[0]
            user_data.append({'user_id': user_id, 'video_id': video_id, 'rating': rating})
    user_data_df = pd.DataFrame(user_data)
    user_data_df.to_csv(output_path, index=False)
    return user_data_df

# 3. Контентная фильтрация
def content_based_filtering(df, user_data, user_id, top_n=5):
    user_videos = user_data[user_data['user_id'] == user_id]['video_id']
    valid_videos = user_videos[user_videos.isin(df['video_id'])]
    
    if len(valid_videos) == 0:
        return df.sort_values(by=["norm_views", "norm_likes"], ascending=False)['video_id'].values[:top_n]
    
    tfidf = TfidfVectorizer(max_features=1000)
    tfidf_matrix = tfidf.fit_transform(df['text'])
    
    user_indices = df[df['video_id'].isin(valid_videos)].index
    if len(user_indices) == 0:
        return df.sort_values(by=["norm_views", "norm_likes"], ascending=False)['video_id'].values[:top_n]
    
    user_profile = np.asarray(tfidf_matrix[user_indices].mean(axis=0))
    
    similarities = cosine_similarity(user_profile, tfidf_matrix)[0]
    top_indices = similarities.argsort()[-top_n:][::-1]
    return df.iloc[top_indices]['video_id'].values

# 4. Коллаборативная фильтрация (SVD)
def collaborative_filtering(user_data, df, user_id, top_n=5):
    user_data = user_data[user_data['video_id'].isin(df['video_id'])]
    if len(user_data[user_data['user_id'] == user_id]) == 0:
        return df.sort_values(by=["norm_views", "norm_likes"], ascending=False)['video_id'].values[:top_n]
    
    reader = Reader(rating_scale=(0, 1))
    data = Dataset.load_from_df(user_data[['user_id', 'video_id', 'rating']], reader)
    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
    
    algo = SVD(n_factors=20, n_epochs=20, random_state=42)
    algo.fit(trainset)
    
    video_ids = df['video_id'].values
    predictions = [algo.predict(user_id, vid).est for vid in video_ids]
    top_indices = np.argsort(predictions)[-top_n:][::-1]
    return df.iloc[top_indices]['video_id'].values

# 5. Взвешенная гибридизация
def hybrid_recommendation(df, user_data, user_id, top_n=5):
    num_interactions = len(user_data[user_data['user_id'] == user_id])
    
    if num_interactions < 5:
        w_cf, w_cbf = 0.0, 1.0
    elif num_interactions < 70:
        w_cf = 0.1 + (num_interactions - 5) * 0.01
        w_cbf = 1.0 - w_cf
    else:
        w_cf, w_cbf = 0.7, 0.3
    
    cf_recs = collaborative_filtering(user_data, df, user_id, top_n=10)
    cbf_recs = content_based_filtering(df, user_data, user_id, top_n=10)
    
    scores = {}
    for vid in set(cf_recs).union(cbf_recs):
        cf_score = df[df['video_id'] == vid]['norm_likes'].values[0] if vid in cf_recs else 0
        cbf_score = df[df['video_id'] == vid]['norm_likes'].values[0] if vid in cbf_recs else 0
        scores[vid] = w_cf * cf_score + w_cbf * cbf_score
    
    sorted_vids = sorted(scores, key=scores.get, reverse=True)[:top_n]
    if not sorted_vids:
        return df.sort_values(by=["norm_views", "norm_likes"], ascending=False)['video_id'].values[:top_n]
    return sorted_vids

# 6. Оценка метрик
def evaluate_metrics(df, user_data, test_data, top_n=5):
    precision, recall, map_score = [], [], []
    
    for user_id in test_data['user_id'].unique():
        recs = hybrid_recommendation(df, user_data, user_id, top_n)
        relevant = test_data[test_data['user_id'] == user_id]['video_id'].values
        
        relevant_recs = len(set(recs).intersection(relevant))
        precision.append(relevant_recs / top_n)
        recall.append(relevant_recs / len(relevant) if len(relevant) > 0 else 0)
        
        ap = 0
        for k, rec in enumerate(recs, 1):
            if rec in relevant:
                ap += relevant_recs / k
        map_score.append(ap / len(relevant) if len(relevant) > 0 else 0)
    
    return {
        'Precision@5': np.mean(precision),
        'Recall@5': np.mean(recall),
        'MAP': np.mean(map_score)
    }

def main():
    file_path = 'RU_youtube_trending_data.csv'
    df = preprocess_data(file_path)
    
    try:
        user_data = pd.read_csv('user_data.csv')
    except FileNotFoundError:
        user_data = generate_user_data(df)
    
    train_data = user_data.sample(frac=0.8, random_state=42)
    test_data = user_data.drop(train_data.index)
    
    metrics = evaluate_metrics(df, train_data, test_data)
    print("Результаты тестирования:")
    print(f"Precision@5: {metrics['Precision@5']:.2f}")
    print(f"Recall@5: {metrics['Recall@5']:.2f}")
    print(f"MAP: {metrics['MAP']:.2f}")

if __name__ == "__main__":
    main()
