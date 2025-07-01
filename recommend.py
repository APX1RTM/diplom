import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
from nltk.corpus import stopwords
from pymystem3 import Mystem
import random

mystem = Mystem()
stop_words = set(stopwords.words('russian'))

# 1. Загрузка и предобработка данных
def preprocess_data(file_path):
    df = pd.read_csv(file_path)
    
    df = df.drop_duplicates(subset=['video_id'])
    df['tags'] = df['tags'].fillna(df['title'])
    df['text'] = df['title'] + ' ' + df['tags']
    
    def clean_text(text):
        words = mystem.lemmatize(text.lower())
        words = [word for word in words if word.isalnum() and word not in stop_words]
        return ' '.join(words)
    
    df['text'] = df['text'].apply(clean_text)
    
    df = pd.get_dummies(df, columns=['categoryId'])
    
    df['norm_likes'] = df['likes'] / df['likes'].max()
    df['norm_views'] = df['view_count'] / df['view_count'].max()
    
    return df

# 2. Генерация синтетических пользовательских данных
def generate_user_data(df, num_users=100):
    video_ids = df['video_id'].values
    user_data = []
    for user_id in range(num_users):
        num_interactions = random.randint(5, 70)
        viewed_videos = random.sample(list(video_ids), num_interactions)
        for video_id in viewed_videos:
            rating = df[df['video_id'] == video_id]['norm_likes'].values[0]
            user_data.append({'user_id': user_id, 'video_id': video_id, 'rating': rating})
    return pd.DataFrame(user_data)

# 3. Контентная фильтрация
def content_based_filtering(df, user_data, user_id, top_n=5):
    tfidf = TfidfVectorizer(max_features=1000)
    tfidf_matrix = tfidf.fit_transform(df['text'])
    
    user_videos = user_data[user_data['user_id'] == user_id]['video_id']
    user_indices = df[df['video_id'].isin(user_videos)].index
    if len(user_indices) == 0:
        return []
    user_profile = tfidf_matrix[user_indices].mean(axis=0)
    
    similarities = cosine_similarity(user_profile, tfidf_matrix)[0]
    top_indices = similarities.argsort()[-top_n:][::-1]
    return df.iloc[top_indices]['video_id'].values

# 4. Коллаборативная фильтрация (SVD)
def collaborative_filtering(user_data, df, user_id, top_n=5):
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
    
    return sorted(scores, key=scores.get, reverse=True)[:top_n]

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
