import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Глобальная история прослушанных песен
listening_history = []

# Подготовка данных
def prepare_data(dataset_path):
    df = pd.read_csv(dataset_path)
    features = ['valence', 'danceability', 'energy']
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[features])
    return df, scaled_features, scaler, features

class MusicRNN(nn.Module):
    def __init__(self, input_size, hidden_size, embedding_size):
        super(MusicRNN, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, embedding_size)
        )
    
    def forward(self, x):
        output, (hn, cn) = self.rnn(x)
        return self.fc(hn[-1])

# Обучение RNN модели
def train_model(model, X, epochs=1000, batch_size=32, lr=0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    X_tensor = torch.FloatTensor(X).unsqueeze(1)
    y_tensor = torch.FloatTensor(X)
    dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            embeddings = model(batch_X)
            loss = criterion(embeddings, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch+1)%1000 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(loader):.4f}")
    return model

# Получение рекомендаций через kNN
def get_music_recommendations_knn(song_name, dataset_path, n_recommendations=25, weights=None):
    df, scaled_features, scaler, features = prepare_data(dataset_path)
    
    if weights is None:
        weights = [1.0] * len(features)
    
    if len(weights) != len(features):
        raise ValueError(f"Количество весов ({len(weights)}) должно совпадать с количеством признаков ({len(features)})")
    
    # Проверка наличия песни в датасете
    if song_name.lower() not in df['name'].str.lower().values:
        return f"Ошибка: Песня '{song_name}' не найдена в датасете."
    
    song_idx = df.index[df['name'].str.lower() == song_name.lower()].tolist()[0]
    
    weighted_features = scaled_features * np.array(weights)
    song_features = weighted_features[song_idx].reshape(1, -1)
    
    nbrs = NearestNeighbors(n_neighbors=n_recommendations + 1, algorithm='kd_tree').fit(weighted_features)
    distances, indices = nbrs.kneighbors(song_features)
    recommended_indices = indices[0][1:]  # Исключаем саму песню
    
    recommendations = df.iloc[recommended_indices].copy()
    recommendations['artists'] = recommendations['artists'].apply(
        lambda x: x[0] if isinstance(x, list) else x.strip("[]").strip("'") if isinstance(x, str) else x
    )
    return recommendations.reset_index(drop=True)

# Получение рекомендаций через RNN с учетом истории
def get_music_recommendations_rnn(song_name, knn_recommendations, history=None, n_recommendations=10):
    df = knn_recommendations
    features = ['instrumentalness', 'liveness', 'loudness', 'tempo']
    scaler = StandardScaler()
    
    if history:
        history_df = pd.DataFrame(history)
        df = pd.concat([df, history_df], ignore_index=True)
    
    scaled_features = scaler.fit_transform(df[features])
    
    try:
        song_idx = df.index[df['name'].str.lower() == song_name.lower()].tolist()[0]
    except IndexError:
        return "Песня не найдена в рекомендациях kNN"
    
    input_size = len(features)
    hidden_size = 64
    embedding_size = len(features)
    model = MusicRNN(input_size, hidden_size, embedding_size)
    model = train_model(model, scaled_features)
    
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(scaled_features).unsqueeze(1)
        embeddings = model(X_tensor).numpy()
    
    song_embedding = embeddings[song_idx].reshape(1, -1)
    similarities = cosine_similarity(song_embedding, embeddings)[0]
    recommended_indices = np.argsort(similarities)[::-1][1:n_recommendations + 1]
    recommendations = df.iloc[recommended_indices].copy()
    recommendations['artists'] = recommendations['artists'].apply(
        lambda x: x[0] if isinstance(x, list) else x.strip("[]").strip("'") if isinstance(x, str) else x
    )

    if 'popularity' in recommendations.columns:
        recommendations = recommendations.sort_values('popularity', ascending=False)

    return recommendations.reset_index(drop=True)

# Анализ эмоционального состояния на основе истории
def analyze_emotional_state(history):
    if not history:
        return "История прослушиваний пуста, невозможно определить эмоциональное состояние."

    history_df = pd.DataFrame(history)
    required_columns = ['valence', 'energy', 'danceability']
    
    if not all(col in history_df.columns for col in required_columns):
        return "Недостаточно данных в истории для анализа эмоций."

    emotions = {
        (0, 0, 0): "Грусть/Апатия",
        (0, 0, 1): "Меланхолия/Задумчивость",
        (0, 1, 0): "Тревожность/Беспокойство",
        (0, 1, 1): "Агрессия/Напряжение",
        (1, 0, 0): "Спокойствие/Умиротворение",
        (1, 0, 1): "Мечтательность/Лёгкость",
        (1, 1, 0): "Радость/Энергичность",
        (1, 1, 1): "Эйфория/Танцевальность"
    }

    def get_emotion(row):
        x_pos = 1 if row['valence'] > 0.5 else 0
        y_pos = 1 if row['energy'] > 0.5 else 0
        z_pos = 1 if row['danceability'] > 0.5 else 0
        return emotions[(x_pos, y_pos, z_pos)]

    history_df['emotion'] = history_df.apply(get_emotion, axis=1)
    emotion_counts = history_df['emotion'].value_counts()
    total_songs = len(history_df)
    dominant_emotion = emotion_counts.idxmax()
    dominant_count = emotion_counts[dominant_emotion]
    
    if dominant_count / total_songs > 0.5:
        return f"Ваше текущее эмоциональное состояние: {dominant_emotion}. " \
               f"Большинство прослушанных песен ({dominant_count} из {total_songs}) " \
               f"отражают это настроение."
    
    top_emotions = emotion_counts.head(2).index.tolist()
    if len(top_emotions) > 1:
        return f"Ваше текущее эмоциональное состояние: смесь {top_emotions[0]} и {top_emotions[1]}. " \
               f"Ваши последние песни ({emotion_counts[top_emotions[0]]} и {emotion_counts[top_emotions[1]]} из {total_songs}) " \
               f"указывают на смешенное настроение."
    else:
        return f"Ваше текущее эмоциональное состояние: {dominant_emotion}. " \
               f"Песни в истории ({total_songs}) в основном отражают это настроение."

# Интерактивная рекомендация
def interactive_recommendation(song_name, knn_result):
    global listening_history
    print(song_name)

    if isinstance(knn_result, str):
        print(knn_result)
        return None, None

    rnn_result = get_music_recommendations_rnn(song_name, knn_result, history=listening_history, n_recommendations=25)
    
    if isinstance(rnn_result, str):
        print(rnn_result)
        return None, None
    
    if listening_history:
        history_names = [song['name'].lower() for song in listening_history]
        rnn_result = rnn_result[~rnn_result['name'].str.lower().isin(history_names)]
    
    if rnn_result.empty:
        print("Нет новых рекомендаций, все предложенные песни уже в истории")
        return None, knn_result
    
    print("\nАнализ вашего эмоционального состояния:")
    print(analyze_emotional_state(listening_history))
    
    recommendation_idx = 0
    
    while recommendation_idx < len(rnn_result):
        song = rnn_result.iloc[recommendation_idx]
        print(f"\nРекомендуемая песня: '{song['name']}' от {song['artists']}")
        
        while True:
            choice = input("Хотите послушать эту песню? (да/нет): ").lower()
            if choice in ['да', 'нет']:
                break
            print("Пожалуйста, введите 'да' или 'нет'")
        
        if choice == 'да':
            listening_history.append(rnn_result.iloc[recommendation_idx].to_dict())
            print(f"Песня '{song['name']}' добавлена в историю прослушиваний")
            return song['name'], knn_result
        else:
            print("Песня пропущена")
            recommendation_idx += 1
    
    print("Больше нет доступных рекомендаций для этой песни")
    return None, knn_result

# Подробный анализ изменения эмоционального состояния
def detailed_emotional_analysis(history):
    if not history:
        return "История прослушиваний пуста, невозможно проанализировать изменение настроения."

    history_df = pd.DataFrame(history)
    required_columns = ['valence', 'energy', 'danceability']
    
    if not all(col in history_df.columns for col in required_columns):
        return "Недостаточно данных в истории для анализа эмоций."

    emotions = {
        (0, 0, 0): "Грусть/Апатия",
        (0, 0, 1): "Меланхолия/Задумчивость",
        (0, 1, 0): "Тревожность/Беспокойство",
        (0, 1, 1): "Агрессия/Напряжение",
        (1, 0, 0): "Спокойствие/Умиротворение",
        (1, 0, 1): "Мечтательность/Лёгкость",
        (1, 1, 0): "Радость/Энергичность",
        (1, 1, 1): "Эйфория/Танцевальность"
    }

    def get_emotion(row):
        x_pos = 1 if row['valence'] > 0.5 else 0
        y_pos = 1 if row['energy'] > 0.5 else 0
        z_pos = 1 if row['danceability'] > 0.5 else 0
        return emotions[(x_pos, y_pos, z_pos)]

    history_df['emotion'] = history_df.apply(get_emotion, axis=1)
    
    # Создаем временную последовательность
    analysis = []
    analysis.append("Подробный анализ изменения вашего настроения в течение сессии:\n")
    
    for i, song in enumerate(history):
        emotion = history_df['emotion'].iloc[i]
        song_name = song['name']
        artist = song['artists']
        valence = song['valence']
        energy = song['energy']
        danceability = song['danceability']
        
        # Описание текущей песни и эмоции
        song_analysis = f"\nПесня {i+1}: '{song_name}' от {artist}\n"
        song_analysis += f"  Эмоция: {emotion}\n"
        song_analysis += f"  Характеристики: Позитивность={valence:.2f}, Энергия={energy:.2f}, Танцевальность={danceability:.2f}\n"
        
        # Подробное объяснение текущей эмоции
        if valence > 0.5:
            song_analysis += "  - Высокая позитивность: эта песня отражает оптимизм или радость.\n"
        else:
            song_analysis += "  - Низкая позитивность: песня может ассоциироваться с грустью или задумчивостью.\n"
            
        if energy > 0.5:
            song_analysis += "  - Высокая энергия: песня передает активность или возбуждение.\n"
        else:
            song_analysis += "  - Низкая энергия: песня способствует расслаблению или спокойствию.\n"
            
        if danceability > 0.5:
            song_analysis += "  - Высокая танцевальность: песня побуждает к движению или легкости.\n"
        else:
            song_analysis += "  - Низкая танцевальность: песня более созерцательная, менее ритмичная.\n"
        
        analysis.append(song_analysis)
        
        # Анализ переходов между эмоциями
        if i > 0:
            prev_emotion = history_df['emotion'].iloc[i-1]
            prev_song = history[i-1]['name']
            prev_artist = history[i-1]['artists']
            if emotion != prev_emotion:
                transition = f"\nПереход настроения с '{prev_emotion}' (песня '{prev_song}' от {prev_artist}) на '{emotion}' (песня '{song_name}' от {artist}):\n"
                
                # Анализируем изменения характеристик
                valence_change = valence - history[i-1]['valence']
                energy_change = energy - history[i-1]['energy']
                danceability_change = danceability - history[i-1]['danceability']
                
                transition += f"  Изменения характеристик:\n"
                transition += f"    - Позитивность: {'увеличилась' if valence_change > 0 else 'уменьшилась'} на {abs(valence_change):.2f}\n"
                transition += f"    - Энергия: {'увеличилась' if energy_change > 0 else 'уменьшилась'} на {abs(energy_change):.2f}\n"
                transition += f"    - Танцевальность: {'увеличилась' if danceability_change > 0 else 'уменьшилась'} на {abs(danceability_change):.2f}\n"
                
                # Интерпретация перехода
                if prev_emotion in ["Грусть/Апатия", "Меланхолия/Задумчивость"] and emotion in ["Радость/Энергичность", "Эйфория/Танцевальность"]:
                    transition += "  - Этот переход может указывать на значительное улучшение настроения, возможно, вы почувствовали прилив оптимизма или энергии.\n"
                elif prev_emotion in ["Радость/Энергичность", "Эйфория/Танцевальность"] and emotion in ["Грусть/Апатия", "Меланхолия/Задумчивость"]:
                    transition += "  - Переход к более меланхоличному настроению может отражать желание задуматься или отдохнуть после активного состояния.\n"
                elif prev_emotion in ["Спокойствие/Умиротворение", "Мечтательность/Лёгкость"] and emotion in ["Радость/Энергичность", "Эйфория/Танцевальность"]:
                    transition += "  - Переход от спокойствия к энергичности может указывать на желание активности или воодушевления.\n"
                elif prev_emotion in ["Радость/Энергичность", "Эйфория/Танцевальность"] and emotion in ["Спокойствие/Умиротворение", "Мечтательность/Лёгкость"]:
                    transition += "  - Снижение активности может означать, что вы искали расслабления или более мягкого настроения.\n"
                elif prev_emotion in ["Тревожность/Беспокойство", "Агрессия/Напряжение"] and emotion in ["Спокойствие/Умиротворение", "Мечтательность/Лёгкость"]:
                    transition += "  - Переход от напряжения к спокойствию может указывать на желание снять стресс или успокоиться.\n"
                elif prev_emotion in ["Спокойствие/Умиротворение", "Мечтательность/Лёгкость"] and emotion in ["Тревожность/Беспокойство", "Агрессия/Напряжение"]:
                    transition += "  - Увеличение напряжения может отражать внутреннее беспокойство или внешние обстоятельства, вызывающие возбуждение.\n"
                else:
                    transition += "  - Этот переход отражает смену настроения, возможно, вызванную разнообразием музыкальных предпочтений или внешними факторами.\n"
                
                analysis.append(transition)
    
    # Итоговое резюме
    emotion_counts = history_df['emotion'].value_counts()
    total_songs = len(history_df)
    dominant_emotion = emotion_counts.idxmax()
    dominant_count = emotion_counts[dominant_emotion]
    
    summary = "\nРезюме:\n"
    summary += f"Всего прослушано песен: {total_songs}\n"
    summary += f"Преобладающая эмоция: {dominant_emotion} ({dominant_count} из {total_songs})\n"
    
    if len(emotion_counts) > 1:
        summary += "Ваше настроение изменялось в течение сессии:\n"
        for emotion, count in emotion_counts.items():
            summary += f"  - {emotion}: {count} песен ({count/total_songs*100:.1f}%)\n"
        
        # Описание последовательности изменений
        summary += "\nПоследовательность настроений:\n"
        for i, emotion in enumerate(history_df['emotion']):
            song_name = history[i]['name']
            artist = history[i]['artists']
            summary += f"  - Песня {i+1}: '{song_name}' от {artist} — {emotion}\n"
    else:
        summary += f"Ваше настроение оставалось стабильным: {dominant_emotion}\n"
    
    analysis.append(summary)
    
    # Визуализация изменений настроения
    if len(history) > 1:
        plt.figure(figsize=(10, 6))
        emotions_list = history_df['emotion'].tolist()
        time_points = range(1, len(emotions_list) + 1)
        
        # Создаем числовое представление эмоций для графика
        emotion_to_num = {emo: idx for idx, emo in enumerate(emotions.values())}
        y_values = [emotion_to_num[emo] for emo in emotions_list]
        
        plt.plot(time_points, y_values, marker='o', linestyle='-', color='b')
        plt.xticks(time_points)
        plt.yticks(range(len(emotion_to_num)), emotion_to_num.keys())
        plt.xlabel('Песня (порядковый номер)')
        plt.ylabel('Эмоция')
        plt.title('Изменение настроения в течение сессии')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('emotion_analysis.png')
    
    return "\n".join(analysis)

if __name__ == "__main__":
    dataset_path = "data.csv"
    song_name = input()
    weights = [2.0, 1.2, 1.0]  # valence, danceability, energy
    
    current_song = song_name
    while True:
        knn_result = get_music_recommendations_knn(current_song, dataset_path, weights=weights)
        if isinstance(knn_result, str):
            print(knn_result)
            break  # Прерываем цикл, если песня не найдена
        
        print(f"\nНа основе '{current_song}' найдено {len(knn_result)} похожих песен для анализа")
        next_song, knn_result = interactive_recommendation(current_song, knn_result)
        if next_song:
            current_song = next_song
        continue_choice = input("\nХотите получить ещё одну рекомендацию? (да/нет): ").lower()
        if continue_choice != 'да':
            break
    
    print("\nИстория прослушанных песен:")
    for song in listening_history:
        print(f"- {song['name']} от {song['artists']}")
    
    print("\nПолный анализ изменения вашего настроения:")
    print(detailed_emotional_analysis(listening_history))