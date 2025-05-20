import unittest
from unittest.mock import patch
import pandas as pd
import numpy as np
import torch
import os
from sklearn.preprocessing import StandardScaler
from vkr1 import (prepare_data, MusicRNN, get_music_recommendations_knn, 
                 analyze_emotional_state, train_model, get_music_recommendations_rnn, 
                 interactive_recommendation, detailed_emotional_analysis, listening_history)

class TestVKR1(unittest.TestCase):

    @patch('pandas.read_csv')
    def test_prepare_data(self, mock_read_csv):
        mock_df = pd.DataFrame({
            'valence': [0.5, 0.6, 0.7],
            'danceability': [0.4, 0.5, 0.6],
            'energy': [0.3, 0.4, 0.5],
            'name': ['song1', 'song2', 'song3']
        })
        mock_read_csv.return_value = mock_df
        df, scaled_features, scaler, features = prepare_data('dummy_path')
        self.assertTrue(df.equals(mock_df))
        self.assertEqual(features, ['valence', 'danceability', 'energy'])
        expected_scaled = StandardScaler().fit_transform(mock_df[features])
        np.testing.assert_array_almost_equal(scaled_features, expected_scaled)
        self.assertIsInstance(scaler, StandardScaler)

    def test_music_rnn_forward(self):
        input_size = 3
        hidden_size = 64
        embedding_size = 3
        model = MusicRNN(input_size, hidden_size, embedding_size)
        x = torch.randn(1, 1, input_size)
        output = model(x)
        self.assertEqual(output.shape, (1, embedding_size))

    @patch('torch.utils.data.DataLoader')
    def test_train_model(self, mock_loader):
        model = MusicRNN(input_size=3, hidden_size=64, embedding_size=3)
        X = np.random.rand(10, 3)
        mock_loader.return_value = [(torch.FloatTensor(X[:2]).unsqueeze(1), torch.FloatTensor(X[:2]))]
        trained_model = train_model(model, X, epochs=1, batch_size=2)
        self.assertIsInstance(trained_model, MusicRNN)

    @patch('vkr1.train_model')
    def test_get_music_recommendations_rnn(self, mock_train_model):
        mock_train_model.return_value = MusicRNN(4, 64, 4)
        knn_recommendations = pd.DataFrame({
            'name': ['song1', 'song2'],
            'instrumentalness': [0.1, 0.2],
            'liveness': [0.3, 0.4],
            'loudness': [-5, -6],
            'tempo': [120, 130],
            'artists': ['artist1', 'artist2']
        })
        result = get_music_recommendations_rnn('song1', knn_recommendations)
        self.assertIsInstance(result, pd.DataFrame)

    @patch('builtins.input', side_effect=['да'])
    def test_interactive_recommendation(self, mock_input):
        global listening_history
        listening_history.clear()  # Сбрасываем историю перед тестом
        knn_result = pd.DataFrame({
            'name': ['song1', 'song2'],  # Включаем 'song1' (входная) и 'song2' (рекомендация)
            'artists': ['artist1', 'artist2'],
            'valence': [0.5, 0.6],
            'energy': [0.4, 0.7],
            'danceability': [0.3, 0.8],
            'instrumentalness': [0.1, 0.2],
            'liveness': [0.3, 0.4],
            'loudness': [-5, -6],
            'tempo': [120, 130]
        })
        next_song, result = interactive_recommendation('song1', knn_result)
        self.assertEqual(next_song, 'song2')  # Ожидаем, что выберется 'song2'
        self.assertEqual(len(listening_history), 1)
        self.assertEqual(listening_history[0]['name'], 'song2')

    @patch('vkr1.prepare_data')
    def test_get_music_recommendations_knn(self, mock_prepare_data):
        mock_df = pd.DataFrame({
            'name': ['song1', 'song2', 'song3'],
            'valence': [0.5, 0.6, 0.7],
            'danceability': [0.4, 0.5, 0.6],
            'energy': [0.3, 0.4, 0.5],
            'artists': ['artist1', 'artist2', 'artist3']
        })
        mock_scaled_features = StandardScaler().fit_transform(mock_df[['valence', 'danceability', 'energy']])
        mock_scaler = StandardScaler().fit(mock_df[['valence', 'danceability', 'energy']])
        mock_features = ['valence', 'danceability', 'energy']
        mock_prepare_data.return_value = (mock_df, mock_scaled_features, mock_scaler, mock_features)
        recommendations = get_music_recommendations_knn('song1', 'dummy_path', n_recommendations=2)
        expected = mock_df.iloc[[1, 2]].reset_index(drop=True)
        expected['artists'] = expected['artists'].apply(lambda x: x.strip("[]").strip("'") if isinstance(x, str) else x)
        pd.testing.assert_frame_equal(recommendations, expected)

    @patch('builtins.input', side_effect=['нет'])
    def test_interactive_recommendation_skip(self, mock_input):
        global listening_history
        listening_history.clear()
        knn_result = pd.DataFrame({
            'name': ['song1', 'song2'],
            'artists': ['artist1', 'artist2'],
            'valence': [0.5, 0.6],
            'energy': [0.4, 0.7],
            'danceability': [0.3, 0.8],
            'instrumentalness': [0.1, 0.2],
            'liveness': [0.3, 0.4],
            'loudness': [-5, -6],
            'tempo': [120, 130]
        })
        next_song, result = interactive_recommendation('song1', knn_result)
        self.assertIsNone(next_song)
        self.assertEqual(len(listening_history), 0)

    def test_get_music_recommendations_rnn_song_not_found(self):
        knn_recommendations = pd.DataFrame({
            'name': ['song2'],
            'instrumentalness': [0.2],
            'liveness': [0.4],
            'loudness': [-6],
            'tempo': [130],
            'artists': ['artist2']
        })
        result = get_music_recommendations_rnn('song1', knn_recommendations)
        self.assertEqual(result, "Песня не найдена в рекомендациях kNN")

    import matplotlib
    matplotlib.use('Agg')
    def test_detailed_emotional_analysis_visualization(self):
        history = [
            {'name': 'song1', 'valence': 0.4, 'energy': 0.3, 'danceability': 0.2, 'artists': 'artist1'},
            {'name': 'song2', 'valence': 0.6, 'energy': 0.7, 'danceability': 0.8, 'artists': 'artist2'}
        ]
        result = detailed_emotional_analysis(history)
        self.assertTrue(os.path.exists('emotion_analysis.png'))

    def test_analyze_emotional_state(self):
        history = [
            {'name': 'song1', 'valence': 0.6, 'energy': 0.7, 'danceability': 0.8},
            {'name': 'song2', 'valence': 0.7, 'energy': 0.8, 'danceability': 0.9},
        ]
        result = analyze_emotional_state(history)
        expected = "Ваше текущее эмоциональное состояние: Эйфория/Танцевальность. Большинство прослушанных песен (2 из 2) отражают это настроение."
        self.assertEqual(result, expected)

        history_mixed = [
            {'name': 'song1', 'valence': 0.4, 'energy': 0.3, 'danceability': 0.2},
            {'name': 'song2', 'valence': 0.6, 'energy': 0.7, 'danceability': 0.8},
        ]
        result_mixed = analyze_emotional_state(history_mixed)
        self.assertIn("смесь", result_mixed)
        self.assertIn("Грусть/Апатия", result_mixed)
        self.assertIn("Эйфория/Танцевальность", result_mixed)

        result_empty = analyze_emotional_state([])
        self.assertEqual(result_empty, "История прослушиваний пуста, невозможно определить эмоциональное состояние.")

        history_incomplete = [{'name': 'song1', 'valence': 0.5}]
        result_incomplete = analyze_emotional_state(history_incomplete)
        self.assertEqual(result_incomplete, "Недостаточно данных в истории для анализа эмоций.")

    def test_detailed_emotional_analysis(self):
        history = [
            {'name': 'song1', 'valence': 0.4, 'energy': 0.3, 'danceability': 0.2, 'artists': 'artist1'},
            {'name': 'song2', 'valence': 0.6, 'energy': 0.7, 'danceability': 0.8, 'artists': 'artist2'}
        ]
        result = detailed_emotional_analysis(history)
        self.assertIn('Грусть/Апатия', result)
        self.assertIn('Эйфория/Танцевальность', result)
        self.assertIn('Переход настроения', result)

if __name__ == '__main__':
    unittest.main()