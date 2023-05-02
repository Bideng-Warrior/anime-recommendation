# -*- coding: utf-8 -*-
"""model & evaluation.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/17W3J1UOGR1ObmxSCYV4wtJdruXoioRCP
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

"""DATA PREPARATION"""

data = pd.read_csv('all_anime_clean.csv')

data

"""COLLABORATIVE FILTERING"""

data = data.drop(['Unnamed: 0'], axis=1)

data

user_ids = data['user_id'].unique().tolist()
user_to_user_encoded = {x: i for i, x in enumerate(user_ids)}
user_encoded_to_user = {i: x for i, x in enumerate(user_ids)}
data['user'] = data['user_id'].map(user_to_user_encoded)

anime_ids = data['anime_id'].unique().tolist()
anime_to_anime_encoded = {x: i for i, x in enumerate(anime_ids)}
anime_encoded_to_anime = {i: x for i, x in enumerate(anime_ids)}
data['anime'] = data['anime_id'].map(anime_to_anime_encoded)

num_users = len(user_to_user_encoded)
num_animes = len(anime_encoded_to_anime)
min_rating = min(data['rating'])
max_rating = max(data['rating'])
print('Number of users: {}, Number of animes: {}, Min rating: {}, Max rating: {}'.format(num_users, num_animes, min_rating, max_rating))

"""DATA SPLIT"""

data = data.sample(frac=1, random_state=30)
data

x = data[['user', 'anime']].values
y = data['rating'].apply(lambda x: (x - min_rating) / (max_rating - min_rating)).values
train_indices = int(0.8 * data.shape[0])
x_train, x_val, y_train, y_val = (
    x[:train_indices],
    x[train_indices:],
    y[:train_indices],
    y[train_indices:]
)
print(x, y)
print(x.shape)
print(y.shape)
print(x_val.shape)
print(y_val.shape)

"""MODELLING"""

class RecommenderNet(tf.keras.Model):

  def __init__(self, num_users, num_animes, embedding_size, **kwargs):
    super(RecommenderNet, self).__init__(**kwargs)
    self.num_users = num_users
    self.num_animes = num_animes
    self.embedding_size = embedding_size
    self.user_embedding = layers.Embedding(
        num_users,
        embedding_size,
        embeddings_initializer='he_normal',
        embeddings_regularizer=keras.regularizers.l2(1e-6)
    )
    self.user_bias = layers.Embedding(num_users, 1)
    self.anime_embedding = layers.Embedding(
        num_animes,
        embedding_size,
        embeddings_initializer='he_normal',
        embeddings_regularizer=keras.regularizers.l2(1e-6)
    )
    self.anime_bias = layers.Embedding(num_animes, 1)

  def call(self, inputs):
    user_vector = self.user_embedding(inputs[:, 0])
    user_bias = self.user_bias(inputs[:, 0])
    anime_vector = self.anime_embedding(inputs[:, 1])
    anime_bias = self.anime_bias(inputs[:, 1])
    dot_user_anime = tf.tensordot(user_vector, anime_vector, 2)
    x = dot_user_anime + user_bias + anime_bias
    return tf.nn.sigmoid(x)

model = RecommenderNet(num_users, num_animes, 50)

model.compile(
    loss = tf.keras.losses.BinaryCrossentropy(),
    optimizer = keras.optimizers.Adam(learning_rate = 0.001),
    metrics = [tf.keras.metrics.RootMeanSquaredError()]
)

hist = model.fit(x = x_train, y = y_train, batch_size = 8, epochs = 20, validation_data = (x_val, y_val))

"""RMSE : Metrics Visualization"""

plt.plot(hist.history['root_mean_squared_error'])
plt.plot(hist.history['val_root_mean_squared_error'])
plt.title('model_metrics')
plt.ylabel('root_mean_squared_error')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc = 'upper left')
plt.show()

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('model_metrics')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc = 'upper left')
plt.show()

anime_recommend = pd.DataFrame({
    'anime_id' : data.anime_id,
    'anime_genre' : data.Tags,
    'anime_name' : data.Name
})
anime_recommend

animes_rcm = anime_recommend
rcm = pd.read_csv('all_anime_clean.csv')

rcm['rating'] = rcm['rating'].values.astype(np.float32)

user_id = rcm.user_id.sample(1).iloc[0]
animes_watched_by_user = rcm[rcm.user_id == user_id]

animes_not_watched = animes_rcm[~animes_rcm['anime_id'].isin(animes_watched_by_user.anime_id.values)]['anime_id']
animes_not_watched = list(
    set(animes_not_watched)
    .intersection(set(anime_to_anime_encoded.keys()))
)

animes_not_watched = [[anime_to_anime_encoded.get(x)] for x in animes_not_watched]
user_encoder = user_to_user_encoded.get(user_id)
user_animes_array = np.hstack(
    ([[user_encoder]] * len(animes_not_watched), animes_not_watched)
)

ratings = model.predict(user_animes_array).flatten()

top_ratings_indices = ratings.argsort()[-10:][::-1]
recommended_animes_ids = [anime_encoded_to_anime.get(animes_not_watched[x][0]) for x in top_ratings_indices]

print('Showing recommendations for users : {}'.format(user_id))
print('===' * 10)

print('Anime with high ratings from user')
print('----' * 10)

top_animes_user = (
    animes_watched_by_user.sort_values(
        by='rating',
        ascending=False
    )
    .head(5)
    .anime_id.values
)

top_animes_user_df = anime_recommend[anime_recommend['anime_id'].isin(top_animes_user)]
top_animes_user_df = top_animes_user_df.drop_duplicates(subset=['anime_name'])
for row in top_animes_user_df.itertuples():
    print(row.anime_name, ':', row.anime_genre)

print('----' * 10)
print('Top 10 anime recommendations')
print('----' * 10)

recommended_animes = animes_rcm[animes_rcm['anime_id'].isin(recommended_animes_ids)]
recommended_animes = recommended_animes.drop_duplicates(subset=['anime_name'])
for row in recommended_animes.itertuples():
    print(row.anime_name, ':', row.anime_genre)

data_filtered = data[data['user_id'] == 48841]
data_filtered

