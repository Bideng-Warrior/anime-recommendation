## ANIME RECOMMENDATION WITH COLLABORATIVE FILTERING

<div id="badges">
  <a href="[https://www.kaggle.com/maoel31](https://www.kaggle.com/datasets/hernan4444/animeplanet-recommendation-database-2020)">
    <img src="https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=Kaggle&logoColor=white" alt="Kaggle Badge"/>
  </a>
</div>
<br>

![A-whisket-away](https://user-images.githubusercontent.com/58927608/235576018-f245709b-4bc6-4277-9b09-18438654fa17.jpg)

### Data Wrangling
- Data integration : combining ```anime_id```, ```user_id, rating```, ```watching_status``` and ```watched_episodes``` from different csv file
- Data cleaning : remove null value, duplicate, and ambiguous data
- Data reduction : remove ```unnamed : 0``` feature

This step can be seen in data-merging.ipynb and data-cleaning.ipynb

### EDA
- Show stastic data from dataset : count, mean, std, min, 25%, 50%, 75%, and max
- Correlation between numerical feature
- User ratings
- Most common tags with wordcloud
- Data outlier

This step can be seen in exploratory-data-analysis.ipynb

### Data Preparation
- encoding ```user_id``` to integer
- encoding ```anime_id``` to integer
- shuffle data
- ```x``` data : ```user```, ```anime```, ```y``` data : ```rating```
- split data into 80L20 ratio : ```x_train```, ```y_train```, ```x_val```, ```y_val```

This step can be seen in model & evaluation.ipynb

### Modelling
- Model using custom neural network ```RecommenderNet``` from tf.keras.Model, layers used in model like ```Embedding, tensordot and sigmoid```
- Make embedding layers for user_id include : num_users, embedding_size, embedding_initializer ```he_normal``` and embedding_regularizer with learning rate ```(1e-6) / 0.000001```
- user_bias layer for learn bias term for each user in dataset, bias added to the dot of user and anime embedding to user-spesific preferences
- Make embedding layers for anime_id : num_animes, embedding_size, embedding_initializer ```he_normal``` and embedding_regularizer with learning rate ```(1e-6) / 0.000001```
- anime_bias layer is same with user_bias
- make call function for calculate the predicted rating for a user-anime pair. This method takes as input a pair of user-anime represented as integer IDs, and returns a tensor of predicted ratings for each pair.
  - ```user_vector``` : for converts user_id to vector
  - ```user_bias``` : layer assigns a scalar bias value to each user
  - ```anime_vector``` : converts anime_id to vector
  - ```anime_bias``` : same with user_bias but for each anime
  - ```dot_user_anime``` with tensordot : calculate dot product between user and anime vector and gives measure of similarity to embedding space
  - tf.nn.sigmoid : normalize predicted rating between 0 and 1
- make ```model``` variable with parameters : (```num_users```, ```num_animes```, 50 embedfing size)
- model compile :
  - ```loss : BinaryCrossentropy```
  - ```optimizer : Adam(learning_rate = 0.001)```
  - ```metrics : RootMeanSquredError```
- train model :
  - ```batch_size = 8```
  - ```epochs = 20```

This step can be seen in model & evaluation.ipynb

### Evaluation


