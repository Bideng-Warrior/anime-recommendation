## ANIME RECOMMENDATION WITH COLLABORATIVE FILTERING

This dataset contains information about 16.621 anime, 175.731 recommendations and the preference from 74.129 different users of animes scrapped from anime-planet. In particular, this dataset contain:

- Information about the anime like Tags, synopsis, average score, etc.
- List of animes recommended given another anime and the count of user that are agreed with the recommendation.
- HTML with anime information to do data scrapping. These files contain information such as reviews, synopsis, information about the staff, anime statistics, genre, etc.
- the anime list per user. Include dropped, watched, want to watch, currently watching, stalled and Won't watch.
- ratings given by users to the animes that they has watched completely.

<div id="badges">
  <a href="https://www.kaggle.com/maoel31](https://www.kaggle.com/datasets/hernan4444/animeplanet-recommendation-database-2020">
    <img src="https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=Kaggle&logoColor=white" alt="Kaggle Badge"/>
  </a>
</div>
<br>

![A-whisket-away](https://user-images.githubusercontent.com/58927608/235576018-f245709b-4bc6-4277-9b09-18438654fa17.jpg)

### Data Wrangling
- Data integration : combining ```anime_id```, ```user_id, rating```, ```watching_status``` and ```watched_episodes``` from different csv file
- Data cleaning : remove null value, duplicate, and ambiguous data
- Data reduction : remove ```unnamed : 0``` feature

This step can be seen in [data-merging.ipynb](https://github.com/Bideng-Warrior/anime-recommendation/blob/main/data-merging.ipynb) and [data-cleaning.ipynb](https://github.com/Bideng-Warrior/anime-recommendation/blob/main/data-merging.ipynb)

### EDA
- Show stastic data from dataset : count, mean, std, min, 25%, 50%, 75%, and max
- Correlation between numerical feature
- User ratings
- Most common tags with wordcloud
- Data outlier

This step can be seen in [exploratory-data-analysis.ipynb](https://github.com/Bideng-Warrior/anime-recommendation/blob/main/exploratory-data-analysis.ipynb)

### Data Preparation
- encoding ```user_id``` to integer
- encoding ```anime_id``` to integer
- shuffle data
- ```x``` data : ```user```, ```anime```, ```y``` data : ```rating```
- split data into 80L20 ratio : ```x_train```, ```y_train```, ```x_val```, ```y_val```

This step can be seen in [model & evaluation.ipynb](https://github.com/Bideng-Warrior/anime-recommendation/blob/main/model%20%26%20evaluation.ipynb)

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
- model summary :

![image](https://user-images.githubusercontent.com/58927608/235581360-2ad7f2bb-1a1b-4746-ba99-fd0f171bf899.png)

This step can be seen in [model & evaluation.ipynb](https://github.com/Bideng-Warrior/anime-recommendation/blob/main/model%20%26%20evaluation.ipynb)

### Evaluation

Root Mean Squared Error :

![image](https://user-images.githubusercontent.com/58927608/235582372-01feff87-9219-4775-9ee2-aaf4f7cd3401.png)

- Visualize metrics root mean squared error and loss over epochs, result :
  - RMSE :
  
    ![image](https://user-images.githubusercontent.com/58927608/235582512-68cab90c-3feb-4530-bbfa-0b9a90cc5808.png)
  
  - Loss :
    
    ![image](https://user-images.githubusercontent.com/58927608/235582561-189667b3-5d38-4cfe-95d7-0592f69bf9db.png)
   
   It seems that the training process is making some progress but the validation loss and validation root mean squared error (RMSE) are not decreasing much after the third epoch.
   Looking at the output, it appears that the model starts with a loss of 0.6304 and RMSE of 0.2328 on the training set and a loss of 0.6155 and RMSE of 0.2170 on the validation set. As the model is trained for more epochs, the loss and RMSE values decrease for both the training and validation sets, indicating that the model is learning to predict better. However, the improvement seems to level off after around 10 epochs, with little change in the validation set performance after this point.
   However, the fact that the validation set performance is fairly close to the training set performance throughout training suggests that the model is not overfitting, which is a positive sign.
   
   _we could have improved it further, but we discovered errors in the data, so we didn't do it KEKW_
 
This step can be seen in [model & evaluation.ipynb](https://github.com/Bideng-Warrior/anime-recommendation/blob/main/model%20%26%20evaluation.ipynb)

### Testing Recommendation
- This is anime recommendation for user_id 48841
  
  data user_id 48841 :
  
  |        | anime_id | user_id | Name                                              | Tags                                              | watching_status | watched_episodes | rating | user  | anime |
  |-------:|---------:|--------:|---------------------------------------------------|---------------------------------------------------|-----------------|------------------|--------|-------|-------|
  | 497036 |    14843 | 48841   | How NOT to Summon a Demon Lord: Omega             | Action, Comedy, Ecchi, Fantasy, Shounen, Demon... | 2               | 9                | 4.0    | 32885 | 686   |
  | 497037 |    16049 | 48841   | Full Dive: This Ultimate Next-Gen Full Dive RP... | Action, Comedy, Fantasy, MMORPG, RPG, Virtual ... | 3               | 7                | 1.5    | 32885 | 226   |
  | 497043 |     4336 | 48841   | Hunter x Hunter (2011)                            | Action, Adventure, Drama, Fantasy, Shounen, Mo... | 2               | 52               | 5.0    | 32885 | 28    |
  | 497040 |    14132 | 48841   | I’ve Been Killing Slimes for 300 Years and Max... | Comedy, Fantasy, Slice of Life, Isekai, Magic,... | 2               | 10               | 5.0    | 32885 | 229   |
  | 497039 |     2703 | 48841   | Fairy Tail                                        | Action, Adventure, Comedy, Drama, Fantasy, Sho... | 2               | 130              | 5.0    | 32885 | 30    |
  | 497045 | 14420    | 48841   | To Your Eternity                                  | Adventure, Drama, Fantasy, Shounen, Animal Tra... | 2               | 10               | 4.0    | 32885 | 616   |
  | 497044 | 14666    | 48841   | The Slime Diaries: That Time I Got Reincarnate... | Adventure, Comedy, Fantasy, Shounen, Slice of ... | 2               | 8                | 3.5    | 32885 | 1274  |
  | 497041 | 7516     | 48841   | Magi: Adventure of Sinbad                         | Action, Adventure, Comedy, Fantasy, Shounen, A... | 3               | 3                | 1.0    | 32885 | 835   |
  | 497042 | 14827    | 48841   | My Hero Academia 5                                | Action, Sci Fi, Shounen, Superheroes, Based on... | 2               | 12               | 5.0    | 32885 | 218   |
  | 497038 | 12179    | 48841   | Cautious Hero: The Hero Is Overpowered but Ove... | Action, Adventure, Comedy, Fantasy, Demons, Go... | 3               | 2                | 0.5    | 32885 | 613   |
  
- Recommendation result :

  |                                                                                                                                                                                                                                                               |
  |---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | **Anime with high ratings from user**                                                                                                                                                                                                                         |
  |                                                                                                                                                                                                                                                               |
  | Hunter x Hunter (2011) : Action, Adventure, Drama, Fantasy, Shounen, Monsters, Superpowers, Based on a Manga                                                                                                                                                  |
  | My Hero Academia 5 : Action, Sci Fi, Shounen, Superheroes, Based on a Manga                                                                                                                                                                                   |
  | Fairy Tail : Action, Adventure, Comedy, Drama, Fantasy, Shounen, Elemental Powers, Guilds, Magic, Based on a Manga                                                                                                                                            |
  | I’ve Been Killing Slimes for 300 Years and Maxed Out My Level : Comedy, Fantasy, Slice of Life, Isekai, Magic, Overpowered Main Characters, Person in a Strange World, Reincarnation, RPG, Slimes, Slow Life, Witches, Based on a Light Novel                 |
  | How NOT to Summon a Demon Lord: Omega : Action, Comedy, Ecchi, Fantasy, Shounen, Demon King, Demons, Isekai, Magic, Overpowered Main Characters, Person in a Strange World, RPG, Summoned Into Another World, Trapped in a Video Game, Based on a Light Novel |
  |                                                                                                                                                                                                                                                               |
  | **Top 10 anime recommendations**                                                                                                                                                                                                                              |
  |                                                                                                                                                                                                                                                               |
  | Attack on Titan The Final Season : Action, Drama, Fantasy, Horror, Shounen, Dark Fantasy, Military, War, Based on a Manga                                                                                                                                     |
  | Demon Slayer: Kimetsu no Yaiba Movie - Mugen Train : Action, Drama, Fantasy, Shounen, Demons, Historical, Martial Arts, Orphans, Siblings, Swordplay, Trains, Based on a Manga                                                                                |
  | Fruits Basket the Final : Comedy, Drama, Fantasy, Romance, Shoujo, Animal Transformation, Curse, Dysfunctional Families, Love Triangle, Based on a Manga                                                                                                      |
  | Gintama.: Shirogane no Tamashii-hen 2 : Action, Comedy, Drama, Sci Fi, Shounen, Aliens, Crude, Feudal Japan, Gag, Samurai, Slapstick, Swordplay, Based on a Manga                                                                                             |
  | My Hero Academia Movie 2: Heroes:Rising : Action, Sci Fi, Shounen, Superheroes, Superpowers, Based on a Manga                                                                                                                                                 |
  | Natsume's Book of Friends Season 5 Specials : Shoujo, Slice of Life, Bodyguards, Cats, Countryside, Iyashikei, Japanese Mythology, Orphans, Supernatural, Youkai, Based on a Manga                                                                            |
  | Yona of the Dawn: Zeno Arc : Action, Adventure, Drama, Fantasy, Romance, Shoujo, Death of a Loved One, Orphans, Political, Royalty, Based on a Manga                                                                                                          |
  | Legend of the Condor Hero III : Martial Arts                                                                                                                                                                                                                  |
  | Dou Po Cangqiong 4th Season : Action, Chinese Animation, Xianxia, Based on a Web Novel, CG Animation                                                                                                                                                          |
  | Yamishibai: Japanese Ghost Stories 8th Season : Horror, Episodic, Short Episodes, Supernatural, Urban Legend, Original Work                                                                                                                                   |
  
This step can be seen in [model & evaluation.ipynb](https://github.com/Bideng-Warrior/anime-recommendation/blob/main/model%20%26%20evaluation.ipynb)

Reference :
- [What does RMSE really mean?](https://towardsdatascience.com/what-does-rmse-really-mean-806b65f2e48e)
- [Anime-Planet Recommendation Database 2020](https://www.kaggle.com/datasets/hernan4444/animeplanet-recommendation-database-2020)
       





