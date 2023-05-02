## ANIME RECOMMENDATION WITH COLLABORATIVE FILTERING

<div id="badges">
  <a href="[https://www.kaggle.com/maoel31](https://www.kaggle.com/datasets/hernan4444/animeplanet-recommendation-database-2020)">
    <img src="https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=Kaggle&logoColor=white" alt="Kaggle Badge"/>
  </a>
</div>
<br>

![dear-pecinta-anime-netflix-akan-produksi-3-anime-baru-bareng-studio-colorido51_700](https://user-images.githubusercontent.com/58927608/235571648-7faed3db-9f0e-4d91-9312-18733dbd3722.jpg)

### Data Wrangling
- Data integration : combining anime_id, user_id, rating, watching_status and watched_episodes from different csv file
- Data cleaning : remove null value, duplicate, and ambiguous data
- Data reduction : remove unnamed : 0 feature

This step can be seen in data-merging.ipynb and data-cleaning.ipynb

### EDA
- Show stastic data from dataset : count, mean, std, min, 25%, 50%, 75%, and max
- Correlation between numerical feature
- User ratings
- Most common tags with wordcloud
- Data outlier

This step can be seen in exploratory-data-analysis.ipynb

### Data Preparation
- encoding user_id to integer
- encoding anime_id to integer
- shuffle data
- x data : user, anime, y data : rating
- split data into 80L20 ratio : x_train, y_train, x_val, y_val

This step can be seen in model & evaluation.ipynb

### Modelling
- encoding user_id to integer
- encoding anime_id to integer
- shuffle data
- x data : user, anime, y data : rating
- split data into 80L20 ratio : x_train, y_train, x_val, y_val

This step can be seen in model & evaluation.ipynb

