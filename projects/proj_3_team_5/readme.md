# Clustering Project on Beer Profiles and Ratings
This project applies unsupervised machine learning techniques—particularly clustering—to explore and analyze the **Beer Profile and Ratings Dataset**.

## Dataset: Beer Profile and Ratings

We will use the [Beer Profile and Ratings Dataset](https://www.kaggle.com/datasets/ruthgn/beer-profile-and-ratings-data-set), available via:

```python
kagglehub.dataset_download("ruthgn/beer-profile-and-ratings-data-set")
```

This data set contains tasting profiles and consumer reviews for 3197 unique beers from 934 different breweries. It was created by integrating information from two existing data sets on Kaggle:

- Beer Tasting Profiles Dataset
- 1.5 Million Beer Reviews

The purpose of the data integration is to create a new data set that contains comprehensive consumer review (appearance, aroma, palate, taste and overall review scores) for different brews, combined with their detailed tasting profiles—this is that data set.

### Column description:

In the below table contains descriptions of each of the 25 features from the Dataset

| Column name            | Datatype  | Description                                      | Sample Entry                                  |
|------------------------|-----------|--------------------------------------------------|-----------------------------------------------|
| Name                   | object    | Name of the beer                                 | Danish Red Lager                              |
| Style                  | object    | Style or category of the beer                    | Lager - Vienna                                |
| Brewery                | object    | Name of the brewery                              | Figueroa Mountain Brewing Co.                 |
| Beer Name (Full)       | object    | Full name including brewery and beer name        | Figueroa Mountain Brewing Co. Danish Red Lager|
| Description            | object    | Textual description and notes about the beer     | Notes: Danish Style Red Lager (Dansk Red)...  |
| ABV                    | float64   | Alcohol by volume percentage                     | 5.0                                           |
| Min IBU                | int64     | Minimum International Bitterness Units           | 15                                            |
| Max IBU                | int64     | Maximum International Bitterness Units           | 30                                            |
| Astringency            | int64     | Astringency level (sensory score)                | 19                                            |
| Body                   | int64     | Body level (sensory score)                       | 43                                            |
| Fruits                 | int64     | Fruitiness score (sensory attribute)             | 17                                            |
| Hoppy                  | int64     | Hop character score                              | 38                                            |
| Spices                 | int64     | Spiciness score                                  | 2                                             |
| Malty                  | int64     | Maltiness score                                  | 88                                            |
| review_aroma           | float64   | Average user rating for aroma (0–5 scale)        | 3.666667                                      |
| review_appearance      | float64   | Average user rating for appearance (0–5 scale)   | 3.500000                                      |
| review_palate          | float64   | Average user rating for palate/mouthfeel         | 3.833333                                      |
| review_taste           | float64   | Average user rating for taste                    | 3.833333                                      |
| review_overall         | float64   | Average overall rating                           | 3.666667                                      |
| number_of_reviews      | int64     | Number of user reviews for this beer             | 3                                             |
