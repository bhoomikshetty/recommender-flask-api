import pandas as pd
import json
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.porter import  PorterStemmer
from sklearn.metrics.pairwise import cosine_similarity
import pickle


def train():

    movies = pd.read_csv("tmdb_5000_movies.csv")
    credits = pd.read_csv("tmdb_5000_credits.csv")

    # Joining both the data frames.
    movies = movies.merge(credits, on = 'title')
    
    # Required columns( this is kept in mind such that anyone would recommend a movie on the basis of below parameters.)
    # Genre
    # Id
    # Keywords(tags)
    # title
    # overview
    # production_companies
    # cast
    # crew for (Director and producer)
    movies = movies[['title', 'id','genres', 'keywords','overview', 'production_companies', 'cast', 'crew']]
    print(movies.shape)


    movies.head(100)
    print(movies.isnull().sum())
    movies.dropna(inplace = True)

    print(movies.isnull().sum())
    movies.shape
    movies.duplicated().sum()
    genres = movies.iloc[0].genres


    def get_genre_names(obj):
        L = []
        for i in json.loads(obj): #string to json format
            L.append(i["name"])
        return L


    movies['genres'] = movies['genres'].apply(get_genre_names)
    movies.head(100)

    movies['keywords'] = movies['keywords'].apply(get_genre_names)
    movies.head(100)

    json.loads(movies['cast'][0])[0]

    def get_cast_names(obj):
        L = []
        for i in json.loads(obj):
            L.append(i['name'])
            if (len(L) == 3):
                return L
        return L

    movies['cast'] = movies['cast'].apply(get_cast_names)
    movies.head(100)

    movies['production_companies'] = movies['production_companies'].apply(get_genre_names)

    def clean_crew_data(obj):
        L = []; # Could have returned directly the name but we need list to easily concancate to find "TAGS" easily.
        for i in json.loads(obj):
            if i['job'] == 'Director' :
                L.append(i['name'])
                return L
        return L
        
    movies['crew'] = movies['crew'].apply(clean_crew_data)
    movies.head(10)
    movies['overview'] = movies['overview'].apply(lambda x:x.split())
    movies['genres'] = movies['genres'].apply(lambda x: [i.replace(" ","") for i in x])
    movies['keywords'] = movies['keywords'].apply(lambda x: [i.replace(" ","") for i in x])
    movies['overview'] = movies['overview'].apply(lambda x: [i.replace(" ","") for i in x])
    movies['production_companies'] = movies['production_companies'].apply(lambda x: [i.replace(" ","") for i in x])
    movies['cast'] = movies['cast'].apply(lambda x: [i.replace(" ","") for i in x])
    movies['crew'] = movies['crew'].apply(lambda x: [i.replace(" ","") for i in x])


    # Creating new tag column
    movies['tags'] = movies['genres'] +movies['keywords'] +movies['overview'] +movies['production_companies'] + movies['cast'] + movies['crew']; 
    new_movies = movies[['title', 'id', 'tags']]

    pd.options.mode.chained_assignment = None
    new_movies.loc[:,'tags'] = new_movies['tags'].apply(lambda x: " ".join(x))
    new_movies.loc[:,'tags'] = new_movies['tags'].apply(lambda x: x.lower())

    cv = CountVectorizer(max_features=10000, stop_words='english')
    vectors_of_movies_according_to_the_most_frequent_words = cv.fit_transform(new_movies['tags']).toarray()
    vector_of_freq_words = cv.get_feature_names_out()

    ps = PorterStemmer()

    def stem_tags(text):
        L = []
        for i in text.split():
            L.append(ps.stem(i))
        return " ".join(L)

    new_movies.loc[:,'tags'] = new_movies['tags'].apply(stem_tags)

    similarity = cosine_similarity(vectors_of_movies_according_to_the_most_frequent_words)
    
    open('similarity.pkl', 'wb').write(pickle.dumps(similarity))
    