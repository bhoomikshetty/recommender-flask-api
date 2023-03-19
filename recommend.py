import train as tr
import pandas as pd
import os.path

def recommend_movie(movieId, number):

    os.path.exists("similarity.pkl") or tr.train()

    similarity = pd.read_pickle('similarity.pkl')
    new_movies = pd.read_csv("tmdb_5000_movies.csv")

    result = []
    if list(new_movies['id']).count(int(movieId)) == 0:
        return {'status': 404,'message': 'Movie not found'}
    
    movie_index = new_movies[new_movies['id'] == int(movieId)].index[0]
    similar_movies = sorted(list(enumerate(similarity[movie_index])), reverse=True, key=lambda x:x[1])[1:int(number)]
    for i in similar_movies:
        result.append(int(new_movies.iloc[i[0]].id))
    
    return {'status': 200,'message': result}
