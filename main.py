from flask import Flask, jsonify
import recommend as rec
import os

app = Flask(__name__)

@app.route("/")
def root():
    result = {
        'status' : 200, 'message' : 'Content based recommendation system.'
    }
    return jsonify(result)

@app.route("/recommend/<movieid>")
@app.route("/recommend/<movieid>/<num>",)
def recommend_movie(movieid, num = None):
    return jsonify(rec.recommend_movie(movieid, num or 5))

@app.route("/train")
def train():
    return rec.train()

if __name__ == "__main__":
    app.run(debug=True, port = 8080, host='0.0.0.0')