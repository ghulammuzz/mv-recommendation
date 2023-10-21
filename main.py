from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import pickle

app = FastAPI()

df = pd.read_csv("mov.csv")

with open('recommendation_model.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)

class Item(BaseModel):
    idx: int

@app.post("/recommendation/")
def get_recommendations(item: Item):
    n = item.idx
    if n > df.shape[0]:
        return {"error": "Requested index is greater than the number of items available."}
    movie_title = df.iloc[n]['title']
    indices = pd.Series(df.index, index=df['title'])
    idx = indices[movie_title]
    sim_scores = list(enumerate(loaded_model[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]  # Ambil 5 film paling mirip
    movie_indices = [i[0] for i in sim_scores]
    recommended_movies = df['title'].iloc[movie_indices].tolist()
    return {"recommendations": recommended_movies}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
