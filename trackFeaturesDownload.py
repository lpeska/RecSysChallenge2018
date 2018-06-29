"""
    shows deep stats for the MPD

    usage:

        python deeper_stats.py path-to-mpd-data/
"""
import sys
import json
import re
import collections
import os
import gzip
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import spotipy
import spotipy.oauth2 as oauth2


def process_counter(path, filename, outFile = "titleVecotrs.csv", vecFile = "embed_word2vec_2000000_32.csv"):
    """
    titleTrackCounter = pd.read_csv(filename, delimiter=";", header=0, encoding = "ISO-8859-1", error_bad_lines=False)
    titleVectors = {}
    embeddingsVector, names, embedNameDict, embedNameDictReverse = processEmbedings(vecFile)


    for index, row in titleTrackCounter.iterrows():
        try:
            w2vTrackID = embedNameDictReverse[row.TrackID.replace("spotify:track:","")]
            vector = embeddingsVector.xs(w2vTrackID)
            if row.title in titleVectors:
                titleVectors[row.title] = titleVectors[row.title] + (vector * row.Count)
            else:
                titleVectors[row.title] = (vector * row.Count)
        except:
            pass
    titleDF = pd.DataFrame.from_dict(titleVectors, orient="index")
    """
    trackDF = pd.read_csv("trackCounter.csv", header=0, sep=";")
    print(trackDF.head(5))
    trackDF.sort_values("Count", inplace=True, ascending=False)
    print(trackDF.head(5))


    """
    SPOTIPY_CLIENT_ID = 'your_client_id'
    SPOTIPY_CLIENT_SECRET = 'your_client_secret'
    SPOTIPY_REDIRECT_URI = 'http://localhost:8080'
    SCOPE = 'user-library-read'
    CACHE = '.spotipyoauthcache'

    sp_oauth = oauth2.SpotifyOAuth(SPOTIPY_CLIENT_ID, SPOTIPY_CLIENT_SECRET, SPOTIPY_REDIRECT_URI, scope=SCOPE,cache_path=CACHE)
    token_info = sp_oauth.get_cached_token()
    access_token = token_info['access_token']
    """
    access_token = "BQB3RWRT5c0r5J9AAye825SXbXtq0xykw2EuUu7DMSq4TUWf9SmjUGbatH4aZkC3fXzMCV8Z58ANXeMLXRwqW1W39yhaKH3SMoP03sQTm3kJa98yuLjF7uQuj6wDspw_Tdmm_Ed3CDJVF3f14GvcjZJGCZ38p-U"




    sp = spotipy.Spotify(auth=access_token)
    with open("trackAcoustics.csv","a") as f:
        for k in range(43281,trackDF.shape[0]//50):
            bound1 = k*50
            bound2 = (k+1) * 50
            print(bound1, bound2)
            tracks = trackDF.TrackID.iloc[bound1: bound2]
            out = sp.audio_features(tracks)
            #print(out)

            feat = [",".join([i["id"],
                              str(i["acousticness"]),
                              str(i["danceability"]),
                              str(i["loudness"]),
                              str(i["valence"]),
                              str(i["tempo"]),
                              str(i["time_signature"]),
                              str(i["speechiness"]),
                              str(i["energy"]),
                              str(i["liveness"]),
                              str(i["mode"]),
                              str(i["instrumentalness"]),
                              str(i["duration_ms"])])+"\n" for i in out if not i is None]
            f.writelines(feat)
            print(k, trackDF.Count.iloc[bound2])




    """
    nnModel = NearestNeighbors(n_neighbors=5, metric="cosine")
    nnModel.fit(titleDF)

    distances, indices = nnModel.kneighbors(titleDF.iloc[0:150])
    distances = 1-distances
    for i in range(indices.shape[0]):
        dct = {}
        key =  titleDF.index[i]
        for j in range(indices.shape[1]):
            dct[ titleDF.index[indices[i,j]] ] = distances[i,j]

        print(key)
        print(dct)

    def idconvert(z):
        return titleDF.index[z]

    idconvertVec = np.vectorize(idconvert)
    indices = idconvertVec(indices)
    print(indices)

    #titleDF.to_csv(outFile, header=False)
    """



if __name__ == '__main__':
    path = ""
    filename = "titleTrack.csv"
    process_counter(path, filename)
