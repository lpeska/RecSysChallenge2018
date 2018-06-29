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

ildForPlaylist = {}

def process_mpd(path, filename, outFile = "submission.csv"):
    count = 0
    fullpath = os.sep.join((path, filename))
    f = open(fullpath)
    js = f.read()
    f.close()
    mpd_slice = json.loads(js)



    audioTrack = pd.read_csv("trackAudioStd.csv", delimiter=",", header=0, index_col=0)
    audioTrack.set_index("id", inplace=True)

    #audioLearned = pd.read_csv("trackAudioStd.csv", delimiter=",", header=0, index_col=0)
    #audioLearned.set_index("id", inplace=True)

    embeddingsLabel = pd.read_csv("names_audioFeatureSimEmbed2_32.csv", delimiter=";", header=None)
    embeddingsScore = pd.read_csv("audioFeatureSimEmbed2_32.csv", delimiter=";", header=None)
    print(len(embeddingsScore), len(embeddingsLabel))
    maxNames = len(embeddingsLabel)
    maxVals = len(embeddingsScore)
    embeddingsScore.drop(range(maxNames, maxVals) , inplace=True)

    embeddingsScore["label"] = embeddingsLabel[0]
    embeddingsScore.set_index("label", inplace=True)
    del embeddingsLabel

    with open(outFile,"w") as outF:
        outF.write("pid,acousticness,danceability,speechiness,energy,liveness,instrumentalness,mode,loudness,tempo,duration_ms,ALL,REFINED,AUTHOR,LENGTH\n")

        for playlist in mpd_slice['playlists']:
            process_playlist(playlist, outF, audioTrack, embeddingsScore)

        count += 1
        if count % 1000 == 0:
            print(count)


def process_playlist(playlist, outF, audioTrack, embeddingsScore):
    if len( playlist['tracks']) > 0:
        diffAuthor = 0
        emb1 = []
        emb2 = []
        emb1a = []
        emb2a = []
        for track1 in playlist['tracks']:
            tr1 = track1["track_uri"].replace("spotify:track:","")
            for track2 in playlist['tracks']:
                tr2 = track2["track_uri"].replace("spotify:track:", "")
                if track1['artist_uri'] != track2['artist_uri']:
                    diffAuthor += 1

                if (tr1 in audioTrack.index) & (tr2 in audioTrack.index):
                    emb1.append(audioTrack.xs(tr1))
                    emb2.append(audioTrack.xs(tr2))

                if (tr1 in embeddingsScore.index) & (tr2 in embeddingsScore.index):
                    emb1a.append(embeddingsScore.xs(tr1))
                    emb2a.append(embeddingsScore.xs(tr2))

        emb1 = np.asarray(emb1)
        emb2 = np.asarray(emb2)
        embDiff = (emb1 - emb2) **2
        diffPerFeature = np.mean(embDiff, axis=0)
        diffAll = np.mean(embDiff)

        emb1a = np.asarray(emb1a)
        emb2a = np.asarray(emb2a)
        embDiffa = (emb1a - emb2a) **2
        diffRefined = np.mean(embDiffa)

        diffAuthor = diffAuthor / len(playlist['tracks']) **2
        vals = diffPerFeature.tolist()
        vals.extend([diffAll,diffRefined,diffAuthor, len(playlist['tracks'])])

        outF.write(str(playlist["pid"])+","+",".join([str(i) for i in vals])+"\n")





if __name__ == '__main__':
    path = "challenge_set"
    filename = "challenge_set.json"
    outFile = "analyseDistribution2.csv"
    process_mpd(path, filename, outFile)
