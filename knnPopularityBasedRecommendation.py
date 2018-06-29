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

total_playlists = 0
total_tracks = 0
tracks = set()
artists = set()
albums = set()
titles = set()
ntitles = set()
full_title_histogram = collections.Counter()
title_histogram = collections.Counter()
artist_histogram = collections.Counter()
album_histogram = collections.Counter()
track_histogram = collections.Counter()
max_files_for_quick_processing = 5


MOST_POPULAR_WEIGHT = 0.0002
TITLE_WEIGHT = 0.1
ALBUM_WEIGHT = 1.0
ARTIST_WEIGHT = 0.1

def normalize_name(name):
    name = name.lower()
    name = re.sub(r"[.,\/#!$%\^\*;:{}=\_`~()@]", ' ', name)
    name = re.sub(r'\s+', ' ', name).strip()
    return name

def process_mpd(path, filename, outFile = "submission.csv"):
    count = 0
    fullpath = os.sep.join((path, filename))
    f = open(fullpath)
    js = f.read()
    f.close()
    mpd_slice = json.loads(js)



    mostPopularTracks = pd.read_csv("trackCounter.csv", delimiter=";", header=0)
    mostPopularTracks.Count = np.round(mostPopularTracks.Count * MOST_POPULAR_WEIGHT)
    mostPopularTracks.sort_values("Count", axis=0, ascending=False, inplace=True)
    mostPopularTracks = mostPopularTracks.iloc[:1000]

    albumTrack = pd.read_csv("albumTrack.csv", delimiter=";", header=0)
    albumTrack.Count = np.round(albumTrack.Count.multiply( ALBUM_WEIGHT ))
    albumTrack.drop(albumTrack[albumTrack.Count < 1].index, inplace=True)
    print(albumTrack.head())
    print(albumTrack.shape)

    artistTrack = pd.read_csv("artistTrack.csv", delimiter=";", header=0)
    artistTrack.Count = np.round(artistTrack.Count.multiply( ARTIST_WEIGHT ) )
    artistTrack.drop(artistTrack[artistTrack.Count < 1].index, inplace=True)
    print(artistTrack.head())
    print(artistTrack.shape)

    titleDF = pd.read_csv("titleVecotrs.csv", header=None, index_col=0)
    print(titleDF.head(2))
    titleNNModel = NearestNeighbors(n_neighbors=50, metric="cosine")
    titleNNModel.fit(titleDF)


    titleTrack = pd.read_csv("titleTrack.csv", delimiter=";", header=0, encoding = "ISO-8859-1", error_bad_lines=False)
    titleTrack.Count = np.round(titleTrack.Count.multiply( TITLE_WEIGHT )  )
    titleTrack.drop(titleTrack[titleTrack.Count < 1].index, inplace=True)


    print(titleTrack.head())
    print(titleTrack.shape)




    with open(outFile,"w") as outF:
        outF.write("\nteam_info,main,KSI CUNI.cz,peska@ksi.mff.cuni.cz\n\n")

        for playlist in mpd_slice['playlists']:
            process_playlist(playlist, outF, mostPopularTracks, albumTrack, artistTrack, titleTrack, titleDF, titleNNModel)

        count += 1
        if count % 1000 == 0:
            print(count)


    f_in = open(outFile, "rb")
    f_out = gzip.open(outFile+".gz", 'wb')
    f_out.writelines(f_in)
    f_out.close()
    f_in.close()



def process_playlist(playlist, outF, mostPopularTracks, albumTrack, artistTrack, titleTrack, titleDF, titleNNModel):
    simTitleConst = 0.1
    pid = playlist["pid"]
    try:
        nnameList = normalize_name(playlist['name']).split(" ")
        titleList = [item for item in nnameList if len(item) > 2 ]
    except:
        titleList = []
    #print(titleList)

    def idconvert(z):
        return titleDF.index[z]

    idconvertVec = np.vectorize(idconvert)


    extTitleList = collections.Counter()
    for t in titleList:
        try:
            tVec = titleDF.xs(t)
            if isinstance(tVec, pd.core.frame.DataFrame):
                pass
            else:
                tVec = [tVec]
            if tVec.shape[0] > 0:
                distances, indices = titleNNModel.kneighbors(tVec)
                sim = (1 - distances) * simTitleConst
                sim = sim.flatten()
                indices = indices.flatten()
                indices = idconvertVec(indices)
                extTitleList.update(dict(zip(indices, sim)))

        except:
            pass


    recommendedItems = collections.Counter()
    for index, row in mostPopularTracks.iterrows():
        recommendedItems[row["TrackID"]] += row["Count"]

    #print(recommendedItems.most_common(10))

    albums = collections.Counter()
    artists = collections.Counter()
    titles = collections.Counter()

    #print(extTitleList)
    for t in titleList:
        titles[t] += 1
    for i, j in extTitleList.items():
        titles[i] += j
    #print(titles)

    for track in playlist['tracks']:
        albums[track['album_uri']] += 1
        artists[track['artist_uri']] += 1 #rerun; there was an error


    for idx, val in albums.most_common():
        at = albumTrack.loc[albumTrack.AlbumID == idx]
        for index, row in at.iterrows():
            recommendedItems[row["TrackID"]] += row["Count"]*val

    for idx, val in artists.most_common():
        at = artistTrack.loc[artistTrack.ArtistID == idx]
        for index, row in at.iterrows():
            recommendedItems[row["TrackID"]] += row["Count"]*val

    for idx, val in titles.most_common():
        #print(idx, val)
        at = titleTrack.loc[titleTrack.title == idx]
        #print(at.shape)
        for index, row in at.iterrows():
            recommendedItems[row["TrackID"]] += row["Count"]*val


    ri = [tid for (tid, _) in recommendedItems.most_common(1000) ]

    #print(recommendedItems.most_common(10))

    for track in playlist['tracks']:
        if track['track_uri'] in ri:
            ri.remove(track['track_uri'])

    outF.write(str(pid) + ", " + ", ".join(ri[0:500]) + "\n" )




if __name__ == '__main__':
    path = "challenge_set"
    filename = "challenge_set.json"
    outFile = "solution_sameTitleAlbumArtistSimTitle.csv"
    process_mpd(path, filename, outFile)
