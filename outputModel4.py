"""
    shows deep stats for the MPD
    - calculates similarity and multiplies it with the track popularity
    - if the track was unknown, decrease its similarity weight
    usage:

        python deeper_stats.py path-to-mpd-data/
"""
#import sys
import json
import re
import collections
import os
import gzip
import pandas as pd
import numpy as np
#from sklearn.neighbors import NearestNeighbors
import pickle
from collections import defaultdict
import heapq
import math

MOST_POPULAR_WEIGHT = 0.000002

TITLE_WEIGHT = 0.01

ALBUM_WEIGHT = 0.1

ARTIST_WEIGHT = 0.01

#"challenge_track_predImprSim_256418_64.csv",
#"challenge_track_scoreImprSim_256418_64.csv",
#"challenge_track_names.csv",
#1.0,

relevantTrackKNNs = ["challenge_track_predNNSim_256418_64.csv", "challenge_track_predImprSim_256418_64.csv"]
relevantTrackScores = [ "challenge_track_score2NNSim_256418_64.csv", "challenge_track_scoreImprSim_256418_64.csv"]
relevantTrackNames = ["challenge_track_names.csv", "challenge_track_names.csv"]
trackWeights = [ 1.0, 1.0]

relevantAlbKNNs = ["challenge_track_predNNAlbFinal_250561_64.csv"]
relevantAlbScores = ["challenge_track_scoreNNAlbFinal_250561_64.csv"]
relevantAlbNames = ["challenge_album_names.csv"]
albWeights = [0.1]

trackDT = zip(relevantTrackKNNs, relevantTrackScores, relevantTrackNames, trackWeights)
albDT = zip(relevantAlbKNNs, relevantAlbScores, relevantAlbNames, albWeights)


def softmax(x):
    """
    Compute softmax values for each sets of scores in x.
    """
    return x / x.sum()
    #scoreMatExp = np.exp(np.asarray(x))
    #return scoreMatExp / scoreMatExp.sum()


def normalize_name(name):
    name = name.lower()
    name = re.sub(r"[.,\/#!$%\^\*;:{}=\_`~()@]", ' ', name)
    name = re.sub(r'[^\x00-\x7F]','', name)
    name = re.sub(r'\s+', ' ', name).strip()
    return name


def processKNNs(params):
    nnName, scoreName, labelName, weight = params
    embeddingsVector = pd.read_csv(nnName, delimiter=";", header=None)
    embeddingsScore = pd.read_csv(scoreName, delimiter=";", header=None)
    embeddingsLabel = pd.read_csv(labelName, delimiter=";", header=None)
    print(len(embeddingsVector), len(embeddingsLabel))

    maxNames = len(embeddingsLabel)
    maxVals = len(embeddingsVector)
    embeddingsVector.drop(range(maxNames, maxVals) , inplace=True)
    embeddingsScore.drop(range(maxNames, maxVals), inplace=True)
    embeddingsScore *= weight

    embeddingsVector["label"] = embeddingsLabel[0]
    embeddingsVector.set_index("label", inplace=True)

    embeddingsScore["label"] = embeddingsLabel[0]
    embeddingsScore.set_index("label", inplace=True)

    #print(embeddingsVector.iloc[0:5,0:5])

    del embeddingsLabel
    return embeddingsVector, embeddingsScore


def save_obj(obj, name ):
    with open('obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


def dsum(dct1, dct2):
        for k, v in dct2.items():
            dct1[k] += v
        return dct1


def process_mpd(path, filename, outFile, trackDT, albDT):
    count = 0
    fullpath = os.sep.join((path, filename))
    f = open(fullpath)
    js = f.read()
    f.close()
    mpd_slice = json.loads(js)
    trackKNNs = dict()
    albTrackKNNs = defaultdict(int)

    tracks = pd.read_csv("trackCounter.csv", delimiter=";", header=0)
    tracks.set_index("TrackID", inplace=True)
    tracks.drop(tracks[tracks.Count < 2].index, inplace=True)

    #tracks.sort_values("Count", axis=0, ascending=False, inplace=True)
    #mostPopularTracks = tracks.iloc[:1000]
    #tracks = tracks.Count.to_dict()

    albumTrack = pd.read_csv("albumTrack.csv", delimiter=";", header=0)
    #albumTrack.Count = albumTrack.Count.multiply( ALBUM_WEIGHT )
    albumTrack.set_index("AlbumID", inplace=True)
    albumTrack.drop(albumTrack[albumTrack.Count < 10].index, inplace=True)
    albumTrack.sort_values("Count", axis=0, ascending=False, inplace=True)

    print(albumTrack.head())
    print(albumTrack.shape)


    artistTrack = pd.read_csv("artistTrack.csv", delimiter=";", header=0)

    #artistTrack.drop(artistTrack[artistTrack.Count < 1].index, inplace=True)
    artistTrack.set_index("ArtistID", inplace=True)
    artistTrack.drop(artistTrack[artistTrack.Count < 2].index, inplace=True)
    #meanArtist = artistTrack.groupby(artistTrack.index)[['Count']].mean()
    #artistTrack.Count = artistTrack.Count.multiply(ARTIST_WEIGHT)


    print(artistTrack.head())
    print(artistTrack.shape)


    titleTrack = pd.read_csv("titleTrack.csv", delimiter=";", header=0, encoding = "ISO-8859-1", error_bad_lines=False)
    #titleTrack.Count = titleTrack.Count.multiply( TITLE_WEIGHT )
    titleTrack.set_index("title", inplace=True)
    titleTrack.drop(titleTrack[titleTrack.Count < 2].index, inplace=True)
    print(titleTrack.head())
    print(titleTrack.shape)


    audioTrackPD = pd.read_csv("trackAudioStd.csv", delimiter=",", header=0, index_col=0)
    audioTrackPD.set_index("id", inplace=True)
    adt = defaultdict(np.float32)
    audioTrack = audioTrackPD.to_dict(into=adt)
    del audioTrackPD

    audioPlaylistAnalysis = pd.read_csv("analyseDistribution2.csv", sep=",", header=0, index_col=0)
    audioPlaylistAnalysis = audioPlaylistAnalysis.loc[audioPlaylistAnalysis.LENGTH > 1]
    audioPlaylistAnalysis.acousticness = audioPlaylistAnalysis.acousticness * 1 / np.log(audioPlaylistAnalysis.LENGTH)
    audioPlaylistAnalysis.danceability = audioPlaylistAnalysis.danceability * 1 / np.log(audioPlaylistAnalysis.LENGTH)
    audioPlaylistAnalysis.speechiness = audioPlaylistAnalysis.speechiness * 1 / np.log(audioPlaylistAnalysis.LENGTH)
    audioPlaylistAnalysis.energy = audioPlaylistAnalysis.energy * 1 / np.log(audioPlaylistAnalysis.LENGTH)
    audioPlaylistAnalysis.liveness = audioPlaylistAnalysis.liveness * 1 / np.log(audioPlaylistAnalysis.LENGTH)
    audioPlaylistAnalysis.instrumentalness = audioPlaylistAnalysis.instrumentalness * 1 / np.log(audioPlaylistAnalysis.LENGTH)
    audioPlaylistAnalysis["mode"] = audioPlaylistAnalysis["mode"] * 1 / np.log(audioPlaylistAnalysis.LENGTH)
    audioPlaylistAnalysis.loudness = audioPlaylistAnalysis.loudness * 1 / np.log(audioPlaylistAnalysis.LENGTH)
    audioPlaylistAnalysis.tempo = audioPlaylistAnalysis.tempo * 1 / np.log(audioPlaylistAnalysis.LENGTH)
    audioPlaylistAnalysis.duration_ms = audioPlaylistAnalysis.duration_ms * 1 / np.log(audioPlaylistAnalysis.LENGTH)
    audioPlaylistAnalysis.AUTHOR = audioPlaylistAnalysis.AUTHOR * 1 / np.log(audioPlaylistAnalysis.LENGTH)
    audioPlaylistAnalysis.REFINED = audioPlaylistAnalysis.REFINED * 1 / np.log(audioPlaylistAnalysis.LENGTH)

    audioTrack2PD = pd.read_csv("audioFeatureSimEmbed2_32.csv", delimiter=";", header=None, index_col=None)
    audioTrack2PD_names = pd.read_csv("names_audioFeatureSimEmbed2_32.csv", delimiter=";", header=None, index_col=None)
    maxNames = len(audioTrack2PD_names)
    maxVals = len(audioTrack2PD)
    audioTrack2PD.drop(range(maxNames, maxVals), inplace=True)
    audioTrack2PD["label"] = audioTrack2PD_names[0]
    audioTrack2PD.set_index("label", inplace=True)
    audioTrack2 = dict(zip(audioTrack2PD.index.values,audioTrack2PD.values.tolist()))

    del audioTrack2PD


    for params in albDT:
        knn,score = processKNNs(params)
        c = 0
        for alb in knn.index.values:

            albDict = albTrackKNNs.get(alb,defaultdict(int))

            simAlbs = knn.xs(alb).values.tolist()[0:10]
            simScore = score.xs(alb).values.tolist()[0:10]
            zipAlbs = zip(simAlbs, simScore)
            for (a,s) in zipAlbs:
                try:
                    trs = albumTrack.xs(a)
                    if isinstance(trs, pd.core.frame.DataFrame):
                        #trs = trs.iloc[:10]
                        #print(trs.head())
                        dct = dict(zip(trs.TrackID.tolist(), [s]*len(trs)))
                        albDict = dsum(albDict, dct)

                        #for index, row in trs.iterrows():
                        #    albDict[row["TrackID"]] = albDict.get(row["TrackID"],0) + (row["Count"] * s)
                    else:
                        albDict[trs["TrackID"]] = albDict.get(trs["TrackID"], 0) + s
                except:
                    pass

            albTrackKNNs[alb] = albDict
            c += 1
            if c % 1000 == 0:
                print(c)
        del knn
        del score
    print(len(albTrackKNNs))

    for params in trackDT:
        knn,score = processKNNs(params)
        c = 0
        for tr in knn.index.values:
            trDict = trackKNNs.get(tr, defaultdict(int))
            simTracks = knn.xs(tr).values.tolist()[0:750]
            simScore = score.xs(tr).values.tolist()[0:750]

            dct = dict(zip(simTracks, simScore))
            trDict = dsum(trDict, dct)

            trackKNNs[tr] = trDict
            c += 1
            if c % 1000 == 0:
                print(c)
        del knn
        del score
    print(len(trackKNNs))


    tracks.sort_values("Count", axis=0, ascending=False, inplace=True)
    mostPopularTracks = tracks.iloc[:1000]


    #with open(outFile,"w") as outF:
    with open(outFile, "w") as outF:
        outF.write("\nteam_info,creative,KSI CUNI.cz,peska@ksi.mff.cuni.cz\n\n")

        for playlist in mpd_slice['playlists']:


            count += 1
            if count < 1000:
                continue
            process_playlist(playlist, outF, tracks, mostPopularTracks, titleTrack, artistTrack, albumTrack, trackKNNs, albTrackKNNs , audioTrack, audioTrack2, audioPlaylistAnalysis )  # audioTrack, nnAudio


            if count % 10 == 0:
                print(count)


    f_in = open(outFile, "rb")
    f_out = gzip.open(outFile+".gz", 'wb')
    f_out.writelines(f_in)
    f_out.close()
    f_in.close()



def process_playlist(playlist, outF, tracks, mostPopularTracks, titleTrack, artistTrack, albumTrack, trackKNNs, albTrackKNNs , audioTrack, audioTrack2, audioPlaylistAnalysis ):
    artists = collections.Counter()
    albums = collections.Counter()
    titles = collections.Counter()
    trackURIs = []

    pid = playlist["pid"]
    try:
        nnameList = normalize_name(playlist['name']).split(" ")
        titleList = [item for item in nnameList if len(item) > 2 ]
    except:
        titleList = []



    recommendedItems = defaultdict(int)

    #for index, row in mostPopularTracks.iterrows():
    #    recommendedItems[index] = row["Count"]

    for t in titleList:
        titles[t] = titles.get(t,0)+ 1

    for track in playlist['tracks']:
        baseTrackUri = track['track_uri'].replace("spotify:track:", "")
        baseAlbUri = track['album_uri'].replace("spotify:album:", "")
        baseArtUri = track['artist_uri'].replace("spotify:artist:", "")
        trackURIs.append(baseTrackUri)
        artists[baseArtUri] += 1
        albums[baseAlbUri] += 1

        if baseTrackUri in trackKNNs:
            recommendedItems = dsum(recommendedItems, trackKNNs[baseTrackUri])

        if baseAlbUri in albTrackKNNs:
            recommendedItems = dsum(recommendedItems, albTrackKNNs[baseAlbUri])


    #Weight artists more if they are consistent in the playlist
    artistsWeight = 1
    if playlist["pid"] in audioPlaylistAnalysis.index:
            if audioPlaylistAnalysis["AUTHOR"][playlist["pid"]] <= 0.05:
                artistsWeight = 1 / (0.5 + audioPlaylistAnalysis["AUTHOR"][playlist["pid"]])
    err = 0
    for idx, val in artists.most_common():
        try:
            at = artistTrack.xs(idx, drop_level=False)
            #atMean = meanArtist.Count[idx]
            #print(type(at))
            if isinstance(at, pd.core.frame.DataFrame):
                if len(at) > 1000:
                    at = at.nlargest(1000, "Count", keep='first')
                keys = at.TrackID.values
                vals = at.Count.values
                vals = softmax(vals) * artistsWeight * val * ARTIST_WEIGHT

                dct = dict(zip(keys, vals))
                recommendedItems = dsum(recommendedItems,dct)

                #for index, row in at.iterrows():
                #    recommendedItems[row["TrackID"]] += row["Count"]* artistsWeight * val/atMean
            else:
                recommendedItems[at["TrackID"]] +=  artistsWeight * val #*at["Count"]/atMean
        except:
            err += 1
    print("artists err. for " + str(err))

    err = 0
    for idx, val in albums.most_common():
        try:
            at = albumTrack.xs(idx, drop_level=False)
            #atMean = meanArtist.Count[idx]
            #print(type(at))
            if isinstance(at, pd.core.frame.DataFrame):
                if len(at) > 1000:
                    at = at.nlargest(1000, "Count", keep='first')
                keys = at.TrackID.values
                vals = at.Count.values
                vals = softmax(vals) * val * ALBUM_WEIGHT

                dct = dict(zip(keys, vals))
                recommendedItems = dsum(recommendedItems,dct)

                #for index, row in at.iterrows():
                #    recommendedItems[row["TrackID"]] += row["Count"]* artistsWeight * val/atMean
            else:
                recommendedItems[at["TrackID"]] +=  artistsWeight * val #*at["Count"]/atMean
        except:
            err += 1
    print("album err. for " + str(err))


    for idx, val in titles.most_common():
        # print(idx, val)
        try:
            at = titleTrack.xs(idx)
            # print(at.shape)
            if len(at) > 1000:
                at = at.nlargest(1000, "Count", keep='first')
            keys = at.TrackID.values
            vals = at.Count.values
            vals = softmax(vals) * val * TITLE_WEIGHT

            dct = dict(zip(keys, vals))
            recommendedItems = dsum(recommendedItems, dct)

            #for index, row in at.iterrows():
            #    recommendedItems[row["TrackID"]] += row["Count"] * val
        except:
            print("title err. for" + idx)
            pass

    #add content-based KNN features if their overall variance w.r.t. audio embednings is low, or the playlist has only one song
    def at2DictAcc(key):
        return audioTrack2.get(key, [0.0]*32)
    at2DictVector = np.vectorize(at2DictAcc, otypes=[object])
    doAudioKNN = False
    if (playlist["pid"] in audioPlaylistAnalysis.index):
        if audioPlaylistAnalysis["REFINED"][playlist["pid"]] <= 0.05:
            doAudioKNN = True
    elif len(trackURIs) == 1:
        doAudioKNN = True

    if doAudioKNN:
        #ct = at2DictVector(trackURIs)
        ct = list(map(at2DictAcc, trackURIs))
        #print(ct)

        currTracksMean = np.asarray(ct)
        #print(currTracksMean)
        if currTracksMean.ndim == 2:
            currTracksMean = np.mean(currTracksMean,axis=0)
        #print(currTracksMean)
        vals = np.asarray(list(audioTrack2.values()))
        keys = np.asarray(list(audioTrack2.keys()))
        audioSimVal = 1/(0.5 + np.sum((vals - currTracksMean) ** 2,axis=1))
        #print(audioSimVal.shape)
        topk = heapq.nlargest(1000, range(len(audioSimVal)), audioSimVal.take)
        #print(topk)
        dct = dict(zip(keys[topk], audioSimVal[topk]))
        recommendedItems = dsum(recommendedItems, dct)



    #re-rank songs if the stability w.r.t. audio feature is high within playlist
    def accDictAcc(feature,key):
        return audioTrack[feature][key]
    accDictVector = np.vectorize(accDictAcc)

    #weight feature more, if it is consistent throughout the playlist
    if playlist["pid"] in audioPlaylistAnalysis.index:
        for feature in ["acousticness","danceability","speechiness","energy","liveness","instrumentalness","mode","loudness","tempo","duration_ms"]:
            if audioPlaylistAnalysis[feature][playlist["pid"]] <= 0.05:
                currTracksMean = np.mean(accDictVector([feature], trackURIs))
                keys = list(recommendedItems.keys())
                recTrackFeatures = accDictVector([feature], keys)
                weight = 1 / (0.5 + (recTrackFeatures - currTracksMean)**2)
                dct = dict(zip(keys,weight))
                recommendedItems = dsum(recommendedItems, dct)


    #add popularity bias here
    notFound = 0
    total = 0
    for idx,val in recommendedItems.items():
        total += 1
        try:
            recommendedItems[idx] *= (1+ math.log(tracks.Count[idx]))
        except:
            notFound += 1
    print("Not found "+str(notFound)+" out of "+str(total)+" tracks")

    vals = mostPopularTracks.Count.values
    keys = mostPopularTracks.index.values
    vals = softmax(vals) * MOST_POPULAR_WEIGHT
    dct = dict(zip(keys.tolist(),vals.tolist()))
    recommendedItems = dsum(recommendedItems, dct)

    ri = list(sorted(recommendedItems, key=recommendedItems.__getitem__, reverse=True))[0:1000]
    ri = ["spotify:track:"+i for i in ri]
    #ri = [tid for (tid, _) in recommendedItems.most_common(1000)]
    # print(recommendedItems.most_common(10))

    for track in playlist['tracks']:
        if track['track_uri'] in ri:
            ri.remove(track['track_uri'])

    if "spotify:track:UNK" in ri:
        ri.remove("spotify:track:UNK")

    outF.write(str(pid) + ", " + ", ".join(ri[0:500]) + "\n")




if __name__ == '__main__':
    path = "challenge_set"
    filename = "challenge_set.json"
    #outFile = "solution_Word2Vec_improvedAllVecs.csv"
    outFile = "solution_OM4LogPOP.csv"

    process_mpd(path, filename, outFile, trackDT, albDT)
