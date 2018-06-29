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


def process_mpd(path, filename, outFile = "submission.csv"):
    count = 0
    fullpath = os.sep.join((path, filename))
    f = open(fullpath)
    js = f.read()
    f.close()
    mpd_slice = json.loads(js)

    trackCounts = pd.read_csv("trackCounter.csv", delimiter=";", header=0)
    print(trackCounts.head())
    trackCounts.sort_values("Count", axis=0, ascending=False, inplace=True)
    print(trackCounts.head())
    recommendedItems = trackCounts.TrackID[0:1000].tolist()



    with open(outFile,"w") as outF:
        outF.write("\nteam_info,main,KSI CUNI.cz,peska@ksi.mff.cuni.cz\n\n")

        for playlist in mpd_slice['playlists']:
            process_playlist(playlist, outF, recommendedItems)

        count += 1
        if count % 1000 == 0:
            print(count)


    f_in = open(outFile, "rb")
    f_out = gzip.open(outFile+".gz", 'wb')
    f_out.writelines(f_in)
    f_out.close()
    f_in.close()


def process_playlist(playlist, outF, recommendedItems):
    pid = playlist["pid"]

    ri = [i for i in recommendedItems]

    for track in playlist['tracks']:
        if track['track_uri'] in ri:
            ri.remove(track['track_uri'])

    outF.write(str(pid) + ", " + ", ".join(ri[0:500]) + "\n" )

    """
    global total_playlists, total_tracks

    total_playlists += 1
    # print playlist['playlist_id'], playlist['name']

    titles.add(playlist['name'])
    nname = normalize_name(playlist['name'])
    ntitles.add(nname)
    title_histogram[nname] += 1
    full_title_histogram[playlist['name'].lower()] += 1

    for track in playlist['tracks']:
        total_tracks += 1
        albums.add(track['album_uri'])
        tracks.add(track['track_uri'])
        artists.add(track['artist_uri'])

        full_name = track['track_name'] + " by " + track['artist_name']
        artist_histogram[track['artist_uri']] += 1
        album_histogram[track['album_uri']] += 1
        track_histogram[track['track_uri']] += 1
    """


def normalize_name(name):
    name = name.lower()
    name = re.sub(r"[.,\/#!$%\^\*;:{}=\_`~()@]", ' ', name)
    name = re.sub(r'\s+', ' ', name).strip()
    return name



if __name__ == '__main__':
    path = "challenge_set"
    filename = "challenge_set.json"
    outFile = "solution_mostPopular.csv"
    process_mpd(path, filename, outFile)
