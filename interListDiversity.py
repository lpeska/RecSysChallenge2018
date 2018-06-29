import sys
import json
import re
import collections
import os

ildForPlaylist = {}
total_playlists = 0
total_tracks = 0
max_files_for_quick_processing = 3

def process_mpd(path):
    count = 0
    filenames = os.listdir(path)
    for filename in sorted(filenames):
        if filename.startswith("mpd.slice.") and filename.endswith(".json"):
            fullpath = os.sep.join((path, filename))
            f = open(fullpath)
            js = f.read()
            f.close()
            mpd_slice = json.loads(js)
            process_info(mpd_slice['info'])
            for playlist in mpd_slice['playlists']:
                process_playlist(playlist)
            count += 1

            if quick and count > max_files_for_quick_processing:
                break

    show_summary()


def process_playlist(playlist):
    global total_playlists, total_tracks

    total_playlists += 1
    # print playlist['playlist_id'], playlist['name']
    sim = 0
    for track1 in playlist['tracks']:
        for track2 in playlist['tracks']:

            if track1['album_uri'] == track2['album_uri']:
                sim += 1
            if track1['artist_uri'] == track2['artist_uri']:
                sim += 1

    averageSim = sim / len(playlist['tracks']) **2
    ildForPlaylist[playlist["pid"]] = averageSim



def process_info(info):
    for k, v in info.items():
        print ("%-20s %s" % (k + ":", v))
    print()


def show_summary():
    import csv
    print()
    print("number of playlists", total_playlists)
    print("number of tracks", total_tracks)
    with open('ildPlaylists.csv', 'w') as csv_file:
        csv_file.write("PID;ILD\n")
        for key, value in ildForPlaylist.items():
            csv_file.write(str(key)+";"+str(value)+"\n")

    import pandas as pd
    df_ild = pd.Series(ildForPlaylist)

    print(df_ild.describe())

if __name__ == '__main__':
    quick = False
    path = "data"
    process_mpd(path)