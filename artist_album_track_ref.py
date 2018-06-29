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

total_playlists = 0
total_tracks = 0
tracks = set()
artists = set()
albums = set()
titles = set()
ntitles = set()

title_track_histogram = collections.Counter()
artist_track_histogram = collections.Counter()
album_track_histogram = collections.Counter()

max_files_for_quick_processing = 5


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


def show_summary():
    print()
    print ("number of playlists", total_playlists)
    print ("number of tracks", total_tracks)
    print ("avg playlist length", float(total_tracks) / total_playlists)
    print()


    print ("top playlist titles")

    with open('titleTrack.csv', 'w') as csv_file:
        csv_file.write("title;TrackID;Count\n")
        for key, value in title_track_histogram.items():
            try:
                csv_file.write(str(key[0])+";"+str(key[1])+";"+str(value)+"\n")
            except:
                pass

    print()
    print("top albums")

    with open('albumTrack.csv', 'w') as csv_file:
        csv_file.write("AlbumID;TrackID;Count\n")
        for key, value in album_track_histogram.items():
            csv_file.write(str(key[0])+";"+str(key[1])+";"+str(value)+"\n")

    print()
    print ("top artists")


    with open('artistTrack.csv', 'w') as csv_file:
        csv_file.write("ArtistID;TrackID;Count\n")
        for key, value in artist_track_histogram.items():
            csv_file.write(str(key[0]) + ";" + str(key[1]) + ";" + str(value) + "\n")


def normalize_name(name):
    name = name.lower()
    name = re.sub(r"[.,\/#!$%\^\*;:{}=\_`~()@]", ' ', name)
    name = re.sub(r'\s+', ' ', name).strip()
    return name


def process_playlist(playlist):
    global total_playlists, total_tracks

    total_playlists += 1
    # print playlist['playlist_id'], playlist['name']

    nnameList = normalize_name(playlist['name']).split(" ")
    nnameList = [item for item in nnameList if len(item) > 2 ]

    for track in playlist['tracks']:
        total_tracks += 1
        albumID = track['album_uri']
        trackID = track['track_uri']
        artistID = track['artist_uri']

        #populate collections
        artist_track_histogram[(artistID, trackID)] += 1
        album_track_histogram[(albumID, trackID)] += 1
        for name in nnameList:
            title_track_histogram[(name, trackID)] += 1


def process_info(info):
    for k, v in info.items():
        print ("%-20s %s" % (k + ":", v))
    print()


if __name__ == '__main__':
    quick = False
    path = "data"
    process_mpd(path)
