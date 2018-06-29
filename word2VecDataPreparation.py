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



max_files_for_quick_processing = 5


def process_mpd(path):
    count = 0
    with open("word2vec_tracks.txt","w") as trackFile:
        with open("word2vec_albums.txt", "w") as albumFile:
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
                        process_playlist(playlist, trackFile, albumFile)
                    count += 1

                    if quick and count > max_files_for_quick_processing:
                        break



def normalize_name(name):
    name = name.lower()
    name = re.sub(r"[.,\/#!$%\^\*;:{}=\_`~()@]", ' ', name)
    name = re.sub(r'\s+', ' ', name).strip()
    return name


def process_playlist(playlist, trackFile, albumFile):
    albs = []
    tracks = []
    for track in playlist['tracks']:

        albs.append( track['album_uri'].replace("spotify:album:",""))
        tracks.append(track['track_uri'].replace("spotify:track:",""))

    trackFile.write(" ".join(tracks))
    albumFile.write(" ".join(albs))

def process_info(info):
    for k, v in info.items():
        print ("%-20s %s" % (k + ":", v))
    print()


if __name__ == '__main__':
    quick = False
    path = "data"
    process_mpd(path)
