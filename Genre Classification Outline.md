Problem: Spotify has thousands of different genre categorization for their songs and artists, and does not make the genre of each song publicly available to developers.
Solution: Everynoise; a website aimed at mapping and categorizing all the songs on Spotify, and their corresponding account The Sounds of Spotify, which contains playlists for all of the genres.

Outline of Consideration Points
1) Use webscraping to gather the top 1000 genres (out of 6000 currently tracked) from everynoise.com/everynoise1d.cgi?scope=all (the list of all genres in order of popularity)
  a. The corresponding playlist links will be collected
  b. The IDs needed to be compatible with spotipy API will be parsed from the collected playlist links
2) All playlists will be collected on the same day, as the playlists are itterated on and updated frequently. Thus, I can provide a snapshot of the genre on a certain date in time, acknowledging that music changes.
3) The playlists contain anywhere from 50-500 songs, and the spotify API limits querying songs to 100 at a time.  
  a. Option 1: split all songs into different playlists, possibly manually
  b. Option 2: use some kind of get/fetching operation to get 100 max at once and continue to iterate through the playlist
4) Alternate possibility would be to consolidate the playlists into different overarching genres, which would require more manual work and less webscrapping and automated work.
