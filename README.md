# DATA340NLP-SemesterProject
Spotify Lyrical Analysis and Recommendation System
(Updated October 3rd)

Goals and Outline:
1) Predict whether a song belongs to a specific genre using lyrics instead of production, melody, or instruments.
- Analysis of lyrics of songs belonging to different genres, as categorized by Spotify --> do the categories share similar lyrical patterns?
    - semantics of lyrics (positive, negative, etc.)
    - patterns of lyrics (rhyme scheme, lengths of stanzas, etc.)
    - content of lyrics (words and tokens)
- Create a ml model to categorize songs into different genres
    - Create a custom score for each song, according to each of the three criteria above, and cutoffs/bounds for each criteria according to each genre
    - optimize accuracy
      
2) Create a customized song recommendation system based on lyrical preferences of users
- analyze the person's liked songs, playlist songs, or songs that they input into the system
- use their custom score (the method for creating the score will come from part one) to find songs in the genres they prefer that have similar characteristics, according to the criterion above
