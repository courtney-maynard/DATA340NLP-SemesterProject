# DATA340NLP-SemesterProject

Beyond The Beat: Leveraging Lyric Content and Sentiment to Classify Songs Into Genres

Abstract: Music recommendation systems utilize audio features to recommend songs for users, but neglect the lyrical features of songs, such as lyric content, structure, and sentiment. Natural Language Processing techniques can give insight into whether song lyrical information can be used to categorize songs into genres to enable better music recommendation systems. Previous literature reveals gaps in multi-class genre classification, utilizing non-bag-of-words datasets and machine learning architectures that preserve lyric structure, and combining lyric sentiment analysis with lyric word content. Using a dataset of ten musical genres with intact lyrics for each song and comparing Logistic Regression (LR), Bi-Directional Long Short Term Memory (Bi-LSTM), and Hierarchical Attention Network (HAN) machine learning techniques on lyric content indicates the best performance by the HAN model. Comparing lyric sentiment at the song level and lyric level saw a nearly two-fold increase in accuracy for lyric-level models. Combining the best-performing models from the content-only models with the best-performing song level and lyric level sentiment models, Decision Tree Classifier and Recurrent Neural Network respectively, revealed the low confidence of the sentiment models, which decreased overall ensemble performance when combined with LR and Bi-LSTM models. The highest performing model was an ensemble with a HAN and lyric-level Recurrent Neural Network (RNN), with an accuracy of 43.07% averaged across the ten genres. Future work can fine-tune the ensemble models for higher accuracy. 

Full Paper and Code Documentation Available In this Repo.


