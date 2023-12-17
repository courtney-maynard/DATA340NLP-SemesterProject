# DATA340NLP-SemesterProject



Beyond The Beat: Leveraging Lyric Content and Sentiment to Classify Songs Into Genres


Courtney Maynard
Department of Data Science, College of William and Mary
DATA 340: Natural Language Processing
Dr. James Tucker
December 19th, 2023



























ABSTRACT
Music recommendation systems utilize audio features to recommend songs for users, but neglect lyrical features of songs, such as lyric content, structure, and sentiment. Natural Language Processing techniques can give insight into whether song lyrical information can be used to categorize songs into genres to enable better music recommendation systems. Previous literature reveals gaps in multi-class genre classification, utilizing non-bag-of-words datasets and machine learning architectures that preserve lyric structure, and combining lyric sentiment analysis with lyric word content. Using a dataset of ten musical genres with intact lyrics for each song and comparing Logistic Regression (LR), Bi-Directional Long Short Term Memory (Bi-LSTM), and Hierarchical Attention Network (HAN) machine learning techniques on lyric content indicates the best performance by the HAN model. Comparing lyric sentiment at the song level and lyric level saw a nearly two-fold increase in accuracy for lyric-level models. Combining the best-performing models from the content-only models with the best-performing song level and lyric level sentiment models, Decision Tree Classifier and Recurrent Neural Network respectively, revealed the low confidence of the sentiment models, which decreased overall ensemble performance when combined with LR and Bi-LSTM models. The highest performing model was an ensemble with a HAN and lyric-level Recurrent Neural Network (RNN), with an accuracy of 43.07% averaged across the ten genres. Future work can fine-tune the ensemble models for higher accuracy. 
1. INTRODUCTION
The popularity of using music recommendation systems to discover new music has increased in recent years, spurring many questions regarding how music is classified into genres or interests. With an increase in interest in music categorization, there has been more scholarly attention on music genre classification, particularly regarding using auditory signals. Though many studies have indicated that auditory features such as bass, timbre, and length of song can be indicative of song genre, there is less research regarding song lyrics as a genre indicator. This report seeks to investigate whether lyric content and sentiment with preserved lyrical structure can be used to classify music into various genres. While some previous works have used machine learning techniques to classify songs using lyric content and structure, there is a lack of research into whether the perceived emotional content or sentiment of a song is indicative of its genre. 
The logic and techniques behind popular music recommendation systems remain largely unknown. While it is known that audio features are used by music recommendation systems, such as Spotify, it is unknown whether individual users are receiving songs that may appeal to them lyrically. Discovering whether lyrics can be used to classify songs into different genres leads to the potential to build more personally tailored music recommendation systems for users who prefer to listen to music with a focus on lyrics, rather than on audio or production. 
2. LITERATURE REVIEW 
Many studies have focused on the audio features of songs, with the exploration of the usefulness of lyrics for song genre classification emerging primarily within the past five years. Within the context of this study, relevant literature falls under the categories of dataset evaluation, emotional and sentiment analysis techniques, embedding techniques, and model architecture variations.
2.1 Dataset Evaluation
One hindrance to large-scale lyrical analysis is the availability of intact lyric datasets (Tsaptsinos, 2017). A majority of research on song genre classification utilizes bag-of-word lyrics or subsets of full lyric corpi due to copyright restrictions or incomplete datasets (Tsaptsinos, 2017). In some studies, researchers have collected personal lyrical data and connected it with existing song genre data sets, such as the Million Song Data Set or various crowd-sourced datasets on Kaggle, as a way of circumventing the bag of words problem for lyric analysis (Liang, Gu, & O’Connor, 2011) (Kumar, Rajpal, & Rathore, 2018) (Dammann & Haugh, 2017). Genre classification is a difficult task due to the large number of genres and ‘micro-genres’ that songs are classified into; Spotify, for example, has over 6,000 micro-genres, such as ‘East Coast Hip Hop’ or ‘South African Pop Dance.’ Due to the proliferation of these micro-genres, all previous studies approach the classification problem in regards to larger, more expansive genres such as ‘Indie’ or ‘Rock’ with a note that classification for micro-genres is an area of further exploration. Most previous studies attempted to generalize the songs into fewer than five genres as additional genres weren’t available in their chosen dataset. This is problematic for genre classification algorithm application to a wider variety of songs and genres. One previous study classified songs into 20 genres and 117 genres in separate instances to determine if their implementation performed better for an increased number of genres, and both variations of their implementation architecture did not (Tsaptsinos, 2017). This decrease in accuracy may have been because there were more genres for the model to select from, so overall performance decreased.
2.2 Emotional and Sentiment Analysis
Though several studies combined lyrics with other song aspects, such as album artwork or audio features, there was only one study that combined lyrical content with lyric sentiments. The study found that using lyric sentiments, as derived from the emotional valence score of Affective Norms for English Words (ANEW) word lists, provided no signal as to the genre of the song (Liang, Gu, & O’Connor, 2011). Investigation into the ANEW word lists for sentiment analysis indicates that the technique is commonly used in poetry classification and poetry generation, with varying levels of success, since the dataset of emotional words is only 2500 words (Satrio Utomo, Sarno & Suhariyanto, 2018). Since lyrical data contains one hundred times more words, it is likely that many of the songs, and their sentiment, can not be accurately represented through those words. Other poetry generation studies have used a technique and tool called SentiStrength, which is a Java class that estimates the negative and positive sentiment values of a particular segment of text by evaluating the valence scores of words within the text segment and returning the strength of a segment’s negative, positive, or neutral sentiment (Misztal-Radecka & Indurkhya, 2014).
2.3 Embedding Techniques
The way that words are embedded into vectors before a machine learning technique is applied can have a significant impact on the model’s performance and is a variable in many studies regarding song classification with lyrics. Several studies used custom or pre-trained GloVe embedding techniques (Boonyanit & Dahl) (Tsaptsinos, 2017), while others utilized Word2Vec (Kumar, Rajpal, & Rathore, 2018). The study that utilized Word2Vec used many different machine learning techniques and compared whether a simple average Word2Vec implementation or Word2Vec with a TFIDF vectorizer obtained a higher accuracy. They determined that the TFIDF outperformed the simple Word2Vec embedding with an accuracy of 74%, as compared to 65% on their best machine learning technique for a three-layer deep learning model (Kumar, Rajpal, & Rathore, 2018). The authors applied Word2Vec at a song level with Continuous Bag of Words rather than SkipGram. Other studies utilized a bag of words approach, with variations of processing techniques, such as stemming and not removing the stopwords (Mayer, Neumayer, & Rauber, 2008) (Liang, Gu, & O’Connor, 2011) (Dammann & Haugh, 2017). One limitation of the bag of words method is that it does not grasp the structural information of the songs, rather it may grasp only the topical content of the songs (Mayer, Neumayer, & Rauber, 2008).
2.4 Machine Learning Architectures
A majority of genre classification studies experiment with different machine learning methods and model architectures to determine which techniques can best predict what genre a song belongs to. Studies have utilized Naive Bayes, K-Nearest Neighbor, Support Vector Machines, Decision Tree Classifiers, Random Forest, eXtreme Gradient Boosting, Logistic Regression, and Deep Neural Networks, to varying degrees of success (Mayer, Neumayer, & Rauber, 2008) (Dammann & Haugh, 2017) (Kumar, Rajpal, & Rathore, 2018) (Liang, Gu, & O’Connor, 2011). Two different studies used a Long Short-Term Memory model and a Hierarchical Attention Network, respectively, to capture the word order and structure of the songs, not just the word content. The study which utilized the LSTM and bi-directional LSTM models classified songs into three genres and had a maximum accuracy of 68% (Boonyanit & Dahl). The study that pioneered the use of a HAN for this task, and is the only study to do so, classified 20 genres and determined that the LSTM outperformed the HAN, which had two layers, one for attention and the word level and the other for attention at the line/segment level. The LSTM had an accuracy of 49.77%, while the HAN had an accuracy of 49.50%, so the difference was not great (Tsaptsinos, 2017).
2.5 Research Gaps
There are significant opportunities for increased research, which have culminated in this study. There is a lack of studies utilizing a non-bag-of-words approach to lyric classification, specifically regarding the analysis of the emotional content and sentiments of lyrics. The study that found that the sentiments of the lyrics did not improve the model’s ability to classify a song into a genre used an un-intact bag of words corpus for their analysis (Liang, Gu, & O’Connor, 2011). More meaning sentimentally and content-wise can be derived from lyrics that are analyzed in regards to their structure, whether that be in song segments or song lines, rather than in a bag of words context. There is an opportunity to investigate whether sentiment analysis at a structural level of a song, such as a line level, could indicate song genre. Additionally, there is only one study regarding Hierarchical Attention Networks applied to song lyric context, and this study utilized GloVe embeddings. More can be discovered regarding the use of different types of embeddings, such as Word2Vec, in tandem with Long Short Term Memory models and Hierarchical Attention Networks. Lastly, there are no studies that combine the use of a model that analyzes/preserves word order and/or structure with the sentiment analysis of a song. This gap presents itself keenly for an investigation into using the LSTM model and HAN machine learning techniques on Word2Vec embeddings performed at the lyric level, and then in an ensemble model with the sentiment analysis of a song at the song and lyric levels to classify song genres. Will analyzing song lyrics at a line structural level by sentiment and content improve the ability to classify songs by genre as opposed to using general representations of song content and sentiment?
3. METHODOLOGY
3.1 Dataset
The dataset chosen for this project is open-source on the popular data-hosting platform, Kaggle. I chose this dataset due to its size and full lyrics, allowing the models to potentially learn from the structure of the lyrics. The original dataset consisted of over two hundred thousand songs across ten genres: Rock, Metal, Pop, Indie, Folk, Electronic, R&B, Jazz, Hip-Hop, and Country. Other variables, Artist Name, Song Title, and Song Language, were not utilized in genre predictions. 

Figure One: Dataset Distribution of Song Genres
The number of songs per genre was disproportionate, as evidenced by Figure One above. Some data-cleaning procedures were undertaken before creating a training set of songs in which all genres were evenly represented. Firstly, only English songs were chosen. Several criteria led to the removal of portions of the raw song lyric data, including:
Lyrical structure markers: [Verse 1:]
Special characters and symbols that were substitutions for a quotation mark
Introductory or post-song information: ‘Lyrics By:’, ‘Music By:’
Artist name or song title preceding the song
Links to the source of lyrics
Additionally, songs with repetitive sections, such as repeated ‘ohs’ or ‘ahs’ followed by a number indicating how many times that particular lyric or word was repeated, were removed from the corpus entirely. Similarly, other songs had lyrical structure placements without lyrics in place, such as marking ‘Chorus’ where the lines for the chorus occur. The lack of lyrics in place of the structural or repetition markers posed a potential problem for classification, as the lyrics are not preserved in full for those songs. Lastly, one of each set of duplicate songs was removed. 
Since the data was disproportionated per genre, I chose 1500 random songs from each genre to use in the project. Selecting a fraction of the total songs allowed for consistent representation of each category to ensure there were no classification biases based upon oversampling or undersampling, and the smaller set helped with decreasing the amount of time to run models later in the project. 
3.2 Embeddings
One consideration when utilizing word embeddings in machine learning models is the data that the embedding model has been trained upon. I chose to create a custom Word2Vec model on the entire song lyric corpus from the dataset because existing models are trained on data that may not include song lyrics. Since lyrical data is different in structure and content than other data, such as legal texts or Wikipedia pages, creating a custom Word2Vec model will likely provide a better understanding and representation of the lyrical content of the songs.
To construct the Word2Vec model, I split each set of lyrics from its original state as a string into many lists, where each list is a sentence corresponding to a line. This line representation will be referred to throughout the paper as line, lyric, line-level, or lyric-level, depending on the context. Each line was tokenized so each song then becomes represented by a list containing many lists full of tokens, where each sublist is a singular lyric. I chose to keep potential stop words and punctuation since punctuation could be indicative of genre, or of sentiment, such as with repeated exclamation points or other punctuation symbols. I trained and saved the custom Word2Vec embeddings, which have a vector dimension of 100 and a window size of ten (the average length of a lyric), to capture the words before and after each predicted word. 
3.3 Lyric Sentiment 
Song sentiment was analyzed in two ways to then be fed into models that used the sentiment to classify the songs into genres. To determine the sentiment of each song, I utilized the SentiStrength tool popular in poetry generation studies; the version I used was available through a Python wrapper class I downloaded from GitHub. To allow for the greatest range of sentiments to be expressed, I utilized the ‘scale’ option, which meant that each song or song lyric received a value ranging from -4, strongly negative, to 4, strongly positive.

Figure Two:  Lyric Sentiment Classification Construction
At the song level, the entire song as one string was passed into the SentiStrength classifier, and a number was returned representing the sentiment of the overall song. Four models were used with this sentiment classification to predict song genre: Logistic Regression, K-Nearest Neighbors, Naive Bayes Classifier, and Decision Tree Classifier, seen in Figure Two, above.
At the lyric level, each line of the song was fed into the SentiStrength classifier and received a score. Thus, a list containing a sentiment-corresponding number for each line represented each song. Because some songs have fewer lines than others, the ends of many song-representative lists were padded with zeros up to the maximum length of all songs, which was forty-one lyric lines. Three models were used with lyric-level classification: Recurrent Neural Network, Naive Bayes Classifier, and Decision Tree Classifier. 
3.4 Lyric Content
To determine whether lyric content may indicate genre, I trained three different models on song representations of line-level embeddings. 

Figure Three:  Lyric Content Classification Construction
I applied the custom Word2Vec model to each tokenized line to create the line-level word embeddings. To get a representation of the document (the song) to then be used in the models for classification, the embeddings must be transformed into a song-level embeddings vector. Figure Three, above, indicates the flow of model creation; for the Logistic Regression (LR) and Bi-Directional Long Short Term Memory (Bi-LSTM) models, I compared the average of the sentence-level embeddings with the Term Frequency- Inverse Document Frequency (TF-IDF) weightings of the sentence-level embeddings. TF-IDF works by comparing the occurrence of a word in a singular document, contextually a singular song, to how often it occurs across the entire corpus, the song dataset.

Figure Four: HAN Model Architecture
The three models analyzed were Logistic Regression, Bi-Directional LSTM, and Hierarchical Attention Network. Logistic Regression was chosen as a baseline performance metric, while Bi-LSTM and HAN were evaluated due to their ability to preserve the structure of lines and documents in various capacities, and were the main research gaps being addressed. The Logistic Regression models were optimized for the lowest loss, after training for only two epochs, and no other hyperparameters were modified. The Bi-LSTM had an ADAM optimizer, categorical cross-entropy loss, and softmax activation for genre prediction, and was trained with a batch size of sixteen for fifty epochs. The HAN had an RMSprop optimizer, categorical cross-entropy loss, and was trained with a batch size of 16 and 3 epochs for the lowest validation loss. The HAN accepted sentences with a maximum length of three hundred words, and the architecture is shown in Figure Four. Since HAN’s internal architecture already provides attention to the sentence and then document level of a song, it was not necessary to use an average or TF-IDF weighting of the word embeddings to represent each song (Yang et al.).
3.5 Ensemble Models
The goal of the study is to determine if song content and song sentiment, which both preserve the structure of the song in their classification, can be used to classify songs into genres. I created six ensemble models that combined the best genre-classifying models from each of the base lyric content models (LR, Bi-LSTM, and HAN) with the best song-level and lyric-level sentiment models.

Figure Five: Ensemble Models Construction
For reproducibility and consistency, the same 80/20 train-test split was used throughout all of the created independent and ensemble models, and the fitted models from the independent experiments were saved for use in the ensemble models. The ensemble models voted based on the weighted average of predicted class probabilities from both models, with equal consideration of the predictions of the sentiment and content models. 
4. RESULTS
4.1 Independent Lyric Sentiment Models
All of the song-level lyric sentiment models performed similarly poorly, only slightly above how they would perform by random chance. 
Model Type
Accuracies
Logistic Regression
15.20%
K- Nearest Neighbors
115.30%
Naive Bayes
15.10%
Decision Tree
15.30%

Table One: Lyric Sentiment, Song-Level Model Performance
The best model was the Decision Tree Classifier, with an accuracy of 15.3%, however, it was only able to identify five out of the ten genres.
Figure Six: Lyric Sentiment Confusion Matrix for Best Song-Level Sentiment Model
Similarly to all models throughout the analysis, the model correctly classified more Hip-Hop and Metal songs than any other genre. 
Model Type
Accuracies
Recurrent Neural Network
28.53%
Naive Bayes
25.13%
Decision Tree
24.53%

Table Two: Lyric Sentiment, Lyric-Level, Model Performance
The lyric level sentiment models performed significantly better than the song level models, with nearly double the accuracy for the best performing Recurrent Neural Network. 
Figure Seven: Lyric Sentiment Confusion Matrix for Best Lyric-Level Sentiment Model
The RNN performed best in the Hip-Hop genre, like the previous sentiment models, with an 84.33% accuracy in correctly classifying Hip-Hop songs. It performed extremely poorly in the Electronic and R&B categories. 
4.2 Independent Lyric Content Models
Content Model
Average of Line Embeddings
TF-IDF of Line Embeddings
Logistic Regression
12.90%
36.33%
Bi-Directional LSTM
31.60%
24.83%
Hierarchical Attention Network
38.60%

Table Three: Lyric Content Model Performance

The highest-performing model from the content models was the Hierarchical Attention Network, with an accuracy of 38.60 percent, averaged across the individual accuracies of the classification for each genre. For the Logistic Regression model, using TF-IDF weightings of the line-level embeddings led to improved classification over a simple average, while for the Bi-Directional LSTM, the average was a better classifier than the TF-IDF weightings. 

Figure Eight: Hierarchical Attention Network  Confusion Matrix For Best Content Model
Across all five models, the genre with the highest accuracy was Hip-Hop, and the worst was Rock. The best content model for each model type also performed moderately well in classifying country, metal, and jazz songs. The confusion matrices for LR and Bi-LSTM can be found in the appendix.
4.3 Ensemble Models
Using the weaker song-level sentiment analysis models in combination with the content models did not improve the ability to predict genre. 
Model Type
Best Content Only
With Song - Level (DTC) Sentiment 
With Line - Level (RNN) Sentiment
Logistic Regression
36.33%
25.46%
25.46%
Bi-LSTM
31.60%
27.57%
33.23%
HAN
38.60%
38.60%
43.07%

Table Four: Ensemble Models Performance
For the Logistic Regression models, using the sentiment classification decreased the accuracy of the ensemble model, while the ensemble of Bi-LSTM and line-level RNN saw a slight improvement. The HAN with line-level RNN together had a significant improvement and was the best-performing model across all independent and ensemble models, indicating that combining sentiment analysis with content analysis of lyrics can improve genre classification. 
Notably, the HAN and RNN ensemble model was the only model to correctly predict more Country songs than any other genre, with an 87.33% accuracy in predicting Country songs, the highest from any genre in any model. It also performed moderately well on some of the more difficult-to-classify genres, based on the poor accuracy of many models, such as electronic and indie. 


Figure Nine: Best Performing Ensemble Model for each Base Content Model Combination

5. DISCUSSION

6. CONCLUSION
Utilizing song lyrics represented through lyric level sentiment and content analysis increases the ability to classify songs into genres as compared to general song representations of sentiment and lyrics. Exploring the relationship between words in songs at the word, line, and song level, as well as comparison to the corpus, allowed the HAN model to discover trends and similarities within different genres. The existing model as created by this study works best for genres that have clear differences to most music listeners, such as Country and Hip-Hop, indicating that lyrics are possibly as indicative of a song’s genre as its sound is.
Future research can investigate which words or phrases are the strongest indicators of song genre and where in the song these words occur, to determine if there distinct patterns in songs of different genres that the model may be picking up on. Additionally, more feature engineering and fine-tuning can be done to increase the performance of the HAN and RNN ensemble models, such as creating a more layered neural network and experimentation with activation and optimization functions. Lastly, it is worth looking into why some genres, such as Rock, Indie, and Pop, are harder to classify than others; possible factors could be similar lyric structure or content.
REFERENCES
Boonyanit, A., & Dahl, A. (n.d.). Music Genre Classification using Song Lyrics. Stanford CS224N Course. https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1214/reports/final_reports/report003.pdf.
Dammann, T., & Haugh, K. (2017). Genre Classification of Spotify Songs using Lyrics, Audio Previews, and Album Artwork. Stanford CS229 Course. https://cs229.stanford.edu/proj2017/final-reports/5242682.pdf.
Kumar, A., Rajpal, A., & Rathore, D. (2018). Genre classification using word embeddings and deep learning. 2018 International Conference on Advances in Computing, Communications and Informatics (ICACCI). https://doi.org/10.1109/icacci.2018.8554816
Liang, D., Gu, H., & O’Connor, B. (2011). Music Genre Classification with the Million Song Dataset. Carnegie Mellon 15-826 Course. https://www.ee.columbia.edu/~dliang/files/FINAL.pdf.
Mayer, R., Neumayer, R., & Rauber, A. (2008). Rhyme and Style Features for Musical Genre Classification by Song Lyrics. ISMIR 2008 - 9th International Conference on Music Information Retrieval. 337-342. 
McVicar, M., Di Giorgi, B., Dundar, B., & Mauch, M. (2021, November 29). Lyric document embeddings for music tagging. https://arxiv.org/pdf/2112.11436.pdf
Misztal-Radecka, J., & Indurkhya, B. (2014). Poetry generation system with an emotional personality. International Conference on Innovative Computing and Cloud Computing. https://computationalcreativity.net/iccc2014/wp-content/uploads/2014/06/6.3_Misztal.pdf

Satrio Utomo, T., Sarno R., & Suhariyanto. (2018). Emotion Label from ANEW Dataset for Searching Best Definition from WordNet. 2018 International Seminar on Application for Technology of Information and Communication, Semarang, Indonesia. 10.1109/ISEMANTIC.2018.8549769.
Tsaptsinos, A. (2017, July 15). Lyrics-based music genre classification using a hierarchical attention network. https://arxiv.org/abs/1707.04678 
Yang, Z., Yang, D., Dyer, C., He, X., Smola, A., & Hovy, E. (n.d.). Hierarchical Attention Networks for Document Classification. Carnegie Mellon University. https://www.cs.cmu.edu/~./hovy/papers/16HLT-hierarchical-attention-networks.pdf. 
APPENDIX


