import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import re
import string
from wordcloud import WordCloud
from nlp_id.lemmatizer import Lemmatizer
from nlp_id.stopword import StopWord
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

st.header('Dashboard Analisis Sentimen')

# Fungsi untuk case folding dan pembersihan teks
def case_folding(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(" \d+", '', text)
    text = text.strip()
    text = text.replace('\\t', " ").replace('\\n', " ").replace('\\u', " ").replace('\\', " ")
    text = text.encode('ascii', 'replace').decode('ascii')
    text = text.translate(str.maketrans(" ", " ", string.punctuation))
    text = re.sub('\s+', ' ', text)
    return text

lemmatizer = Lemmatizer()
stopword = StopWord()

# Mengunggah dan memproses kamus slang
slang_dictionary = pd.read_csv('https://raw.githubusercontent.com/nikovs/data-science-portfolio/master/topic%20modelling/colloquial-indonesian-lexicon.csv')
slang_dict = pd.Series(slang_dictionary['formal'].values, index=slang_dictionary['slang']).to_dict()

def Slangwords(text):
    for word in text.split():
        if word in slang_dict.keys():
            text = text.replace(word, slang_dict[word])
    return text

def case_folding_final(text):
    text = re.sub(" \d+", '', text)
    text = text.strip()
    return text

def tokenization(teks):
    text_list = teks.split(" ")
    return text_list

def preprocess_text(df):
    df['review_processed'] = ''
    df['Tokenizing'] = ''
    for i, row in df.iterrows():
        text = row['review']
        clean_text = case_folding(text)
        clean_text = lemmatizer.lemmatize(clean_text)
        clean_text = stopword.remove_stopword(clean_text)
        clean_text = Slangwords(clean_text)
        clean_text = case_folding_final(clean_text)
        tokens = tokenization(clean_text)
        df.at[i, 'review_processed'] = clean_text
        df.at[i, 'Tokenizing'] = ', '.join(tokens)  # Pisahkan token dengan koma
    return df

def sentiment_analysis_lexicon_indonesia(text, list_positive, list_negative):
    score = 0
    for word in text:
        if word in list_positive:
            score += 1
        if word in list_negative:
            score -= 1
    polarity = ''
    if score > 0:
        polarity = 'positif'
    elif score < 0:
        polarity = 'negatif'
    else:
        polarity = 'netral'
    return score, polarity

# Unggah berkas CSV
st.subheader('Unggah Berkas CSV')
upl = st.file_uploader('Unggah berkas CSV', type=['csv'])

if upl:
    df = pd.read_csv(upl)
    st.subheader('Info Data')
    st.write(df.head())

    # Unggah kamus kata positif dan negatif
    st.subheader('Unggah Kamus Kata Positif (Format .txt)')
    pos = st.file_uploader('Positive Lexicon', key='positive', type=['txt'])

    st.subheader('Unggah Kamus Kata Negatif (Format .txt)')
    neg = st.file_uploader('Negatif Lexicon', key='negative', type=['txt'])

    # Periksa apakah kamus lexicon sudah diunggah
    if not pos or not neg:
        st.warning('Harap unggah kedua kamus lexicon (positif dan negatif) untuk melanjutkan.')
    else:
        # Membaca dan memproses kamus lexicon
        list_positive = [line.strip() for line in pos.read().decode('utf-8').splitlines()]
        list_negative = [line.strip() for line in neg.read().decode('utf-8').splitlines()]

        df_clean = preprocess_text(df)
        hasil = df_clean['Tokenizing'].apply(lambda x: sentiment_analysis_lexicon_indonesia(x.split(', '), list_positive, list_negative))
        hasil = list(zip(*hasil))
        df_clean['skor_polaritas'] = hasil[0]
        df_clean['polaritas'] = hasil[1]

        # Ubah skor polaritas menjadi array 2D
        X_polarity = df_clean['skor_polaritas'].values.reshape(-1, 1)

        # Inisialisasi TF-IDF Vectorizer
        tfidf_vectorizer = TfidfVectorizer(lowercase=False)
        X_tfidf = tfidf_vectorizer.fit_transform(df_clean['Tokenizing'].astype(str))

        # Gabungkan fitur TF-IDF dan skor polaritas
        X_polarity = np.maximum(X_polarity, 0)  # Pastikan tidak ada nilai negatif
        X_combined = np.hstack((X_tfidf.toarray(), X_polarity))

        # Target variabel
        y = df_clean['polaritas'].map({'positif': 0, 'netral': 1, 'negatif': 2})

        # Inisialisasi dan pelatihan model Naive Bayes (Multinomial)
        nb_classifier = MultinomialNB()
        nb_classifier.fit(X_combined, y)

        # Prediksi sentimen menggunakan Naive Bayes
        y_pred_nb = nb_classifier.predict(X_combined)

        # Map hasil prediksi angka ke label sentimen
        sentiment_labels = {0: 'positif', 1: 'netral', 2: 'negatif'}
        y_pred_labels = [sentiment_labels[pred] for pred in y_pred_nb]

        # Hitung akurasi
        accuracy = accuracy_score(y, y_pred_nb)
        st.write(f"Akurasi Model Naive Bayes: {accuracy:.2f}")

        # Pie chart untuk distribusi prediksi
        pred_counts = pd.Series(y_pred_labels).value_counts()
        labels = ['positif', 'netral', 'negatif']
        sizes = [pred_counts.get(label, 0) for label in labels]
        colors = ['#008000', '#FFFF00', '#FF0000']  # Hijau, Kuning, Merah
        explode = [0.05] * len(labels)
        fig1, ax1 = plt.subplots()
        ax1.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
                shadow=False, startangle=90)
        ax1.axis('equal')
        st.pyplot(fig1)

        # Tabel distribusi count
        st.subheader('Distribusi Count Prediksi Sentimen')

        sentiment_counts = pd.Series(y_pred_labels).value_counts().reindex(labels, fill_value=0)
        sentiment_counts_df = sentiment_counts.reset_index()
        sentiment_counts_df.columns = ['Sentimen', 'Count']

        # Menampilkan tabel distribusi count dengan tampilan default
        st.table(sentiment_counts_df)

        # Menggabungkan semua informasi ke dalam satu DataFrame
        df_results = pd.concat([df[['user', 'produk', 'review']],
                                pd.Series(df_clean['Tokenizing'], name='Tokenizing'),
                                pd.Series(df_clean['skor_polaritas'], name='Skor Polaritas'),
                                pd.Series(df_clean['polaritas'], name='Sentiment Lexicon'),
                                pd.Series(y_pred_labels, name='Predicted Sentiment')], axis=1)
        
        # Menampilkan data hasil prediksi
        st.write('Data Hasil Prediksi:')
        st.write(df_results)

         # Word Cloud berdasarkan hasil prediksi
        positive_reviews = df_clean[df_clean['polaritas'] == 'positif']
        negative_reviews = df_clean[df_clean['polaritas'] == 'negatif']

        positive_words = ' '.join([' '.join(review.split()) for review in positive_reviews['Tokenizing']])
        negative_words = ' '.join([' '.join(review.split()) for review in negative_reviews['Tokenizing']])

        positive_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(positive_words)
        negative_wordcloud = WordCloud(width=800, height=400, background_color='black', colormap='Reds').generate(negative_words)

        fig2, ax2 = plt.subplots()
        ax2.imshow(positive_wordcloud, interpolation='bilinear')
        ax2.axis('off')
        st.subheader("WordCloud Ulasan Positif")
        st.pyplot(fig2)

        fig3, ax3 = plt.subplots()
        ax3.imshow(negative_wordcloud, interpolation='bilinear')
        ax3.axis('off')
        st.subheader("WordCloud Ulasan Negatif")
        st.pyplot(fig3)

        
        @st.cache_data
        def convert_df(df):
            return df.to_csv(index=False)

        csv = convert_df(df_results)

        st.download_button(
            label="Unduh Data Hasil analisis",
            data=csv,
            file_name='data_hasil_prediksi.csv',
            mime='text/csv'
        )

