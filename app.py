from flask import Flask, render_template, request, send_file, Response
import nltk
from gtts import gTTS
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx
import pyttsx3
import os
import tempfile
from io import BytesIO
import re
import base64
import gtts


app = Flask(__name__)


def read_article(content):
    sentences = content.split(". ")
    sentences.pop()
    return sentences


def sentence_similarity(sent1, sent2, stopwords=None):
    if stopwords is None:
        stopwords = []

    sent1 = [w.lower() for w in sent1]
    sent2 = [w.lower() for w in sent2]

    all_words = list(set(sent1 + sent2))

    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)

    for w in sent1:
        if w in stopwords:
            continue
        vector1[all_words.index(w)] += 1

    for w in sent2:
        if w in stopwords:
            continue
        vector2[all_words.index(w)] += 1

    return 1 - cosine_distance(vector1, vector2)


def build_similarity_matrix(sentences, stopwords):
    similarity_matrix = np.zeros((len(sentences), len(sentences)))

    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i == j:
                continue
            similarity_matrix[i][j] = sentence_similarity(sentences[i], sentences[j], stopwords)

    return similarity_matrix


def generate_summary(content, top_n=5):
    stop_words = stopwords.words('english')
    summarize_text = []

    sentences = read_article(content)
    sentence_similarity_matrix = build_similarity_matrix(sentences, stop_words)
    sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_matrix)
    scores = nx.pagerank(sentence_similarity_graph)
    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)

    for i in range(top_n):
        summarize_text.append(ranked_sentences[i][1])

    return ". ".join(summarize_text)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/summarize', methods=['POST'])
def summarize():
    content = request.form['content']
    summary = generate_summary(content, top_n=3)

    # Initialize the TTS engine
    engine = pyttsx3.init()

    # Create a temporary audio file
    # with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_audio:
    #     audio_path = temp_audio.name
    #     engine.save_to_file(summary, audio_path)
    #     engine.runAndWait()
    myobj = gTTS(text=summary, lang='en', slow=False)
    myobj.save("summary.mp3")

    return render_template('summary.html', summary=summary, audio_path='/summary.mp3')


@app.route('/audio/<path:audio_path>')
def stream_audio(audio_path):
    return send_file(audio_path, mimetype='audio/mpeg')


if __name__ == '__main__':
    app.run(debug=True)
