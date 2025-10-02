from flask import Flask, render_template, request
from nltk.corpus import wordnet
import nltk

nltk.download('wordnet')
nltk.download('omw-1.4')

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    results = []
    word = ''
    if request.method == 'POST':
        word = request.form['word']
        synsets = wordnet.synsets(word)
        if synsets:
            for i, syn in enumerate(synsets):
                synonyms = set(lemma.name() for lemma in syn.lemmas())
                antonyms = set(lemma.antonyms()[0].name() for lemma in syn.lemmas() if lemma.antonyms())
                hypernyms = [h.name().split('.')[0] for h in syn.hypernyms()]
                hyponyms = [h.name().split('.')[0] for h in syn.hyponyms()]
                results.append({
                    'sense': i + 1,
                    'definition': syn.definition(),
                    'examples': syn.examples(),
                    'synonyms': ', '.join(synonyms),
                    'antonyms': ', '.join(antonyms) if antonyms else 'None',
                    'hypernyms': ', '.join(hypernyms) if hypernyms else 'None',
                    'hyponyms': ', '.join(hyponyms) if hyponyms else 'None'
                })
        else:
            results = None
    return render_template('index.html', word=word, results=results)

if __name__ == '__main__':
    app.run(debug=True)
