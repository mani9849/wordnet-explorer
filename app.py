import streamlit as st
from nltk.corpus import wordnet as wn
import nltk
import pandas as pd
import networkx as nx
from pyvis.network import Network
import tempfile
import streamlit.components.v1 as components

# Ensure required corpora are available
nltk_data_items = ['wordnet', 'omw-1.4', 'punkt']
for item in nltk_data_items:
    try:
        nltk.data.find(f'corpora/{item}')
    except LookupError:
        nltk.download(item)

st.set_page_config(page_title="WordNet Explorer", layout="wide")

# -------- Utility functions --------
def synset_info(syn):
    return {
        "synset_name": syn.name(),
        "pos": syn.pos(),
        "definition": syn.definition(),
        "examples": syn.examples(),
        "lemmas": [l.name() for l in syn.lemmas()]
    }

def synonyms_for_synset(syn):
    return list({l.name() for l in syn.lemmas()})

def antonyms_for_synset(syn):
    ants = []
    for l in syn.lemmas():
        for ant in l.antonyms():
            ants.append(ant.name())
    return list(set(ants))

def hypernyms(syn): return syn.hypernyms()
def hyponyms(syn): return syn.hyponyms()
def meronyms(syn): return syn.part_meronyms() + syn.substance_meronyms()
def holonyms(syn): return syn.part_holonyms() + syn.substance_holonyms()

def safe_first_synset(word):
    s = wn.synsets(word)
    return s[0] if s else None

# Build graph recursively (limited depth)
def build_relation_graph(start_synset, depth=2, max_nodes=200):
    G = nx.DiGraph()
    visited = set()
    def add_rel(syn, d):
        if syn is None: return
        if syn.name() in visited:
            return
        visited.add(syn.name())
        G.add_node(syn.name(), label=syn.definition(), title=syn.definition(), pos=syn.pos())
        if d <= 0 or len(visited) > max_nodes:
            return
        for h in syn.hypernyms():
            G.add_node(h.name(), label=h.definition(), title=h.definition(), pos=h.pos())
            G.add_edge(syn.name(), h.name(), rel='hypernym')
            add_rel(h, d-1)
        for h in syn.hyponyms():
            G.add_node(h.name(), label=h.definition(), title=h.definition(), pos=h.pos())
            G.add_edge(syn.name(), h.name(), rel='hyponym')
            add_rel(h, d-1)
    add_rel(start_synset, depth)
    return G

def nx_to_pyvis(G, notebook=True, physics=True):
    net = Network(height="600px", width="100%", notebook=notebook)
    for n, d in G.nodes(data=True):
        title = d.get('title', '')
        label = n.split('.')[0]
        net.add_node(n, label=label, title=title)
    for u, v, d in G.edges(data=True):
        rel = d.get('rel', '')
        net.add_edge(u, v, title=rel, value=1)
    if physics:
        net.force_atlas_2based()
    return net

# -------- UI layout --------
st.title("🧠 WordNet Explorer")
st.markdown("Search a word, explore its senses (synsets), relations, and similarity metrics using WordNet.")

with st.sidebar:
    st.header("Controls")
    word = st.text_input("Enter a word", value="bank")
    depth = st.slider("Graph depth (relation hop)", 1, 4, 2)
    show_graph = st.checkbox("Show relation graph", value=True)
    show_similarity = st.checkbox("Show similarity demo", value=True)

if not word:
    st.warning("Please enter a word in the sidebar to begin.")
    st.stop()

synsets = wn.synsets(word)

if not synsets:
    st.info("No synsets found for this word in WordNet.")
    st.stop()

# Choose sense
syn_labels = [f"{i+1}. {s.name()} ({s.pos()}) — {s.definition()}" for i, s in enumerate(synsets)]
selected_idx = st.radio("Choose a sense to inspect:", list(range(len(synsets))), format_func=lambda i: syn_labels[i])
selected_syn = synsets[selected_idx]

info = synset_info(selected_syn)
st.markdown(f"**Selected synset**: `{selected_syn.name()}`  
**POS**: {info['pos']}  
**Definition**: {info['definition']}")
if info['examples']:
    st.markdown("**Examples:**")
    for ex in info['examples']:
        st.write(f"- {ex}")

st.markdown("**Lemmas:** " + ", ".join(info['lemmas']))
st.markdown("**Synonyms:** " + (", ".join(synonyms_for_synset(selected_syn)) or "—"))
st.markdown("**Antonyms:** " + (", ".join(antonyms_for_synset(selected_syn)) or "—"))

if show_similarity:
    st.subheader("Similarity demo")
    other = st.text_input("Compare with another word:", "river")
    if other:
        s2 = safe_first_synset(other)
        if s2:
            wup = selected_syn.wup_similarity(s2)
            path = selected_syn.path_similarity(s2)
            st.write(f"Comparing `{selected_syn.name()}` with `{s2.name()}`")
            st.metric("WUP similarity", f"{wup:.3f}" if wup is not None else "N/A")
            st.metric("Path similarity", f"{path:.3f}" if path is not None else "N/A")
        else:
            st.info(f"No synset found for '{other}'.")

if show_graph:
    st.header("Relation Graph")
    graph = build_relation_graph(selected_syn, depth)
    if graph and len(graph.nodes) > 0:
        net = nx_to_pyvis(graph)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
            path = tmp.name
            net.show(path)
            html = open(path, 'r', encoding='utf-8').read()
            components.html(html, height=650, scrolling=True)
    else:
        st.info("No graph available for this synset.")
