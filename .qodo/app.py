# app.py
import streamlit as st
from nltk.corpus import wordnet as wn
import nltk
import pandas as pd
import networkx as nx
from pyvis.network import Network
import tempfile
import streamlit.components.v1 as components
from typing import List, Dict, Any
import io

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
        # hypernyms
        for h in syn.hypernyms():
            G.add_node(h.name(), label=h.definition(), title=h.definition(), pos=h.pos())
            G.add_edge(syn.name(), h.name(), rel='hypernym')
            add_rel(h, d-1)
        # hyponyms
        for h in syn.hyponyms():
            G.add_node(h.name(), label=h.definition(), title=h.definition(), pos=h.pos())
            G.add_edge(syn.name(), h.name(), rel='hyponym')
            add_rel(h, d-1)
        # meronyms
        for m in syn.part_meronyms() + syn.substance_meronyms():
            G.add_node(m.name(), label=m.definition(), title=m.definition(), pos=m.pos())
            G.add_edge(syn.name(), m.name(), rel='meronym')
            add_rel(m, d-1)
        # holonyms
        for h in syn.part_holonyms() + syn.substance_holonyms():
            G.add_node(h.name(), label=h.definition(), title=h.definition(), pos=h.pos())
            G.add_edge(syn.name(), h.name(), rel='holonym')

    add_rel(start_synset, depth)
    return G

def nx_to_pyvis(G, notebook=True, physics=True):
    net = Network(height="600px", width="100%", notebook=notebook)
    # add nodes
    for n, d in G.nodes(data=True):
        title = d.get('title', '')
        label = n.split('.')[0]  # show lemma part
        net.add_node(n, label=label, title=title)
    # add edges
    for u, v, d in G.edges(data=True):
        rel = d.get('rel', '')
        net.add_edge(u, v, title=rel, value=1)
    if physics:
        net.force_atlas_2based()
    return net

# Cached helpers to speed up repeated lookups
@st.cache_data(show_spinner=False)
def get_synsets_cached(word: str):
    return wn.synsets(word)

@st.cache_data(show_spinner=False)
def build_graph_cached(syn_name: str, depth: int):
    syns = wn.synsets(syn_name.split('.')[0])
    # find the exact synset if possible
    syn = None
    for s in syns:
        if s.name() == syn_name:
            syn = s
            break
    if syn is None and syns:
        syn = syns[0]
    if syn is None:
        return None
    return build_relation_graph(syn, depth=depth)

# -------- UI layout --------
st.title("🧠 WordNet Explorer")
st.markdown("Search a word, explore its senses (synsets), relations, and similarity metrics using WordNet.")

with st.sidebar:
    st.header("Controls")
    word = st.text_input("Enter a word", value="bank")
    depth = st.slider("Graph depth (relation hop)", 1, 4, 2)
    max_nodes = st.slider("Graph max nodes (safety)", 50, 1000, 300)
    show_graph = st.checkbox("Show relation graph", value=True)
    show_similarity = st.checkbox("Show similarity demo", value=True)
    export_csv = st.checkbox("Enable CSV export", value=True)
    st.markdown("---")
    st.markdown("**Notes**: Graph uses hypernym/hyponym and part meronym/holonym relations. Similarity uses WordNet synset similarity (WUP & path).")

# Main content
if not word:
    st.warning("Please enter a word in the sidebar to begin.")
    st.stop()

synsets = get_synsets_cached(word)

col1, col2 = st.columns([2, 3])

with col1:
    st.subheader(f"Senses for '{word}' ({len(synsets)} found)")
    if not synsets:
        st.info("No synsets found for this word in WordNet.")
    else:
        # show as radio selection so user can pick sense
        syn_labels = [f"{i+1}. {s.name()} ({s.pos()}) — {s.definition()}" for i, s in enumerate(synsets)]
        selected_idx = st.radio("Choose a sense to inspect:", list(range(len(synsets))), format_func=lambda i: syn_labels[i])
        selected_syn = synsets[selected_idx]
        info = synset_info(selected_syn)
        st.markdown(f"**Selected synset**: `{selected_syn.name()}`  \n**POS**: {info['pos']}  \n**Definition**: {info['definition']}")
        if info['examples']:
            st.markdown("**Examples:**")
            for ex in info['examples']:
                st.write(f"- {ex}")
        st.markdown("**Lemmas (lemmas names):**")
        st.write(", ".join(info['lemmas']))

        st.markdown("### Relations")
        cols = st.columns(2)
        with cols[0]:
            st.write("**Synonyms:**")
            st.write(", ".join(synonyms_for_synset(selected_syn)) or "—")
            st.write("**Antonyms:**")
            st.write(", ".join(antonyms_for_synset(selected_syn)) or "—")
        with cols[1]:
            st.write("**Hypernyms:**")
            st.write(", ".join([h.name() for h in selected_syn.hypernyms()]) or "—")
            st.write("**Hyponyms:**")
            st.write(", ".join([h.name() for h in selected_syn.hyponyms()]) or "—")

        st.write("**Meronyms:**")
        mer = meronyms(selected_syn)
        st.write(", ".join([m.name() for m in mer]) or "—")
        st.write("**Holonyms:**")
        hol = holonyms(selected_syn)
        st.write(", ".join([h.name() for h in hol]) or "—")

with col2:
    st.subheader("Extras")
    if show_similarity:
        st.markdown("#### Similarity demo (first senses used unless you change selection)")
        other_word = st.text_input("Compare with (word):", value="river", key="compare_word")
        if other_word:
            s1 = selected_syn
            s2 = safe_first_synset(other_word)
            if s2:
                wup = s1.wup_similarity(s2)
                path = s1.path_similarity(s2)
                st.write(f"Comparing `{s1.name()}` with `{s2.name()}`")
                st.metric("WUP similarity", f"{wup:.3f}" if wup is not None else "N/A")
                st.metric("Path similarity", f"{path:.3f}" if path is not None else "N/A")
                st.write("**Note**: Similarity values depend on sense selection; polysemous words may have low similarity for mismatched senses.")
            else:
                st.info(f"No synset found for '{other_word}'.")
    st.markdown("---")
    st.subheader("Current synset table")
    # create DataFrame for export / preview
    rows = []
    for s in synsets:
        rows.append({
            "synset": s.name(),
            "pos": s.pos(),
            "definition": s.definition(),
            "lemmas": ", ".join([l.name() for l in s.lemmas()]),
            "examples": " | ".join(s.examples())
        })
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True)

    if export_csv:
        csv_buf = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download synsets as CSV", data=csv_buf, file_name=f"{word}_synsets.csv", mime="text/csv")

# Graph area (full width)
if show_graph and synsets:
    st.header("Relation Graph")
    # Build graph for the selected synset
    try:
        graph = build_graph_cached(selected_syn.name(), depth)
        if graph is None or len(graph.nodes) == 0:
            st.info("No graph could be built for this synset.")
        else:
            # Convert to pyvis and render
            net = nx_to_pyvis(graph)
            # tweak options
            net.set_options("""
            var options = {
              "nodes": {
                "font": {"size": 14},
                "scaling": {"label": true}
              },
              "edges": {
                "smooth": false
              },
              "physics": {
                "barnesHut": { "gravitationalConstant": -8000 }
              }
            }
            """)
            # save to temp and load HTML
            with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
                path = tmp.name
                net.show(path)
                html = open(path, 'r', encoding='utf-8').read()
                components.html(html, height=650, scrolling=True)
    except Exception as e:
        st.error(f"Graph rendering failed: {e}")

st.markdown("---")
st.markdown("Made with `nltk` WordNet • Tips: try polysemous words like `bank`, `bass`, `spring` to see distinct senses.")
