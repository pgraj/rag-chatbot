"""
vector_explorer.py
──────────────────
Visual learning tool for understanding embeddings and vector databases.
Run with: py -m streamlit run vector_explorer.py

Install dependencies first:
pip install streamlit plotly scikit-learn pandas numpy sentence-transformers chromadb
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import chromadb

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Vector Space Explorer",
    page_icon="🧠",
    layout="wide"
)

st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .stMetric { background: #1e2130; border-radius: 8px; padding: 12px; }
    .explain-box {
        background: #1a1f2e;
        border-left: 4px solid #4f8ef7;
        padding: 12px 16px;
        border-radius: 4px;
        margin-bottom: 12px;
        font-size: 0.9em;
        color: #c8d0e0;
    }
    .highlight { color: #4f8ef7; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_resources():
    chroma = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma.get_or_create_collection("course_docs")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return collection, embedder

@st.cache_data
def load_vectors():
    collection, _ = load_resources()
    results = collection.get(include=["documents", "embeddings"])
    docs = results["documents"]
    vecs = np.array(results["embeddings"])
    labels = [f"Chunk {i+1}: {d[:40]}..." for i, d in enumerate(docs)]
    short_labels = [f"C{i+1}" for i in range(len(docs))]
    return docs, vecs, labels, short_labels

collection, embedder = load_resources()
docs, vecs, labels, short_labels = load_vectors()
n_chunks = len(docs)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🧠 Vector Space Explorer")
st.caption("Learn how RAG embeddings work through interactive visualisations")

# ── Metrics row ───────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("📄 Total Chunks", n_chunks)
c2.metric("📐 Vector Dimensions", vecs.shape[1])
c3.metric("🔢 Total Numbers Stored", f"{n_chunks * vecs.shape[1]:,}")
c4.metric("📦 Model", "all-MiniLM-L6-v2")

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# TAB LAYOUT
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📡 What is a Vector?",
    "🗺️ PCA — 2D Map",
    "🌀 t-SNE — Cluster View",
    "🔥 Similarity Heatmap",
    "🔎 Query Explorer"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — What is a Vector?
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("What is a Vector?")

    st.markdown("""
    <div class="explain-box">
    When the embedding model reads a sentence, it converts it into a list of 384 numbers.
    Each number captures a tiny aspect of meaning. Together they form a <span class="highlight">vector</span>
    — a coordinate in 384-dimensional space. Similar sentences land close together in that space.
    </div>
    """, unsafe_allow_html=True)

    chunk_idx = st.slider("Select a chunk to inspect", 1, n_chunks, 1) - 1

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("**📄 Text Content**")
        st.info(docs[chunk_idx])

        st.markdown("**🔢 Raw Vector (first 50 of 384 dimensions)**")
        vec_preview = vecs[chunk_idx][:50]
        vec_df = pd.DataFrame({
            "Dimension": [f"D{i}" for i in range(50)],
            "Value": vec_preview
        })
        fig = px.bar(
            vec_df, x="Dimension", y="Value",
            color="Value",
            color_continuous_scale="RdBu",
            color_continuous_midpoint=0,
            title=f"Vector values — Chunk {chunk_idx+1}",
            height=300
        )
        fig.update_layout(
            paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
            font_color="#c8d0e0", showlegend=False,
            xaxis=dict(tickangle=45, tickfont=dict(size=8))
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("**📊 Vector Statistics**")
        v = vecs[chunk_idx]
        stats_df = pd.DataFrame({
            "Statistic": ["Min value", "Max value", "Mean", "Std Dev", "L2 Norm"],
            "Value": [
                round(float(v.min()), 4),
                round(float(v.max()), 4),
                round(float(v.mean()), 4),
                round(float(v.std()), 4),
                round(float(np.linalg.norm(v)), 4)
            ]
        })
        st.dataframe(stats_df, use_container_width=True, hide_index=True)

        st.markdown("""
        <div class="explain-box">
        <b>What these numbers mean:</b><br><br>
        • <span class="highlight">Values range from -1 to +1</span> — positive means the chunk
          aligns with that dimension of meaning, negative means it doesn't.<br><br>
        • <span class="highlight">L2 Norm ≈ 1.0</span> — vectors are normalised (unit length),
          so distance = direction of meaning, not size of text.<br><br>
        • <span class="highlight">384 dimensions</span> — each captures a different semantic concept
          (formality, topic, sentiment, specificity, etc.)
        </div>
        """, unsafe_allow_html=True)

        # Distribution of values
        fig2 = px.histogram(
            x=v, nbins=40,
            title="Distribution of vector values",
            labels={"x": "Value", "y": "Count"},
            color_discrete_sequence=["#4f8ef7"],
            height=250
        )
        fig2.update_layout(
            paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
            font_color="#c8d0e0"
        )
        st.plotly_chart(fig2, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — PCA
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("PCA — Principal Component Analysis")

    st.markdown("""
    <div class="explain-box">
    <b>The problem:</b> You can't visualise 384 dimensions.<br>
    <b>The solution:</b> PCA finds the 2 directions of <span class="highlight">maximum variance</span>
    and projects all vectors onto them. It's like photographing a 3D object from its most
    informative angle. Chunks that are close on this map are semantically similar.
    </div>
    """, unsafe_allow_html=True)

    pca = PCA(n_components=2)
    coords_2d = pca.fit_transform(vecs)
    variance = pca.explained_variance_ratio_

    pca_df = pd.DataFrame({
        "PC1": coords_2d[:, 0],
        "PC2": coords_2d[:, 1],
        "Label": labels,
        "Chunk": short_labels,
        "Preview": [d[:80] + "..." for d in docs]
    })

    col1, col2 = st.columns([3, 1])

    with col1:
        fig = px.scatter(
            pca_df, x="PC1", y="PC2",
            text="Chunk",
            hover_name="Label",
            hover_data={"Preview": True, "PC1": ":.3f", "PC2": ":.3f"},
            title=f"PCA 2D — explains {variance[0]*100:.1f}% + {variance[1]*100:.1f}% = {sum(variance)*100:.1f}% of variance",
            color="PC1",
            color_continuous_scale="Viridis",
            height=550
        )
        fig.update_traces(
            textposition="top center",
            marker=dict(size=12, line=dict(width=1, color="#ffffff"))
        )
        fig.update_layout(
            paper_bgcolor="#0f1117", plot_bgcolor="#1a1f2e",
            font_color="#c8d0e0",
            xaxis=dict(gridcolor="#2a2f3e", title=f"PC1 ({variance[0]*100:.1f}% variance)"),
            yaxis=dict(gridcolor="#2a2f3e", title=f"PC2 ({variance[1]*100:.1f}% variance)")
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("**📐 Variance Explained**")
        var_df = pd.DataFrame({
            "Component": ["PC1", "PC2", "Remaining"],
            "Variance %": [
                round(variance[0]*100, 1),
                round(variance[1]*100, 1),
                round((1 - sum(variance))*100, 1)
            ]
        })
        fig_var = px.pie(
            var_df, values="Variance %", names="Component",
            color_discrete_sequence=["#4f8ef7", "#f7a24f", "#3a3f52"],
            height=300
        )
        fig_var.update_layout(
            paper_bgcolor="#0f1117", font_color="#c8d0e0"
        )
        st.plotly_chart(fig_var, use_container_width=True)

        st.markdown("""
        <div class="explain-box">
        <b>How to read this:</b><br><br>
        • <span class="highlight">Clusters</span> = chunks with similar meaning<br><br>
        • <span class="highlight">Distance</span> = semantic difference<br><br>
        • <span class="highlight">Variance %</span> = how much meaning is preserved after compression from 384D → 2D<br><br>
        • PCA is <b>linear</b> — great for overview, misses complex clusters
        </div>
        """, unsafe_allow_html=True)

    # Scree plot
    pca_full = PCA(n_components=min(20, n_chunks))
    pca_full.fit(vecs)
    scree_df = pd.DataFrame({
        "Component": [f"PC{i+1}" for i in range(len(pca_full.explained_variance_ratio_))],
        "Variance %": pca_full.explained_variance_ratio_ * 100,
        "Cumulative %": np.cumsum(pca_full.explained_variance_ratio_) * 100
    })

    fig_scree = go.Figure()
    fig_scree.add_bar(x=scree_df["Component"], y=scree_df["Variance %"],
                      name="Individual", marker_color="#4f8ef7")
    fig_scree.add_scatter(x=scree_df["Component"], y=scree_df["Cumulative %"],
                          name="Cumulative", line=dict(color="#f7a24f", width=2),
                          mode="lines+markers")
    fig_scree.update_layout(
        title="Scree Plot — how many dimensions capture most meaning?",
        paper_bgcolor="#0f1117", plot_bgcolor="#1a1f2e",
        font_color="#c8d0e0",
        xaxis=dict(gridcolor="#2a2f3e"),
        yaxis=dict(gridcolor="#2a2f3e", title="Variance %"),
        height=300
    )
    st.plotly_chart(fig_scree, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — t-SNE
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("t-SNE — Cluster Visualisation")

    st.markdown("""
    <div class="explain-box">
    t-SNE is better than PCA at revealing <span class="highlight">natural clusters</span>.
    Instead of finding the axis of maximum variance, it tries to preserve neighbourhood
    relationships — chunks that are close in 384D space stay close in 2D.
    It's slower but reveals groupings that PCA misses.
    </div>
    """, unsafe_allow_html=True)

    perplexity = st.slider(
        "Perplexity (controls cluster tightness — try different values)",
        min_value=2, max_value=min(30, n_chunks-1), value=min(5, n_chunks-1)
    )

    with st.spinner("Running t-SNE... (this takes a few seconds)"):
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, n_iter=1000)
        coords_tsne = tsne.fit_transform(vecs)

    tsne_df = pd.DataFrame({
        "X": coords_tsne[:, 0],
        "Y": coords_tsne[:, 1],
        "Label": labels,
        "Chunk": short_labels,
        "Preview": [d[:80] + "..." for d in docs]
    })

    fig = px.scatter(
        tsne_df, x="X", y="Y",
        text="Chunk",
        hover_name="Label",
        hover_data={"Preview": True},
        color="X",
        color_continuous_scale="Plasma",
        title=f"t-SNE 2D (perplexity={perplexity}) — semantic clusters",
        height=600
    )
    fig.update_traces(
        textposition="top center",
        marker=dict(size=14, line=dict(width=1, color="#ffffff"))
    )
    fig.update_layout(
        paper_bgcolor="#0f1117", plot_bgcolor="#1a1f2e",
        font_color="#c8d0e0",
        xaxis=dict(gridcolor="#2a2f3e"),
        yaxis=dict(gridcolor="#2a2f3e")
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    <div class="explain-box">
    <b>PCA vs t-SNE:</b><br><br>
    • <span class="highlight">PCA</span> — preserves global structure, fast, deterministic.
      Good for seeing the overall spread.<br><br>
    • <span class="highlight">t-SNE</span> — preserves local neighbourhoods, reveals clusters,
      slow, non-deterministic (changes each run).<br><br>
    • In RAG, what matters is that similar course content (e.g. all aged care chunks)
      clusters together — those are the chunks that will be retrieved together when a
      student asks about aged care.
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Cosine Similarity Heatmap
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Cosine Similarity Heatmap")

    st.markdown("""
    <div class="explain-box">
    <b>Cosine similarity</b> measures the angle between two vectors.
    <span class="highlight">Score = 1.0</span> means identical meaning.
    <span class="highlight">Score = 0.0</span> means completely unrelated.
    This is exactly the calculation ChromaDB runs when you submit a query —
    it finds the chunks with the highest cosine similarity to your question.
    </div>
    """, unsafe_allow_html=True)

    sim_matrix = cosine_similarity(vecs)

    fig = go.Figure(data=go.Heatmap(
        z=sim_matrix,
        x=short_labels,
        y=short_labels,
        colorscale="RdBu",
        zmid=0.5,
        zmin=0, zmax=1,
        hoverongaps=False,
        hovertemplate="<b>%{x}</b> vs <b>%{y}</b><br>Similarity: %{z:.3f}<extra></extra>"
    ))

    fig.update_layout(
        title="Cosine Similarity Matrix — every chunk vs every chunk",
        paper_bgcolor="#0f1117", plot_bgcolor="#1a1f2e",
        font_color="#c8d0e0",
        height=600,
        xaxis=dict(tickangle=45),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Most and least similar pairs
    st.markdown("**🔝 Most Similar Chunk Pairs**")
    pairs = []
    for i in range(n_chunks):
        for j in range(i+1, n_chunks):
            pairs.append({
                "Chunk A": short_labels[i],
                "Chunk B": short_labels[j],
                "Similarity": round(sim_matrix[i][j], 4),
                "Preview A": docs[i][:60] + "...",
                "Preview B": docs[j][:60] + "..."
            })

    pairs_df = pd.DataFrame(pairs).sort_values("Similarity", ascending=False)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Most similar (closest meaning)**")
        st.dataframe(pairs_df.head(5)[["Chunk A", "Chunk B", "Similarity"]], 
                     use_container_width=True, hide_index=True)
    with col2:
        st.markdown("**Least similar (most different meaning)**")
        st.dataframe(pairs_df.tail(5)[["Chunk A", "Chunk B", "Similarity"]], 
                     use_container_width=True, hide_index=True)

    st.markdown("""
    <div class="explain-box">
    <b>Why this matters for RAG:</b> When you ask "What are the entry requirements for nursing?",
    ChromaDB computes the cosine similarity between your question vector and every chunk vector,
    then returns the top 3 matches. The heatmap above shows you which chunks are
    <span class="highlight">naturally close neighbours</span> — they will tend to get retrieved together.
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Live Query Explorer
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.subheader("🔎 Live Query Explorer")

    st.markdown("""
    <div class="explain-box">
    Type a question and watch it get embedded into the same vector space as your chunks.
    You'll see <span class="highlight">exactly where your query lands</span> on the PCA map
    and which chunks it pulls toward — this is the retrieval step of RAG made visible.
    </div>
    """, unsafe_allow_html=True)

    query = st.text_input(
        "Type a question:",
        placeholder="e.g. What are the entry requirements for the Diploma of Nursing?"
    )

    if query:
        with st.spinner("Embedding your query..."):
            q_vec = embedder.encode([query])[0]
            q_vec_np = np.array([q_vec])

        # Cosine similarity to all chunks
        sims = cosine_similarity(q_vec_np, vecs)[0]
        top3_idx = np.argsort(sims)[::-1][:3]

        # PCA projection — fit on chunks + query together
        all_vecs = np.vstack([vecs, q_vec_np])
        pca2 = PCA(n_components=2)
        all_coords = pca2.fit_transform(all_vecs)

        chunk_coords = all_coords[:-1]
        query_coord = all_coords[-1]

        # Build plot dataframe
        plot_df = pd.DataFrame({
            "X": chunk_coords[:, 0],
            "Y": chunk_coords[:, 1],
            "Label": short_labels,
            "Similarity": [round(s, 3) for s in sims],
            "Preview": [d[:80] + "..." for d in docs],
            "Type": ["Top Match" if i in top3_idx else "Chunk" for i in range(n_chunks)]
        })

        col1, col2 = st.columns([3, 1])

        with col1:
            fig = go.Figure()

            # Regular chunks
            regular = plot_df[plot_df["Type"] == "Chunk"]
            fig.add_scatter(
                x=regular["X"], y=regular["Y"],
                mode="markers+text",
                text=regular["Label"],
                textposition="top center",
                marker=dict(size=10, color="#4f8ef7", opacity=0.6),
                name="Chunks",
                hovertemplate="<b>%{text}</b><br>Sim: %{customdata}<br>%{hovertext}<extra></extra>",
                customdata=regular["Similarity"],
                hovertext=regular["Preview"]
            )

            # Top matches
            top = plot_df[plot_df["Type"] == "Top Match"]
            fig.add_scatter(
                x=top["X"], y=top["Y"],
                mode="markers+text",
                text=top["Label"],
                textposition="top center",
                marker=dict(size=16, color="#f7a24f",
                            line=dict(width=2, color="#ffffff")),
                name="Top 3 Matches",
                hovertemplate="<b>%{text}</b><br>Sim: %{customdata}<br>%{hovertext}<extra></extra>",
                customdata=top["Similarity"],
                hovertext=top["Preview"]
            )

            # Draw lines from query to top 3
            for idx in top3_idx:
                fig.add_shape(type="line",
                    x0=query_coord[0], y0=query_coord[1],
                    x1=chunk_coords[idx, 0], y1=chunk_coords[idx, 1],
                    line=dict(color="#f7a24f", width=1.5, dash="dot")
                )

            # Query point
            fig.add_scatter(
                x=[query_coord[0]], y=[query_coord[1]],
                mode="markers+text",
                text=["YOUR QUERY"],
                textposition="bottom center",
                marker=dict(size=20, color="#ff4f4f", symbol="star",
                            line=dict(width=2, color="#ffffff")),
                name="Your Query"
            )

            fig.update_layout(
                title="Your query in vector space — dotted lines show retrieval",
                paper_bgcolor="#0f1117", plot_bgcolor="#1a1f2e",
                font_color="#c8d0e0",
                xaxis=dict(gridcolor="#2a2f3e"),
                yaxis=dict(gridcolor="#2a2f3e"),
                height=550,
                legend=dict(bgcolor="#1a1f2e", bordercolor="#4f8ef7")
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("**🏆 Top 3 Retrieved Chunks**")
            for rank, idx in enumerate(top3_idx):
                st.markdown(f"**#{rank+1} — {short_labels[idx]}**")
                st.markdown(f"Similarity: `{sims[idx]:.4f}`")
                st.success(docs[idx][:200] + "...")
                st.divider()

        # Query vector preview
        st.markdown("**🔢 Your Query as a Vector (first 50 dims)**")
        qv_df = pd.DataFrame({
            "Dimension": [f"D{i}" for i in range(50)],
            "Value": q_vec[:50]
        })
        fig_q = px.bar(
            qv_df, x="Dimension", y="Value",
            color="Value",
            color_continuous_scale="RdBu",
            color_continuous_midpoint=0,
            title=f'"{query[:50]}..." as a 384-dim vector',
            height=280
        )
        fig_q.update_layout(
            paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
            font_color="#c8d0e0", showlegend=False,
            xaxis=dict(tickangle=45, tickfont=dict(size=8))
        )
        st.plotly_chart(fig_q, use_container_width=True)

    else:
        st.info("👆 Type a question above to see it plotted in vector space alongside your course chunks.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown("""
<div style="text-align:center; color:#555; font-size:0.8em;">
RAG Vector Explorer &nbsp;·&nbsp; Built with ChromaDB + sentence-transformers + Plotly
</div>
""", unsafe_allow_html=True)
