"""
Information Retrieval Search Engine - Advanced Streamlit Frontend
Web GUI with file upload, highlighting, wildcard search, and spelling correction
"""

import streamlit as st
from search_engine import SearchEngine, SAMPLE_CORPUS
import os

# Page configuration
st.set_page_config(
    page_title="Advanced IR Search Engine",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        text-align: center;
        padding: 20px;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 30px;
    }
    .doc-card {
        padding: 15px;
        border-left: 4px solid #667eea;
        background-color: #f0f2f6;
        border-radius: 5px;
        margin: 10px 0;
    }
    .step-box {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 5px;
        font-family: monospace;
        font-size: 14px;
        white-space: pre-wrap;
        margin: 10px 0;
        max-height: 400px;
        overflow-y: auto;
    }
    .matched-terms {
        background-color: #e3f2fd;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .suggestion-box {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    mark {
        background-color: #ffeb3b;
        padding: 2px 4px;
        border-radius: 3px;
        font-weight: bold;
    }
    .feature-badge {
        display: inline-block;
        padding: 4px 8px;
        background-color: #4caf50;
        color: white;
        border-radius: 12px;
        font-size: 12px;
        margin: 2px;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'engine' not in st.session_state:
    st.session_state.engine = None
if 'corpus' not in st.session_state:
    st.session_state.corpus = SAMPLE_CORPUS
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []

# Function to load search engine
def load_search_engine(corpus):
    """Load and return the search engine with given corpus"""
    return SearchEngine(corpus)

# Function to read uploaded files
def read_uploaded_files(uploaded_files):
    """Read content from uploaded text files"""
    corpus = []
    for uploaded_file in uploaded_files:
        try:
            content = uploaded_file.read().decode('utf-8')
            # Split by lines or paragraphs (each line/paragraph as a document)
            lines = [line.strip() for line in content.split('\n') if line.strip()]
            corpus.extend(lines)
        except Exception as e:
            st.error(f"Error reading {uploaded_file.name}: {str(e)}")
    return corpus

# ==================== HEADER ====================

st.markdown("""
    <div class="main-header">
        <h1>🔍 Advanced Information Retrieval Search Engine</h1>
        <p>Boolean Retrieval • Phrase Queries • Soundex Phonetic Search</p>
        <div>
            <span class="feature-badge">📁 File Upload</span>
            <span class="feature-badge">✨ Highlighting</span>
            <span class="feature-badge">🔮 Wildcard Search</span>
            <span class="feature-badge">📝 Spelling Correction</span>
        </div>
    </div>
""", unsafe_allow_html=True)

# ==================== SIDEBAR ====================

with st.sidebar:
    st.header("📚 About")
    st.write("""
    This advanced IR system implements:
    - **Preprocessing**: Tokenization, Stop-word removal, Porter Stemming
    - **Indexing**: Inverted Index & Positional Index
    - **Retrieval Models**: Boolean, Phrase, Soundex
    - **Advanced Features**:
      - 📁 Dynamic file upload
      - ✨ Keyword highlighting
      - 🔮 Wildcard search (*)
      - 📝 Spelling correction
    """)
    
    st.divider()
    
    # ==================== FILE UPLOAD ====================
    
    st.header("📁 Upload Documents")
    st.write("Upload `.txt` files to build a custom corpus")
    
    uploaded_files = st.file_uploader(
        "Choose text files",
        type=['txt'],
        accept_multiple_files=True,
        help="Upload one or more .txt files. Each line will be treated as a separate document."
    )
    
    # Process uploaded files
    if uploaded_files:
        if uploaded_files != st.session_state.uploaded_files:
            st.session_state.uploaded_files = uploaded_files
            
            with st.spinner("📖 Reading files and building indexes..."):
                new_corpus = read_uploaded_files(uploaded_files)
                
                if new_corpus:
                    st.session_state.corpus = new_corpus
                    st.session_state.engine = load_search_engine(new_corpus)
                    st.success(f"✅ Loaded {len(new_corpus)} documents from {len(uploaded_files)} file(s)")
                else:
                    st.error("❌ No valid content found in uploaded files")
    else:
        # Use default corpus if no files uploaded
        if st.session_state.corpus != SAMPLE_CORPUS:
            st.session_state.corpus = SAMPLE_CORPUS
            st.session_state.engine = None
        
        st.info("💡 Using default corpus. Upload files to use custom documents.")
    
    st.divider()
    
    # Load engine if not loaded
    if st.session_state.engine is None:
        st.session_state.engine = load_search_engine(st.session_state.corpus)
    
    engine = st.session_state.engine
    
    # ==================== METRICS ====================
    
    st.header("📊 Index Statistics")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Documents", len(engine.get_corpus()))
    with col2:
        st.metric("Total Terms", engine.get_total_terms())
    
    st.metric("Inverted Index Terms", len(engine.get_inverted_index()))
    st.metric("Positional Index Terms", len(engine.get_positional_index()))
    st.metric("Soundex Codes", len(engine.get_soundex_index()))
    
    st.divider()
    
    # Show/Hide Indexes
    show_indexes = st.checkbox("Show Index Structures", value=False)

# ==================== MAIN CONTENT ====================

# Search Interface
col1, col2 = st.columns([3, 1])

with col1:
    query = st.text_input(
        "🔎 Enter your search query:",
        placeholder="e.g., 'information AND retrieval', 'quick brown fox', 'comput*', 'Robert'",
        help="Enter your query based on the selected retrieval model"
    )

with col2:
    model = st.selectbox(
        "Select Retrieval Model:",
        ["Boolean Retrieval", "Phrase Query", "Soundex Search"],
        help="Choose the search algorithm to use"
    )

# Model descriptions with advanced features
if model == "Boolean Retrieval":
    st.info("💡 **Boolean Retrieval**: Use AND, OR, NOT operators. Supports wildcards (e.g., 'comput*' matches 'computer', 'computing')")
elif model == "Phrase Query":
    st.info("💡 **Phrase Query**: Search for exact phrase matches with adjacent words (e.g., 'quick brown fox')")
else:
    st.info("💡 **Soundex Search**: Find phonetically similar names (e.g., 'Robert' matches 'Rania', 'Mohamed' matches 'Muhammad')")

# Search button
search_clicked = st.button("🚀 Search", type="primary", use_container_width=True)

# ==================== DISPLAY INDEXES ====================

if show_indexes:
    st.divider()
    st.header("📊 Index Structures")
    
    tab1, tab2, tab3 = st.tabs(["Inverted Index", "Positional Index", "Soundex Index"])
    
    with tab1:
        st.subheader("Inverted Index (Term → Documents)")
        inv_index = engine.get_inverted_index()
        
        # Display first 20 terms
        for i, (term, docs) in enumerate(list(inv_index.items())[:20]):
            st.text(f"{term}: {docs}")
        
        if len(inv_index) > 20:
            st.text(f"... and {len(inv_index) - 20} more terms")
    
    with tab2:
        st.subheader("Positional Index (Term → {Doc: [Positions]})")
        pos_index = engine.get_positional_index()
        
        # Display first 15 terms
        for i, (term, docs_pos) in enumerate(list(pos_index.items())[:15]):
            st.text(f"{term}:")
            for doc_id, positions in docs_pos.items():
                st.text(f"  Doc {doc_id}: {positions}")
        
        if len(pos_index) > 15:
            st.text(f"... and {len(pos_index) - 15} more terms")
    
    with tab3:
        st.subheader("Soundex Index (Code → Terms)")
        soundex_index = engine.get_soundex_index()
        
        # Display first 20 codes
        for i, (code, data) in enumerate(list(soundex_index.items())[:20]):
            terms = sorted(list(data['terms']))
            docs = sorted(list(data['docs']))
            st.text(f"{code}: {terms}")
            st.text(f"  Documents: {docs}")
        
        if len(soundex_index) > 20:
            st.text(f"... and {len(soundex_index) - 20} more codes")

# ==================== SEARCH EXECUTION ====================

if search_clicked and query:
    st.divider()
    
    # Extract query terms for highlighting
    query_terms = query.lower().replace(' and ', ' ').replace(' or ', ' ').replace('not ', '').split()
    query_terms = [term.replace('*', '') for term in query_terms if term]
    
    # Execute search based on model
    suggestion = None
    matched_terms = None
    
    if model == "Boolean Retrieval":
        result_docs, steps, suggestion = engine.boolean_retrieval(query)
    elif model == "Phrase Query":
        result_docs, steps, suggestion = engine.phrase_query(query)
    else:  # Soundex Search
        result_docs, steps, matched_terms = engine.soundex_search(query)
    
    # ==================== SPELLING SUGGESTION ====================
    
    if suggestion:
        st.markdown(f"""
            <div class="suggestion-box">
                <h4>💡 Did you mean: <strong>"{suggestion}"</strong>?</h4>
                <p>No results found for your query. Try the suggested spelling above.</p>
            </div>
        """, unsafe_allow_html=True)
    
    # ==================== PROCESSING STEPS ====================
    
    st.header("📝 Processing Steps")
    
    with st.expander("Show detailed processing steps", expanded=True):
        st.markdown('<div class="step-box">', unsafe_allow_html=True)
        for step in steps:
            st.text(step)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Display matched terms for Soundex
    if matched_terms:
        st.subheader("🔊 Phonetic Matches")
        for query_term, similar_terms in matched_terms.items():
            if similar_terms:
                st.markdown(f'<div class="matched-terms">', unsafe_allow_html=True)
                st.write(f"**'{query_term}'** → {', '.join(similar_terms)}")
                st.markdown('</div>', unsafe_allow_html=True)
    
    # ==================== RETRIEVED DOCUMENTS ====================
    
    st.divider()
    st.header(f"📄 Retrieved Documents ({len(result_docs)})")
    
    if len(result_docs) == 0:
        st.warning("❌ No documents found matching your query.")
    else:
        for doc_id in result_docs:
            # Highlight query terms in document
            if model == "Soundex Search" and matched_terms:
                # For Soundex, highlight all phonetically similar terms
                highlight_terms = []
                for terms in matched_terms.values():
                    highlight_terms.extend(terms)
                highlighted_text = engine.highlight_terms(doc_id, highlight_terms)
            else:
                # For Boolean and Phrase, highlight query terms
                highlighted_text = engine.highlight_terms(doc_id, query_terms)
            
            st.markdown(f'<div class="doc-card">', unsafe_allow_html=True)
            st.markdown(f"**Document {doc_id}**")
            st.markdown(highlighted_text, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

elif search_clicked and not query:
    st.warning("⚠️ Please enter a search query.")

# ==================== DOCUMENT CORPUS ====================

st.divider()
st.header("📚 Document Corpus")

with st.expander("View all documents in the corpus"):
    corpus = engine.get_corpus()
    
    # Show warning if corpus is large
    if len(corpus) > 50:
        st.warning(f"⚠️ Large corpus detected ({len(corpus)} documents). Showing first 50 documents.")
        corpus = corpus[:50]
    
    for idx, doc in enumerate(corpus):
        st.markdown(f'<div class="doc-card">', unsafe_allow_html=True)
        st.markdown(f"**Doc {idx}:** {doc}")
        st.markdown('</div>', unsafe_allow_html=True)

# ==================== EXAMPLE QUERIES ====================

st.divider()
st.header("💡 Example Queries")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Boolean Retrieval")
    st.code("information AND retrieval")
    st.code("fox OR dog")
    st.code("NOT python")
    st.code("comput*")
    st.caption("🔮 Wildcard: comput* matches computer, computing, etc.")

with col2:
    st.subheader("Phrase Query")
    st.code("quick brown fox")
    st.code("search engine")
    st.code("machine learning")
    st.code("natural language")

with col3:
    st.subheader("Soundex Search")
    st.code("Robert")
    st.code("Herman")
    st.code("Mohamed")
    st.code("Smith")
    st.caption("🔊 Finds phonetically similar names")

# ==================== ADVANCED FEATURES INFO ====================

st.divider()
st.header("✨ Advanced Features")

feat_col1, feat_col2 = st.columns(2)

with feat_col1:
    st.subheader("🔮 Wildcard Search")
    st.write("""
    Use the `*` character to match multiple terms with a common prefix:
    - `comput*` matches: computer, computing, computational
    - `inform*` matches: information, informational, informatics
    """)
    
    st.subheader("✨ Keyword Highlighting")
    st.write("""
    Search results automatically highlight matched terms in yellow, making it easy to see why a document was retrieved.
    """)

with feat_col2:
    st.subheader("📝 Spelling Correction")
    st.write("""
    If your search returns no results, the system uses Levenshtein Distance (Edit Distance) to suggest the closest matching term from the index.
    """)
    
    st.subheader("📁 Dynamic Corpus")
    st.write("""
    Upload your own `.txt` files via the sidebar to search through custom documents instead of the default corpus.
    """)

# ==================== FOOTER ====================

st.divider()
st.markdown("""
    <div style="text-align: center; color: #666; padding: 20px;">
        <p><strong>Advanced Information Retrieval Search Engine</strong></p>
        <p>Powered by NLTK, Streamlit | Features: File Upload, Highlighting, Wildcard Search, Spelling Correction</p>
    </div>
""", unsafe_allow_html=True)