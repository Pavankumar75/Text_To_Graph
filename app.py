import streamlit as st
import pandas as pd
from langchain_core.documents import Document
from langchain_neo4j import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_ollama import ChatOllama
from langchain_neo4j.chains.graph_qa.cypher import GraphCypherQAChain
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from streamlit_agraph import agraph, Node, Edge, Config
import neo4j.exceptions
import re

CYPHER_GENERATION_TEMPLATE = """Task:Generate Cypher statement to query a graph database.
Instructions:
Use only the provided relationship types and property keys in the schema.
Do not use any other relationship types or property keys that are not provided.
Schema:
{schema}
Note: Do not include any explanations or apologies in your responses.
Do not respond to any questions that might ask anything else than for you to construct a Cypher statement.
Do not include any text except the generated Cypher statement.

The question is:
{question}"""

QA_GENERATION_TEMPLATE = """Task: Answer the question based ONLY on the following graph search results.
Instructions:
- If the answer is not in the context, say "I don't know based on the provided data."
- Do not use your internal knowledge.
- Keep the answer concise.

Context:
{context}

Question:
{question}

Answer:"""

# --- Configuration & UI Setup ---
st.set_page_config(page_title="Text-to-Graph RAG (Ollama)", layout="wide")

st.title("🕸️ Dynamic Social-Graph RAG")
st.markdown("Extract entities & relationships using **Ollama**, visualize with **Graphviz**, and chat with your graph (Offline or Neo4j).")

# --- Sidebar: Connection Settings ---
with st.sidebar:
    st.header("🔌 Settings")
    
    use_neo4j = st.toggle("Use Neo4j Database", value=False)
    
    if use_neo4j:
        neo4j_uri = st.text_input("Neo4j URI", value="bolt://localhost:7687")
        neo4j_user = st.text_input("Neo4j Username", value="neo4j")
        neo4j_password = st.text_input("Neo4j Password", value="Pavankumar@2003", type="password")
        clear_graph = st.checkbox("Clear existing graph?", value=True)
        
        if "neo4j_status" not in st.session_state:
            st.session_state["neo4j_status"] = "Not Connected"

        status_color = "red"
        if st.session_state["neo4j_status"] == "Connected":
            status_color = "green"
        elif st.session_state["neo4j_status"] == "Not Connected":
            status_color = "gray"
            
        st.markdown(f"**Status:** :{status_color}[{st.session_state['neo4j_status']}]")

        if st.button("Test / Connect to Neo4j"):
            try:
                graph = Neo4jGraph(url=neo4j_uri, username=neo4j_user, password=neo4j_password)
                graph.query("RETURN 1")
                st.session_state["neo4j_status"] = "Connected"
                st.success("✅ Connected to Neo4j!")
                st.rerun()
            except Exception as e:
                st.session_state["neo4j_status"] = f"Failed: {str(e)}"
                st.error(f"❌ Connection Failed: {e}")
    else:
        st.info("Running in **Offline Mode**. Graph will be stored in memory.")

    st.divider()
    ollama_url = st.text_input("Ollama Base URL", value="http://localhost:11434")
    model_name = st.text_input("Ollama Model", value="llama3.2:3b")

# --- Main Interface ---
col1, col2 = st.columns([1, 1])

default_text = """Ned Stark, the Lord of Winterfell, travels to the capital city of King's Landing to serve as Hand of the King for Robert Baratheon. Following the suspicious death of Robert Baratheon, the Lannister family, led by Cersei Lannister, seizes the Iron Throne for her son, Joffrey Baratheon. This act of usurpation triggers the War of the Five Kings, involving factions like the House Stark, House Lannister, and House Baratheon. Simultaneously, Daenerys Targaryen resides in the continent of Essos, where she commands a legion of Unsullied and hatches three dragons. In the far North, Jon Snow joins the Night’s Watch to defend The Wall against the White Walkers, an ancient undead threat led by the Night King. The narrative culminates when Arya Stark defeats the Night King, and Bran Stark is ultimately elected King of the Six Kingdoms."""

with col1:
    st.subheader("📝 Input Text")
    
    # Input Mode Selection
    input_mode = st.radio("Input Mode:", ["Text Area", "CSV File", "JSON File"], horizontal=True)
    
    input_text = ""
    uploaded_file = None
    
    input_text = ""
    uploaded_file = None
    
    # State variables for filtering
    date_col = None
    start_date = None
    end_date = None
    
    if input_mode == "Text Area":
        input_text = st.text_area("Enter text:", value=default_text, height=300)
    elif input_mode in ["CSV File", "JSON File"]:
        if input_mode == "CSV File":
            uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
        else:
            uploaded_file = st.file_uploader("Upload JSON", type=["json"])
            
        if uploaded_file:
            # Read Data for Preview & Setup
            uploaded_file.seek(0)
            if input_mode == "CSV File":
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_json(uploaded_file)

            # Auto-Detect Date Column
            # Heuristic: Look for 'date', 'time', 'timestamp', 'published', 'created' in column name or strictly datetime types
            potential_date_cols = []
            for col in df.columns:
                # 1. Check if already datetime
                if pd.api.types.is_datetime64_any_dtype(df[col]):
                    potential_date_cols.append(col)
                    continue
                
                # 2. Check name heuristics
                col_lower = str(col).lower()
                keywords = ["date", "time", "timestamp", "published", "created", "at"]
                if any(k in col_lower for k in keywords):
                    # Try converting a sample to check validity (fail fast if not date)
                    try:
                        # Convert sample to minimize performance hit
                        sample = df[col].dropna().head(10)
                        if not sample.empty:
                            pd.to_datetime(sample, errors='raise')
                            potential_date_cols.append(col)
                    except:
                        pass
            
            if potential_date_cols:
                # Default to the first detected column (usually best match) or allow fallback logic if needed
                date_col = potential_date_cols[0]
                
                # Convert to datetime for internal consistency (standardize format)
                df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                
                # Normalize Timezone
                if pd.api.types.is_datetime64_any_dtype(df[date_col]):
                     try:
                         df[date_col] = df[date_col].dt.tz_localize(None)
                     except:
                         pass

                # Persist for Query Logic (Store FULL DataFrame)
                st.session_state["source_df"] = df
                st.session_state["date_col"] = date_col
                
                # Optional: Just show a small info badge
                st.caption(f"📅 Date column detected: `{date_col}`. You can ask date-specific questions.")
            
            st.write(f"**Preview ({len(df)} rows):**")
            # Create a display copy to show cleaner dates
            display_df = df.head(10).copy()
            if date_col and date_col in display_df.columns:
                 display_df[date_col] = display_df[date_col].dt.strftime('%Y-%m-%d %H:%M:%S')
            st.dataframe(display_df)

    st.divider()
    st.markdown("**Schema Definition (Optional)**")
    allowed_nodes_input = st.text_input("Allowed Node Types (comma-separated)", placeholder="Person, City, Organization")
    allowed_rels_input = st.text_input("Allowed Relationship Types (comma-separated)", placeholder="LIVES_IN, WORKS_FOR")

    process_btn = st.button("🚀 Process Graph", type="primary")

# --- Logic ---

def get_llm():
    return ChatOllama(model=model_name, base_url=ollama_url, temperature=0)

def extract_date_from_query(query):
    """Extracts a date from a natural language query using regex."""
    date_pattern = r'\b(\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{2,4})\b'
    match = re.search(date_pattern, query, re.IGNORECASE)
    if match:
        try:
            return pd.to_datetime(match.group(0))
        except (ValueError, TypeError):
            return None
    return None

def query_pandas_context(query, df, date_col, date_val):
    """Filters the DataFrame for a specific date and uses it as context for the LLM."""
    try:
        # Remove the date part from the query to clean it for the LLM
        date_pattern = r'\b(\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{2,4})\b'
        clean_query = re.sub(date_pattern, '', query, flags=re.IGNORECASE).strip()
        # Clean up trailing prepositions often associated with dates
        clean_query = re.sub(r'\s+(on|for|at|during|in)$', '', clean_query, flags=re.IGNORECASE).strip()

        # Ensure datetime and remove timezone for comparison
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        if pd.api.types.is_datetime64_any_dtype(df[date_col]):
             df[date_col] = df[date_col].dt.tz_localize(None)

        mask = df[date_col].dt.date == pd.to_datetime(date_val).date()
        filtered_df = df.loc[mask]
        
        if filtered_df.empty:
            return f"I detected a date-based query for {date_val.date()}, but no records match that date."
        
        # Construct context from filtered rows
        records = [str({k: v for k, v in row.items() if k != date_col}) for _, row in filtered_df.head(30).iterrows()]
        context = "\n".join(records)
            
        llm = get_llm()
        prompt = PromptTemplate.from_template(
            """Task: Answer the user's question based ONLY on the following records.
            
            Records:
            {context}
            
            Question: {query}
            Instructions: Provide a detailed answer. If listing items, list up to 10 distinct items.
            Answer:"""
        )
        chain = prompt | llm | StrOutputParser()
        return chain.invoke({"context": context, "query": clean_query})
        
    except Exception as e:
        return f"Error filtering data: {str(e)}"

def visualize_graph(documents):
    nodes = []
    edges = []
    seen_nodes = set()
    
    for doc in documents:
        for rel in doc.relationships:
            source_id = rel.source.id
            target_id = rel.target.id
            source_label = rel.source.type
            target_label = rel.target.type
            rel_type = rel.type
            
            # Add nodes
            if source_id not in seen_nodes:
                nodes.append(Node(id=source_id, label=source_id, size=25, shape="dot", title=source_label))
                seen_nodes.add(source_id)
            
            if target_id not in seen_nodes:
                nodes.append(Node(id=target_id, label=target_id, size=25, shape="dot", title=target_label))
                seen_nodes.add(target_id)
            
            # Add edge
            edges.append(Edge(source=source_id, target=target_id, label=rel_type))
            
    return nodes, edges

def process_documents(documents, use_db, uri, user, pwd, clear, allowed_nodes=[], allowed_rels=[]):
    try:
        # LLM & Transformer
        llm = get_llm()
        
        # Configure Transformer with Optional Constraints
        if allowed_nodes or allowed_rels:
            llm_transformer = LLMGraphTransformer(
                llm=llm, 
                allowed_nodes=allowed_nodes if allowed_nodes else None,
                allowed_relationships=allowed_rels if allowed_rels else None
            )
        else:
            llm_transformer = LLMGraphTransformer(llm=llm)
        
        st.info(f"Extracting graph from {len(documents)} context chunks... (this uses local LLM, please wait)")
        progress_bar = st.progress(0)
        
        all_graph_docs = []
        
        # Process in chunks to update progress
        chunk_size = 20 # Reduced chunk size for local LLM stability
        total_chunks = len(documents) // chunk_size + (1 if len(documents) % chunk_size > 0 else 0)
        
        status_text = st.empty()
        
        for i in range(0, len(documents), chunk_size):
            chunk_num = (i // chunk_size) + 1
            status_text.text(f"Processing chunk {chunk_num}/{total_chunks}...")
            
            chunk = documents[i : i + chunk_size]
            try:
                chunk_graph_docs = llm_transformer.convert_to_graph_documents(chunk)
                all_graph_docs.extend(chunk_graph_docs)
            except Exception as e:
                st.warning(f"⚠️ Error processing chunk {chunk_num}: {e}")
            
            # Update progress
            current_progress = min((i + chunk_size) / len(documents), 1.0)
            progress_bar.progress(current_progress)
            
        progress_bar.empty()
        status_text.empty()
        
        if not all_graph_docs:
            st.warning("No entities extracted.")
            return None, None
            
        total_nodes = sum(len(d.nodes) for d in all_graph_docs)
        total_rels = sum(len(d.relationships) for d in all_graph_docs)
        st.success(f"Extracted {total_nodes} nodes and {total_rels} relationships from {len(documents)} records.")
        
        # Storage
        if use_db:
            try:
                graph = Neo4jGraph(url=uri, username=user, password=pwd)
                if clear:
                    graph.query("MATCH (n) DETACH DELETE n")
                graph.add_graph_documents(all_graph_docs)
                st.success("Synced to Neo4j.")
                return graph, all_graph_docs
            except Exception as e:
                st.error(f"Neo4j Error: {e}")
                return None, all_graph_docs
        else:
            return "OFFLINE_GRAPH", all_graph_docs
            
    except Exception as e:
        st.error(f"Extraction Error: {e}")
        return None, None

def query_graph(query, graph_obj, graph_docs, use_db):
    # 1. Check for Temporal Query on Source Data
    if "source_df" in st.session_state and st.session_state.get("date_col"):
        detected_date = extract_date_from_query(query)
        if detected_date:
            st.info(f"📅 Temporal Search Active: Using data for {detected_date.date()}")
            return query_pandas_context(query, st.session_state["source_df"], st.session_state["date_col"], detected_date)

    llm = get_llm()
    
    if use_db and graph_obj != "OFFLINE_GRAPH":
        # Online RAG (Cypher)
        CYPHER_PROMPT = PromptTemplate(input_variables=["schema", "question"], template=CYPHER_GENERATION_TEMPLATE)
        QA_PROMPT = PromptTemplate(input_variables=["context", "question"], template=QA_GENERATION_TEMPLATE)

        chain = GraphCypherQAChain.from_llm(
            graph=graph_obj, 
            llm=llm, 
            cypher_prompt=CYPHER_PROMPT,
            qa_prompt=QA_PROMPT,
            verbose=True,
            allow_dangerous_requests=True,
            return_intermediate_steps=True,
            top_k=20
        )
        try:
            return chain.invoke({"query": query})
        except Exception as e:
             return {"result": f"Error querying graph: {e}", "intermediate_steps": []}
    else:
        # Offline RAG (Context Injection)
        triples = []
        for doc in graph_docs:
            for rel in doc.relationships:
                triples.append(f"({rel.source.id}) -[{rel.type}]-> ({rel.target.id})")
        
        context_str = "\n".join(triples)
        template = """Answer the question based only on the following graph relationships:
        {context}
        
        Question: {question}
        Instructions: Provide a detailed answer. If listing items or relationships, list up to 10 items.
        Answer:"""
        
        prompt = PromptTemplate.from_template(template)
        chain = prompt | llm | StrOutputParser()
        return chain.invoke({"context": context_str, "question": query})

# --- Execution ---

if process_btn:
    # Validate Neo4j Connection BEFORE extraction (Fail Fast)
    if use_neo4j:
        with st.spinner("Verifying Neo4j Connection..."):
            try:
                graph = Neo4jGraph(url=neo4j_uri, username=neo4j_user, password=neo4j_password)
                graph.query("RETURN 1")
            except Exception as e:
                st.error(f"❌ Could not connect to Neo4j: {e}")
                st.stop()

    with st.spinner("Processing..."):
        
        # Determine allowed lists
        allowed_nodes = [x.strip() for x in allowed_nodes_input.split(",")] if allowed_nodes_input else []
        allowed_rels = [x.strip() for x in allowed_rels_input.split(",")] if allowed_rels_input else []
        
        # Prepare content
        docs = []
        if input_mode == "Text Area":
            if input_text:
                docs = [Document(page_content=input_text)]
        elif input_mode in ["CSV File", "JSON File"]:
            if uploaded_file is not None:
                # Convert CSV/JSON rows to Documents
                # Strategy: key: value, key: value...
                uploaded_file.seek(0)
                
                if input_mode == "CSV File":
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_json(uploaded_file)
                
                # Fill NAs to avoid 'nan' content
                df.fillna("", inplace=True)
                
                # Batched processing for speed
                # Group sets of rows into a single document to reduce LLM overhead
                batch_size = 10
                current_batch = []
                
                for _, row in df.iterrows():
                    # Construct a text representation of the row
                    # Use newlines to separate fields to avoid run-on sentence confusion for the LLM
                    text_parts = [f"{col}: {val}" for col, val in row.items()]
                    row_text = "\n".join(text_parts)
                    current_batch.append(row_text)
                    
                    if len(current_batch) >= batch_size:
                        # Separate records clearly with a separator line
                        docs.append(Document(page_content="\n\n---\n\n".join(current_batch)))
                        current_batch = []
                
                # Add remaining
                if current_batch:
                    docs.append(Document(page_content="\n\n---\n\n".join(current_batch)))

        if not docs:
            st.warning("No content to process!")
        else:
            graph_obj, graph_docs = process_documents(docs, use_neo4j, neo4j_uri if use_neo4j else "", neo4j_user if use_neo4j else "", neo4j_password if use_neo4j else "", clear_graph if use_neo4j else False, allowed_nodes, allowed_rels)
        
        if graph_docs:
            st.session_state["graph_data"] = {
                "obj": graph_obj,
                "docs": graph_docs,
                "use_db": use_neo4j
            }

# --- Results & Chat ---

with col2:
    st.subheader("🔍 Graph Knowledge")
    
    if "graph_data" in st.session_state:
        data = st.session_state["graph_data"]
        
        # Chat
        query = st.text_input("Ask a question:", placeholder="Who is the Lord of Winterfell?")
        if query:
            with st.spinner("Thinking..."):
                ans_result = query_graph(query, data["obj"], data["docs"], data["use_db"])
                
                # Handle different return types (String vs Dict)
                if isinstance(ans_result, str):
                    st.markdown(f"**Answer:** {ans_result}")
                elif isinstance(ans_result, dict):
                    # Online Mode (Cypher) or Dict Error
                    final_ans = ans_result.get("result", "No result returned.")
                    steps = ans_result.get("intermediate_steps", [])
                        
                    if steps:
                            with st.expander("🕵️ Generated Cypher Query"):
                                st.code(steps[0]["query"], language="cypher")
                                st.write("**Context:**")
                                st.json(steps[0]["context"])

                    st.markdown(f"**Answer:** {final_ans}")
        
        st.divider()
        
        # Visualization
        with st.expander("🕸️ Graph Visualization", expanded=True):
            nodes, edges = visualize_graph(data["docs"])
            config = Config(width="100%", height=500, directed=True, nodeHighlightBehavior=True, highlightColor="#F7A7A6", collapsible=True)
            igraph = agraph(nodes=nodes, edges=edges, config=config)
            
        # Raw Data
        with st.expander("📄 Raw Triples"):
            for doc in data["docs"]:
                for rel in doc.relationships:
                    st.text(f"{rel.source.id} -> {rel.type} -> {rel.target.id}")
                    
    else:
        st.info("Process text to see the graph.")
