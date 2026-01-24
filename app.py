import streamlit as st
from langchain_core.documents import Document
from langchain_community.graphs import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_ollama import ChatOllama
from langchain_neo4j.chains.graph_qa.cypher import GraphCypherQAChain
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from streamlit_agraph import agraph, Node, Edge, Config

# --- Configuration & UI Setup ---
st.set_page_config(page_title="Text-to-Graph RAG (Ollama)", layout="wide")

st.title("🕸️ Dynamic Text-to-Graph RAG")
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
    else:
        st.info("Running in **Offline Mode**. Graph will be stored in memory.")

    st.divider()
    ollama_url = st.text_input("Ollama Base URL", value="http://localhost:11434")
    model_name = st.text_input("Ollama Model", value="llama3:latest")

# --- Main Interface ---
col1, col2 = st.columns([1, 1])

default_text = """Ned Stark, the Lord of Winterfell, travels to the capital city of King's Landing to serve as Hand of the King for Robert Baratheon. Following the suspicious death of Robert Baratheon, the Lannister family, led by Cersei Lannister, seizes the Iron Throne for her son, Joffrey Baratheon. This act of usurpation triggers the War of the Five Kings, involving factions like the House Stark, House Lannister, and House Baratheon. Simultaneously, Daenerys Targaryen resides in the continent of Essos, where she commands a legion of Unsullied and hatches three dragons. In the far North, Jon Snow joins the Night’s Watch to defend The Wall against the White Walkers, an ancient undead threat led by the Night King. The narrative culminates when Arya Stark defeats the Night King, and Bran Stark is ultimately elected King of the Six Kingdoms."""

with col1:
    st.subheader("📝 Input Text")
    input_text = st.text_area("Enter text:", value=default_text, height=300)
    process_btn = st.button("🚀 Process Graph", type="primary")

# --- Logic ---

def get_llm():
    return ChatOllama(model=model_name, base_url=ollama_url, temperature=0)

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

def process_text(text, use_db, uri, user, pwd, clear):
    try:
        # LLM & Transformer
        llm = get_llm()
        llm_transformer = LLMGraphTransformer(llm=llm)
        
        st.info("Extracting graph... (this uses Llama 3, please wait)")
        raw_docs = [Document(page_content=text)]
        graph_docs = llm_transformer.convert_to_graph_documents(raw_docs)
        
        if not graph_docs:
            st.warning("No entities extracted.")
            return None, None
            
        st.success(f"Extracted {len(graph_docs[0].nodes)} nodes and {len(graph_docs[0].relationships)} relationships.")
        
        # Storage
        if use_db:
            try:
                graph = Neo4jGraph(url=uri, username=user, password=pwd)
                if clear:
                    graph.query("MATCH (n) DETACH DELETE n")
                graph.add_graph_documents(graph_docs)
                st.success("Synced to Neo4j.")
                return graph, graph_docs
            except Exception as e:
                st.error(f"Neo4j Error: {e}")
                return None, graph_docs
        else:
            return "OFFLINE_GRAPH", graph_docs
            
    except Exception as e:
        st.error(f"Extraction Error: {e}")
        return None, None

def query_graph(query, graph_obj, graph_docs, use_db):
    llm = get_llm()
    
    if use_db and graph_obj != "OFFLINE_GRAPH":
        # Online RAG (Cypher)
        chain = GraphCypherQAChain.from_llm(
            graph=graph_obj, 
            llm=llm, 
            verbose=True,
            allow_dangerous_requests=True
        )
        return chain.invoke({"query": query}).get("result")
    else:
        # Offline RAG (Context Injection)
        # Serialize triples into context
        triples = []
        for doc in graph_docs:
            for rel in doc.relationships:
                triples.append(f"({rel.source.id}) -[{rel.type}]-> ({rel.target.id})")
        
        context_str = "\n".join(triples)
        
        template = """Answer the question based only on the following graph relationships:
        {context}
        
        Question: {question}
        Answer:"""
        
        prompt = PromptTemplate.from_template(template)
        chain = prompt | llm | StrOutputParser()
        
        return chain.invoke({"context": context_str, "question": query})

# --- Execution ---

if process_btn:
    with st.spinner("Processing..."):
        graph_obj, graph_docs = process_text(input_text, use_neo4j, neo4j_uri if use_neo4j else "", neo4j_user if use_neo4j else "", neo4j_password if use_neo4j else "", clear_graph if use_neo4j else False)
        
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
                ans = query_graph(query, data["obj"], data["docs"], data["use_db"])
                st.markdown(f"**Answer:** {ans}")
        
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
