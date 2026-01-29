"""
Streamlit RAG Application
Simple interface for document upload, querying, and response display with token tracking.
"""

import os
import tempfile
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from vector_store import VectorStore
from llm_inference import LLMInference

# Load environment variables from .env file
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="RAG Document Assistant",
    page_icon="📚",
    layout="wide"
)

# All 20 Test questions for batch evaluation
TEST_QUESTIONS = [
    "How can the reliability of LLM-as-a-Judge systems be systematically improved and evaluated across diverse assessment scenarios?",
    "How do memory, tool use, and planning capabilities enable LLM agents to provide personalized and scalable educational support?"
    "How does replacing sign-based gradient scaling with a momentum-based adaptive matrix improve convergence and transferability in adversarial attacks?",
    "How does the LLM compare method estimate problem difficulty without ground truth while remaining robust to hallucinations?",
    "Why does teacher-based data filtering provably improve representation learning performance under noisy multimodal data conditions?",
    "How can mass conservation and quasipositivity be embedded directly into learned reaction terms to ensure both physical consistency and well-posedness of reaction–diffusion systems? ",
    "Why is Maximum Mean Discrepancy (MMD) insufficient for evaluating graph generative models, and how does the proposed representation-aware evaluation (RGM) address this limitation? ",
    "How does the combination of Legendre Memory and flow-based probabilistic modeling enable FLAME to achieve efficient zero-shot forecasting for both deterministic and probabilistic time series tasks? ",
    "In what ways do decision tree–based surrogate models improve interpretability and scalability over Gaussian Process surrogates in preferential Bayesian optimization? ",
    "How does implicit bias in Hopfield network training lead to the emergence of invariance and enable efficient learning of graph isomorphism classes from limited samples? ",
    "How does CADYT leverage Gaussian Process–based continuous-time modeling to improve causal structure discovery compared to discrete-time Dynamic Bayesian Network approaches, especially under irregular sampling conditions?",
    "How do quantum canaries enable black-box auditing of quantum machine learning models, and how is the canary offset mathematically linked to empirical lower bounds on privacy budget leakage? ",
    "In what ways does the REPO mechanism reduce extraneous cognitive load in large language models, and how does learning non-linear token positions improve long-context and noisy-context reasoning performance? ",
    "How does the SuperWing dataset’s geometry parameterization strategy improve generalization for data-driven aerodynamic surrogate models compared to baseline-perturbation wing datasets? ",
    "How does GRAFT achieve strict temporal and regional alignment between external textual data (news, social media, policy) and power load signals, and why is position-aware cross-attention critical for multi-scale forecasting accuracy? ",
    "How does the Dual-Axis RCCL framework combine local valence environments (GCN) and ring/cage topologies (NBG) to achieve representation-complete and convergent learning across vast organic chemical space? ",
    "Why does relaxation to prior perturbation (RTPP) provide greater stability than relaxation to prior spread (RTPS) when assimilating real observations in purely data-driven ML-based weather forecasting systems like ClimaX-LETKF? ",
    "How does AnySleep’s channel-agnostic attention mechanism enable robust sleep staging at sub-30-second temporal resolutions across heterogeneous EEG and EOG channel configurations? ",
    "What are the trade-offs between rhythm-specific controllability and signal fidelity when using class-conditioned VAEs versus sinus-only VAEs for synthetic atrial electrogram generation? ",
    "Why are minimal input-perturbation-based counterfactual explanations insufficient for clinical time series applications, and how do temporal coherence and human-centered intervention constraints address these limitations? "

]


def init_session_state():
    """Initialize session state variables."""
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = VectorStore(
            chunk_size=800,
            chunk_overlap=120,
            top_k=15
        )
    
    if "llm" not in st.session_state:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if api_key:
            st.session_state.llm = LLMInference(
                api_key=api_key,
                max_tokens=2000,
                history_file="qa_history.json"
            )
        else:
            st.session_state.llm = None
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "batch_running" not in st.session_state:
        st.session_state.batch_running = False
    
    if "batch_progress" not in st.session_state:
        st.session_state.batch_progress = 0
    
    if "batch_results" not in st.session_state:
        st.session_state.batch_results = []


def render_sidebar():
    """Render the sidebar with file upload and stats."""
    with st.sidebar:
        st.header("📁 Document Upload")
        
        # File uploader
        uploaded_files = st.file_uploader(
            "Upload documents",
            type=["pdf", "docx", "txt", "md", "csv"],
            accept_multiple_files=True,
            help="Supported: PDF, DOCX, TXT, MD, CSV"
        )
        
        if uploaded_files:
            if st.button("Process Documents", type="primary"):
                process_uploaded_files(uploaded_files)
        
        # Vector store stats
        st.divider()
        st.header("📊 Vector Store")
        
        stats = st.session_state.vector_store.get_stats()
        col1, col2 = st.columns(2)
        col1.metric("Documents", stats["total_documents"])
        col2.metric("Chunks", stats["total_chunks"])
        
        # Show indexed documents
        if stats["documents"]:
            with st.expander("Indexed Documents"):
                for doc in stats["documents"]:
                    st.write(f"• {doc['filename']} ({doc['chunk_count']} chunks)")
        
        # Token usage
        if st.session_state.llm:
            st.divider()
            st.header("🔢 Token Usage")
            
            token_summary = st.session_state.llm.get_token_summary()
            cumulative = token_summary.get("cumulative_token_usage", {})
            
            st.metric("Total Queries", token_summary.get("total_queries", 0))
            
            col1, col2 = st.columns(2)
            col1.metric("Input Tokens", cumulative.get("total_input_tokens", 0))
            col2.metric("Output Tokens", cumulative.get("total_output_tokens", 0))
            
            st.metric("Total Tokens", cumulative.get("total_tokens", 0))
        
        # Batch Test Section
        st.divider()
        st.header("🧪 Batch Testing")
        
        stats = st.session_state.vector_store.get_stats()
        
        if stats["total_documents"] == 0:
            st.warning("Upload documents first")
        elif st.session_state.batch_running:
            st.info(f"Running... {st.session_state.batch_progress}/{len(TEST_QUESTIONS)}")
        else:
            st.write(f"**{len(TEST_QUESTIONS)} questions ready**")
            if st.button("▶️ Run All Test Questions", type="secondary", use_container_width=True):
                run_batch_test()
        
        # Clear buttons
        st.divider()
        col1, col2 = st.columns(2)
        
        if col1.button("Clear Documents"):
            st.session_state.vector_store.clear()
            st.session_state.messages = []
            st.session_state.batch_results = []
            st.rerun()
        
        if col2.button("Clear History"):
            if st.session_state.llm:
                st.session_state.llm.clear_history()
            st.session_state.messages = []
            st.session_state.batch_results = []
            st.rerun()


def process_uploaded_files(uploaded_files):
    """Process uploaded files and add to vector store."""
    with st.spinner("Processing documents..."):
        for uploaded_file in uploaded_files:
            try:
                # Save to temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name
                
                # Add to vector store
                result = st.session_state.vector_store.add_document(tmp_path)
                
                # Clean up temp file
                os.unlink(tmp_path)
                
                if "error" in result:
                    st.sidebar.error(f"Error: {uploaded_file.name} - {result['error']}")
                else:
                    st.sidebar.success(f"Added: {uploaded_file.name} ({result['chunk_count']} chunks)")
                    
            except Exception as e:
                st.sidebar.error(f"Error processing {uploaded_file.name}: {str(e)}")
    
    st.rerun()


def run_batch_test():
    """Run all 20 test questions and store results."""
    st.session_state.batch_running = True
    st.session_state.batch_progress = 0
    st.session_state.batch_results = []
    
    # Clear chat history for fresh batch test
    st.session_state.llm.clear_chat_history()
    
    # Create a placeholder for progress display
    progress_container = st.container()
    
    with progress_container:
        progress_bar = st.progress(0)
        status_text = st.empty()
        current_question = st.empty()
        current_answer = st.empty()
        
        total_questions = len(TEST_QUESTIONS)
        
        for i, question in enumerate(TEST_QUESTIONS):
            # Update progress
            st.session_state.batch_progress = i + 1
            progress_bar.progress((i + 1) / total_questions)
            status_text.text(f"Processing question {i + 1}/{total_questions}...")
            current_question.info(f"**Q{i + 1}:** {question[:100]}...")
            
            try:
                # Get context from vector store
                context, sources = st.session_state.vector_store.get_context_for_llm(question)
                
                if not context:
                    answer = "No relevant context found in documents."
                    token_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
                else:
                    # Generate response (non-streaming for batch)
                    result = st.session_state.llm.generate_response(
                        question=question,
                        context=context,
                        sources=sources,
                        use_history=True  # No history for batch test - each question independent
                    )
                    answer = result.get("answer", "Error generating response")
                    token_usage = result.get("token_usage", {})
                
                # Store result
                st.session_state.batch_results.append({
                    "question_num": i + 1,
                    "question": question,
                    "answer": answer,
                    "token_usage": token_usage,
                    "chunks_retrieved": len(sources)
                })
                
                # Show current answer preview
                current_answer.success(f"✓ Q{i + 1} Completed - {token_usage.get('total_tokens', 0)} tokens")
                
            except Exception as e:
                st.session_state.batch_results.append({
                    "question_num": i + 1,
                    "question": question,
                    "answer": f"Error: {str(e)}",
                    "token_usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                    "chunks_retrieved": 0
                })
                current_answer.error(f"✗ Q{i + 1} Error: {str(e)}")
            
            # Small delay to avoid rate limiting
            time.sleep(0.5)
        
        # Complete
        progress_bar.progress(1.0)
        status_text.text(f"✅ Batch test completed! All {total_questions} questions processed.")
        current_question.empty()
        current_answer.empty()
    
    st.session_state.batch_running = False
    st.rerun()


def render_chat():
    """Render the main chat interface."""
    st.title("📚 RAG Document Assistant")
    
    # Check for API key
    if not st.session_state.llm:
        st.error("⚠️ ANTHROPIC_API_KEY not found in environment variables.")
        st.info("Please create a `.env` file with your API key:\n```\nANTHROPIC_API_KEY=your-key-here\n```")
        return
    
    # Check for documents
    stats = st.session_state.vector_store.get_stats()
    if stats["total_documents"] == 0:
        st.info("👈 Upload documents in the sidebar to get started.")
        return
    
    # Show batch test results if available
    if st.session_state.batch_results:
        render_batch_results()
        st.divider()
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show debug info and token usage for assistant messages
            if message["role"] == "assistant":
                if "chunks_retrieved" in message:
                    st.caption(f"🔍 Retrieved {message['chunks_retrieved']} chunks for context")
                
                if "token_usage" in message:
                    usage = message["token_usage"]
                    st.caption(
                        f"🔢 Tokens: {usage.get('input_tokens', 0)} in / "
                        f"{usage.get('output_tokens', 0)} out / "
                        f"{usage.get('total_tokens', 0)} total"
                    )
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            # Get context from vector store
            context, sources = st.session_state.vector_store.get_context_for_llm(prompt)
            
            # Debug: Show number of chunks retrieved
            chunks_retrieved = len(sources)
            st.caption(f"🔍 Retrieved {chunks_retrieved} chunks for context")
            
            if not context:
                response = "I couldn't find relevant information in the uploaded documents."
                st.markdown(response)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                    "chunks_retrieved": 0
                })
            else:
                # Stream response
                response_placeholder = st.empty()
                full_response = ""
                
                for chunk in st.session_state.llm.generate_response_stream(prompt, context, sources):
                    full_response += chunk
                    response_placeholder.markdown(full_response + "▌")
                
                response_placeholder.markdown(full_response)
                
                # Get token usage
                result = st.session_state.llm.get_last_result()
                token_usage = result.get("token_usage", {}) if result else {}
                
                # Show token usage
                st.caption(
                    f"🔢 Tokens: {token_usage.get('input_tokens', 0)} in / "
                    f"{token_usage.get('output_tokens', 0)} out / "
                    f"{token_usage.get('total_tokens', 0)} total"
                )
                
                # Save to session with debug info
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_response,
                    "token_usage": token_usage,
                    "chunks_retrieved": chunks_retrieved
                })


def render_batch_results():
    """Render the batch test results."""
    st.header(f"🧪 Batch Test Results ({len(st.session_state.batch_results)} Questions)")
    
    results = st.session_state.batch_results
    
    # Summary stats
    total_input = sum(r["token_usage"].get("input_tokens", 0) for r in results)
    total_output = sum(r["token_usage"].get("output_tokens", 0) for r in results)
    total_tokens = sum(r["token_usage"].get("total_tokens", 0) for r in results)
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Questions", len(results))
    col2.metric("Input Tokens", f"{total_input:,}")
    col3.metric("Output Tokens", f"{total_output:,}")
    col4.metric("Total Tokens", f"{total_tokens:,}")
    
    # Results in expandable sections
    for result in results:
        q_num = result["question_num"]
        question = result["question"]
        answer = result["answer"]
        token_usage = result["token_usage"]
        chunks = result["chunks_retrieved"]
        
        with st.expander(f"Q{q_num}: {question[:80]}..."):
            st.markdown(f"**Question:**\n{question}")
            st.divider()
            st.markdown(f"**Answer:**\n{answer}")
            st.divider()
            st.caption(
                f"🔍 Chunks: {chunks} | "
                f"🔢 Tokens: {token_usage.get('input_tokens', 0):,} in / "
                f"{token_usage.get('output_tokens', 0):,} out / "
                f"{token_usage.get('total_tokens', 0):,} total"
            )
    
    # Clear batch results button
    if st.button("🗑️ Clear Batch Results"):
        st.session_state.batch_results = []
        st.rerun()


def main():
    """Main application entry point."""
    init_session_state()
    render_sidebar()
    render_chat()


if __name__ == "__main__":
    main()