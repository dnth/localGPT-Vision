import os
import json
import uuid
import hashlib
from typing import Dict, Any, List, Tuple
from datetime import datetime

import streamlit as st
from markupsafe import Markup
import markdown

# Reuse existing modules
from models.indexer import index_documents
from models.retriever import retrieve_documents
from models.responder import generate_response
from byaldi import RAGMultiModalModel

# Environment setup
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Streamlit-specific folders (separate namespace from Flask)
UPLOAD_FOLDER = "uploaded_documents_streamlit"
STATIC_FOLDER = "static"  # Keep static root same to reuse any existing assets
SESSION_FOLDER = "sessions_streamlit"
INDEX_FOLDER = os.path.join(os.getcwd(), ".byaldi_streamlit")  # Separate Byaldi index root

# Ensure required directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)
os.makedirs(SESSION_FOLDER, exist_ok=True)
os.makedirs(os.path.join(STATIC_FOLDER, "images_streamlit"), exist_ok=True)
os.makedirs(INDEX_FOLDER, exist_ok=True)

# Global cache for Byaldi models
RAG_MODELS: Dict[str, Any] = {}

# ----------- Cached loaders (Streamlit) ----------- #
@st.cache_resource(show_spinner=False)
def _cached_rag_from_index(path: str, version: float) -> Any:
    # version is included to allow invalidation when index updates
    return RAGMultiModalModel.from_index(path)

def _index_dir_version(index_dir: str) -> float:
    try:
        # Use directory mtime as a coarse version for cache invalidation
        return os.path.getmtime(index_dir)
    except Exception:
        return 0.0

# ------------- Utility functions ------------- #

def get_file_icon(filename: str) -> str:
    """Return an appropriate emoji icon based on file extension."""
    ext = os.path.splitext(filename.lower())[1]
    icon_map = {
        '.pdf': '📄',
        '.doc': '📝', '.docx': '📝',
        '.txt': '📄',
        '.md': '📝',
        '.png': '🖼️', '.jpg': '🖼️', '.jpeg': '🖼️', '.gif': '🖼️', '.bmp': '🖼️', '.svg': '🖼️',
        '.mp4': '🎥', '.avi': '🎥', '.mov': '🎥', '.mkv': '🎥',
        '.mp3': '🎵', '.wav': '🎵', '.flac': '🎵',
        '.zip': '📦', '.rar': '📦', '.7z': '📦',
        '.xlsx': '📊', '.xls': '📊', '.csv': '📊',
        '.ppt': '📊', '.pptx': '📊',
        '.py': '🐍', '.js': '📜', '.html': '🌐', '.css': '🎨',
        '.json': '📋', '.xml': '📋', '.yaml': '📋', '.yml': '📋',
    }
    return icon_map.get(ext, '📄')

def get_file_size(filepath: str) -> str:
    """Get human-readable file size."""
    try:
        size_bytes = os.path.getsize(filepath)
        if size_bytes == 0:
            return "0 B"
        size_names = ["B", "KB", "MB", "GB"]
        i = 0
        while size_bytes >= 1024 and i < len(size_names) - 1:
            size_bytes /= 1024.0
            i += 1
        return f"{size_bytes:.1f} {size_names[i]}"
    except OSError:
        return "Unknown"

def get_file_modified_time(filepath: str) -> str:
    """Get human-readable file modification time."""
    try:
        mtime = os.path.getmtime(filepath)
        return datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
    except OSError:
        return "Unknown"

def extract_original_filename(hashed_filename: str) -> str:
    """Extract original filename from hashed filename format: base_hash.ext"""
    # Remove the hash suffix (last 32 characters before extension)
    base, ext = os.path.splitext(hashed_filename)
    if len(base) > 32 and base[-33] == '_':
        return base[:-33] + ext
    return hashed_filename

def render_indexed_files(indexed_files: List[str], session_id: str):
    """Render indexed files in a simplified list view."""
    if not indexed_files:
        st.info("No files have been indexed yet. Upload some documents to get started!")
        return
    
    # Put the indexed files in a collapsible expander
    with st.expander(f"📚 Indexed Files ({len(indexed_files)})", expanded=True):
        # Simple list view of files
        for filename in indexed_files:
            original_name = extract_original_filename(filename)
            file_icon = get_file_icon(filename)
            
            # Get file size for display
            file_path = os.path.join(session_uploads_folder(session_id), filename)
            file_size = get_file_size(file_path) if os.path.exists(file_path) else "Unknown"
            
            # Display as a simple row with icon, name, and size
            st.markdown(f"{file_icon} **{original_name}** `({file_size})`")
    

def session_json_path(session_id: str) -> str:
    return os.path.join(SESSION_FOLDER, f"{session_id}.json")

def session_images_folder(session_id: str) -> str:
    return os.path.join(STATIC_FOLDER, "images_streamlit", session_id)

def session_uploads_folder(session_id: str) -> str:
    return os.path.join(UPLOAD_FOLDER, session_id)

def index_path_for_session(session_id: str) -> str:
    # Per-session working directory where Byaldi will create .byaldi/<session_id> inside
    return os.path.join(INDEX_FOLDER, session_id)

def load_rag_model_for_session(session_id: str) -> None:
    # Byaldi index lives under: .byaldi_streamlit/<session_id>/.byaldi/<session_id>
    workdir = index_path_for_session(session_id)
    index_dir = os.path.join(workdir, ".byaldi", session_id)
    if os.path.exists(index_dir) and os.path.isdir(index_dir):
        try:
            with st.spinner(f"Loading RAG model for session {session_id[:8]}..."):
                version = _index_dir_version(index_dir)
                RAG = _cached_rag_from_index(index_dir, version)
                RAG_MODELS[session_id] = RAG
        except Exception as e:
            st.error(f"Error loading RAG model for session {session_id}: {e}")
    else:
        # Provide a clearer message to help users understand what's missing
        st.warning(f"No Byaldi index found for session {session_id}. Expected at {index_dir}. Please upload files and build the index.")

def load_existing_indexes() -> None:
    if os.path.exists(INDEX_FOLDER):
        for sid in os.listdir(INDEX_FOLDER):
            idx_dir = os.path.join(INDEX_FOLDER, sid)
            if os.path.isdir(idx_dir):
                load_rag_model_for_session(sid)

def list_chat_sessions() -> List[Dict[str, str]]:
    sessions = []
    if os.path.exists(SESSION_FOLDER):
        for fname in os.listdir(SESSION_FOLDER):
            if fname.endswith(".json"):
                sid = fname[:-5]
                try:
                    with open(os.path.join(SESSION_FOLDER, fname), "r") as f:
                        data = json.load(f)
                    sessions.append({"id": sid, "name": data.get("session_name", "Untitled Session")})
                except Exception:
                    continue
    # Sort by name or by file creation time; here keep as-is
    return sessions

def read_session_data(session_id: str) -> Dict[str, Any]:
    path = session_json_path(session_id)
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {
        "session_name": "Untitled Session",
        "chat_history": [],
        "indexed_files": [],
    }

def write_session_data(session_id: str, data: Dict[str, Any]) -> None:
    path = session_json_path(session_id)
    with open(path, "w") as f:
        json.dump(data, f)

def ensure_streamlit_state_defaults():
    # Settings persisted in st.session_state
    st.session_state.setdefault("indexer_model", "vidore/colpali")
    st.session_state.setdefault("generation_model", "gemini2")
    st.session_state.setdefault("resized_height", 280)
    st.session_state.setdefault("resized_width", 280)

    # Current logical chat session identifier
    if "current_session_id" not in st.session_state:
        st.session_state.current_session_id = str(uuid.uuid4())
        # Create initial JSON file
        sessions = [f for f in os.listdir(SESSION_FOLDER) if f.endswith(".json")] if os.path.exists(SESSION_FOLDER) else []
        session_number = len(sessions) + 1
        session_data = {
            "session_name": f"Session {session_number}",
            "chat_history": [],
            "indexed_files": [],
        }
        write_session_data(st.session_state.current_session_id, session_data)

def create_new_session():
    new_id = str(uuid.uuid4())
    sessions = [f for f in os.listdir(SESSION_FOLDER) if f.endswith(".json")] if os.path.exists(SESSION_FOLDER) else []
    session_number = len(sessions) + 1
    data = {
        "session_name": f"Session {session_number}",
        "chat_history": [],
        "indexed_files": [],
    }
    write_session_data(new_id, data)
    st.session_state.current_session_id = new_id

def rename_current_session(new_name: str):
    sid = st.session_state.current_session_id
    data = read_session_data(sid)
    data["session_name"] = new_name or "Untitled Session"
    write_session_data(sid, data)

def delete_session(session_id: str):
    # Remove JSON
    try:
        path = session_json_path(session_id)
        if os.path.exists(path):
            os.remove(path)
    except Exception as e:
        st.warning(f"Failed to remove session JSON: {e}")

    # Remove uploaded docs
    try:
        up = session_uploads_folder(session_id)
        if os.path.exists(up):
            import shutil
            shutil.rmtree(up)
    except Exception as e:
        st.warning(f"Failed to remove uploads: {e}")

    # Remove retrieved images
    try:
        img_dir = session_images_folder(session_id)
        if os.path.exists(img_dir):
            import shutil
            shutil.rmtree(img_dir)
    except Exception as e:
        st.warning(f"Failed to remove images: {e}")

    # Remove cached RAG
    RAG_MODELS.pop(session_id, None)

    # Note: we do NOT remove the Byaldi index on disk to allow reloading later if needed.
    # If desired, uncomment to remove:
    # try:
    #     idx_dir = index_path_for_session(session_id)
    #     if os.path.exists(idx_dir):
    #         import shutil
    #         shutil.rmtree(idx_dir)
    # except Exception as e:
    #     st.warning(f"Failed to remove index: {e}")

    # If deleting current, switch to a fresh session
    if st.session_state.current_session_id == session_id:
        create_new_session()

def index_uploaded_files_for_current_session(saved_files: List[str]) -> Tuple[bool, str]:
    sid = st.session_state.current_session_id
    session_folder = session_uploads_folder(sid)
    os.makedirs(session_folder, exist_ok=True)

    # Build/refresh Byaldi index
    try:
        index_name = sid
        idx_path = index_path_for_session(sid)
        indexer_model = st.session_state.indexer_model
        # Build index so that Byaldi writes to: idx_path/.byaldi/<sid>
        RAG = index_documents(session_folder, index_name=index_name, index_path=idx_path, indexer_model=indexer_model)
        if RAG is None:
            return False, "Indexing failed: RAG model is None"

        # Refresh cached handle by retrieving via cached loader with new version
        index_dir = os.path.join(idx_path, ".byaldi", sid)
        try:
            version = _index_dir_version(index_dir)
            cached_rag = _cached_rag_from_index(index_dir, version)
            RAG_MODELS[sid] = cached_rag
        except Exception:
            # Fallback to RAG returned by indexer if cache retrieval fails
            RAG_MODELS[sid] = RAG

        # Update persisted session state
        data = read_session_data(sid)
        data["indexed_files"].extend(saved_files)
        write_session_data(sid, data)
        return True, "Files indexed successfully."
    except Exception as e:
        return False, f"Error indexing files: {e}"

def retrieve_image_paths_for_current_session(query: str, k: int = 3) -> List[str]:
    sid = st.session_state.current_session_id
    rag = RAG_MODELS.get(sid)
    if rag is None:
        # Attempt to load from disk
        load_rag_model_for_session(sid)
        rag = RAG_MODELS.get(sid)
        if rag is None:
            st.error("RAG model not found for this session. Please upload and index documents first.")
            return []
    return retrieve_documents(rag, query, sid, k=k)  # returns relative paths from static folder

def render_images_horizontally(images: List[str], max_width: int = 200):
    """
    Render a list of images horizontally in a single row using st.columns().
    
    Args:
        images: List of relative image paths from static folder
        max_width: Maximum width for each image in pixels
    """
    if not images:
        return
    
    # Create columns for horizontal layout - one column per image
    num_images = len(images)
    if num_images == 1:
        # Single image - use a narrower column to prevent it from being too large
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            img_path = os.path.join(STATIC_FOLDER, images[0])
            if os.path.exists(img_path):
                st.image(img_path, width=max_width)
    else:
        # Multiple images - create equal columns
        cols = st.columns(num_images)
        for i, rel_path in enumerate(images):
            img_path = os.path.join(STATIC_FOLDER, rel_path)
            if os.path.exists(img_path):
                with cols[i]:
                    st.image(img_path, width=max_width)

def render_chat_history(chat_history: List[Dict[str, Any]]):
    for msg in chat_history:
        role = msg.get("role")
        content = msg.get("content", "")
        images = msg.get("images", [])
        if role == "user":
            st.chat_message("user").write(content)
        else:
            # Assistant content is HTML (from Markdown)
            with st.chat_message("assistant"):
                # Render HTML content
                st.markdown(content, unsafe_allow_html=True)
                # Render images horizontally if any
                if images:
                    render_images_horizontally(images, max_width=280)

def sidebar_ui():
    st.sidebar.title("Sessions")
    sessions = list_chat_sessions()
    # Select session
    if sessions:
        names = [f"{s['name']} ({s['id'][:8]})" for s in sessions]
        ids = [s["id"] for s in sessions]
        try:
            curr_index = ids.index(st.session_state.current_session_id)
        except ValueError:
            curr_index = 0
        selected = st.sidebar.selectbox("Select session", options=list(range(len(ids))), format_func=lambda i: names[i], index=curr_index)
        st.session_state.current_session_id = ids[selected]
    else:
        st.sidebar.info("No sessions found.")

    # Create, rename, delete
    col1, col2, col3 = st.sidebar.columns(3)
    if col1.button("New"):
        create_new_session()
        st.experimental_rerun()
    new_name = st.sidebar.text_input("Rename session", value=read_session_data(st.session_state.current_session_id).get("session_name", "Untitled Session"))
    if col2.button("Rename"):
        rename_current_session(new_name)
        st.experimental_rerun()
    if col3.button("Delete"):
        delete_session(st.session_state.current_session_id)
        st.experimental_rerun()

    st.sidebar.markdown("---")
    st.sidebar.subheader("Settings")
    st.session_state.indexer_model = st.sidebar.text_input("Indexer model", value=st.session_state.indexer_model, help="Byaldi indexer, e.g., vidore/colpali")
    # Map display names to internal keys
    display_to_key = {
        "Qwen": "qwen",
        "Gemini (legacy)": "gemini",
        "Gemini 2": "gemini2",
        "Llama Vision": "llama-vision",
        "Pixtral": "pixtral",
        "Molmo": "molmo",
        "Groq Llama Vision": "groq-llama-vision",
        "Ollama Llama Vision": "ollama-llama-vision",
    }
    key_to_display = {v: k for k, v in display_to_key.items()}
    display_options = list(display_to_key.keys())
    current_display = key_to_display.get(st.session_state.generation_model, "Qwen")
    selected_display = st.sidebar.selectbox("Generation model", options=display_options, index=display_options.index(current_display))
    st.session_state.generation_model = display_to_key[selected_display]
    st.session_state.resized_height = st.sidebar.number_input("Resized height", min_value=64, max_value=2048, value=int(st.session_state.resized_height), step=16)
    st.session_state.resized_width = st.sidebar.number_input("Resized width", min_value=64, max_value=2048, value=int(st.session_state.resized_width), step=16)

    st.sidebar.markdown("---")
    st.sidebar.caption("Indexes stored under .byaldi_streamlit. Uploaded documents under uploaded_documents_streamlit. Retrieved images under static/images_streamlit.")

def upload_ui():
    st.subheader("Upload and Index Documents")
    files = st.file_uploader("Select files", type=None, accept_multiple_files=True)
    if st.button("Index Files"):
        if not files:
            st.warning("No files selected.")
        else:
            with st.spinner("Uploading and indexing files..."):
                sid = st.session_state.current_session_id
                up_dir = session_uploads_folder(sid)
                os.makedirs(up_dir, exist_ok=True)
                saved_names = []
                for f in files:
                    # Use content hash to avoid collisions, but preserve original extension and name
                    content = f.read()
                    f.seek(0)
                    name = f.name
                    safe_name = name.replace("/", "_").replace("\\", "_")
                    h = hashlib.md5(content).hexdigest()
                    base, ext = os.path.splitext(safe_name)
                    out_name = f"{base}_{h}{ext}"
                    out_path = os.path.join(up_dir, out_name)
                    with open(out_path, "wb") as out:
                        out.write(content)
                    saved_names.append(out_name)
                ok, msg = index_uploaded_files_for_current_session(saved_names)
                if ok:
                    st.success(msg)
                else:
                    st.error(msg)

    # Show indexed files with enhanced display
    data = read_session_data(st.session_state.current_session_id)
    render_indexed_files(data.get("indexed_files", []), st.session_state.current_session_id)

def chat_ui():
    st.subheader("Chat")
    sid = st.session_state.current_session_id
    data = read_session_data(sid)
    render_chat_history(data.get("chat_history", []))

    # Streamlit chat input for new message
    query = st.chat_input("Ask about your documents...")
    if query:
        # Immediately display the user message
        with st.chat_message("user"):
            st.write(query)
        
        # Add user message to chat history and persist immediately
        data["chat_history"].append({"role": "user", "content": query})
        
        # If first interaction, set session_name
        if len(data["chat_history"]) == 1:
            data["session_name"] = query[:50]
        
        write_session_data(sid, data)
        
        # Show loading indicator for assistant response
        with st.chat_message("assistant"):
            with st.spinner("Retrieving documents and generating response..."):
                # Create placeholder for the response
                response_placeholder = st.empty()
                images_placeholder = st.empty()
                
                try:
                    # Retrieve relevant images
                    retrieved_images = retrieve_image_paths_for_current_session(query, k=3)
                    # Convert to absolute filesystem paths for the responder
                    full_image_paths = [os.path.join(STATIC_FOLDER, rel) for rel in retrieved_images]

                    response_text, used_images_abs = generate_response(
                        full_image_paths,
                        query,
                        sid,
                        int(st.session_state.resized_height),
                        int(st.session_state.resized_width),
                        st.session_state.generation_model
                    )

                    # Parse markdown to HTML for persistence to match Flask behavior
                    parsed_response = Markup(markdown.markdown(response_text))

                    # Use the original retrieved images instead of the processed ones for display
                    # This ensures fullscreen shows the original full-resolution images
                    relative_images = retrieved_images

                    # Update chat history with assistant response and persist
                    data["chat_history"].append({"role": "assistant", "content": str(parsed_response), "images": relative_images})
                    write_session_data(sid, data)

                    # Display the final response in the placeholder
                    with response_placeholder.container():
                        st.markdown(str(parsed_response), unsafe_allow_html=True)
                    
                    # Display retrieved images in the images placeholder
                    if retrieved_images:
                        with images_placeholder.container():
                            render_images_horizontally(retrieved_images, max_width=280)
                            
                except Exception as e:
                    # Display error in the response placeholder
                    with response_placeholder.container():
                        st.error(f"An error occurred while generating the response: {e}")

def main():
    st.set_page_config(page_title="LocalGPT-Vision (Streamlit)", layout="wide")
    ensure_streamlit_state_defaults()

    # Remove eager preload; lazy-load on demand for better memory and startup time.
    if "initialized" not in st.session_state:
        st.session_state.initialized = True

    # Sidebar
    sidebar_ui()

    # Main sections
    st.title("LocalGPT-Vision (Streamlit)")
    st.caption("Streamlit implementation with separate storage: sessions_streamlit, uploaded_documents_streamlit, static/images_streamlit, .byaldi_streamlit")

    upload_ui()
    st.markdown("---")
    chat_ui()

if __name__ == "__main__":
    main()