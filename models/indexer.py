# models/indexer.py

import os
from byaldi import RAGMultiModalModel
from models.converters import convert_docs_to_pdfs
from logger import get_logger

logger = get_logger(__name__)

import shutil
import tempfile

def _create_filtered_folder(source_folder):
    """
    Create a temporary folder containing only files with supported extensions.
    This prevents Byaldi from encountering unsupported file types.
    
    Args:
        source_folder (str): The path to the source folder.
        
    Returns:
        str: Path to the temporary folder with filtered files.
    """
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp(prefix="byaldi_filtered_")
    logger.info(f"[Indexer] Created temporary folder for filtered files: {temp_dir}")
    
    # Define supported extensions based on Byaldi's capabilities
    # Byaldi typically supports: PDF, images (jpg, jpeg, png, gif, bmp, tiff), and documents that can be converted to PDF
    supported_extensions = {'.pdf', '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.doc', '.docx'}
    
    try:
        for item in os.listdir(source_folder):
            item_path = os.path.join(source_folder, item)
            
            # Skip directories
            if os.path.isdir(item_path):
                logger.debug(f"[Indexer] Skipping directory: {item}")
                continue
            
            # Check if file has a supported extension
            _, ext = os.path.splitext(item)
            if ext.lower() in supported_extensions:
                # Copy file to temp directory
                dest_path = os.path.join(temp_dir, item)
                shutil.copy2(item_path, dest_path)
                logger.debug(f"[Indexer] Copied supported file: {item}")
            else:
                logger.warning(f"[Indexer] Skipping unsupported file: {item} (extension: '{ext}')")
                
    except Exception as e:
        logger.error(f"[Indexer] Error filtering files: {e}")
        # Clean up temp directory on error
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise
        
    return temp_dir

def _session_workdir(index_root: str, session_id: str) -> str:
    """
    Create and return a per-session working directory that will contain
    Byaldi's expected `.byaldi/<index_name>` layout.

    Example returned path: <index_root>/<session_id>
    Inside it, Byaldi will create: <index_root>/<session_id>/.byaldi/<session_id>/
    """
    workdir = os.path.join(index_root, session_id)
    os.makedirs(workdir, exist_ok=True)
    return workdir

def _byaldi_index_dir(index_root: str, session_id: str) -> str:
    """
    Return the absolute path to the actual Byaldi index directory that
    RAGMultiModalModel.from_index expects, i.e.:
      <index_root>/<session_id>/.byaldi/<session_id>
    """
    return os.path.join(index_root, session_id, ".byaldi", session_id)

def index_documents(folder_path, index_name='document_index', index_path=None, indexer_model='vidore/colpali'):
    """
    Indexes documents in the specified folder using Byaldi.

    Args:
        folder_path (str): The path to the folder containing documents to index.
        index_name (str): The name of the index to create or update (use session_id).
        index_path (str): The per-session index root path (e.g., .byaldi_streamlit/<session_id>).
        indexer_model (str): The name of the indexer model to use.

    Returns:
        RAGMultiModalModel: The RAG model with the indexed documents.
    """
    temp_folder = None
    try:
        logger.info(f"[Indexer] Starting document indexing | folder={folder_path} | index_name={index_name} | index_path={index_path} | model={indexer_model}")

        # Create a filtered folder with only supported files
        temp_folder = _create_filtered_folder(folder_path)
        
        # Convert non-PDF documents to PDFs in the temp folder
        convert_docs_to_pdfs(temp_folder)
        logger.info("[Indexer] Conversion of non-PDF documents to PDFs completed.")

        # Initialize RAG model
        RAG = RAGMultiModalModel.from_pretrained(indexer_model)
        if RAG is None:
            raise ValueError(f"Failed to initialize RAGMultiModalModel with model {indexer_model}")
        logger.info(f"[Indexer] RAG model initialized with {indexer_model}.")

        if not index_path:
            # Fallback to current working directory; Byaldi will write to CWD/.byaldi/<index_name>
            index_root = os.getcwd()
            logger.warning("[Indexer] index_path not provided; defaulting to current working directory for Byaldi output.")
        else:
            index_root = os.path.abspath(os.path.join(index_path, os.pardir)) if os.path.basename(index_path) else os.path.abspath(index_path)

        # Ensure per-session working directory exists: <index_root>/<index_name>
        # For streamlit we pass index_path = .byaldi_streamlit/<session_id>, so:
        #   workdir = .byaldi_streamlit/<session_id>
        workdir = _session_workdir(os.path.dirname(index_path) if index_path else index_root, index_name)
        logger.info(f"[Indexer] Using per-session workdir: {workdir}")

        # Run Byaldi so it writes into: workdir/.byaldi/<index_name>
        original_cwd = os.getcwd()
        try:
            os.chdir(workdir)
            logger.info(f"[Indexer] Changed CWD to workdir: {workdir}")

            RAG.index(
                input_path=temp_folder,  # Use the filtered temp folder instead
                index_name=index_name,
                store_collection_with_index=True,
                overwrite=True
            )
        finally:
            os.chdir(original_cwd)
            logger.info(f"[Indexer] Restored CWD to: {original_cwd}")

        # Log final index dir
        final_index_dir = _byaldi_index_dir(os.path.dirname(index_path) if index_path else index_root, index_name)
        logger.info(f"[Indexer] Indexing completed. Byaldi index located at: {final_index_dir}")

        return RAG
    except Exception as e:
        logger.error(f"[Indexer] Error during indexing: {str(e)}", exc_info=True)
        raise
    finally:
        # Clean up temporary folder
        if temp_folder and os.path.exists(temp_folder):
            try:
                shutil.rmtree(temp_folder)
                logger.info(f"[Indexer] Cleaned up temporary folder: {temp_folder}")
            except Exception as e:
                logger.warning(f"[Indexer] Failed to clean up temporary folder: {e}")