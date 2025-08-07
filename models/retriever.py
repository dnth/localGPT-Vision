# models/retriever.py

import base64
import os
from PIL import Image
from io import BytesIO
from logger import get_logger
import time
import hashlib
import shutil

logger = get_logger(__name__)

def retrieve_documents(RAG, query, session_id, k=3, app_type='flask'):
    """
    Retrieves relevant documents based on the user query using Byaldi.
    Now attempts to map back to original full-resolution images when possible.

    Args:
        RAG (RAGMultiModalModel): The RAG model with the indexed documents.
        query (str): The user's query.
        session_id (str): The session ID to store images in per-session folder.
        k (int): The number of documents to retrieve.
        app_type (str): 'flask' or 'streamlit' to determine folder structure.

    Returns:
        list: A list of image filenames corresponding to the retrieved documents.
    """
    try:
        logger.info(f"Retrieving documents for query: {query}")
        results = RAG.search(query, k=k)
        images = []
        
        # Use different folder structure based on app type
        if app_type == 'streamlit':
            session_images_folder = os.path.join('static', 'images_streamlit', session_id)
            session_uploads_folder = os.path.join('uploaded_documents_streamlit', session_id)
            relative_path_prefix = 'images_streamlit'
        else:
            session_images_folder = os.path.join('static', 'images', session_id)
            session_uploads_folder = os.path.join('uploaded_documents', session_id)
            relative_path_prefix = 'images'
            
        os.makedirs(session_images_folder, exist_ok=True)
        
        for i, result in enumerate(results):
            if result.base64:
                image_data = base64.b64decode(result.base64)
                image = Image.open(BytesIO(image_data))
                
                # Try to find the original image file by matching content or metadata
                original_image_path = _find_original_image(result, session_uploads_folder, image_data)
                
                if original_image_path and os.path.exists(original_image_path):
                    # Use the original full-resolution image
                    logger.info(f"Found original image: {original_image_path}")
                    
                    # Copy original to the display folder with a consistent name
                    image_hash = hashlib.md5(image_data).hexdigest()
                    original_ext = os.path.splitext(original_image_path)[1] or '.png'
                    image_filename = f"retrieved_{image_hash}_original{original_ext}"
                    display_image_path = os.path.join(session_images_folder, image_filename)
                    
                    if not os.path.exists(display_image_path):
                        shutil.copy2(original_image_path, display_image_path)
                        logger.debug(f"Copied original image to display folder: {display_image_path}")
                    
                    relative_path = os.path.join(relative_path_prefix, session_id, image_filename)
                    images.append(relative_path)
                    logger.info(f"Using original image: {relative_path}")
                else:
                    # Fallback to RAG-processed image
                    image_hash = hashlib.md5(image_data).hexdigest()
                    image_filename = f"retrieved_{image_hash}.png"
                    image_path = os.path.join(session_images_folder, image_filename)
                    
                    if not os.path.exists(image_path):
                        image.save(image_path, format='PNG')
                        logger.debug(f"Retrieved and saved processed image: {image_path}")
                    
                    relative_path = os.path.join(relative_path_prefix, session_id, image_filename)
                    images.append(relative_path)
                    logger.info(f"Using processed image: {relative_path}")
            else:
                logger.warning(f"No base64 data for document {result.doc_id}, page {result.page_num}")
        
        logger.info(f"Total {len(images)} documents retrieved. Image paths: {images}")
        return images
    except Exception as e:
        logger.error(f"Error retrieving documents: {e}")
        return []

def _find_original_image(result, uploads_folder, processed_image_data):
    """
    Attempt to find the original image file that corresponds to a RAG result.
    
    Args:
        result: The RAG search result
        uploads_folder: Path to the folder containing original uploaded files
        processed_image_data: The processed image data from RAG
        
    Returns:
        str or None: Path to the original image file if found
    """
    try:
        if not os.path.exists(uploads_folder):
            return None
            
        # Get image extensions to check
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
        
        # Look for image files in the uploads folder
        for filename in os.listdir(uploads_folder):
            file_path = os.path.join(uploads_folder, filename)
            if not os.path.isfile(file_path):
                continue
                
            _, ext = os.path.splitext(filename.lower())
            if ext not in image_extensions:
                continue
                
            try:
                # Try to match by comparing image content/metadata
                with open(file_path, 'rb') as f:
                    original_data = f.read()
                
                # Simple heuristic: if the original file is significantly larger than processed,
                # and has similar aspect ratio, it's likely the source
                original_image = Image.open(BytesIO(original_data))
                processed_image = Image.open(BytesIO(processed_image_data))
                
                # Check if aspect ratios are similar (within 10% tolerance)
                orig_ratio = original_image.width / original_image.height
                proc_ratio = processed_image.width / processed_image.height
                
                if abs(orig_ratio - proc_ratio) / orig_ratio < 0.1:
                    # Check if original is higher resolution
                    orig_pixels = original_image.width * original_image.height
                    proc_pixels = processed_image.width * processed_image.height
                    
                    if orig_pixels > proc_pixels * 1.2:  # At least 20% more pixels
                        logger.info(f"Found potential original image: {filename} ({original_image.size} vs {processed_image.size})")
                        return file_path
                        
            except Exception as e:
                logger.debug(f"Error checking image {filename}: {e}")
                continue
                
        return None
    except Exception as e:
        logger.error(f"Error finding original image: {e}")
        return None