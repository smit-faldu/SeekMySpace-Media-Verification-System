import streamlit as st
import os
import tempfile
import traceback

from verifier import MediaVerifier
from logger import get_logger

logger = get_logger(__name__)

st.set_page_config(page_title="SeekMySpace Media Verifier", page_icon="🏢", layout="wide")

@st.cache_resource
def get_verifier() -> MediaVerifier:
    logger.info("Initializing cached MediaVerifier.")
    return MediaVerifier()

try:
    verifier = get_verifier()
except Exception as e:
    st.error(f"Critical System Failure: Unable to initialize the Media Verifier. Error: {e}")
    st.stop()

st.title("🏢 SeekMySpace Auto-Verification System")
st.markdown("Upload images or videos of commercial spaces to automatically verify their quality and relevance.")

uploaded_file = st.file_uploader("Upload Media (Images/Videos)", type=["jpg", "jpeg", "png", "mp4", "mov", "avi"])

if uploaded_file is not None:
    logger.info(f"File uploaded via UI: {uploaded_file.name}")
    
    # Save uploaded file to a temporary location safely
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix="." + uploaded_file.name.split('.')[-1])
    try:
        tfile.write(uploaded_file.read())
        file_path = tfile.name
    finally:
        tfile.close()

    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Uploaded Media")
        try:
            if file_path.lower().endswith(('.mp4', '.mov', '.avi')):
                st.video(file_path)
            else:
                # Removed deprecated use_container_width=True to avoid warning.
                st.image(file_path)
        except Exception as e:
            logger.error(f"Failed to render uploaded media {file_path}: {e}", exc_info=True)
            st.warning("Preview unavailable for this file format or file is corrupted.")
            
    with col2:
        st.subheader("Verification Analysis")
        try:
            with st.spinner("Analyzing media quality, metadata, and relevance..."):
                result = verifier.verify_media(file_path)
                
            score = result.get('score', 0.0)
            decision = result.get('decision', 'Error')
            reasoning = result.get('reasoning', '')
            details = result.get('details', {})
            
            # Display Decision
            if decision == "Auto Approved":
                st.success(f"Decision: **{decision}**")
            elif decision == "Error":
                st.error(f"Error during verification: {reasoning}")
                st.stop()
            else:
                st.warning(f"Decision: **{decision}**")
                
            # Display Overall Score with Progress Bar
            st.metric(label="Overall Score", value=f"{score:.1f}/100")
            st.progress(min(score / 100.0, 1.0))
            
            # Details Breakdown
            st.markdown("### Breakdown")
            
            st.info(f"**Metadata Score:** {details.get('metadata_score', 0)}/20  \n"
                    f"GPS Data: {'Yes' if details.get('gps') else 'No'} | "
                    f"Timestamp Data: {'Yes' if details.get('timestamp') else 'No'}")
                    
            st.info(f"**Quality Score:** {details.get('quality_score', 0.0):.1f}/30")
            
            st.info(f"**Relevance Score:** {details.get('relevance_score', 0)}/60")
            
            st.markdown("### Reasoning")
            st.write(reasoning)

        except Exception as e:
            logger.error(f"UI Error during verification: {e}", exc_info=True)
            st.error("An unexpected error occurred during analysis. Please try a different file.")
            with st.expander("Error Details"):
                st.code(traceback.format_exc())

    # Clean up temp file
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.debug(f"Temporary file removed: {file_path}")
    except Exception as e:
        logger.warning(f"Failed to remove temporary file {file_path}: {e}")
