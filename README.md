# SeekMySpace Media Verification System

This is an automated media verification system for commercial spaces, built as part of the SeekMySpace assignment. It evaluates uploaded images and videos based on metadata presence, visual quality, and scene relevance using local heuristics and a Zero-Shot Image Classifier (CLIP), strictly without using external LLM APIs.

## Setup and Running

1. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Or .venv\Scripts\activate on Windows
   pip install -r requirements.txt
   ```

2. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```

## Scoring Logic (0-100)

The system assigns a total score up to 100 based on three categories:

1. **Metadata (10 points)**: Presence of GPS (+5) and Timestamp (+5).
2. **Quality (30 points)**:
   - **Blur** (up to 15 points): Evaluated using the Variance of the Laplacian (higher variance = sharper).
   - **Exposure** (up to 15 points): Evaluated using grayscale histogram bounds to heavily penalize extreme underexposure or overexposure.
3. **Relevance (60 points)**: Evaluated using a Zero-Shot Image Classifier (`openai/clip-vit-base-patch32`). Instead of just counting objects, it assesses the semantic scene. 
   - **High Relevance** (e.g., "professional corporate office"): Up to 60 points based on confidence.
   - **Moderate Relevance** (e.g., "office cubicles"): Up to 40 points based on confidence.
   - **Negative/Low Relevance** (e.g., "messy bedroom", "selfie"): Up to 10 points or 0 if highly confident.

**Decision Threshold**: If the final score is `>= 70`, the media is `Auto Approved`. Otherwise, it is `Flagged for Manual Review`.



## Follow-up Questions

### 1. What are the biggest weaknesses of your verification system?
The reliance on simple heuristics for brightness/blur can be thrown off by intentional artistic choices or unusual lighting. For relevance, zero-shot CLIP models are incredibly powerful for semantic understanding but are computationally heavier than simple CNNs and can sometimes be overly sensitive to the exact phrasing of the prompt labels. Finally, the metadata check is naive—it only checks for the *presence* of EXIF data, not its validity or geographical accuracy.

### 2. How can a malicious user bypass your checks?
A malicious user can easily upload a high-quality stock photo of a modern office that includes forged EXIF metadata (adding GPS/timestamp tags manually using EXIF modification tools). Since the system only checks for the presence of GPS data and the visual relevance of the office, this would bypass the checks and easily score > 70.

### 3. If GPS metadata is missing, how would your system behave and why?
The system will simply deduct the 5 points allocated for GPS data. The media can still be `Auto Approved` if its quality and relevance scores are excellent (Total = 95). This ensures users who strip EXIF data for privacy or use devices that don't record GPS aren't immediately blocked, but their media is scrutinized more closely on visual metrics.

### 4. How would you verify that the media actually belongs to the listed property?
1. **Distance Check**: Compute the Haversine distance between the extracted GPS coordinates and the listed property's address coordinates. Reject if distance > threshold.
2. **Visual Consistency**: Use a Siamese network (or feature matching) to compare new uploads against already verified existing photos of that property.
3. **Watermarking/Live Capture**: Enforce capturing via an in-app camera that stamps the media with unmodifiable location data rather than allowing gallery uploads.

### 5. What trade-offs did you make between accuracy, complexity, and performance?
I chose a Transformer-based model (CLIP `vit-base-patch32`) over a lightweight object detector (like YOLO). While YOLO is blazing fast, an office is more than just a sum of chairs and laptops; CLIP provides a much better semantic understanding of "commercial space" vs "residential living room". To balance this heavier relevance model, I used extremely fast mathematical heuristics (Laplacian variance and Histograms) for blur and exposure instead of a deep learning-based image quality assessor (IQA) to keep the overall Streamlit app performant.

### 6. If you had to scale this to 1 million uploads per day, what would break first?
The synchronous ML inference pipeline. Running a Vision Transformer synchronously on a web server for every upload (especially processing multiple frames per video) will bottleneck CPU/GPU resources, exhaust memory, and cause severe request timeouts.

### 7. How would you redesign this system if metadata was not available at all?
I would shift all weight (100 points) to Visual Quality and Relevance. I would also integrate an Image Forensics model to detect Photoshop manipulations (Error Level Analysis) or AI generation artifacts (like Stable Diffusion/Midjourney artifacts) to ensure the image is genuine. I might also rely exclusively on "Live Capture" from mobile apps to guarantee authenticity at the time of capture.

### 8. What additional signals would you use if you had access to historical user data?
- **User Trust Score**: Historically reliable users (whose media is rarely flagged or rejected by manual reviewers) get a lower approval threshold.
- **Upload Frequency**: A surge of uploads from a single IP or user account can heavily indicate spam, triggering automatic review.
- **Device Signatures**: Match the EXIF "Make/Model" against the historical devices the user has previously used.

### 9. How would you reduce false positives (rejecting valid media)?
- Fine-tune the relevance threshold. An empty commercial space might lack obvious furniture and trigger a negative label.
- Implement an asynchronous appeal logic: when a flag occurs, let the user optionally provide additional context immediately before it goes to a human manual reviewer.

### 10. What would your next version (v2) of this system look like?
**v2 Architecture**:
- **Async Processing**: A FastAPI backend dumping upload events into an SQS/RabbitMQ queue, picked up by dedicated GPU workers doing the ML inference, reporting back via WebSockets/Webhooks.
- **Fine-Tuned Models**: Replacing zero-shot CLIP with a fine-tuned ViT or ResNet trained specifically on SeekMySpace data to classify "Good Commercial Space" vs "Bad/Irrelevant".
- **Advanced Verification**: GPS validation (Haversine distance to listed property radius) and Reverse Image Search (via perceptual hashing like pHash) to prevent stock photo reuse.
