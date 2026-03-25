# SeekMySpace Media Verification System

This is a basic automated media verification system for commercial spaces, built as part of the SeekMySpace assignment. It evaluates uploaded images and videos based on quality, metadata presence, and relevance using local logic and a lightweight ML model (YOLOv8 nano), strictly without using LLM APIs.

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

The system assigns a score up to 100 based on three categories:
1. **Metadata (20 points)**: Presence of GPS (+10) and Timestamp (+10).
2. **Quality (30 points)**:
   - **Blur** (up to 15 points) using Variance of Laplacian (higher variance = sharper).
   - **Brightness** (up to 15 points) using grayscale mean bounds ([40, 220]).
3. **Relevance (50 points)**: Evaluated using YOLOv8 (`yolov8n.pt`) to detect office-related objects (laptops, chairs, TVs, dining tables, etc.). More unique target classes detected yields higher scores (3+ unique objects = 50, 2 = 35, 1 = 20).

**Decision Threshold**: If the final score is `> 70`, the media is `Auto Approved`. Otherwise, it is `Flagged for Manual Review`.

---

## Follow-up Questions

### 1. What are the biggest weaknesses of your verification system?
The reliance on simple heuristics for brightness/blur can be thrown off by intentional artistic choices or unusual lighting. For relevance, YOLOv8n is lightweight but might miss smaller objects or falsely classify similar objects. Also, the metadata check is naive—it only checks for the *presence* of EXIF data, not its validity or logical consistency (e.g., matching the property's actual coordinates).

### 2. How can a malicious user bypass your checks?
A malicious user can upload a high-quality stock photo of an office that includes forged EXIF metadata (adding GPS/timestamp tags manually using EXIF modification tools). Since the system only checks for the presence of GPS data and the presence of office objects, this would easily bypass the checks and score >70.

### 3. If GPS metadata is missing, how would your system behave and why?
The system will simply deduct the 10 points allocated for GPS data. The media can still be `Auto Approved` if its quality and relevance scores are perfect (Total = 90). This ensures users who strip EXIF data for privacy or use devices that don't record GPS aren't immediately blocked, but their media is scrutinized more closely on other metrics.

### 4. How would you verify that the media actually belongs to the listed property?
1. **Distance Check**: Compute the distance between the extracted GPS coordinates and the listed property's address coordinates. Reject if distance > threshold.
2. **Visual Consistency**: Use a Siamese network (or feature matching) to compare new uploads against already verified existing photos of that property.
3. **Watermarking/Live Capture**: Enforce capturing via an in-app camera that stamps the media with unmodifiable metadata / location data.

### 5. What trade-offs did you make between accuracy, complexity, and performance?
I chose `YOLOv8 nano` for object detection over heavier models (like ResNet-152 or larger YOLO variants) to prioritize blazing-fast inference inside a Streamlit app, sacrificing some edge-case detection accuracy. I also used simple Laplacian variance for blur instead of a deep learning-based image quality assessor (IQA) to keep dependencies light and performance high.

### 6. If you had to scale this to 1 million uploads per day, what would break first?
The synchronous ML inference pipeline. Running YOLOv8 synchronously on a web server for every upload (especially processing multiple frames per video) will bottleneck CPU/GPU resources and cause request timeouts.

### 7. How would you redesign this system if metadata was not available at all?
I would shift all weight (100 points) to Visual Quality and Relevance. I would also integrate an Image Forensics model to detect Photoshop manipulations (Error Level Analysis) or AI generation artifacts to ensure the image is genuine. I might also rely more heavily on "Live Capture" from mobile apps rather than post-capture uploads.

### 8. What additional signals would you use if you had access to historical user data?
- **User Trust Score**: Historically reliable users (whose media is rarely flagged or rejected by manual reviewers) get a lower approval threshold.
- **Upload Frequency**: A surge of uploads from a single IP or user account can heavily indicate spam, triggering automatic review.
- **Device Signatures**: Match the EXIF "Make/Model" against the historical devices the user has used.

### 9. How would you reduce false positives (rejecting valid media)?
- Fine-tune the relevance threshold. A space might just be empty without many objects (e.g., an empty floor plan). I could train a custom CNN to classify "commercial rooms" vs "residential rooms" globally rather than relying solely on individual object detection.
- Implement an asynchronous appeal logic: when a flag occurs, let the user optionally provide additional context immediately before going to a human manual reviewer.

### 10. What would your next version (v2) of this system look like?
**v2 Architecture**:
- **Async Processing**: A FastAPI backend dumping events into an SQS/RabbitMQ queue, picked up by dedicated GPU workers doing the ML inference, reporting back via WebSockets/Webhooks.
- **Better ML models**: A custom fine-tuned ViT (Vision Transformer) trained specifically on SeekMySpace data to classify "Good Commercial Space" vs "Bad/Irrelevant".
- **Advanced Verification**: GPS validation (Haversine distance to listed property radius) and Reverse Image Search (via perceptual hashing like pHash to prevent stock photo reuse).
