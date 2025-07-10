# Multilingual Image Captioning Application (English - Vietnamese)

This project presents a multilingual image captioning application developed with a Gradio interface. It allows users to upload images or select from sample images to generate detailed descriptions in both English and Vietnamese. The application supports comparing descriptive outputs from multiple models and includes a feature for evaluating description quality on a test dataset.

## 1. Models Used

This project integrates and leverages the power of advanced Vision-Language Models (VLMs) for image captioning. The model selection process involved thorough survey and comparison, considering performance, resource requirements, and practical deployment feasibility.

### 1.1. Dataset

[[cite_start]](https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset)The project utilizes a subset of the **Flickr Image Dataset** (originally known as Flickr30k dataset) from Kaggle. Flickr30k is a standard benchmark dataset for sentence-based image description, containing 158,000 image captions. Specifically, Flickr30k Entities, an extension of Flickr30k, augments these captions with coreference chains linking mentions of the same entities across different captions for the same image, and associates them with manually annotated bounding boxes.

In this project, to optimize resources, we used only the **first 50 images** from the Flickr30k dataset, along with their corresponding comments. Each of these 50 images has **more than one comment**. For evaluation purposes, BLEU Score calculation is performed only on the **first 10 images** of this 50-image set to conserve time and computational resources.

### 1.2. Model Survey and Selection Process

Initially, the project surveyed other powerful and popular Vision-Language models such as **BLIP-2**, **OFA**, **GIT**, and **InstructBLIP**. These models are well-known for their superior captioning capabilities, often trained on massive datasets with complex architectures.

However, during practical implementation on the current development environment (personal laptop configuration), loading and running larger versions of these models (e.g., BLIP-2 Flan-T5-XXL, GIT-Large, OFA-Large) presented significant resource challenges, specifically **limitations in RAM and GPU VRAM**. This led to errors like "The paging file is too small" or extremely slow processing times (e.g., OFA taking over 15 minutes for evaluation with Beam Search). Therefore, to ensure feasibility, efficiency, and responsiveness in a local demo application, the project opted for the following three models, which offer a good balance between description quality and resource requirements:

* **`blip-image-captioning-base` (Salesforce BLIP)**:
    * [[cite_start]](https://huggingface.co/Salesforce/blip-image-captioning-base)
    * **Reason for Selection**: This is the base version of BLIP, a flexible VLP framework that excels in both vision-language understanding and generation tasks. It effectively utilizes noisy web data by bootstrapping captions. This model achieves state-of-the-art results on various VL tasks, including image-text retrieval (+2.7% in average recall@1), image captioning (+2.8% in CIDEr), and VQA (+1.6% in VQA score). Compared to larger BLIP-2 versions, BLIP-base requires significantly fewer resources while still producing accurate and fluent descriptions.
    * **Mechanism**: BLIP operates as an encoder-decoder model utilizing Transformers to process both visual and textual information. It learns to align visual features with linguistic structures to generate descriptions.

* **`noamrot/FuseCap` (FuseCap)**:
    * [[cite_start]](https://huggingface.co/noamrot/FuseCap_Image_Captioning)
    * **Reason for Selection**: FuseCap is a novel method designed to enrich existing image captions by fusing detailed information from "vision experts" (e.g., object detectors, attribute recognizers, and Optical Character Recognition (OCR)) with original captions using a Large Language Model (LLM). The model is trained on this enriched caption dataset, allowing it to generate more comprehensive and detailed descriptions than traditional methods. Despite having fewer parameters and utilizing less training data than other state-of-the-art models, FuseCap demonstrates superior performance in generating comprehensive captions. This highlights the potential of a data-centric approach over solely architectural improvements.
    * **Mechanism**: FuseCap functions as a "methodology" for generating richer training data, which is then used to train a captioning model based on the BLIP architecture.The process involves: 1) Extracting visual information from images using vision experts (Faster-RCNN for object detection, attribute recognition, and OCR models like CRAFT and Parseq. 2) Fusing this extracted information with the original captions using a fine-tuned LLM. 3) Training a BLIP-based captioning model with this enriched image-caption dataset.

* **`nlpconnect/vit-gpt2-image-captioning` (ViT-GPT2)**:
    * [[cite_start]](https://huggingface.co/nlpconnect/vit-gpt2-image-captioning)
    * **Reason for Selection**: This represents a classic encoder-decoder architecture for image captioning[cite: 10]. [cite_start]It utilizes a Vision Transformer (ViT) for image encoding and GPT-2 for text decoding. This model provides a solid and relatively lightweight baseline, suitable for demo purposes and performance comparison within a resource-constrained environment.
    * **Mechanism**: Images are fed into the Vision Transformer (encoder) to extract visual features. These features are then passed to GPT-2 (decoder), an autoregressive language model, which generates the text sequence (caption).

### 1.3. Multilingual Support

All selected models (`blip-image-captioning-base`, `noamrot/FuseCap`, `nlpconnect/vit-gpt2-image-captioning`) are primarily trained on English datasets and generate captions in English. To meet the project's multilingual requirement, the application integrates the `deep_translator` library to automatically translate English captions into Vietnamese.

### 1.4. Performance Evaluation and Captioning Capability

Testing on the first 10 images from the `test_images` dataset provided quantitative and qualitative insights into each model's capabilities:

**Captioning Example (Image: `10010052.jpg`)**
<br>
<p align="center">
  <img src="https://github.com/NhatNQuang/image-captioning/blob/main/test_images/10010052.jpg" alt="Sample Image 10010052.jpg" style="width:20%;"/>
</p>
<br>

* **Ground Truths:**
    * A girl is on rollerskates talking on her cellphone standing in a parking lot.
    * A trendy girl talking on her cellphone while gliding slowly down the street.
    * A young adult wearing rollerblades, holding a cellular phone to her ear.
    * There is a young girl on her cellphone while skating.
    * Woman talking on cellphone and wearing rollerskates.

* **Generated Captions:**
    * **`blip-image-captioning-base`:**
        * English: A girl on a skateboard.
        * Vietnamese: Một cô gái trên ván trượt.
    * **`noamrot/FuseCap`:**
        * English: ##yra, a young girl in a blue shirt and black shorts, rides a black skateboard while talking on her cell phone she wears dark sunglasses and has brown hair in the background, there is
        * Vietnamese: ## yra, một cô gái trẻ mặc áo sơ mi màu xanh và quần short đen, cưỡi một chiếc ván trượt màu đen trong khi nói chuyện trên điện thoại di động của mình, cô ấy đeo kính râm tối màu và có mái tóc nâu trong nền, có
    * **`nlpconnect/vit-gpt2-image-captioning`:**
        * English: A young woman is on a cell phone while wearing a blue shirt.
        * Vietnamese: Một phụ nữ trẻ đang sử dụng điện thoại di động trong khi mặc áo sơ xanh.

**Qualitative Evaluation from Example:**
* **`blip-image-captioning-base`:** The caption is concise and generic ("skateboard" instead of "rollerskates"). While contextually relevant, it lacks specific details.
* **`noamrot/FuseCap`:** This model attempts to provide highly detailed descriptions, including shirt color, accessories (sunglasses), and background context. However, it sometimes generates undesirable characters (`##yra`) and can produce overly long or grammatically awkward sentences in certain cases.
* **`nlpconnect/vit-gpt2-image-captioning`:** The caption is also quite concise and basic ("cell phone", "blue shirt"), but it misses crucial contextual or primary action details ("rollerskates", "skating").

**BLEU Score Results (on the first 10 images):**

* **`blip-image-captioning-base`:** BLEU Score: 30.12
* **`noamrot/FuseCap`:** BLEU Score: 11.43
* **`nlpconnect/vit-gpt2-image-captioning`:** BLEU Score: 10.76

**Explanation of BLEU Results:**

BLEU Score measures the n-gram overlap between generated captions and reference (Ground Truth) captions.

* **`blip-image-captioning-base` (BLEU: 30.12):** The highest BLEU score indicates that this model produces captions with the highest word overlap with the Ground Truths. This aligns with the qualitative observation: although concise, its generated words are typically standard and match common phrases found in the Ground Truth.
* **`noamrot/FuseCap` (BLEU: 11.43):** Despite the model's attempt to generate more detailed captions, the low BLEU score indicates significant word-level divergence from the Ground Truths. This can be explained by:
    * **Excessive Detail**: Ground Truths are often concise. FuseCap's addition of many details (colors, brands, specific positions) not present in the Ground Truth reduces word overlap.
    * **Distinct Style**: FuseCap might have a different generation style (e.g., using more specialized phrases or complex sentence structures) compared to traditional Ground Truths.
    * **Technical Issues**: Minor technical issues or generation artifacts (like `##yra`) can also reduce exact word matching.
* **`nlpconnect/vit-gpt2-image-captioning` (BLEU: 10.76):** A low BLEU score, similar to FuseCap, suggests limited word overlap with Ground Truths. While more concise than FuseCap, it might omit key details that Ground Truths focus on or use different synonyms/phrases.

**General Conclusion on Performance:**
BLIP-base demonstrates the best BLEU performance among the three selected models, striking a balance between accuracy and conciseness that aligns well with the provided Ground Truths. Conversely, FuseCap and ViT-GPT2, while generating meaningful descriptions, show lower BLEU scores due to their descriptive style (more detailed/divergent) or limited word overlap with the Ground Truths. This highlights that BLEU is not always a perfect measure for "richness" or "detail," especially when models aim to produce more elaborate captions than the original Ground Truths. [cite_start]The authors of FuseCap also noted that n-gram metrics (like BLEU) might not effectively measure the quality of "enriched" captions, suggesting metrics like CLIPScore or human evaluation as alternatives[cite: 4].

## 2. Project Structure

```bash
.
├── app.py                # Main Gradio application
├── model/                # Code for loading models and caption generation
│   ├── __init__.py
│   ├── caption_generator.py
│   └── translator.py
├── utils/                # Utility functions (data loading, evaluation)
│   ├── __init__.py
│   ├── data_loader.py
│   └── evaluator.py
├── test_images/          # Directory containing test images (50 images)
├── captions.xlsx         # Excel file containing Ground Truth captions for all test images
├── requirements.txt      # List of required Python libraries
└── README.md             # This project's documentation and guide
```

## 3. How to Run the Application

To set up and run the application on your local machine, please follow these steps:

**Clone Repository:**
Open your terminal (e.g., Git Bash) and navigate to your desired directory, then clone the repository:

```bash
git clone https://github.com/NhatNQuang/image-captioning.git
cd image-captioning
```

**Prepare Data and Images:**
- The `test_images/` directory is located in the project's root and should contain your 50 images.
- The `captions.xlsx` file, also located in the project's root, contains the Ground Truth captions for these images. This file must have the columns `image_name`, `comment_number`, and `comment`.
- **Note**: While you provide 50 images, the application will only use the first 10 images from the `test_images/` directory (sorted by filename) for the sample image gallery in the UI and for BLEU evaluation purposes, optimizing resource usage and time.

**Set Up Development Environment:**
Create and activate a Python virtual environment:

```bash
python -m venv venv
source venv/Scripts/activate  # On Windows (Git Bash/CMD/PowerShell)
# Or: source venv/bin/activate  # On Linux/macOS
```

**Install Required Libraries:**
Ensure you have an internet connection to download the necessary libraries and models.

```bash
pip install -r requirements.txt
```

**Run the Gradio Application:**

```bash
python app.py
```

After running, the application will provide a local URL (typically `http://127.0.0.1:7860/`). Open this URL in your web browser to interact with the application.

## 4. Recommendations for Production Deployment

If deploying this application in a real-world production environment, several aspects can be optimized:

**Speed and RAM Optimization:**
- **Quantization**: Utilize quantization techniques (e.g., FP16 or INT8) for models to reduce size and accelerate inference, especially on GPUs. The `bitsandbytes` library can facilitate this.
- **ONNX Runtime / TensorRT**: Convert models to inference-optimized formats like ONNX or TensorRT to achieve maximum performance.
- **Model Pruning / Distillation**: Reduce model size by removing unnecessary parts or training a smaller model to mimic the behavior of a larger one.

**Translation Service:**
- Instead of `deep_translator` (which might have rate limits or stability issues), consider using official services like Google Cloud Translation API or DeepL API for higher translation quality, reliability, and scalability. This would require an API key and might incur costs.

**Model Management:**
- Employ a model management system (e.g., MLflow, BentoML) to track model versions, facilitate deployment, and enable rollbacks.
- Establish a separate API for model inference to decouple backend and frontend logic, allowing easier scaling.
- **Error Handling and Logging**: Implement more robust logging to monitor performance, detect errors, and debug in a production environment.

**Scalability:**
- For increased user traffic, consider deploying the application on cloud platforms (AWS, GCP, Azure) with auto-scaling services to ensure continuous availability.

**Fine-tuning Models:**
- While not strictly required in the initial project scope, fine-tuning the selected models (e.g., BLIP-base) on a more specific and specialized image-caption dataset (if available) can significantly improve the accuracy and relevance of captions for that particular domain. Techniques like LoRA (Low-Rank Adaptation) are highly beneficial for this, as they allow fine-tuning large models with fewer resources and reduced risk of overfitting.

## 5. Example Using 
<br>
<p align="center">
  <img src="https://github.com/NhatNQuang/image-captioning/blob/develop/Image_Captioning_Sample_test.png" alt="Sample using for Image 10010052.jpg" style="width:70%;"/>
</p>
<br>
