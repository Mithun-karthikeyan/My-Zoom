## My Zoom: A Transformer-Based Model for Contextual Feedback Validation

## Project Overview

This project develops a machine learning solution to automatically validate user feedback on EdTech platforms like Zoom. By leveraging a fine-tuned BERT model, it checks whether the free-form feedback text aligns with the selected dropdown reason. This validation process improves the quality of feedback data for actionable insights. A Gradio interface is provided for real-time predictions, and the solution is deployable via Hugging Face Spaces.

>  Achieves over **85% accuracy** (depending on dataset quality) and improves platform feedback pipelines.

---

## Problem Statement

Zoom and similar EdTech tools collect feedback through open text and selected reasons (e.g., dropdowns). However, users often mismatch them:

**Example:**
- Feedback: *“Amazing app!”*
- Reason: *“Audio not working”*

Such mismatches degrade feedback quality. Manual filtering is inefficient and unscalable. This project builds an **automated validation model** to classify each feedback-reason pair as:
- `Aligned`
- `Not Aligned`

---

## Features

- **Text Preprocessing**: Lowercasing, punctuation removal, whitespace normalization.
- **Model Training**: Fine-tunes `bert-base-uncased` for binary classification (`Aligned` vs. `Not Aligned`) on labeled data.
- **Evaluation**: Outputs accuracy, precision, recall, F1-score.
- **Interactive Inference**: Real-time predictions using Gradio.
- **Deployment**: Easily hosted via Hugging Face Spaces.

---

## Technology Stack

| Component           | Tools & Libraries                            |
|---------------------|----------------------------------------------|
| Language            | Python 3.11                                   |
| Model Architecture  | BERT (`bert-base-uncased`)                    |
| ML Framework        | PyTorch 2.3.0, Hugging Face Transformers 4.40.0 |
| Interface           | Gradio 4.31.0                                 |
| Data Handling       | Pandas 2.2.2, Regex                           |
| Metrics             | scikit-learn 1.5.0                            |
| Deployment          | Hugging Face Spaces                           |

---

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/myzoom-feedback-validator.git
   cd myzoom-feedback-validator
   ```

2. **Set Up Virtual Environment** (Recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## Dataset Format

Use CSV files with the following columns:

| Column  | Description                            |
|---------|----------------------------------------|
| `text`  | User-submitted feedback text           |
| `reason`| Reason selected from dropdown menu     |
| `label` | 1 = Aligned, 0 = Not Aligned           |

Example (`train - train.csv` and `evaluation - evaluation.csv`):

```csv
text,reason,label
"Great class!", "Good app for online classes", 1
"Terrible lag", "Audio not working", 1
"Amazing UX", "Video freezing", 0
```

---

## Training the Model

Use the script: `bert_feedback_classifier.py`

### On Google Colab
1. Upload your CSV files.
2. Set runtime to GPU.
3. Install dependencies:
   ```python
   !pip install transformers==4.40.0 torch==2.3.0 pandas==2.2.2 scikit-learn==1.5.0 tqdm
   ```
4. Paste and run `bert_feedback_classifier.py` in a code cell.

### Output:
- Trained model saved in `feedback_model/`

---

## Running Locally (Gradio)

1. Ensure `feedback_model/` is in the root directory.
2. Run the app:
   ```bash
   python myzoom_gradio_app.py
   ```
3. Visit: [http://localhost:7860](http://localhost:7860)

---

## Deploy to Hugging Face Spaces

1. Create a new **Gradio Space** on [Hugging Face Spaces](https://huggingface.co/spaces).
2. Upload:
   - `myzoom_gradio_app.py`
   - `requirements.txt`
   - `feedback_model/`
3. Deploy!

---

## Results

| Metric     | Value   |
|------------|---------|
| Accuracy   | > 85%   |
| Precision  | ~0.87   |
| Recall     | ~0.85   |
| F1 Score   | ~0.86   |

---

## Project Structure

```
myzoom-feedback-validator/
├── bert_feedback_classifier.py     # Model training and evaluation
├── myzoom_gradio_app.py           # Gradio web app
├── requirements.txt               # Dependencies
├── feedback_model/                # Trained BERT model
├── train - train.csv              # Training data (user-supplied)
├── evaluation - evaluation.csv    # Eval data (user-supplied)
└── README.md                      # This file
```

---

## Challenges & Solutions

- **Version Conflicts**: Resolved with strict dependency versions.
- **Imbalanced Data**: Manual balancing + augmentation planned.
- **Model Size**: Consider DistilBERT for faster inference.

---

## Future Enhancements

- Add multi-class classification.
- Integrate with Zoom API.
- Use lightweight models for mobile/cloud optimization.
- Display model confidence scores.

---

## Acknowledgments

- Hugging Face for `transformers`, `Gradio`, and Spaces.
- Google Colab for GPU resources.
- Zoom for inspiring the real-world use case.

