# 🤖 Machine Learning Internship Projects - ELite Tech Intern (ETI)

This repository showcases three key projects completed during my Machine Learning internship at **ELite Tech Intern**. Each task demonstrates a different machine learning or deep learning technique using structured and unstructured data, with thorough implementation and evaluation.

---

## 📌 Overview of Tasks

| Task No. | Title                      | Technique                | Dataset                 | Outcome               |
|----------|----------------------------|---------------------------|--------------------------|------------------------|
| Task 1   | Decision Tree Classifier   | Supervised Learning (Tree-Based) | Iris Dataset             | 97% Accuracy + Tree Visualization |
| Task 2   | Sentiment Analysis with NLP | NLP + Logistic Regression/Random Forest | Twitter Sentiment Dataset | ~90% Accuracy + Interactive UI |
| Task 3   | Image Classification with CNN | Deep Learning (CNN)      | Fashion MNIST            | ~89% Accuracy + Prediction Widget |

---

## 🌳 Task 1: Decision Tree Classification

**Dataset**: Iris Dataset  
**Techniques**: EDA, Decision Tree Classifier, Feature Correlation, Interactive Prediction  
**Libraries**: `sklearn`, `seaborn`, `matplotlib`, `ipywidgets`

### 🔍 Summary
- Performed EDA and plotted heatmaps and class distributions.
- Built a decision tree model using Gini index.
- Evaluated with accuracy (~97%), confusion matrix, and classification report.
- Visualized the decision tree using `plot_tree`.
- Created an **interactive widget** to classify based on custom inputs.

---

## 💬 Task 2: Sentiment Analysis with NLP

**Dataset**: [Twitter Sentiment Analysis](https://raw.githubusercontent.com/dD2405/Twitter_Sentiment_Analysis/master/train.csv)  
**Techniques**: Text Cleaning, TF-IDF Vectorization, Logistic Regression, Random Forest  
**Libraries**: `nltk`, `scikit-learn`, `seaborn`, `ipywidgets`, `re`

### 🔍 Summary
- Cleaned text (lowercase, URL/mention removal, punctuation stripping).
- Balanced classes using resampling techniques.
- Vectorized text using **TF-IDF**.
- Trained Logistic Regression and Random Forest models.
- Evaluated using accuracy (~90%), confusion matrix, classification report.
- Built an **interactive text area** for real-time sentiment prediction.

---

## 🧠 Task 3: Image Classification using CNN

**Dataset**: Fashion MNIST  
**Techniques**: Convolutional Neural Networks (CNN), Model Evaluation, Interactive Image Classification  
**Libraries**: `tensorflow`, `keras`, `matplotlib`, `ipywidgets`

### 🔍 CNN Architecture
Input: 28x28 grayscale image → Reshaped to (28, 28, 1)

Conv2D (32 filters, 3x3, ReLU)

MaxPooling2D (2x2)

Conv2D (64 filters, 3x3, ReLU)

MaxPooling2D (2x2)

Flatten

Dense (128 units, ReLU)

Output Dense (10 units, Softmax)


### 🔍 Summary
- Trained model for 5 epochs with ~91% training accuracy and ~89% test accuracy.
- Evaluated model using a confusion matrix and classification report.
- Implemented **interactive image selector** from test set to preview predictions.

---

## 🧪 Evaluation Metrics Used

All tasks included evaluation using:

- **Accuracy** – Overall correctness of model predictions.
- **Confusion Matrix** – Visual representation of true vs predicted labels.
- **Precision, Recall, F1-Score** – Per-class metrics for better performance insights.

---

## 🔄 Interactive Features

Each notebook includes widgets via **`ipywidgets`** for hands-on testing:
- Task 1: Sliders to predict flower class.
- Task 2: Text area to enter a review and get sentiment.
- Task 3: Dropdown selector to pick test image and get prediction.

---

## 🔧 Future Work

- Tune hyperparameters using GridSearchCV or RandomizedSearchCV.
- Expand datasets or use ensemble models for stronger generalization.
- Incorporate more advanced models (e.g., BERT, ResNet) for NLP and Vision tasks.
- Apply techniques like Dropout, Data Augmentation, and Early Stopping.

---

## 🛠️ Tools & Libraries

- **Languages**: Python  
- **Libraries**: `scikit-learn`, `tensorflow`, `keras`, `nltk`, `seaborn`, `matplotlib`, `ipywidgets`  
- **Platforms**: Jupyter Notebook, VSCode

---

## 📂 Repository Structure

ETI/
│
├── Decision Tree Classification.ipynb
├── Sentiment Analysis NLP.ipynb
├── Image_Classification_CNN.ipynb
└── README.md


---

## Acknowledgments

Special thanks to  **Teachers**, **Friends**, **ETI** for the mentorship and guidance during the internship.

- Dataset Credits: UCI, Zalando, Twitter NLP  
- Frameworks: Scikit-learn, TensorFlow, Keras  
- Visuals: Seaborn, Matplotlib, Plotly


