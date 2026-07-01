# 🎬 Movie Review Sentiment Analyzer

An end-to-end Machine Learning web application that predicts whether a movie review is **Positive** or **Negative**. Built using the classic IMDb dataset, traditional Natural Language Processing (NLP) techniques, and deployed with an interactive interface using Streamlit.

---

## 🚀 Features
* **Custom Text Preprocessing:** Cleans raw internet text by lowering case, removing HTML tags/punctuation, and filtering out English stopwords.
* **TF-IDF Vectorization:** Transforms raw text strings into numerical features optimized for machine learning.
* **Highly Efficient Classifier:** Leverages a trained Logistic Regression model achieving high accuracy with lightning-fast inference times.
* **Interactive UI:** A clean, user-friendly web application interface to test custom reviews on the fly.

---

## 📁 Project Structure
```text
Movie_Review_Sentiment_Analyzer/
│
├── .venv/                  # Virtual environment
├── app.py                  # Streamlit web application script
├── sentiment_model.pkl     # Trained Logistic Regression model
├── tfidf_vectorizer.pkl    # Saved TF-IDF Vectorizer vocabulary
└── README.md               # Project documentation
```

## 🛠️ Installation & Setup
Follow these steps to run the application locally:

## 1. Clone or Open the Project Directory
Navigate to your project directory using your terminal:

```Bash
cd D:\Movie_Review_Sentiment_Analyzer
```
## 2. Activate the Virtual Environment
Activate your Python virtual environment:

 - Windows (PowerShell):

```PowerShell
.venv\Scripts\Activate.ps1
```
 - Windows (Command Prompt):

```DOS
.venv\Scripts\activate.bat
```
## 3. Install Required Dependencies
Ensure you have the necessary libraries installed inside your environment:

```Bash
pip install pandas scikit-learn nltk streamlit
```
## 🖥️ How to Run the App
Once your environment is active and dependencies are installed, launch the web application by running:

```Bash
streamlit run app.py
```
**💡 Note**: A browser window should automatically open at http://localhost:8501. If it doesn't, copy and paste that URL into your web browser.

## ⚙️ How the Pipeline Works
1. **Data Ingestion**: Merges 50,000 individual movie review text files from the IMDb dataset into structured CSV datasets.

2. **Preprocessing**: Strips away non-alphabet characters, web-specific noise (<br />), and non-sentiment driving words (stopwords).

3. **Feature Engineering**: Converts cleaned strings into continuous values using a TF-IDF Matrix with unigrams and bigrams.

4. **Modeling**: Fits a Logistic Regression model on the mathematical weights of the vocabulary to classify textual sentiment.
