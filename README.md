🔍 Advanced Information Retrieval System

A comprehensive Information Retrieval (IR) search engine built with Python and Streamlit. This project demonstrates fundamental and advanced IR concepts, including Inverted Indexing, Positional Indexing, Boolean Retrieval, and Phonetic Search using Soundex.

🌟 Features

Core Retrieval Models

Boolean Retrieval: Supports complex queries using AND, OR, and NOT operators.

Phrase Query: Finds documents containing exact phrases (e.g., "quick brown fox") using positional indexes.

Soundex Search: Implements phonetic matching to find names that sound similar but are spelled differently (e.g., "Smith" vs "Smyth").

Advanced Capabilities

🔮 Wildcard Search: Supports prefix matching using * (e.g., comput* finds computer, computing).

📝 Spelling Correction: Uses Levenshtein Edit Distance to suggest corrections for misspelled queries.

✨ Keyword Highlighting: visualizes search terms directly within the retrieved documents.

📁 Dynamic File Upload: Allows users to upload their own .txt corpora to search through custom data.

📊 Index Visualization: View the internal structure of Inverted, Positional, and Soundex indexes.

🛠️ Installation

Clone the repository:

git clone [https://github.com/yourusername/ir-search-engine.git](https://github.com/yourusername/ir-search-engine.git)
cd ir-search-engine


Install dependencies:

pip install nltk streamlit


Download NLTK data:
The application will attempt to download necessary NLTK data (stopwords, punkt) automatically on the first run.

🚀 Usage

Run the Streamlit application:

streamlit run app.py


Navigate to the Web Interface:
Open your browser and go to http://localhost:8501.

How to Search:

Select a Model: Choose between Boolean, Phrase, or Soundex from the dropdown.

Enter Query:

Boolean: information AND retrieval, python OR java, NOT machine

Phrase: natural language processing

Soundex: Robert (finds Rupert), Herman (finds Hermann)

Wildcard: progra*

📂 Project Structure

├── app.py              # The Frontend: Streamlit GUI handling user input and visualization
├── search_engine.py    # The Backend: Core logic for indexing, preprocessing, and searching
├── requirements.txt    # List of dependencies
└── README.md           # Project documentation


🧠 Algorithms & Logic

1. Preprocessing Pipeline

Every document and query goes through a cleaning process:

Tokenization: Splitting text into individual words using nltk.

Stop Word Removal: Filtering out common words (is, the, and) that add little meaning.

Stemming: Reducing words to their root form (e.g., "running" -> "run") using the Porter Stemmer.

2. Indexing Strategies

Inverted Index: Maps specific terms to the list of documents containing them. (Used for Boolean Search).

Positional Index: Maps terms to specific positions within documents. (Used for Phrase Search).

Soundex Index: Maps 4-character phonetic codes to terms. (Used for Soundex Search).

3. Spelling Correction

If a search returns 0 results, the system calculates the Levenshtein Distance between the query term and the index vocabulary to suggest the closest matching word.

🤝 Contributing

Contributions, issues, and feature requests are welcome!

Fork the Project

Create your Feature Branch (git checkout -b feature/AmazingFeature)

Commit your Changes (git commit -m 'Add some AmazingFeature')

Push to the Branch (git push origin feature/AmazingFeature)

Open a Pull Request

📝 License

Distributed under the MIT License. See LICENSE for more information.

Created for the Information Retrieval Course Project
