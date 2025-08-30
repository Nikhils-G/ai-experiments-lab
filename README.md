# Understanding ds-ai-ml-dl-nn

**Practical-1 . Autoencoders** properly in the hierarchy form :

* **Field** → **Machine Learning (ML)**
* **Subfield** → **Deep Learning (DL)** (because they use neural networks with multiple layers)
* **Category** → **Unsupervised Learning** (since they learn patterns from data without labels)
* **Technique** → **Representation Learning / Dimensionality Reduction** (they learn compressed, meaningful features from data)

> Autoencoders are a **deep learning technique under unsupervised learning** that aim to **compress data into a lower-dimensional representation (encoding) and then reconstruct it back (decoding)**. They are mainly used for **feature extraction, dimensionality reduction, noise removal, and generative tasks**.

---

**Practical-2 . Natural Language Processing (NLP)** in Hierarchy

**Field → Artificial Intelligence (AI)**
**Subfield → Machine Learning (ML)**
**Branch → Deep Learning (DL)** (modern NLP relies heavily on neural networks, especially transformers)
**Domain → Natural Language Processing (NLP)**
**Category → Text Understanding & Generation**
**Technique → Transformer-based Models (e.g., BERT, DistilBERT, GPT, etc.)**

**What NLP is**

Natural Language Processing (NLP) is a domain of AI that focuses on enabling machines to understand, interpret, and generate human language. It bridges the gap between computers and human communication by processing unstructured text and speech data.

**What I Did (Sentiment Analysis)**

* We used **Hugging Face Transformers** to build a **sentiment analysis pipeline**.
* Model used: **DistilBERT fine-tuned on SST-2 dataset** (binary classification: Positive / Negative).
* Input: A text sentence.
* Output: The sentiment label (Positive/Negative) + Confidence Score.

**Why I Did This**

* **To demonstrate NLP in action**: showing how a pre-trained model can quickly classify text sentiment.
* **Hands-on with Transformers**: exploring Hugging Face’s pipeline for easy inference.
* **Practical use-case**: Sentiment analysis is widely used in product reviews, customer feedback, and social media monitoring.

**Final Conclusion**

This experiment helped me understand how **NLP models like BERT/DistilBERT work for real-world text tasks**. By using a simple pipeline, I saw how transfer learning allows models trained on large datasets to be applied to our own sentences with minimal effort. This lays the foundation for exploring more advanced NLP tasks like text summarization, question answering, and chatbots.

---

**LLM Core Demo – Text Generation with GPT-2**

**Objective**
Demonstrate the core idea of **Large Language Models (LLMs): predicting the next word**.  

**Core Idea**

LLMs like GPT-2 are essentially **next-word predictors** trained on massive text corpora.
They generate text step by step, producing human-like sentences.

**Conclusion**

**This experiment demonstrates the core working principle of Large Language Models (LLMs):**

LLMs like GPT-2 are trained on huge text datasets.Their goal is not to "understand" language in a human way, but to predict the most probable next word (token) based on the input sequence.By repeating this prediction step many times, they generate coherent and contextually relevant sentences.The experiment shows how the same model behaves differently depending on the starting prompt.For example, with “Artificial Intelligence is transforming the world because…”, GPT-2 completes the thought in a logical way, whereas with “In 2050, humans and machines will…”, it predicts a futuristic continuation.

---

