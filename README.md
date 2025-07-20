# DocuSummary
A smart MultiModal RAG application that allows you to query your documents!

## Screenshots


### What is RAG?
Retrival Augmented Generation - as the name implies, uses "Retrieval" method to generate answers for user query.
We use a retrieval model to retrive relevent data from our knowledge base. This data is then passed as a context to our Generative model, which answers the user's exact query using the contexts.

### Motivation 
LLMs can generate answers to queries, no issues to that regard. However, the answers generated can be hallucinated to just "it seems fine and logical to read". By having a knowledge base for the LLM to extract data from, the chances of hallucinations are greatly reduced.
Additionally, we can provide personal documents to LLMs and ask questions regarding the documents. The LLMs can piece together the extracted information and provide the output that satisfies our queries.

### Models and working
Our documents are processed and seperated per page. Each page is scanned by a `Vision Language Model` (currently using InternVL3 from OpenGVLabs). The model extracts text data, tabular relations and images description from the page.
Once our documents are processed, we have the textual outputs of the documents. These outputs are converted into `chunks of text`. Each chunk contains only certain part of the textual input. \
The chunks are then converted to `Embedding vectors` using an `Embedding Model` (currently using Qwen 3 0.6B Embedding). \
The Vectors are then stored in a `Vector Store` (currently using ChromaDB).

When a query is asked, the query is converted into vector using Embedding Model. Then, nearest vectors are extracted from the Vector Store using a similarility search function. The relevent texts of the extracted vectors are loaded and then sent to a `Chat` or `Instruct` model, which then pieces the vector chunks as contexts and answers our queries. (Currently using Qwen 3 3B Instruct model for answers generation).

### Limitations
Hardware limitations played a major factor when I created this project, which is why I took certain decisions related to design or components of the project. \
My main motive was to run this project on general consumer hardware (8GB system RAM, 4GB GPU (GTX 1650 tested), Ubuntu (WSL compatible)). Which is why I have to use models with less numbers of parameters, quantizations applied and weird subprocess callbacks.

A Major issue I came across was the CUDA memory not cleaning up after the reference to models were removed, turns out is an issue where the objects sent to CUDA stayed in memory even after their references are deleted. This led to me writing file scripts and calling them using subprocess to circumvent the issue. When the file exits after processing, the models are removed from memory.

Given this issue, it also meant I cannot load the chat model initially. I have to explicitly load the required models when user needs them, thus the overall loading time is high.
