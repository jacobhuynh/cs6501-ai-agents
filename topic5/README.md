# Topic 5: Retrieval-Augmented Generation (RAG)

AI Assistance: Used ClaudeCode to help complete assignment and generate README. Assignment Answers were AI assisted, but not completely AI generated.

### Teammates

- Albert Huang
- Blaise Duncan

## Files

### Scripts & Notebooks

- **[manual_rag_pipeline_universal.ipynb](manual_rag_pipeline_universal.ipynb)** — Google Colab notebook implementing a full RAG pipeline from scratch. Uses `BAAI/bge-base-en-v1.5` (768-dim) for embeddings, FAISS IndexFlatIP for cosine similarity search, and PyMuPDF for PDF extraction. Chunks documents with configurable size and overlap, respecting paragraph and sentence boundaries. Supports two query modes: `direct_query()` (no context, for hallucination testing) and `rag_query()` (full pipeline with retrieved context and source citations). The LLM is `Qwen/Qwen2.5-1.5B-Instruct`. Covers 12 exercises including RAG vs no-RAG comparison, GPT-4o Mini comparison, top-K tuning, query phrasing sensitivity, chunk overlap/size variation, prompt template variations, failure mode cataloging, and cross-document synthesis.

### Data Corpora (`Corpora/`)

- **`Congressional_Record_Jan_2026/`** — 30 Congressional Record PDFs and text files (Jan 2–30, 2026). Used to test RAG on recent events beyond any model's training cutoff.
- **`EU_AI_Act/`** — EU AI Act PDF and text file.
- **`Learjet/`** — 41 ATA aircraft technical manual PDFs and text files.
- **`ModelTService/`** — 8 Model T Ford service manual PDFs (both OCR and scan-only versions) plus text files.
- **`NewModelT/`** — 1 supplementary Model T Ford PDF and text file.

---

# Assignment Answers

## Exercise 1

### Does the model hallucinate specific values without RAG?

Yes the model does hallucinate specific values without RAG. In the Model T question about the spark plug gap, the model answered 0.375 inches, which is an arbitrary number. Similarly, for the Congressional Record questions, the no-RAG answers were made up (Mr. Flood's quote was random).

### Does RAG ground the answers in the actual manual?

Yes, RAG does ground the answers in the actual manual. The RAG answers were able to retrieve and cite actual text from the documents. A major example of this was quoting the specific steps about adjusting the rod and cotter key.

### Are there questions where the model's general knowledge is actually correct?

The model's general knowledge was partially correct when answering the carburetor question. The no-RAG answer gave generic steps for adjusting a carburetor that weren't necessarily wrong, but also not specific to a model T.

## Exercise 2

### Does GPT 4o Mini do a better job than Qwen 2.5 1.5B in avoiding hallucinations?

Yes, GPT-4o Mini does a better job than Qwen 2.5 1.5B in avoiding hallucinations. GPT-4o Mini admitted when it had no information on the topic rather than hallucinating answers like Qwen did. In addition, GPT-4o Mini gave answers that are more likely to be true (like 0.025-0.035" spark plug gap) while Qwen gave arbitrary values (0.375" spark plug gap). GPT-4o Mini did still hallucinate some answers though.

### Which questions does GPT 4o Mini answer correctly? Compare the cut-off date of GPT 4o Mini pre-training and the age of the Model T Ford and Congressional Record corpora.

GPT-4o Mini's training cutoff was October 2023. Since the Model T Ford manual was made around the 1900s, this information has been widely available for some time. Given this is the case, it makes sense that GPT-4o Mini was able to answer with decent accuracy. However, the Congressional Record corpus is from January 2026 (2 years past the training cutoff), which makes sense why GPT-4o Mini had no knowledge on these events.

## Exercise 6

### Which phrasings retrieve the best chunks?

"When do I need to check the engine?" scored a 0.5496 and retrieved very specific chunks about engine diagnostics. "How often should I service the engine?" and "engine oil change lubrication schedule" scored 0.5085 and 0.5109 respectively, and were able to retrieve relevant information about oil change intervals, inspection schedules, and more. "Preventive maintenance requirements" did not perform as well with a score of 0.4125 and retrieved relatively less relevant chunks about dealer inventory and shop equipment.

### Do keyword-style queries work better or worse than natural questions?

I found that the specific vocabulary mattered more than the style of querying. For example, for keyword-style, "Engine oil change lubrication schedule" scored very well, while "engine maintenance intervals" did not. On the other hand, for natural language question, "How often should I service the engine?" scored well, but doesn't seem to be because of the way the query was worded but rather the vocabulary used.

### What does this tell you about potential query rewriting strategies?

When querying, it's best to use specific terminology found in the documents (like "oil change" and "lubrication") rather than abstract terms like "preventive maintenance requirements." In addition, some phrases like "When do I need to check the engine?" had a 0/5 overlap with the formal phrasing despite high scores. This shows that retrieval is very sensitive to wording.

## Exercise 7

### Does higher overlap improve retrieval of complete information?

Higher overlap does not necessarily improve retrieval of complete information. overlap = 0 scored surprisingly well with a top score of 0.6185 and average top 5 score of 0.8070. However, overlap = 128 does better with a top score of 0.6044 and average top 5 score of 0.7024. In the middle though, overlap = 64 does worse than both of these with a top score of 0.7864 and average top 5 score of 0.8088. This shows that there is a sweet spot for the amount of overlap for retrieval of complete information.

### What's the cost? (Index size, redundant information in context)

The cost of going from overlap = 0 to overlap = 256 resulted in over 2 times as many chunks (2320 to 5215) and over 2 times as much time to build the index (4.1s to 9.5s). In addition, when moving to Overlap 256, the information became redundant as ranks 2, 3, and 4 had almost identical text relating to cylinder head bolt holes (just with a slight offset). This wastes the LLM's context window with a lot of duplicate information.

### Is there a point of diminishing returns?

Yes, as overlap = 128 seems to be the sweet spot with the top score of 0.6044 and best average top 5 score of 0.7024. However, increasing the overlap to 256 actually resulted in a lower top score and lower average top 5. In addition, it required almost double the chunk count without improving retrieval quality.

## Exercise 8

### How does chunk size affect retrieval precision (relevant vs. irrelevant content)?

A smaller chunk size of 128 was able to retrieve more precise text but could not always capture the full context. A medium chunk size of 512 seemed to be the perfect balance between retrieving relevant information while still providing enough context to be useful. A large chunk size of 2048 seemed to hurt precision as it was able to retrieve relevant blocks of information, but a significant amount of irrelevant information with it.

### How does it affect answer completeness?

Chunk size of 128 produced incomplete answers. While it would provide relevant information (or phrases), it wouldn't be of any substance. Chunk size of 512 produced the most complete answers. It was able to provide relevant information with multi-step procedures. Chunk size of 2048 produced longer answers but were often hallucinated or off topic. It would take in the irrelevant context and use it to create an answer that has nothing to do with the initial question.

### Is there a sweet spot for your corpus?

A chunk size of 512 seemed to be the sweet spot for the Model T manual. It had the best balance of relevant retrieval and answer completeness. The summary scores also seem to reflect this as 512 had the best average top score.

### Does optimal size depend on the type of question?

Yes the optimal size does depend on the type of question. For simple questions like "what oil should I use," a small chunk size like 128 is sufficient since the answer is short. However, for questions like asking for multi-step procedures, a chunk size of 512 was better as it was able to retrieve full blocks of procedures where a chunk size of 128 would have cut off. As stated before, a chunk size of 2048 gave too much context and would confuse the LLM with irrelevant information.
