# LLM_Playing
Practice on QA system(RAG) and Document based CodeGen with local and remote LLM models. These practices are done by LangChain, LLAMA2, and chatGPT.  
---
#### - `vectorstore_create.ipynb`
- Create local vectorstore from your own documents in advance
- In this way, we can save the time for parsing large documents and creating vectorstore each time running the app.
- If user need different code language paser, please update your LangChain basing on `issue/issue.txt` 

#### - `QA_BasicPractice.ipynb`
- Document QA system with LangChain and LLAMA.

#### - `QABot_UI.ipynb` 
- Gradio practice.

#### -`RetrievalQA_gpt.ipynb`, `RetrievalQA_llama.ipynb` and `RetrievalQABot.py`
- Use LangChain's RetrievalQA() to create a document based CodeGen system. (No memory)
- `RetrievalQA_gpt.ipynb`: use GPT API directly.
- `RetrievalQA_llama.ipynb`: use LLAMA2 or Code-LLAMA in local machine. (CodeLlama-34b 4bit mode on RTX-3090 but maybe take a while to generate output).
- `RetrievalQABot.py`: CodeLlama document based CodeGen system with gradio UI.

#### -`ConversationRAG_gpt.ipynb` and `ConversationRAG_llama.ipynb`
- Use LangChain's ConversationalRetrievalChain() to create a document based CodeGen system. (With memory)
- `ConversationRAG_gpt.ipynb`: use GPT API directly.
- `ConversationRAG_llama.ipynb`: use LLAMA2 or Code-LLAMA in local machine. (With memory but the prompt is not good, need update)(CodeLlama-34b 4bit mode on RTX-3090 but maybe take a while to generate output).
- `ConversationRAGBot.py`: chatGPT or CodeLlama document based CodeGen system with gradio UI.
