# Webscrap-Rag-based-Chatbot



## **1. Architecture Overview**

      ┌───────────────────────────────┐
      │   xyz.com website │
      └───────────────┬───────────────┘
                      │  (scraper.py)
                      ▼
        ┌────────────────────────────┐
        │   knowledge.json (raw)     │
        └───────────────┬────────────┘
                        │ (chunks.py)
                        ▼
        ┌────────────────────────────┐
        │     chunks.json (clean)    │
        └───────────────┬────────────┘
                        │ (build_embeddings.py)
                        ▼
        ┌────────────────────────────┐
        │   FAISS vectorstore index  │
        └───────────────┬────────────┘
                        │
                        ▼
      ┌────────────────────────────────────────┐
      │         Streamlit Frontend (main.py)   │
      │                                        │
      │  - User Onboarding (PII masked)        │
      │  - Chat Input                          │
      │  - LangGraph Pipeline:                 │
      │      Router → Retriever → Output Node  │
      │  - LLM Response + Fallback             │
      └────────────────────────────────────────┘


---

## **2. Key Design Choices**

- **Minimal Stack:**  
  The system uses small, focused libraries: BeautifulSoup for scraping, FAISS for offline vector search, Ollama for local LLM inference, and Streamlit for the frontend. This makes it lightweight, portable, and easy to maintain. The trade-off is fewer out-of-the-box features but greater clarity and simplicity.

- **Fallback Logic for Offline-Friendliness:**  
  When the LLM/API is unavailable, pre-defined helpful responses are served to ensure continuity. This reduces richness of answers but guarantees usability even offline.

- **Dynamic Prompt Handling:**  
  Instead of hardcoding replies for greetings or unknown queries, the LLM generates answers based on context and defined rules. Flexible, but requires careful prompt design to prevent hallucinations.

---

## **3. Threat Model**

- **PII Flow:**  
  - User data (name, email, phone) is collected during onboarding.  
  - Stored locally in `user_data.json` and session state during runtime.  
  - Masked before sending prompts to the LLM, so sensitive info never leaves the system.

- **Mitigation:**  
  - No unmasked PII is transmitted externally.  
  - Session-based masking ensures safe interaction with the LLM.  
  - Local storage minimizes third-party exposure.

---

## **4. Scraping Approach**

- Used Selenium + BeautifulSoup to extract unstructured content from occamsadvisory.com
- Targeted headings, paragraphs, and main text blocks; filtered out boilerplate and duplicates.
- Saved raw data in `knowledge.json`, processed into clean chunks in `chunks.json`.
- Generated vector embeddings with Ollama + FAISS for efficient retrieval during chat.

---

## **5. Failure Modes & Graceful Degradation**

- **Potential Failures:**  
  - LLM/API unavailable → serves fallback text with key company info.  
  - No relevant context found → returns safe “❌ Sorry, I don’t know the answer based on our data.”  
  - Invalid user inputs → form validation prevents onboarding until corrected.

- **Graceful Handling:**  
  - Predefined responses keep the app functional without AI.  
  - Masking prevents PII leaks.  
  - Local chat history persists to maintain session continuity.

---

## **6. Minimal Tests Implemented**

- Email & phone validation: Ensures correct format and prevents invalid onboarding.
- Unknown question handling: Returns safe fallback without hallucinating.
- Chat nudges: Gently guides users to ask questions about services, careers, or contact info.

---

## **7. Usage**

1. **Install dependencies:**

  - pip install -r requirements.txt 	


2. **Installing Ollama for Local LLM:**

- Visit Ollama Official Site  
  Go to https://ollama.com/download to download the installer for your operating system (macOS, Windows, or Linux).

- **Install Ollama**  
  - macOS: Open the downloaded .dmg and drag Ollama to Applications.  
  - Windows: Run the .exe installer and follow the prompts.  
  - Linux: Follow the instructions on the website (package manager installation).  

- **Verify Installation**  
  Open terminal or command prompt and run:
  ```
  ollama list
  ```
  You should see available models, e.g., mistral.

- **Download the Model**  
  If the model you want isn’t installed yet, run:
  ```
  ollama pull mistral
  ```
  This downloads the model locally so that the LLM works offline.

3. **Scrape Website Data:**  
Generates knowledge.json with raw content from the website.

    Run :
    ```
    py scrapper.py
  
    ```


5. **Process Data into Chunks:**  
Generates chunks.json, which is structured for embedding.

     Run :
    ```
    py chunks.py
  
    ```    


7. **Build FAISS Vectorstore:**  
Creates faiss_index for efficient offline retrieval.

     Run :
    ```
    py embeddings.py
  
    ```  
     
    
9. **Run the Streamlit App**

    ```
    py -m streamlit run main.py
  
    ```  
---

## **8. Design Boundaries & Future Considerations**

### **🔹 What did we not build (and why)?**

I focused on keeping the system **lightweight and offline-friendly**.
- No cloud database – all data is stored locally to minimize PII exposure.
- No heavy orchestration tools (like LangChain agents) – simpler code, easier to maintain and debug.
- No multi-language support yet – we kept the scope narrow to English for faster prototyping.

This kept the project small, private, and easy to run without extra infrastructure.

---

### **🔹 How does the system behave if scraping fails or the LLM/API is down?**

- **If scraping fails** → the system falls back to the **last saved knowledge.json**, so the assistant can still function with older data.
- **If the LLM or Ollama is down** → the chatbot switches to **safe predefined fallback responses** (e.g., “Sorry, I don’t know the answer right now.”). This ensures the app never breaks completely.

---

### **🔹 Where could this be gamed or produce unsafe answers?**

- If a user deliberately asks **off-topic or misleading questions**, the LLM could still try to answer beyond the company scope.
- If malicious inputs are given (e.g., prompts to “ignore rules” or “leak data”), the assistant could potentially misbehave.

**To mitigate:**
- **Strict fallbacks** → “I don’t know” when no context is found.
- **PII masking** → user details are never leaked in model prompts.
- **Limited knowledge base** → only content from the official site is embedded.

---

### **🔹 How could we extend this to support OTP verification (without leaking PII)?**

If we needed OTP verification in the future:
- OTP generation and validation should happen **locally** (e.g., sending codes via email/SMS from a local server or company’s secure API).
- PII (email/phone) should never be sent to third-party services (public APIs).
- Verification could use:
- **Local SMTP setup** for emails.
- **Company’s in-house SMS gateway** if available.

This way, OTP adds a layer of authentication **without exposing personal data externally**.

---
