import os
import json
import time
import streamlit as st
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langgraph.graph import StateGraph, END

# Config & Paths

BASE_URL = "https://www.occamsadvisory.com/"
VECTORSTORE_PATH = "faiss_index"
USER_FILE = "user_data.json"
CHAT_FILE = "chat_history.json"

# STEP 1: SCRAPER

def scrape_occams():
    """Scrape Occams Advisory and save into knowledge.json (only if missing)."""
    if os.path.exists("knowledge.json"):
        return

    options = Options()
    options.headless = True
    driver = webdriver.Chrome(options=options)
    driver.get(BASE_URL)
    time.sleep(5)  # let JS load
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(2)
    html = driver.page_source
    driver.quit()

    soup = BeautifulSoup(html, "html.parser")
    data, seen_text = [], set()
    selectors = ["div.et_pb_text_inner", "section.et_pb_section p", "h1, h2, h3"]

    for sel in selectors:
        for tag in soup.select(sel):
            text = tag.get_text(strip=True)
            if text and len(text) > 30 and text not in seen_text:
                seen_text.add(text)
                data.append({"content": text, "url": BASE_URL})

    with open("knowledge.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# STEP 2: CHUNKING

def make_chunks():
    """Convert knowledge.json into chunks.json for embedding."""
    if os.path.exists("chunks.json"):
        return
    with open("knowledge.json", encoding="utf-8") as f:
        knowledge = json.load(f)
    chunks = [entry["content"] for entry in knowledge]
    with open("chunks.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

# STEP 3: EMBEDDINGS + VECTORSTORE

def build_embeddings():
    """Build FAISS vectorstore from chunks.json using Ollama embeddings."""
    if os.path.exists(VECTORSTORE_PATH):
        return
    with open("chunks.json", "r", encoding="utf-8") as f:
        chunk_texts = json.load(f)
    docs = [Document(page_content=text) for text in chunk_texts]
    embeddings = OllamaEmbeddings(model="phi3:mini")
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(VECTORSTORE_PATH)

# USER + CHAT HISTORY

def load_users():
    return json.load(open(USER_FILE)) if os.path.exists(USER_FILE) else []


def save_users(users):
    json.dump(users, open(USER_FILE, "w"), indent=4)


def load_chat(user_email):
    if os.path.exists(CHAT_FILE):
        all_history = json.load(open(CHAT_FILE))
        return all_history.get(user_email, [])
    return []


def save_chat(user_email, history):
    all_history = json.load(open(CHAT_FILE)) if os.path.exists(CHAT_FILE) else {}
    all_history[user_email] = history
    json.dump(all_history, open(CHAT_FILE, "w"), indent=2)


# STREAMLIT APP

def main():
    # --- Preprocessing ---
    scrape_occams()
    make_chunks()
    build_embeddings()

    # --- Load vectorstore AFTER building ---
    embeddings = OllamaEmbeddings(model="phi3:mini")
    vectorstore = FAISS.load_local(
        VECTORSTORE_PATH, embeddings, allow_dangerous_deserialization=True
    )
    retriever = vectorstore.as_retriever()
    llm = OllamaLLM(model="phi3:mini")

    # --- LangGraph pipeline ---
    class State(dict):
        question: str
        context: str
        answer: str

    def mask_pii(text):
        masked = text
        if "user_info" in st.session_state:
            info = st.session_state.user_info
            masked = masked.replace(info.get("name", ""), "[NAME]")
            masked = masked.replace(info.get("email", ""), "[EMAIL]")
            masked = masked.replace(info.get("phone", ""), "[PHONE]")
        return masked

    def router_node(state: State):
        state["route"] = "RETRIEVAL"
        return state

    def retrieval_node(state: State):
        docs = retriever.invoke(mask_pii(state["question"]))
        state["context"] = "\n".join([d.page_content for d in docs]) if docs else None
        return state

    def output_node(state: State):
        try:
            prompt = f"""
            You are an AI Assistantrepresenting Occams Advisory.
        Using the following context, provide a clear, structured answer to the user.
        When answering, speak in first-person as the company ("we", "our").
        Be friendly, professional, and concise.
        Rules:
           -If the user greets with (eg. "hi","hello","hey"), reply with a warm greeting and suggest 
            what they can ask about (services, careers, contact info).
           -If the user greets + asks a real question (e.g., "Hi, I want to know about your company"), 
            combine both: start with a greeting and then answer their question.
           -If the user asks something unrelated to the knowledge base, reply: 
            "❌ Sorry, I don’t know the answer based on our data."
           - Do NOT say "based on the provided context" or similar phrases.
           - Just answer directly like a human from the company would.
           - Do not include any generic signatures, disclaimers, or placeholders like [Your Name] 
             or [Your Position]

            Context:
            {state['context']}
            Question:
            {state['question']}
            """
            state["answer"] = llm.invoke(mask_pii(prompt))
        except Exception:
            state["answer"] = "⚠️ AI assistant is temporarily unavailable."
        if "user_info" in st.session_state:
            state["answer"] = state["answer"].replace(
                "[NAME]", st.session_state.user_info["name"]
            )
        return state

    graph = StateGraph(State)
    graph.add_node("router", router_node)
    graph.add_node("retrieval", retrieval_node)
    graph.add_node("output", output_node)
    graph.set_entry_point("router")
    graph.add_edge("router", "retrieval")
    graph.add_edge("retrieval", "output")
    graph.add_edge("output", END)
    app = graph.compile()


    # STREAMLIT UI
    

    
    if "onboarding_complete" not in st.session_state:
        st.session_state.onboarding_complete = False
    if "user_info" not in st.session_state:
        st.session_state.user_info = {}
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "form_id" not in st.session_state:
        st.session_state.form_id = 0

    
    # Onboarding Completion
    
    if not st.session_state.onboarding_complete:
        st.subheader("Complete Onboarding")
        with st.form(f"onboarding_form_{st.session_state.form_id}"):
            name = st.text_input("👤 Name", key=f"name_{st.session_state.form_id}")
            email = st.text_input("📧 Email", key=f"email_{st.session_state.form_id}")
            phone = st.text_input("📞 Phone", key=f"phone_{st.session_state.form_id}")
            submitted = st.form_submit_button("Submit")
            
            if submitted:
                if not name or not email or not phone:
                    st.error("All fields are required.")
                elif "@" not in email or "." not in email:
                    st.error("Please enter a valid email.")
                elif not phone.isdigit() or len(phone) < 7:
                    st.error("Please enter a valid phone number.")
                else:
                    users = load_users()
                    new_user = {"name": name, "email": email, "phone": phone}

                    already_signed_up = any(
                        u["name"] == new_user["name"] and 
                        u["email"] == new_user["email"] and 
                        u["phone"] == new_user["phone"]
                        for u in users
                    )
                    
                    if already_signed_up:
                        st.warning("⚠️ You have already signed up.")
                    else:
                        users.append(new_user)
                        save_users(users)
                        st.success("✅ Onboarding completed! You can now chat with the assistant.")

                    st.session_state.onboarding_complete = True
                    st.session_state.user_info = new_user
                    st.session_state.chat_history = [] 

    # Signed-In User & Logout
    
    if st.session_state.onboarding_complete:
        user_info = st.session_state.user_info
        if st.button("🚪 Logout"):
            save_chat(user_info["email"], st.session_state.chat_history)
            st.session_state.onboarding_complete = False
            st.session_state.user_info = {}
            st.session_state.chat_history = []
            st.session_state.form_id += 1
            st.rerun()

    # Chat Input & History
    
    if st.session_state.onboarding_complete:
        user_input = st.text_input("Ask a question:")
        if user_input:
            state = {"question": user_input}
            result = app.invoke(state)
            st.session_state.chat_history.append({"user": user_input, "ai": result["answer"]})
            save_chat(user_info["email"], st.session_state.chat_history)

    if st.session_state.chat_history:
        st.subheader("Chat History")
        for chat in st.session_state.chat_history:
            st.markdown(f"**👤 User:** {chat['user']}")
            st.markdown(f"**🤖 AI Bot:** {chat['ai']}")


if __name__ == "__main__":
    main()
