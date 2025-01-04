import os
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
from langchain.document_loaders import PyPDFLoader
from tempfile import NamedTemporaryFile
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

def get_ai_response(prompt):
    # Set API Key OpenAI melalui variabel lingkungan
    os.environ["OPENAI_API_KEY"] = "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"

    # Inisialisasi ChatOpenAI dari LangChain
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # Menghasilkan tanggapan berdasarkan prompt
    response = llm.invoke([HumanMessage(content=prompt)]).content
    # Menghitung estimasi token
    token_usage = {
        "prompt_tokens": len(prompt.split()),
        "completion_tokens": len(response.split()),
        "total_tokens": len(prompt.split()) + len(response.split())
    }
    # Hitung biaya total berdasarkan estimasi token
    cost_per_1k_input_tokens = 0.00015  # USD untuk input tokens GPT-4o-mini
    cost_per_1k_output_tokens = 0.0006  # USD untuk output tokens GPT-4o-mini
    input_cost = (token_usage["prompt_tokens"] / 1000) * cost_per_1k_input_tokens
    output_cost = (token_usage["completion_tokens"] / 1000) * cost_per_1k_output_tokens
    total_cost = input_cost + output_cost
    return response.strip(), token_usage, total_cost

def extract_text_from_pdf(file):
    with NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file.read())
        temp_file_path = temp_file.name
    
    loader = PyPDFLoader(temp_file_path)
    documents = loader.load()
    text = "".join([doc.page_content for doc in documents])
    os.remove(temp_file_path)  # Hapus file sementara setelah selesai
    return text

def validate_sdg(output):
    # Daftar SDG resmi
    sdg_list = [
        "No Poverty", "Zero Hunger", "Good Health and Well-being", "Quality Education", 
        "Gender Equality", "Clean Water and Sanitation", "Affordable and Clean Energy", 
        "Decent Work and Economic Growth", "Industry, Innovation, and Infrastructure", 
        "Reduced Inequalities", "Sustainable Cities and Communities", 
        "Responsible Consumption and Production", "Climate Action", 
        "Life Below Water", "Life on Land", "Peace, Justice, and Strong Institutions", 
        "Partnerships for the Goals"
    ]
    return output if output in sdg_list else "SDG tidak valid"

def generate_wordcloud(text):
    # Tambahkan kata-kata umum bahasa Indonesia ke STOPWORDS
    additional_stopwords = {"dan", "pada", "yang", "untuk", "dengan", "ini", "itu", "adalah", "dari", "di", "ke", "atau"}
    stopwords = STOPWORDS.union(additional_stopwords)

    wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=stopwords).generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)

# Streamlit UI
st.title("AI Chat with LangChain and PDF Support")

# Upload PDF file
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
pdf_text = ""

if uploaded_file is not None:
    pdf_text = extract_text_from_pdf(uploaded_file)

# Batasi output untuk hanya mencakup jawaban SDG Indonesia
if st.button("Dapatkan Tanggapan AI"):
    if pdf_text:
        response, token_usage, total_cost = get_ai_response(
            "Keluarkan hanya satu macam SDG Indonesia dari teks berikut tanpa tambahan kata lain. Pilih dari daftar SDG resmi seperti: 'No Poverty', 'Zero Hunger', 'Peace, Justice, and Strong Institutions', dll. Jawab hanya dengan nama SDG yang paling relevan.\n\n" + pdf_text
        )
        validated_response = validate_sdg(response)

        st.write("**Hasil Tanggapan AI:**")
        st.write(validated_response)

        # Tampilkan jumlah token yang digunakan
        st.write("**Token yang digunakan:**")
        st.write(f"Prompt Tokens: {token_usage['prompt_tokens']}, Completion Tokens: {token_usage['completion_tokens']}, Total Tokens: {token_usage['total_tokens']}")

        # Tampilkan biaya total
        st.write("**Biaya total (estimasi):**")
        st.write(f"${total_cost:.6f}")

        # Generate word cloud for keywords from the PDF
        st.write("**Wordcloud dari kata kunci:**")
        generate_wordcloud(pdf_text)
    else:
        st.warning("Harap unggah file PDF terlebih dahulu.")