from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
import os
import json
import base64
import zipfile
from fpdf import FPDF
from docx import Document
import requests
from datetime import datetime

# FAISS + embedding model
import faiss
import pickle
from sentence_transformers import SentenceTransformer

app = Flask(__name__)
CORS(app)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@app.route("/query", methods=["POST"])
def handle_query():
    data = request.get_json()

    print("🚀 AIVS BUILD: 2024-04-07 19:00 - full rebuild")

    name = data.get("full_name")
    email = data.get("email")
    query = data.get("query")

    try:
        job_code = int(data.get("job_code", 9999))
    except (ValueError, TypeError):
        job_code = 9999

    discipline = data.get("discipline", "").lower().replace(" ", "_")
    search_type = data.get("search_type", "")
    timeline = data.get("timeline", "") or "Not specified"
    source_context = str(data.get("source_context", ""))
    supervisor_email = data.get("supervisor_email")
    supervisor_name = data.get("supervisor_name")
    hr_email = data.get("hr_email")

    if not all([name, email, query]):
        return jsonify({"status": "error", "message": "Missing required fields"}), 400

    if 1000 <= job_code < 2000:
        role_type = "executive"
    elif 2000 <= job_code < 3000:
        role_type = "hr"
    elif 3000 <= job_code < 4000:
        role_type = "supervisor"
    elif 4000 <= job_code < 5000:
        role_type = "staff"
    else:
        role_type = "general"
    # Load FAISS index and docs
    try:
        index = faiss.read_index(f"indexes/{discipline}_index.faiss")
        with open(f"indexes/{discipline}_docs.pkl", "rb") as f:
            docs = pickle.load(f)
    except Exception as e:
        print("❌ FAISS load error:", str(e))
        return jsonify({"status": "error", "message": f"FAISS load error: {str(e)}"}), 500

    # Embed the query and search
    embed_model = SentenceTransformer("paraphrase-albert-small-v2")
    query_vector = embed_model.encode([query])
    D, I = index.search(query_vector, k=10)

    # Collect top chunks
    context_chunks = [
        docs[i]['text'] for i in I[0]
        if i < len(docs) and len(docs[i]['text'].strip()) > 100
    ][:5]

    context_text = "\n\n".join(context_chunks)
   
    prompt = f"""
{source_context}

📚 Context from strategic materials:
{context_text}

You are a strategic advisor or director responding to an executive-level UK business issue.

Context:
- Discipline: {discipline.title()}
- Search Type: {search_type}
- Timeline: {timeline}

Please return your answer in this exact JSON format:
{{
  "enquirer_reply": "A thoughtful, actionable paragraph that addresses the problem..",
  "action_sheet": [
    "Step 1: ...",
    "Step 2: ...",
    "Step 3: ..."
  ]
}}

Query:
{query}
"""

    print("📤 Prompt being sent to GPT:\n", prompt)

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        content = response.choices[0].message.content
        parsed = json.loads(content)

        if isinstance(parsed.get("action_sheet"), dict):
            parsed["action_sheet"] = list(parsed["action_sheet"].values())

    except Exception as e:
        print("❌ GPT Error:", str(e))
        return jsonify({"status": "error", "message": f"OpenAI error: {str(e)}"}), 500

    enquirer_text = parsed.get("enquirer_reply", "No enquirer reply provided.")
    action_text = parsed.get("action_sheet", [])
    action_text_formatted = "\n".join(f"{i+1}. {item}" for i, item in enumerate(action_text))

    def write_outputs(recipient_label, include_action, timestamp):
        full_text = f"""AIVS REPORT – {recipient_label.upper()} – {timestamp}

==================================================

🔍 QUERY:
{query}

📘 Discipline: {discipline.title()}
📅 Timeline: {timeline}
🧭 Search Type: {search_type}

==================================================

📩 ENQUIRER REPLY:
{enquirer_text}
"""

        if include_action:
            full_text += "\n\n✅ ACTION SHEET:\n" + "\n".join(f"  - {item}" for item in action_text)

        pdf_file = f"{recipient_label}_{timestamp}.pdf"
        docx_file = f"{recipient_label}_{timestamp}.docx"
        zip_file = f"{recipient_label}_{timestamp}.zip"

        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, full_text.encode("latin-1", "replace").decode("latin-1"))
        pdf.output(pdf_file)

        doc = Document()
        doc.add_paragraph(full_text)
        doc.save(docx_file)

        with zipfile.ZipFile(zip_file, 'w') as zipf:
            zipf.write(pdf_file)
            zipf.write(docx_file)

        return zip_file

    # Generate timestamp once
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    files_to_send = {
        email: write_outputs("user", include_action=False, timestamp=timestamp)
    }

    if supervisor_email:
        files_to_send[supervisor_email] = write_outputs("supervisor", include_action=True, timestamp=timestamp)

    if hr_email:
        files_to_send[hr_email] = write_outputs("hr", include_action=True, timestamp=timestamp)

    try:
        for recipient, zip_path in files_to_send.items():
            with open(zip_path, "rb") as f:
                zip_encoded = base64.b64encode(f.read()).decode()

            postmark_payload = {
                "From": os.getenv("POSTMARK_FROM_EMAIL"),
                "To": recipient,
                "Subject": "Your AI Response",
                "TextBody": "Please find your AI-generated response attached.",
                "Attachments": [
                    {
                        "Name": os.path.basename(zip_path),
                        "Content": zip_encoded,
                        "ContentType": "application/zip"
                    }
                ]
            }

            r = requests.post(
                "https://api.postmarkapp.com/email",
                headers={
                    "X-Postmark-Server-Token": os.getenv("POSTMARK_API_KEY"),
                    "Content-Type": "application/json"
                },
                json=postmark_payload
            )

            if r.status_code != 200:
                raise Exception(f"Postmark error: {r.status_code} - {r.text}")

    except Exception as e:
        print("❌ Postmark Error:", str(e))
        return jsonify({"status": "error", "message": f"Postmark error: {str(e)}"}), 500

    print("✅ All responses sent")
    return jsonify({"status": "success", "message": "Response emailed to all recipients successfully."})

@app.route("/ping")
def ping():
    return jsonify({"status": "ok", "message": "API is live and reachable."})