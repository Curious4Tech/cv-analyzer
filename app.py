import os
import re
from dotenv import load_dotenv
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.storage.blob import BlobServiceClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from openai import AzureOpenAI


# Load environment variables from .env
load_dotenv()
app = Flask(__name__)

# 1. Flask Config
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"pdf", "docx", "png", "jpg", "jpeg"}
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # 5MB limit

# Ensure the uploads directory exists (for both production and local)
if not os.path.exists(app.config["UPLOAD_FOLDER"]):
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# 2. Retrieve Credentials (using valid key names)
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")
AZURE_KEY = os.getenv("AZURE_KEY")
AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
AZURE_BLOB_CONTAINER_NAME = os.getenv("AZURE_BLOB_CONTAINER_NAME", "qualified-resumes")

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")


# 3. Validate Credentials
if not AZURE_ENDPOINT or not AZURE_KEY:
    raise ValueError("AZURE_ENDPOINT and AZURE_KEY must be set.")
if not AZURE_STORAGE_CONNECTION_STRING:
    raise ValueError("azure-storage-connection-string must be set.")
if not AZURE_OPENAI_ENDPOINT or not AZURE_OPENAI_KEY or not AZURE_OPENAI_DEPLOYMENT:
    print("Warning: Azure OpenAI credentials not fully set. AI feedback may fail.")

# 4. Initialize Clients
document_intelligence_client = DocumentIntelligenceClient(
    endpoint=AZURE_ENDPOINT, credential=AzureKeyCredential(AZURE_KEY)
)
azure_openai_client = AzureOpenAI(
    api_key=AZURE_OPENAI_KEY,
    api_version=AZURE_OPENAI_API_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT
)

# 5. Headings for Section Parsing
SECTION_HEADINGS = [
    "work experience",
    "skills",
    "education",
    "projects",
    "certifications"
]

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def analyze_document(file_path):
    with open(file_path, "rb") as f:
        file_bytes = f.read()

    # Decide model
    model_id = "prebuilt-layout"
    poller = document_intelligence_client.begin_analyze_document(
        model_id, body=file_bytes
    )
    result = poller.result()
    extracted_lines = []
    for page in result.pages:
        for line in page.lines:
            extracted_lines.append(line.content.strip())

    cleaned_text = clean_up_text("\n".join(extracted_lines))
    sections_dict = extract_sections(cleaned_text)
    return cleaned_text, sections_dict

def clean_up_text(text):
    placeholders = ["Lorem ipsum", "Header", "Footer", "Page "]
    for placeholder in placeholders:
        text = text.replace(placeholder, "")

    lines = text.splitlines()
    unique_lines = []
    seen = set()

    for line in lines:
        line = line.strip()
        if len(line) < 3:
            continue
        if line in seen:
            continue
        seen.add(line)
        unique_lines.append(line)

    return "\n".join(unique_lines)

def extract_sections(cleaned_text):
    lower_text = cleaned_text.lower()
    pattern = r"(?P<heading>(" + "|".join(SECTION_HEADINGS) + r"))\s*(?P<content>[\s\S]*?)(?=(?:" + "|".join(SECTION_HEADINGS) + r")|$)"
    sections = {}
    matches = re.finditer(pattern, lower_text, flags=re.IGNORECASE)
    
    for match in matches:
        heading = match.group("heading").strip().title()
        content = match.group("content").strip()
        sections[heading] = content
    
    return sections

def compare_texts(resume_text, job_desc_text):
    tfidf_vectorizer = TfidfVectorizer().fit([resume_text, job_desc_text])
    tfidf_matrix = tfidf_vectorizer.transform([resume_text, job_desc_text])
    cos_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    return cos_sim[0][0]

def upload_to_blob(file_path, filename):
    blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
    container_client = blob_service_client.get_container_client(AZURE_BLOB_CONTAINER_NAME)
    try:
        container_client.create_container()
    except Exception:
        pass
    with open(file_path, "rb") as data:
        container_client.upload_blob(name=filename, data=data, overwrite=True)

def generate_feedback_with_azure(resume_text, job_desc_text, sections=None):
    if not (AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_KEY and AZURE_OPENAI_DEPLOYMENT):
        return "Azure OpenAI credentials are missing or incomplete. Cannot generate AI feedback."

    sections_info = ""
    if sections:
        for heading, content in sections.items():
            sections_info += f"\n[{heading}]\n{content}\n"

    user_prompt = f"""
    The following is a resume parsed into sections:
    {sections_info if sections else resume_text}
    Job Description:
    {job_desc_text}

    Please provide 3-4 specific, actionable suggestions to improve this resume so it better matches the job posting.
    - Identify missing or underemphasized skills and key requirements.
    - Suggest improvements in formatting, structure, or readability.
    - Highlight areas where the candidate can add more details or quantify achievements.
    """

    try:
        messages = [
            {"role": "system", "content":"You are an expert career advisor and resume reviewer. Your task is to provide exactly four bullet points of feedback, no more and no less. Each bullet point must be clear, specific, and actionable, addressing how the candidate can improve their resume to better match the job description. You should reference best practices in resume writing and offer relevant examples when possible. Do not exceed four bullet points."},
            {"role": "user", "content": user_prompt.strip()}
        ]
        response = azure_openai_client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT,
            messages=messages,
            temperature=0.7,
            max_tokens=500,
            n=1
        )
        feedback = response.choices[0].message.content.strip()
        return feedback if feedback else "No feedback generated."
    except Exception as e:
        return f"Error generating AI feedback: {str(e)}"


@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        job_description = request.form.get("job_description", "").strip()
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"})

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No selected file"})

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(file_path)

            extracted_text, sections_dict = analyze_document(file_path)

            qualification_status = None
            similarity_score = None
            upload_status = None
            feedback_message = None

            if job_description:
                similarity_score = compare_texts(extracted_text, job_description)
                feedback_message = generate_feedback_with_azure(extracted_text, job_description, sections=sections_dict)

                threshold = 0.65
                if similarity_score >= threshold:
                    qualification_status = "Qualified"
                    upload_to_blob(file_path, filename)
                    upload_status = "Your resume has been uploaded to our database."
                    # Optionally, send_email_to_hr(filename)
                else:
                    qualification_status = "Not Qualified"
                    upload_status = "Your resume requires improvements to better match the job requirements."

            formatted_similarity = f"{similarity_score * 100:.1f}%" if similarity_score is not None else None

            return render_template(
                "result.html",
                text=extracted_text,
                similarity=formatted_similarity,
                status=qualification_status,
                upload_status=upload_status,
                feedback=feedback_message
            )

    return render_template("upload.html")
if not os.path.exists(app.config["UPLOAD_FOLDER"]):
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
if __name__ == "__main__":
    app.run(debug=False)
