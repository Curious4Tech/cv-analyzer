# CV Analyzer - AI-Powered Resume Screening  

CV Analyzer is a **Flask-based web app** that automates resume screening using **Azure Document Intelligence** for text extraction, **TF-IDF & cosine similarity** for job matching, and **Azure OpenAI** for AI-driven feedback. Qualified resumes are stored in **Azure Blob Storage** for HR teams.

---

## 🚀 1. Setup Guide  

### **Clone & Navigate**  
```bash
git clone https://github.com/Curious4Tech/cv-analyzer.git
cd cv-analyzer
```

### **Create & Activate Virtual Environment**  
```bash
python -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows
```

### **Install Dependencies**  
```bash
pip install -r requirements.txt
```

### **Set Up Environment Variables**  
Create or edit the  `.env` file and add your Azure credentials:  
```env
AZURE_ENDPOINT=your_azure_endpoint
AZURE_KEY=your_azure_key
AZURE_STORAGE_CONNECTION_STRING=your_blob_storage_connection
AZURE_OPENAI_ENDPOINT=your_openai_endpoint
AZURE_OPENAI_KEY=your_openai_key
AZURE_OPENAI_DEPLOYMENT=your_deployment_name
```

---

## 🏗️ 2. Project Structure  

```
cv-analyzer/
├── app.py              # Main Flask app
├── requirements.txt    # Dependencies
├── .env                # Environment variables
├── templates/          # HTML templates (upload/result pages)
├── uploads/            # Temporary resume storage, it will be created for you automatically for the first time
└── README.md           # Documentation
```

---

## ▶️ 3. Run the App  
```bash
python app.py
```

![image](https://github.com/user-attachments/assets/d7ae3785-0405-4c7e-8ed2-46aaa53fe81d)

Visit `http://127.0.0.1:5000/` to access the **upload page**.

---

## 📌 4. How It Works  
1. **Upload a resume** (PDF, DOCX, PNG, JPG).  
2. **Enter job description** and submit by clicking on **Analyze Resume**.

![image](https://github.com/user-attachments/assets/efd04c9f-f56e-4427-8736-8750a98bee36)

4. **AI processing**:  
   - Extracts text & sections (Work Experience, Skills, Education).  
   - Compares with job description using NLP.  
   - Provides AI-powered **feedback** for improvement.  
5. **Results displayed** with similarity score & feedback.  
6. **Qualified resumes are stored** in **Azure Blob Storage**.  

![image](https://github.com/user-attachments/assets/2672c6e0-46f2-43fa-be4f-90021d0bed96)

---

## 🔥 5. Future Enhancements  
- **Advanced OCR** for scanned documents.  
- **NLP-based section extraction** (using spaCy/NLTK).  
- **User authentication** for HR teams.  
- **Dashboard** for resume insights & analytics.  

---

## 🤝 6. Contributing  
Want to improve the app? Feel free to **submit a PR** or open an **issue** on GitHub!  

📌 **Medium:** [Details](https://medium.com/@Curious4Tech/cv-analyzer-ai-powered-resume-screening-for-smarter-hiring-8d3d29ddaf7d)

🚀 Happy coding!
