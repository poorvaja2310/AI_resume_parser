# import os
# import json
# import re
# from pathlib import Path
# import tempfile
# import pandas as pd
# import streamlit as st
# from streamlit_option_menu import option_menu

# # Import the enhanced ResumeParser
# from resume_parser import ResumeParser

# st.set_page_config(page_title="Resume Parser & Ranker", page_icon="📄", layout="wide")

# # ---------------- UI helpers ---------------- #
# st.markdown("""
# <style>
# .metric-card{background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);padding:1rem;border-radius:10px;color:#fff;text-align:center}
# .candidate-card{border:1px solid #ddd;border-radius:10px;padding:1rem;margin:1rem 0;background:#f8f9fa}
# .top-candidate{border:2px solid #28a745;background:#d4edda}
# .low-match{border:2px solid #ff4b4b;background:#ffe6e6}
# </style>
# """, unsafe_allow_html=True)

# def header():
#     st.markdown('<h1 style="text-align:center;margin:0 0 1rem 0">🎯 AI Resume Parser & Candidate Ranker</h1>', unsafe_allow_html=True)

# def metrics_box(title: str, value: str):
#     st.markdown(f'<div class="metric-card"><h3>{value}</h3><p>{title}</p></div>', unsafe_allow_html=True)

# # ---------------- Session ---------------- #
# if "parsed" not in st.session_state:
#     st.session_state.parsed = []
# if "rankings" not in st.session_state:
#     st.session_state.rankings = []
# if "jd_text" not in st.session_state:
#     st.session_state.jd_text = ""
# if "llm_available" not in st.session_state:
#     st.session_state.llm_available = False

# # ---------------- Nav ---------------- #
# header()
# selected = option_menu(
#     None,
#     ["Upload & Parse", "View Results", "Candidate Ranking"],
#     icons=["cloud-upload", "table", "trophy"],
#     orientation="horizontal",
#     default_index=0
# )

# # ---------------- Pages ---------------- #
# if selected == "Upload & Parse":
#     col1, col2 = st.columns([2,1])

#     with col1:
#         up = st.file_uploader(
#             "Upload resumes (PDF, DOCX, TXT, or ZIP folder)",
#             type=["pdf","docx","txt","zip"],
#             accept_multiple_files=True
#         )
        
#         # LLM configuration
#         llm_enabled = st.checkbox("Enable LLM Enhancement", value=True, 
#                                  help="Use TinyLLama for improved parsing accuracy")
        
#         model_path = st.text_input(
#             "TinyLLama model path",
#             value=r"C:\Users\Hp\Downloads\tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
#             help="Path to your TinyLLama .gguf model file"
#         )
        
#         # Check if model exists
#         model_exists = os.path.exists(model_path) if model_path else False
#         if model_path and not model_exists:
#             st.warning("⚠️ Model file not found at the specified path. Using heuristic parsing only.")

#         if st.button("🚀 Parse Resumes", type="primary"):
#             if not up:
#                 st.warning("Upload at least one file.")
#             else:
#                 # Initialize parser with or without LLM
#                 use_llm = llm_enabled and model_exists
#                 parser = ResumeParser(model_path=model_path if use_llm else None)
#                 st.session_state.llm_available = use_llm
                
#                 results = []
#                 prog = st.progress(0.0)

#                 # collect all uploaded files with their original names
#                 all_files = []  # This will store tuples: (temp_path, original_filename)
#                 for uf in up:
#                     suffix = Path(uf.name).suffix.lower()
#                     original_filename = uf.name

#                     if suffix == ".zip":
#                         # unzip into temp folder
#                         with tempfile.TemporaryDirectory() as tmpdir:
#                             zip_path = Path(tmpdir) / uf.name
#                             with open(zip_path, "wb") as f:
#                                 f.write(uf.getbuffer())

#                             import zipfile
#                             with zipfile.ZipFile(zip_path, "r") as zip_ref:
#                                 zip_ref.extractall(tmpdir)

#                             # collect resumes with their original names
#                             for f in Path(tmpdir).rglob("*"):
#                                 if f.suffix.lower() in [".pdf", ".docx", ".txt"]:
#                                     all_files.append((f, f.name))  # Store both path and original name
#                     else:
#                         with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
#                             tmp.write(uf.getbuffer())
#                             tmp_path = Path(tmp.name)
#                             all_files.append((tmp_path, original_filename))  # Store original filename

#                 # helper to shorten JSON
#                 def shorten_resume_json(parsed: dict) -> dict:
#                     # Handle skills - ensure they're strings
#                     skills = parsed.get("skills", [])
#                     if skills and skills != ["Not Available"]:
#                         if isinstance(skills[0], dict):
#                             skills = [s.get('name', '') for s in skills]
#                         skills = [s for s in skills if s != "Not Available"]
                    
#                     # Handle projects - ensure they're strings
#                     projects = parsed.get("projects", [])
#                     if projects and projects != ["Not Available"]:
#                         if isinstance(projects[0], dict):
#                             projects = [p.get('name', '') for p in projects]
#                         projects = [p for p in projects if p != "Not Available"]
                    
#                     # Handle experience
#                     experience = parsed.get("total_experience_years", 0)
#                     if experience == "Not Available":
#                         experience = 0
                    
#                     return {
#                         "name": parsed.get("name"),
#                         "email": parsed.get("email"),
#                         "phone": parsed.get("phone"),
#                         "linkedin": parsed.get("linkedin"),
#                         "github": parsed.get("github"),
#                         "education": parsed.get("education")[0] if parsed.get("education") and parsed.get("education") != ["Not Available"] else "Not Available",
#                         "summary": parsed.get("summary"),
#                         "skills": skills[:10] if skills else [],
#                         "projects": projects[:5] if projects else [],
#                         "total_experience": experience,
#                         "filename": parsed.get("filename")
#                     }

#                 # parse all collected resumes
#                 for i, (fpath, original_filename) in enumerate(all_files):
#                     try:
#                         parsed = parser.parse_resume(str(fpath))
#                         parsed["filename"] = original_filename  # Use the original filename instead of temp name

#                         short_json = shorten_resume_json(parsed)
#                         results.append(short_json)
#                     except Exception as e:
#                         st.error(f"Error parsing {original_filename}: {str(e)}")
#                         results.append({
#                             "filename": original_filename,
#                             "error": str(e),
#                             "name": "Error",
#                             "email": "N/A",
#                             "phone": "N/A",
#                             "skills": [],
#                             "projects": [],
#                             "total_experience": 0,
#                             "education": "N/A"
#                         })
#                     finally:
#                         try:
#                             os.unlink(fpath)
#                         except Exception:
#                             pass

#                     prog.progress((i+1)/len(all_files))

#                 st.session_state.parsed = results
#                 st.success(f"Parsed {len(results)} resumes ✅")
#                 if use_llm:
#                     st.info("✅ LLM enhancement was used for parsing")
#                 else:
#                     st.info("ℹ️ Using heuristic parsing only")

#     with col2:
#         st.info("""
#         **What gets extracted**
#         - Name, Email, Phone, LinkedIn, GitHub  
#         - Education (only highest), Skills (top 10), Projects (top 5), Summary  
#         - Total Experience (years)  

#         **LLM Enhancement Benefits:**
#         - Better name extraction
#         - Improved experience calculation
#         - More accurate skill identification
#         - Better section parsing

#         **Tip:** You can upload resumes individually **or** upload a folder as `.zip` (auto-unzipped).
#         """)

# # ---------------- View Results ---------------- #
# if selected == "View Results":
#     st.header("📂 Parsed Resume Results")

#     if not st.session_state.parsed:
#         st.warning("No resumes parsed yet. Please upload and parse resumes first.")
#     else:
#         results = st.session_state.parsed

#         for res in results:
#             if "error" in res:
#                 st.error(f"Error in {res.get('filename', 'Unknown')}: {res['error']}")
#                 continue
                
#             st.subheader(res.get("filename", "Unnamed Resume"))

#             st.write(f"**Name:** {res.get('name', 'N/A')}")
#             st.write(f"**Email:** {res.get('email', 'N/A')}")
#             st.write(f"**Phone:** {res.get('phone', 'N/A')}")
#             st.write(f"**Experience (Years):** {res.get('total_experience', 'N/A')}")

#             # Handle skills and projects
#             skills = res.get('skills', [])
#             projects = res.get('projects', [])

#             st.write(f"**Top Skills:** {', '.join(skills[:8]) if skills else 'N/A'}")
#             st.write(f"**Projects:** {', '.join(projects[:3]) if projects else 'N/A'}")

#             with st.expander("🔎 Full JSON"):
#                 st.json(res)

#             st.markdown("---")

#         # Bulk download
#         json_export = json.dumps(results, indent=2)
#         st.download_button(
#             "💾 Download All Results (JSON)",
#             json_export,
#             "parsed_resumes.json",
#             "application/json"
#         )

# # ---------------- Candidate Ranking ---------------- #
# elif selected == "Candidate Ranking":
#     st.header("🏆 Candidate Ranking")

#     jd_file = st.file_uploader("📂 Upload Job Description (txt/pdf)", type=["txt", "pdf"], key="jd_upload")
#     jd_text_manual = st.text_area("✍️ Or paste the Job Description here", placeholder="Paste JD here...", height=200)

#     jd_text = None
#     if jd_file:
#         if jd_file.name.endswith(".pdf"):
#             parser = ResumeParser()
#             jd_text = parser._extract_text_from_pdf(jd_file)
#         else:
#             jd_text = jd_file.read().decode("utf-8")
#         st.success("✅ Job Description uploaded!")
#     elif jd_text_manual.strip():
#         jd_text = jd_text_manual
#         st.success("✅ Job Description entered manually!")

#     if jd_text:
#         st.session_state.jd_text = jd_text
#         if "parsed" in st.session_state and st.session_state.parsed:
#             resumes = [r for r in st.session_state.parsed if "error" not in r]
            
#             if not resumes:
#                 st.warning("No valid resumes to rank. Please upload and parse resumes first.")
#                 st.stop()
                
#             # Extract JD domain to check for mismatches
#             jd_lower = jd_text.lower()
#             jd_domain = "other"
#             domain_keywords = {
#                 "software": ["software", "developer", "programming", "coding", "java", "python", "javascript", "react", "node", "backend", "frontend", "fullstack"],
#                 "data": ["data science", "analyst", "machine learning", "ai", "sql", "database", "big data", "analytics", "statistic"],
#                 "design": ["design", "ui", "ux", "graphic", "fashion", "creative", "photoshop", "illustrator", "figma", "adobe"],
#                 "business": ["business", "marketing", "sales", "manager", "finance", "account", "mba", "management", "strategy"],
#                 "engineering": ["engineer", "mechanical", "electrical", "civil", "manufacturing", "hardware", "cad", "solidworks"]
#             }
            
#             for domain, keywords in domain_keywords.items():
#                 if any(keyword in jd_lower for keyword in keywords):
#                     jd_domain = domain
#                     break

#             # ---------- Compute weighted score ----------
#             def calculate_weighted_score(resume: dict, jd_text: str, jd_domain: str):
#                 jd_lower = jd_text.lower()
#                 total_score = 0
#                 match_reasons = []
#                 warning_reasons = []

#                 # Skills matching (most important - 40%)
#                 skills_score = 0
#                 skills_list = resume.get('skills', [])
#                 if skills_list:
#                     # Count how many skills from resume are mentioned in JD
#                     matched_skills = []
#                     for skill in skills_list:
#                         if skill and isinstance(skill, str) and skill.lower() in jd_lower:
#                             matched_skills.append(skill)
                    
#                     if matched_skills:
#                         skills_score = min((len(matched_skills) / max(len(skills_list), 1)) * 40, 40)
#                         match_reasons.append(f"Skills match: {', '.join(matched_skills[:3])}")
#                     else:
#                         warning_reasons.append("No skills match JD requirements")
#                 else:
#                     warning_reasons.append("No skills listed in resume")

#                 # Check for domain mismatch (critical filter)
#                 resume_skills_text = ' '.join([str(s).lower() for s in skills_list]) if skills_list else ""
#                 domain_mismatch = False
                
#                 if jd_domain != "other":
#                     if jd_domain == "software" and not any(keyword in resume_skills_text for keyword in domain_keywords["software"]):
#                         domain_mismatch = True
#                         warning_reasons.append("Domain mismatch: Software skills not found")
#                     elif jd_domain == "design" and not any(keyword in resume_skills_text for keyword in domain_keywords["design"]):
#                         domain_mismatch = True
#                         warning_reasons.append("Domain mismatch: Design skills not found")
#                     elif jd_domain == "data" and not any(keyword in resume_skills_text for keyword in domain_keywords["data"]):
#                         domain_mismatch = True
#                         warning_reasons.append("Domain mismatch: Data skills not found")
#                     elif jd_domain == "business" and not any(keyword in resume_skills_text for keyword in domain_keywords["business"]):
#                         domain_mismatch = True
#                         warning_reasons.append("Domain mismatch: Business skills not found")
#                     elif jd_domain == "engineering" and not any(keyword in resume_skills_text for keyword in domain_keywords["engineering"]):
#                         domain_mismatch = True
#                         warning_reasons.append("Domain mismatch: Engineering skills not found")

#                 # If domain mismatch, significantly reduce score
#                 if domain_mismatch:
#                     total_score = max(skills_score * 0.2, 10)  # Max 20% of skills score or minimum 10
#                     return total_score, match_reasons, warning_reasons

#                 # Education matching (20%)
#                 education_score = 0
#                 education = resume.get('education', '')
#                 if education and education != "Not Available":
#                     if isinstance(education, list):
#                         education_str = ' '.join(education).lower()
#                     else:
#                         education_str = str(education).lower()
                    
#                     # Check if education matches JD requirements
#                     education_terms = ['bachelor', 'master', 'phd', 'mba', 'degree', 'diploma', 'bs', 'ms', 'ba', 'ma']
#                     jd_education_terms = [term for term in education_terms if term in jd_lower]
                    
#                     if jd_education_terms and any(term in education_str for term in jd_education_terms):
#                         education_score = 20
#                         match_reasons.append("Education matches requirements")
#                     else:
#                         warning_reasons.append("Education doesn't match JD requirements")
#                 else:
#                     warning_reasons.append("No education information")

#                 # Experience matching (20%)
#                 experience_score = 0
#                 experience = resume.get('total_experience', 0)
                
#                 # Convert experience to number if it's a string
#                 if isinstance(experience, str):
#                     if experience == "Not Available":
#                         experience = 0
#                     else:
#                         # Extract numbers from string (e.g., "5 years" -> 5)
#                         exp_match = re.search(r'(\d+)', str(experience))
#                         experience = int(exp_match.group(1)) if exp_match else 0
                
#                 if experience and experience > 0:
#                     # Look for experience requirements in JD
#                     exp_patterns = [r'(\d+)[\s\-]*years?', r'(\d+)[\s\-]*yrs?', r'experience.*(\d+)', r'(\d+)\+.*years?']
#                     jd_experience = 0
                    
#                     for pattern in exp_patterns:
#                         match = re.search(pattern, jd_lower)
#                         if match:
#                             jd_experience = int(match.group(1))
#                             break
                    
#                     if jd_experience > 0:
#                         if experience >= jd_experience:
#                             experience_score = 20
#                             match_reasons.append(f"Experience: {experience}+ years (meets {jd_experience}+ requirement)")
#                         else:
#                             experience_score = max(10 * (experience / jd_experience), 5)  # Partial credit
#                             warning_reasons.append(f"Experience: {experience} years (below {jd_experience}+ requirement)")
#                     else:
#                         # No specific experience requirement in JD, give partial credit
#                         experience_score = min(experience * 2, 20)  # 2 points per year up to 20
#                         if experience_score > 0:
#                             match_reasons.append(f"Experience: {experience} years")
#                 else:
#                     warning_reasons.append("No experience information")

#                 # Projects matching (20%)
#                 projects_score = 0
#                 projects_list = resume.get('projects', [])
#                 if projects_list:
#                     # Check if project keywords match JD
#                     project_keywords = ' '.join([str(p).lower() for p in projects_list])
#                     jd_project_terms = ['project', 'development', 'design', 'implementation', 'management', 'lead', 'build', 'create']
#                     matched_projects = [term for term in jd_project_terms if term in jd_lower and term in project_keywords]
                    
#                     if matched_projects:
#                         projects_score = 20
#                         match_reasons.append("Relevant project experience")
#                     else:
#                         # Give partial credit for having projects even if not explicitly matching
#                         projects_score = 10
#                         warning_reasons.append("Projects don't directly match JD keywords")
#                 else:
#                     warning_reasons.append("No projects listed")

#                 total_score = round(skills_score + education_score + experience_score + projects_score, 2)
#                 return min(total_score, 100), match_reasons, warning_reasons  # Cap at 100

#             # Rank candidates
#             ranked = []
#             for res in resumes:
#                 score, match_reasons, warning_reasons = calculate_weighted_score(res, jd_text, jd_domain)
#                 res_copy = res.copy()
#                 res_copy["score"] = score
#                 res_copy["match_reasons"] = match_reasons
#                 res_copy["warning_reasons"] = warning_reasons
#                 ranked.append(res_copy)

#             ranked.sort(key=lambda x: x["score"], reverse=True)

#             # Display JD for reference
#             with st.expander("📋 View Job Description"):
#                 st.text(jd_text[:1000] + "..." if len(jd_text) > 1000 else jd_text)
#                 st.write(f"**Detected Domain:** {jd_domain.upper()}")

#             # ---------- Tabular display ----------
#             st.subheader("📊 Candidate Ranking Table")
#             st.write(f"**Total Candidates:** {len(ranked)}")
#             st.write(f"**JD Domain:** {jd_domain.upper()}")

#             table_data = []
#             for i, r in enumerate(ranked, 1):
#                 skills = r.get('skills', [])
#                 score = r.get("score", 0)
#                 status = "✅ Good match" if score >= 70 else "⚠️ Partial match" if score >= 40 else "❌ Poor match"
                
#                 table_data.append({
#                     "Rank": i,
#                     "Name": r.get("name", "N/A"),
#                     "Experience": f"{r.get('total_experience', 'N/A')} yrs",
#                     "Education": r.get("education", "N/A")[:20] + "..." if r.get("education") and len(r.get("education")) > 20 else r.get("education", "N/A"),
#                     "Top Skills": ", ".join([str(s) for s in skills[:2]]) if skills else "N/A",
#                     "Match Score": f"{score:.1f}/100",
#                     "Status": status
#                 })

#             df = pd.DataFrame(table_data)
#             st.dataframe(df, use_container_width=True)

#             # Show top candidates with match analysis
#             st.subheader("🏅 Top Candidates with Match Analysis")
#             for i, candidate in enumerate(ranked[:10], 1):
#                 score = candidate.get("score", 0)
#                 match_reasons = candidate.get("match_reasons", [])
#                 warning_reasons = candidate.get("warning_reasons", [])
                
#                 # Create a container with appropriate styling based on score
#                 if score < 30:
#                     st.markdown(f'<div class="low-match">', unsafe_allow_html=True)
#                     st.markdown(f"### #{i} - {candidate.get('name', 'N/A')} - Score: {score:.1f}/100")
#                     st.markdown('</div>', unsafe_allow_html=True)
#                 elif score >= 70:
#                     st.markdown(f'<div class="top-candidate">', unsafe_allow_html=True)
#                     st.markdown(f"### #{i} - {candidate.get('name', 'N/A')} - Score: {score:.1f}/100")
#                     st.markdown('</div>', unsafe_allow_html=True)
#                 else:
#                     st.markdown(f"### #{i} - {candidate.get('name', 'N/A')} - Score: {score:.1f}/100")
                
#                 col1, col2 = st.columns(2)
#                 with col1:
#                     st.write(f"**Email:** {candidate.get('email', 'N/A')}")
#                     st.write(f"**Experience:** {candidate.get('total_experience', 'N/A')} years")
#                     st.write(f"**Education:** {candidate.get('education', 'N/A')}")
#                 with col2:
#                     skills = candidate.get('skills', [])
#                     st.write(f"**Skills:** {', '.join([str(s) for s in skills[:5]]) if skills else 'N/A'}")
                
#                 if match_reasons:
#                     st.write("**✅ Strengths:**")
#                     for reason in match_reasons:
#                         st.write(f"- {reason}")
                
#                 if warning_reasons:
#                     st.write("**⚠️ Areas of Concern:**")
#                     for reason in warning_reasons:
#                         st.write(f"- {reason}")
                
#                 st.markdown("---")

#             # Show statistics
#             good_matches = sum(1 for r in ranked if r.get("score", 0) >= 70)
#             partial_matches = sum(1 for r in ranked if 40 <= r.get("score", 0) < 70)
#             poor_matches = sum(1 for r in ranked if r.get("score", 0) < 40)
            
#             st.subheader("📈 Match Statistics")
#             col1, col2, col3 = st.columns(3)
#             with col1:
#                 metrics_box("Good Matches (≥70%)", str(good_matches))
#             with col2:
#                 metrics_box("Partial Matches (40-69%)", str(partial_matches))
#             with col3:
#                 metrics_box("Poor Matches (<40%)", str(poor_matches))

#             # Add download option for rankings
#             ranking_data = []
#             for r in ranked:
#                 ranking_data.append({
#                     "Rank": ranked.index(r) + 1,
#                     "Name": r.get("name", "N/A"),
#                     "Email": r.get("email", "N/A"),
#                     "Score": r.get("score", 0),
#                     "Experience": r.get("total_experience", "N/A"),
#                     "Education": r.get("education", "N/A"),
#                     "Skills": ", ".join([str(s) for s in r.get('skills', [])[:5]]),
#                     "Strengths": "; ".join(r.get("match_reasons", [])),
#                     "Concerns": "; ".join(r.get("warning_reasons", []))
#                 })
            
#             df_export = pd.DataFrame(ranking_data)
#             csv = df_export.to_csv(index=False)
#             st.download_button(
#                 "📥 Download Ranking Results (CSV)",
#                 csv,
#                 "candidate_rankings.csv",
#                 "text/csv"
#             )

#         else:
#             st.warning("⚠️ Please upload resumes first.")
#     else:
#         st.info("📂 Upload a JD file or paste it above to rank candidates.")
import os
import json
import re
from pathlib import Path
import tempfile
import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np

# Import the enhanced ResumeParser
from resume_parser import ResumeParser

st.set_page_config(page_title="Resume Parser & Ranker", page_icon="📄", layout="wide")

# ---------------- UI helpers ---------------- #
st.markdown("""
<style>
.metric-card{background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);padding:1rem;border-radius:10px;color:#fff;text-align:center}
.candidate-card{border:1px solid #ddd;border-radius:10px;padding:1rem;margin:1rem 0;background:#f8f9fa}
.top-candidate{border:2px solid #28a745;background:#d4edda}
.low-match{border:2px solid #ff4b4b;background:#ffe6e6}
.match-reason {color: #28a745; font-weight: 500;}
.warning-reason {color: #ff4b4b; font-weight: 500;}
.llm-loading {background: #fff3cd; border: 1px solid #ffeaa7; border-radius: 5px; padding: 10px; margin: 10px 0;}
</style>
""", unsafe_allow_html=True)

def header():
    st.markdown('<h1 style="text-align:center;margin:0 0 1rem 0">🎯 AI Resume Parser & Candidate Ranker</h1>', unsafe_allow_html=True)

def metrics_box(title: str, value: str):
    st.markdown(f'<div class="metric-card"><h3>{value}</h3><p>{title}</p></div>', unsafe_allow_html=True)

# ---------------- Session ---------------- #
if "parsed" not in st.session_state:
    st.session_state.parsed = []
if "rankings" not in st.session_state:
    st.session_state.rankings = []
if "jd_text" not in st.session_state:
    st.session_state.jd_text = ""
if "llm_available" not in st.session_state:
    st.session_state.llm_available = False
if "llm_model" not in st.session_state:
    st.session_state.llm_model = None
if "llm_initialized" not in st.session_state:
    st.session_state.llm_initialized = False

# ---------------- LLM Ranking Functions ---------------- #
def initialize_llm(model_path: str):
    """Initialize the LLM for ranking"""
    try:
        from llama_cpp import Llama
        with st.spinner("🔄 Loading AI model... This may take a minute."):
            llm = Llama(
                model_path=model_path,
                n_ctx=4096,  # Larger context for ranking
                n_threads=6,
                n_gpu_layers=0,
                verbose=False
            )
        st.session_state.llm_model = llm
        st.session_state.llm_initialized = True
        return True
    except Exception as e:
        st.error(f"Failed to initialize LLM: {e}")
        return False

def llm_rank_candidate(resume_data: dict, jd_text: str, llm) -> dict:
    """Use LLM to analyze and score a candidate against JD"""
    
    # Prepare resume text for LLM with better formatting
    skills_list = resume_data.get('skills', [])
    projects_list = resume_data.get('projects', [])
    
    resume_text = f"""
CANDIDATE PROFILE:
Name: {resume_data.get('name', 'Not specified')}
Email: {resume_data.get('email', 'Not specified')}
Experience: {resume_data.get('total_experience', 0)} years
Education: {resume_data.get('education', 'Not specified')}

TECHNICAL SKILLS:
{', '.join(skills_list) if skills_list else 'No skills listed'}

PROJECT EXPERIENCE:
{', '.join(projects_list) if projects_list else 'No projects listed'}

PROFESSIONAL SUMMARY:
{resume_data.get('summary', 'No summary available')}
"""
    
    prompt = f"""
JOB DESCRIPTION FOR DATA SCIENTIST ROLE:
{jd_text[:1200]}

CANDIDATE RESUME INFORMATION:
{resume_text[:1000]}

ANALYSIS REQUESTED:
As an expert HR recruiter, analyze this candidate's fit for the Data Scientist position. Focus on:
1. Technical skills match (Python, ML, Data Science tools)
2. Experience level vs requirements
3. Education background relevance
4. Project portfolio quality

Provide a score 0-100 and specific, actionable insights.

RETURN FORMAT (JSON only):
{{
  "score": 75,
  "strengths": ["Specific strength 1", "Specific strength 2", "Specific strength 3"],
  "weaknesses": ["Specific area for improvement 1", "Specific gap 2"],
  "overall_assessment": "Concise summary of candidate fit"
}}

Be specific about skills, experience, and education matches/mismatches.
"""
    
    try:
        response = llm(
            prompt,
            max_tokens=1200,
            temperature=0.2,
            stop=["</s>", "```"],
            echo=False
        )
        
        if isinstance(response, dict) and "choices" in response:
            response_text = response['choices'][0].get('text', '').strip()
        else:
            response_text = str(response).strip()
        
        # Clean the response text
        response_text = response_text.replace('```json', '').replace('```', '').strip()
        
        # Extract JSON from response
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            try:
                result = json.loads(json_str)
                
                # Validate and sanitize the response
                score = min(max(float(result.get('score', 50)), 0), 100)
                
                # Get strengths with fallbacks
                strengths = result.get('strengths', [])
                if not strengths or (isinstance(strengths, list) and len(strengths) == 0):
                    strengths = self_generate_strengths(resume_data, jd_text)
                elif isinstance(strengths, str):
                    strengths = [strengths]
                
                # Get weaknesses with fallbacks
                weaknesses = result.get('weaknesses', [])
                if not weaknesses or (isinstance(weaknesses, list) and len(weaknesses) == 0):
                    weaknesses = self_generate_weaknesses(resume_data, jd_text)
                elif isinstance(weaknesses, str):
                    weaknesses = [weaknesses]
                
                # Get assessment with fallback
                assessment = result.get('overall_assessment', '')
                if not assessment:
                    assessment = generate_assessment(resume_data, score)
                
                return {
                    'score': score,
                    'match_reasons': strengths[:3],
                    'warning_reasons': weaknesses[:2],
                    'assessment': assessment
                }
                
            except json.JSONDecodeError:
                # Fallback to heuristic scoring if JSON parsing fails
                return generate_heuristic_analysis(resume_data, jd_text)
        
        # Fallback if no JSON found
        return generate_heuristic_analysis(resume_data, jd_text)
        
    except Exception as e:
        print(f"LLM ranking error: {e}")
        return generate_heuristic_analysis(resume_data, jd_text)

def self_generate_strengths(resume_data: dict, jd_text: str) -> list:
    """Generate strengths based on resume content"""
    strengths = []
    skills = resume_data.get('skills', [])
    experience = resume_data.get('total_experience', 0)
    
    # Check for technical skills
    tech_skills = ['python', 'machine learning', 'data science', 'sql', 'tensorflow', 'pytorch', 'keras']
    found_skills = [skill for skill in skills if any(tech in str(skill).lower() for tech in tech_skills)]
    
    if found_skills:
        strengths.append(f"Strong technical skills in {', '.join(found_skills[:3])}")
    
    # Check experience
    if experience > 2:
        strengths.append(f"Substantial professional experience ({experience} years)")
    elif experience > 0:
        strengths.append(f"Some professional experience ({experience} years)")
    
    # Check education
    education = str(resume_data.get('education', '')).lower()
    if any(degree in education for degree in ['bachelor', 'master', 'phd', 'ms', 'bs']):
        strengths.append("Relevant educational background")
    
    if not strengths:
        strengths.append("Potential candidate with basic qualifications")
    
    return strengths[:3]

def self_generate_weaknesses(resume_data: dict, jd_text: str) -> list:
    """Generate weaknesses based on resume content"""
    weaknesses = []
    skills = resume_data.get('skills', [])
    experience = resume_data.get('total_experience', 0)
    
    # Check JD for required skills
    jd_lower = jd_text.lower()
    required_skills = []
    if 'python' in jd_lower:
        required_skills.append('Python')
    if 'machine learning' in jd_lower:
        required_skills.append('Machine Learning')
    if 'sql' in jd_lower:
        required_skills.append('SQL')
    if 'data science' in jd_lower:
        required_skills.append('Data Science')
    
    # Find missing required skills
    missing_skills = []
    for req_skill in required_skills:
        if not any(req_skill.lower() in str(skill).lower() for skill in skills):
            missing_skills.append(req_skill)
    
    if missing_skills:
        weaknesses.append(f"Missing key skills: {', '.join(missing_skills[:2])}")
    
    # Check experience
    if experience == 0:
        weaknesses.append("No professional experience indicated")
    elif experience < 2:
        weaknesses.append("Limited professional experience")
    
    if not weaknesses:
        weaknesses.append("Some skill gaps may need addressing")
    
    return weaknesses[:2]

def generate_assessment(resume_data: dict, score: float) -> str:
    """Generate assessment based on score"""
    if score >= 80:
        return "Strong candidate with excellent skills match for the position"
    elif score >= 60:
        return "Good candidate with solid qualifications and relevant experience"
    elif score >= 40:
        return "Moderate fit with some relevant skills but may need additional training"
    else:
        return "Limited fit for the current role based on available information"

def generate_heuristic_analysis(resume_data: dict, jd_text: str) -> dict:
    """Generate analysis using heuristic rules when LLM fails"""
    # Calculate heuristic score based on skills match
    skills = resume_data.get('skills', [])
    experience = resume_data.get('total_experience', 0)
    
    # Basic scoring
    score = 50  # Base score
    
    # Add points for experience
    if experience > 5:
        score += 20
    elif experience > 2:
        score += 10
    elif experience > 0:
        score += 5
    
    # Add points for relevant skills
    relevant_skills = ['python', 'machine learning', 'data science', 'sql', 'tensorflow', 'pytorch']
    found_skills = sum(1 for skill in skills if any(rs in str(skill).lower() for rs in relevant_skills))
    score += min(found_skills * 5, 25)
    
    score = min(max(score, 0), 100)
    
    return {
        'score': score,
        'match_reasons': self_generate_strengths(resume_data, jd_text),
        'warning_reasons': self_generate_weaknesses(resume_data, jd_text),
        'assessment': generate_assessment(resume_data, score)
    }

def batch_llm_ranking(resumes: list, jd_text: str, llm) -> list:
    """Rank all candidates using LLM"""
    ranked_candidates = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, resume in enumerate(resumes):
        status_text.text(f"Analyzing candidate {i+1}/{len(resumes)}: {resume.get('name', 'Unknown')}")
        progress_bar.progress((i + 1) / len(resumes))
        
        ranking_result = llm_rank_candidate(resume, jd_text, llm)
        
        candidate_data = resume.copy()
        candidate_data.update(ranking_result)
        ranked_candidates.append(candidate_data)
    
    progress_bar.empty()
    status_text.empty()
    
    # Sort by score descending
    ranked_candidates.sort(key=lambda x: x['score'], reverse=True)
    return ranked_candidates

# ---------------- Nav ---------------- #
header()
selected = option_menu(
    None,
    ["Upload & Parse", "View Results", "Candidate Ranking"],
    icons=["cloud-upload", "table", "trophy"],
    orientation="horizontal",
    default_index=0
)

# ---------------- Pages ---------------- #
if selected == "Upload & Parse":
    col1, col2 = st.columns([2,1])

    with col1:
        up = st.file_uploader(
            "Upload resumes (PDF, DOCX, TXT, or ZIP folder)",
            type=["pdf","docx","txt","zip"],
            accept_multiple_files=True
        )
        
        # LLM configuration
        llm_enabled = st.checkbox("Enable LLM Enhancement", value=True, 
                                 help="Use TinyLLama for improved parsing accuracy")
        
        model_path = st.text_input(
            "TinyLLama model path",
            value=r"C:\Users\Hp\Downloads\tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
            help="Path to your TinyLLama .gguf model file"
        )
        
        # Check if model exists
        model_exists = os.path.exists(model_path) if model_path else False
        if model_path and not model_exists:
            st.warning("⚠️ Model file not found at the specified path. Using heuristic parsing only.")

        if st.button("🚀 Parse Resumes", type="primary"):
            if not up:
                st.warning("Upload at least one file.")
            else:
                # Initialize parser with or without LLM
                use_llm = llm_enabled and model_exists
                parser = ResumeParser(model_path=model_path if use_llm else None)
                st.session_state.llm_available = use_llm
                
                results = []
                prog = st.progress(0.0)

                # collect all uploaded files with their original names
                all_files = []  # This will store tuples: (temp_path, original_filename)
                for uf in up:
                    suffix = Path(uf.name).suffix.lower()
                    original_filename = uf.name

                    if suffix == ".zip":
                        # unzip into temp folder
                        with tempfile.TemporaryDirectory() as tmpdir:
                            zip_path = Path(tmpdir) / uf.name
                            with open(zip_path, "wb") as f:
                                f.write(uf.getbuffer())

                            import zipfile
                            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                                zip_ref.extractall(tmpdir)

                            # collect resumes with their original names
                            for f in Path(tmpdir).rglob("*"):
                                if f.suffix.lower() in [".pdf", ".docx", ".txt"]:
                                    all_files.append((f, f.name))  # Store both path and original name
                    else:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                            tmp.write(uf.getbuffer())
                            tmp_path = Path(tmp.name)
                            all_files.append((tmp_path, original_filename))  # Store original filename

                # helper to shorten JSON
                def shorten_resume_json(parsed: dict) -> dict:
                    # Handle skills - ensure they're strings
                    skills = parsed.get("skills", [])
                    if skills and skills != ["Not Available"]:
                        if isinstance(skills[0], dict):
                            skills = [s.get('name', '') for s in skills]
                        skills = [s for s in skills if s != "Not Available"]
                    
                    # Handle projects - ensure they're strings
                    projects = parsed.get("projects", [])
                    if projects and projects != ["Not Available"]:
                        if isinstance(projects[0], dict):
                            projects = [p.get('name', '') for p in projects]
                        projects = [p for p in projects if p != "Not Available"]
                    
                    # Handle experience - fix parsing
                    experience = parsed.get("total_experience_years", 0)
                    if experience == "Not Available" or not isinstance(experience, (int, float)):
                        experience = 0
                    elif isinstance(experience, str):
                        # Extract numbers from strings like "5 years"
                        exp_match = re.search(r'(\d+)', experience)
                        experience = int(exp_match.group(1)) if exp_match else 0
                    
                    # Handle education - clean up
                    education = parsed.get("education", "Not Available")
                    if isinstance(education, list) and education:
                        education = education[0]
                    if education == "Not Available":
                        education = "Education not specified"
                    
                    return {
                        "name": parsed.get("name", "Not specified"),
                        "email": parsed.get("email", "Not specified"),
                        "phone": parsed.get("phone", "Not specified"),
                        "linkedin": parsed.get("linkedin", "Not specified"),
                        "github": parsed.get("github", "Not specified"),
                        "education": education,
                        "summary": parsed.get("summary", "No summary available"),
                        "skills": skills[:10] if skills else [],
                        "projects": projects[:5] if projects else [],
                        "total_experience": experience,
                        "filename": parsed.get("filename", "Unknown")
                    }

                # parse all collected resumes
                for i, (fpath, original_filename) in enumerate(all_files):
                    try:
                        parsed = parser.parse_resume(str(fpath))
                        parsed["filename"] = original_filename  # Use the original filename instead of temp name

                        short_json = shorten_resume_json(parsed)
                        results.append(short_json)
                    except Exception as e:
                        st.error(f"Error parsing {original_filename}: {str(e)}")
                        results.append({
                            "filename": original_filename,
                            "error": str(e),
                            "name": "Error",
                            "email": "N/A",
                            "phone": "N/A",
                            "skills": [],
                            "projects": [],
                            "total_experience": 0,
                            "education": "N/A"
                        })
                    finally:
                        try:
                            os.unlink(fpath)
                        except Exception:
                            pass

                    prog.progress((i+1)/len(all_files))

                st.session_state.parsed = results
                st.success(f"Parsed {len(results)} resumes ✅")
                if use_llm:
                    st.info("✅ LLM enhancement was used for parsing")
                else:
                    st.info("ℹ️ Using heuristic parsing only")

    with col2:
        st.info("""
        **What gets extracted**
        - Name, Email, Phone, LinkedIn, GitHub  
        - Education (only highest), Skills (top 10), Projects (top 5), Summary  
        - Total Experience (years)  

        **LLM Enhancement Benefits:**
        - Better name extraction
        - Improved experience calculation
        - More accurate skill identification
        - Better section parsing

        **Tip:** You can upload resumes individually **or** upload a folder as `.zip` (auto-unzipped).
        """)

# ---------------- View Results ---------------- #
if selected == "View Results":
    st.header("📂 Parsed Resume Results")

    if not st.session_state.parsed:
        st.warning("No resumes parsed yet. Please upload and parse resumes first.")
    else:
        results = st.session_state.parsed

        for res in results:
            if "error" in res:
                st.error(f"Error in {res.get('filename', 'Unknown')}: {res['error']}")
                continue
                
            st.subheader(res.get("filename", "Unnamed Resume"))

            st.write(f"**Name:** {res.get('name', 'N/A')}")
            st.write(f"**Email:** {res.get('email', 'N/A')}")
            st.write(f"**Phone:** {res.get('phone', 'N/A')}")
            st.write(f"**Experience (Years):** {res.get('total_experience', 'N/A')}")

            # Handle skills and projects
            skills = res.get('skills', [])
            projects = res.get('projects', [])

            st.write(f"**Top Skills:** {', '.join(skills[:8]) if skills else 'N/A'}")
            st.write(f"**Projects:** {', '.join(projects[:3]) if projects else 'N/A'}")

            with st.expander("🔎 Full JSON"):
                st.json(res)

            st.markdown("---")

        # Bulk download
        json_export = json.dumps(results, indent=2)
        st.download_button(
            "💾 Download All Results (JSON)",
            json_export,
            "parsed_resumes.json",
            "application/json"
        )

# ---------------- Candidate Ranking ---------------- #
elif selected == "Candidate Ranking":
    st.header("🏆 AI-Powered Candidate Ranking")
    
    # LLM Model initialization for ranking
    if not st.session_state.llm_initialized:
        st.markdown("""
        <div class="llm-loading">
        <h4>🔑 AI Ranker Setup Required</h4>
        <p>Please provide the path to your TinyLlama model file to enable AI-powered ranking.</p>
        </div>
        """, unsafe_allow_html=True)
        
        model_path = st.text_input(
            "Path to TinyLLama model for ranking",
            value=r"C:\Users\Hp\Downloads\tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
            help="Path to your TinyLLama .gguf model file for AI ranking"
        )
        
        if st.button("🚀 Initialize AI Ranker", type="primary"):
            if model_path and os.path.exists(model_path):
                if initialize_llm(model_path):
                    st.success("✅ AI Ranker initialized successfully!")
                    st.session_state.llm_initialized = True
                else:
                    st.error("❌ Failed to initialize AI Ranker")
            else:
                st.error("❌ Model file not found. Please provide a valid path.")
    
    jd_file = st.file_uploader("📂 Upload Job Description (txt/pdf)", type=["txt", "pdf"], key="jd_upload")
    jd_text_manual = st.text_area("✍️ Or paste the Job Description here", placeholder="Paste JD here...", height=200)

    jd_text = None
    if jd_file:
        if jd_file.name.endswith(".pdf"):
            parser = ResumeParser()
            jd_text = parser._extract_text_from_pdf(jd_file)
        else:
            jd_text = jd_file.read().decode("utf-8")
        st.success("✅ Job Description uploaded!")
    elif jd_text_manual.strip():
        jd_text = jd_text_manual
        st.success("✅ Job Description entered manually!")

    if jd_text and st.session_state.llm_initialized:
        st.session_state.jd_text = jd_text
        if "parsed" in st.session_state and st.session_state.parsed:
            resumes = [r for r in st.session_state.parsed if "error" not in r]
            
            if not resumes:
                st.warning("No valid resumes to rank. Please upload and parse resumes first.")
                st.stop()
            
            if st.button("🧠 Analyze Candidates with AI", type="primary"):
                with st.spinner("🤖 AI is analyzing candidates. This may take a few minutes..."):
                    ranked_candidates = batch_llm_ranking(resumes, jd_text, st.session_state.llm_model)
                    st.session_state.rankings = ranked_candidates
                    st.success(f"✅ AI analysis complete! Ranked {len(ranked_candidates)} candidates.")

        # Display ranking results if available
        if st.session_state.rankings:
            ranked = st.session_state.rankings
            
            # Display JD for reference
            with st.expander("📋 View Job Description"):
                st.text(jd_text[:1000] + "..." if len(jd_text) > 1000 else jd_text)

            # ---------- Tabular display ----------
            st.subheader("📊 AI-Powered Candidate Ranking")
            st.write(f"**Total Candidates:** {len(ranked)}")

            table_data = []
            for i, r in enumerate(ranked, 1):
                skills = r.get('skills', [])
                score = r.get("score", 0)
                status = "✅ Top Match" if score >= 80 else "⚠️ Good Match" if score >= 60 else "🔶 Partial Match" if score >= 40 else "❌ Poor Match"
                
                table_data.append({
                    "Rank": i,
                    "Name": r.get("name", "N/A"),
                    "Experience": f"{r.get('total_experience', 'N/A')} yrs",
                    "AI Score": f"{score:.1f}/100",
                    "Status": status,
                    "Top Skills": ", ".join([str(s) for s in skills[:2]]) if skills else "N/A"
                })

            df = pd.DataFrame(table_data)
            st.dataframe(df, use_container_width=True)

            # Show top candidates with AI analysis
            st.subheader("🏅 AI Analysis of Top Candidates")
            for i, candidate in enumerate(ranked[:8], 1):
                score = candidate.get("score", 0)
                match_reasons = candidate.get("match_reasons", [])
                warning_reasons = candidate.get("warning_reasons", [])
                assessment = candidate.get("assessment", "")
                
                # Create a container with appropriate styling based on score
                if score >= 80:
                    st.markdown(f'<div class="top-candidate">', unsafe_allow_html=True)
                    st.markdown(f"### 🥇 #{i} - {candidate.get('name', 'N/A')} - AI Score: {score:.1f}/100")
                    st.markdown('</div>', unsafe_allow_html=True)
                elif score >= 60:
                    st.markdown(f"### 🥈 #{i} - {candidate.get('name', 'N/A')} - AI Score: {score:.1f}/100")
                elif score >= 40:
                    st.markdown(f"### 🥉 #{i} - {candidate.get('name', 'N/A')} - AI Score: {score:.1f}/100")
                else:
                    st.markdown(f'<div class="low-match">', unsafe_allow_html=True)
                    st.markdown(f"### #{i} - {candidate.get('name', 'N/A')} - AI Score: {score:.1f}/100")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Basic info
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Email:** {candidate.get('email', 'N/A')}")
                    st.write(f"**Experience:** {candidate.get('total_experience', 'N/A')} years")
                with col2:
                    st.write(f"**Education:** {candidate.get('education', 'N/A')}")
                    skills = candidate.get('skills', [])
                    st.write(f"**Skills:** {', '.join([str(s) for s in skills[:5]]) if skills else 'N/A'}")
                
                # AI Analysis
                st.write("**🤖 AI Assessment:**")
                st.info(assessment)
                
                if match_reasons:
                    st.write("**✅ AI-Identified Strengths:**")
                    for reason in match_reasons:
                        st.markdown(f'<p class="match-reason">✓ {reason}</p>', unsafe_allow_html=True)
                
                if warning_reasons:
                    st.write("**⚠️ AI-Identified Concerns:**")
                    for reason in warning_reasons:
                        st.markdown(f'<p class="warning-reason">✗ {reason}</p>', unsafe_allow_html=True)
                
                st.markdown("---")

            # Show statistics
            top_matches = sum(1 for r in ranked if r.get("score", 0) >= 80)
            good_matches = sum(1 for r in ranked if 60 <= r.get("score", 0) < 80)
            partial_matches = sum(1 for r in ranked if 40 <= r.get("score", 0) < 60)
            poor_matches = sum(1 for r in ranked if r.get("score", 0) < 40)
            
            st.subheader("📈 AI Match Statistics")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                metrics_box("Top Matches (≥80%)", str(top_matches))
            with col2:
                metrics_box("Good Matches (60-79%)", str(good_matches))
            with col3:
                metrics_box("Partial Matches (40-59%)", str(partial_matches))
            with col4:
                metrics_box("Poor Matches (<40%)", str(poor_matches))

            # Add download option for rankings
            ranking_data = []
            for i, r in enumerate(ranked, 1):
                ranking_data.append({
                    "Rank": i,
                    "Name": r.get("name", "N/A"),
                    "Email": r.get("email", "N/A"),
                    "AI Score": r.get("score", 0),
                    "Experience": r.get("total_experience", "N/A"),
                    "Education": r.get("education", "N/A"),
                    "Skills": ", ".join([str(s) for s in r.get('skills', [])[:5]]),
                    "Strengths": "; ".join(r.get("match_reasons", [])),
                    "Concerns": "; ".join(r.get("warning_reasons", [])),
                    "AI Assessment": r.get("assessment", "")
                })
            
            df_export = pd.DataFrame(ranking_data)
            csv = df_export.to_csv(index=False)
            st.download_button(
                "📥 Download AI Ranking Results (CSV)",
                csv,
                "ai_candidate_rankings.csv",
                "text/csv"
            )

    elif jd_text and not st.session_state.llm_initialized:
        st.warning("⚠️ Please initialize the AI Ranker first by providing the model path and clicking 'Initialize AI Ranker'.")
    else:
        st.info("📂 Upload a JD file or paste it above to rank candidates.")