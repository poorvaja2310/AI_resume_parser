# Resume Parser

A fast and accurate Python-based resume parser that extracts structured data from PDF, DOCX, and TXT resume files. This parser uses advanced pattern recognition and optional LLM enhancement to provide high-quality data extraction.

## Features

- **Multi-format Support**: Parse PDF, DOCX, and TXT files
- **Fast Processing**: Optimized for speed with precompiled regex patterns
- **Accurate Extraction**: Enhanced algorithms for better data accuracy
- **Comprehensive Data**: Extracts name, contact info, skills, education, experience, and more
- **Optional LLM Enhancement**: Can use local LLM models for improved accuracy
- **Batch Processing**: Parse multiple resumes at once

## Extracted Data Fields

The parser extracts the following information:

- **Personal Information**: Name, email, phone number
- **Social Profiles**: LinkedIn and GitHub URLs
- **Skills**: Technical skills and technologies (80+ vocabulary terms)
- **Education**: Degrees, universities, and academic background
- **Experience**: Work history and professional experience
- **Certifications**: Professional certifications and licenses
- **Summary**: Professional summary or objective statement

## Installation

### Prerequisites

```bash
pip install PyMuPDF python-docx
```

### Optional LLM Support

For enhanced accuracy with local LLM models:

```bash
pip install llama-cpp-python
```

## Quick Start

### Basic Usage

```python
from resume_parser import ResumeParser

# Initialize parser (without LLM)
parser = ResumeParser()

# Parse a single resume
result = parser.parse_resume("path/to/resume.pdf")
print(result)

# Parse multiple resumes
file_paths = ["resume1.pdf", "resume2.docx", "resume3.txt"]
results = parser.parse_multiple_resumes(file_paths)
```

### With LLM Enhancement

```python
# Initialize parser with local LLM model
parser = ResumeParser(model_path="path/to/your/model.gguf")

# Parse resume with LLM enhancement
result = parser.parse_resume("path/to/resume.pdf")
```

## Example Output

```json
{
  "name": "John Smith",
  "email": "john.smith@email.com",
  "phone": "+1-555-123-4567",
  "linkedin": "https://linkedin.com/in/johnsmith",
  "github": "https://github.com/johnsmith",
  "skills": [
    "Python",
    "JavaScript",
    "React",
    "Django",
    "PostgreSQL",
    "AWS",
    "Docker"
  ],
  "education": [
    "Bachelor of Science in Computer Science, Stanford University, 2018-2022"
  ],
  "experience": [
    "Senior Software Engineer at Tech Corp, 2022-Present",
    "Software Developer at StartupXYZ, 2020-2022"
  ],
  "certifications": [
    "AWS Certified Solutions Architect Associate"
  ],
  "summary": "Experienced software engineer with 5+ years in full-stack web development...",
  "file_path": "path/to/resume.pdf",
  "text_length": 2847,
  "filename": "resume.pdf"
}
```

## Advanced Usage

### Custom Configuration

```python
# Initialize with custom LLM settings
parser = ResumeParser(
    model_path="path/to/model.gguf",
    llm_config={
        "n_ctx": 4096,
        "n_threads": 8,
        "temperature": 0.1
    }
)
```

### Batch Processing with Progress

```python
import os
from pathlib import Path

def parse_resume_folder(folder_path):
    parser = ResumeParser()
    resume_files = []
    
    # Find all resume files
    for ext in ['.pdf', '.docx', '.txt']:
        resume_files.extend(Path(folder_path).glob(f"*{ext}"))
    
    results = []
    for i, file_path in enumerate(resume_files, 1):
        print(f"Processing {i}/{len(resume_files)}: {file_path.name}")
        result = parser.parse_resume(str(file_path))
        results.append(result)
    
    return results

# Parse all resumes in a folder
results = parse_resume_folder("./resumes")
```

### Export to Different Formats

```python
import json
import csv

def export_to_json(results, output_file):
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

def export_to_csv(results, output_file):
    if not results:
        return
    
    fieldnames = results[0].keys()
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            # Convert lists to strings for CSV
            row = {}
            for k, v in result.items():
                if isinstance(v, list):
                    row[k] = '; '.join(v) if v != ["Not Available"] else "Not Available"
                else:
                    row[k] = v
            writer.writerow(row)

# Usage
results = parser.parse_multiple_resumes(file_paths)
export_to_json(results, "parsed_resumes.json")
export_to_csv(results, "parsed_resumes.csv")
```

## Supported File Formats

| Format | Extension | Notes |
|--------|-----------|-------|
| PDF | `.pdf` | Uses PyMuPDF for text extraction |
| Word Document | `.docx`, `.doc` | Uses python-docx library |
| Text File | `.txt` | Direct text reading with UTF-8 encoding |

## Skills Vocabulary

The parser recognizes 80+ technical skills including:

- **Programming Languages**: Python, Java, JavaScript, TypeScript, C++, Go, Rust, etc.
- **Web Technologies**: React, Angular, Vue.js, HTML, CSS, Node.js, etc.
- **Databases**: PostgreSQL, MongoDB, MySQL, Redis, etc.
- **Cloud & DevOps**: AWS, Azure, GCP, Docker, Kubernetes, etc.
- **Data Science**: TensorFlow, PyTorch, Pandas, NumPy, etc.

## Performance

- **Speed**: Processes most resumes in under 1 second
- **Accuracy**: Enhanced pattern recognition for better data extraction
- **Memory Efficient**: Optimized text processing and deduplication
- **Scalable**: Can handle batch processing of hundreds of resumes

## Error Handling

The parser includes comprehensive error handling:

- Graceful handling of corrupted or unreadable files
- Fallback parsing when primary methods fail
- Detailed error messages for debugging
- "Not Available" values for missing information

## Troubleshooting

### Common Issues

1. **PDF Text Extraction Fails**
   - Ensure the PDF contains selectable text (not scanned images)
   - Try converting scanned PDFs to text-searchable format first

2. **Poor Name Detection**
   - Check if the resume follows standard formatting
   - Ensure the name appears in the first few lines

3. **Missing Skills**
   - Verify skills are mentioned using standard terminology
   - Check if skills appear in a dedicated skills section

### Debug Mode

```python
# Enable verbose output for debugging
parser = ResumeParser()
result = parser.parse_resume("resume.pdf")

# Check extracted text length
print(f"Extracted text length: {result.get('text_length', 0)} characters")

# Examine raw extracted text (first 500 chars)
text = parser.extract_text("resume.pdf")
print("Raw text preview:", text[:500])
```

## Contributing

To improve the parser:

1. Add new skills to the `skills_vocab` set
2. Enhance regex patterns for better matching
3. Improve section detection algorithms
4. Add support for new file formats

## License

This project is open source and available under the MIT License.

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Ensure all dependencies are properly installed
3. Verify file formats are supported
4. Test with different resume formats to identify patterns

---

**Note**: This parser is designed for speed and accuracy. For best results, ensure resumes follow standard formatting conventions with clear section headers and structured content.