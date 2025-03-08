# utils/utils.py
import os
from typing import List, Dict, Any
from firecrawl import FirecrawlApp
from dotenv import load_dotenv
import json
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import List, Optional
import asyncio

load_dotenv()
import tempfile
import PyPDF2

# Initialize the Firecrawl API client
app = FirecrawlApp(api_key=os.getenv("FIRECRAWL_API_KEY"))
openai_api_key = os.environ.get("OPENAI_API_KEY")
openai_client = OpenAI(api_key=openai_api_key)


# Define Pydantic models for data validation and serialization
class CandidateScore(BaseModel):
    name: str = Field(..., description="Candidate's name")
    relevance: int = Field(
        ...,
        description="How relevant the candidate's resume is to the job description (0-100)",
    )
    experience: int = Field(
        ..., description="Candidate's match in terms of work experience (0-100)"
    )
    skills: int = Field(..., description="Candidate's match based on skills (0-100)")
    overall: int = Field(..., description="Overall score (0-100)")
    comment: str = Field(
        ..., description="A cbrief omment explaining the rationale behind the scores"
    )


# Define Pydantic models for data validation and serialization
class Resume(BaseModel):
    name: str = Field(..., description="Candidate's full name")
    work_experiences: List[str] = Field(..., description="List of work experiences")
    location: str = Field(..., description="Candidate's location")
    skills: List[str] = Field(..., description="List of candidate's skills")
    education: List[str] = Field(..., description="Educational background")
    summary: Optional[str] = Field(
        None, description="A short summary or objective statement"
    )
    certifications: Optional[List[str]] = Field(
        None, description="List of certifications"
    )
    languages: Optional[List[str]] = Field(
        None, description="Languages spoken by the candidate"
    )

# Define Pydantic models for data validation and serialization
class JobDescription(BaseModel):
    title: str
    company: str
    location: str
    requirements: list[str]
    responsibilities: list[str]

# Define Pydantic models for data validation and serialization
class SkillGapAnalysis(BaseModel):
    matching_skills: List[str] = Field(..., description="Skills that match the job requirements")
    partial_matches: List[Dict[str, float]] = Field(..., description="Skills that partially match with confidence score")
    missing_skills: List[str] = Field(..., description="Skills required but not found in resume")
    skill_recommendations: List[Dict[str, str]] = Field(..., description="Recommendations for acquiring missing skills")
    overall_match_percentage: float = Field(..., description="Overall skills match percentage")


# Define Pydantic models for data validation and serialization
class EnhancedCandidateScore(CandidateScore):
    skill_gap_analysis: Optional[SkillGapAnalysis] = Field(None, description="Detailed analysis of skill gaps")
    skill_scores: Dict[str, float] = Field(default_factory=dict, description="Individual scores for each required skill")


# Define Pydantic models for data validation and serialization
async def ingest_inputs(
    job_description: str, resume_files: List[Any]
) -> Dict[str, Any]:
    """
    Ingests the job description and resume files.

    Parameters:
        job_description (str): The job description text or URL.
        resume_files (List[Any]): List of uploaded resume files.

    Returns:
        dict: A dictionary with two keys:
              - "job_description": The processed job description (in markdown).
              - "resumes": A list of resume file names.
    """
    # Determine if job_description is a URL.
    if job_description.startswith("http"):
        try:
            result = app.scrape_url(job_description, params={"formats": ["markdown"]})
            # Check if markdown data is present in the result.
            if not result or "markdown" not in result:
                raise ValueError("Scraping did not return markdown data.")
            job_desc_text = result.get("markdown", "")
        except Exception as e:
            raise Exception(f"Failed to scrape the job description URL: {e}")
    else:
        job_desc_text = job_description
    resumes = [file.name for file in resume_files]
    return {"job_description": job_desc_text, "resumes": resumes}



# Call the OpenAI API to interact with the Language Model
def call_llm(messages: list, response_format: None) -> str:
    """
    Calls the OpenAI GPT-4 model with the provided prompt and returns the response text.

    Parameters:
        messages (list): The messages to send to the LLM.
        response_format (None): The expected response format.

    Returns:
        str: The LLM's response.
    """
    params = {
        "model": "gpt-4-0125-preview",
        "messages": messages,
        "response_format": {"type": "json_object"}
    }

    response = openai_client.chat.completions.create(**params)
    return response.choices[0].message.content


# Parse the job description to extract key requirements
async def parse_job_description(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parses the job description to extract key requirements in a structured format.

    This function takes the ingested job description (which might be scraped from a URL)
    and uses an LLM (GPT-4) to extract and return only the essential job details.
    Extraneous content from the scraped page is removed.

    Parameters:
        data (dict): Dictionary containing the job description details, with a key "job_description".

    Returns:
        dict: A dictionary with the structured job description containing keys:
              "title", "company", "location", "requirements", "responsibilities", "benefits", and "experience".

    Raises:
        Exception: If the LLM call fails or the returned JSON cannot be parsed.
    """
    job_text = data.get("job_description", "")
    if not job_text:
        raise ValueError("No job description text provided.")

    # Build the prompt for the LLM
    prompt = (
        "Extract the key job information from the text below. Return only valid JSON "
        "with the following keys: title, company, location, requirements, responsibilities, benefits, experience. "
        "Do not include any extraneous information.\n\n"
        "Job description:\n" + job_text
    )
    messages = [
        {
            "role": "system",
            "content": (
                "You are an assistant that extracts key job description information from text. "
                "Return only the job details in valid JSON format using the keys: "
                "title, company, location, requirements (as a list), responsibilities (as a list), "
                "benefits (as a list), and experience."
            ),
        },
        {"role": "user", "content": prompt},
    ]

    try:
        llm_output = call_llm(messages, response_format=JobDescription)
        # Parse the JSON returned by the LLM
        structured_jd = json.loads(llm_output)
    except Exception as e:
        raise Exception(f"Error parsing job description: {e}")

    return structured_jd


#Parse the resume files to extract candidate information using the LLM model (GPT-4)
async def parse_resumes(resume_files: List[Any]) -> Dict[str, Any]:
    """
    Parses resume files to extract candidate information.

    This function reads each uploaded resume file and uses an LLM (via the call_llm helper)
    to extract candidate details. The LLM is asked to return only valid JSON following the
    schema defined by the Resume Pydantic model. The expected JSON should include keys such as:

        {
            "name": string,
            "work_experiences": list[string],
            "location": string,
            "skills": list[string],
            "education": list[string],
            "summary": string (optional),
            "certifications": list[string] (optional),
            "languages": list[string] (optional)
        }

    Parameters:
        resume_files (List[Any]): List of uploaded resume file objects (e.g., from Streamlit's file uploader).

    Returns:
        dict: A dictionary with a key "parsed_resumes" that is a list of parsed resume details.

    Raises:
        Exception: If any LLM call or JSON parsing fails.
    """
    parsed_resumes = []
    for resume in resume_files:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(resume.read())
            temp_path = temp_file.name

        # Extract text from PDF
        with open(temp_path, "rb") as file:
            pdf_reader = PyPDF2.PdfReader(file)
            pdf_text = " ".join(page.extract_text() for page in pdf_reader.pages)
        # Build messages for the LLM.
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an assistant that extracts candidate resume details. "
                    "Extract only the information following this JSON schema: "
                ),
            },
            {
                "role": "user",
                "content": f"Extract resume details from the following resume text:\n\n{pdf_text}",
            },
        ]

        try:
            # Call the LLM to process the resume text.
            # Pass the JSON schema (as a string) to instruct the LLM on the expected format.
            llm_response = call_llm(messages, response_format=Resume)
            # Parse the JSON response from the LLM.
            parsed_resume = json.loads(llm_response)
        except Exception as e:
            parsed_resume = {"error": f"Failed to parse resume using LLM: {e}"}

        parsed_resumes.append(parsed_resume)
    return {"parsed_resumes": parsed_resumes}


async def analyze_skill_gaps(job_requirements: Dict[str, Any], candidate_resume: Dict[str, Any]) -> SkillGapAnalysis:
    """
    Performs detailed analysis of skill gaps between job requirements and candidate's resume.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are a skilled technical recruiter analyzing skill gaps. You MUST return a JSON object "
                "that follows this EXACT structure, with no additional fields:\n"
                "{\n"
                '  "matching_skills": ["skill1", "skill2"],\n'
                '  "partial_matches": [{"skill1": 0.8}, {"skill2": 0.6}],\n'
                '  "missing_skills": ["skill3", "skill4"],\n'
                '  "skill_recommendations": [{"python": "Take an advanced Python course"}, {"sql": "Practice with real databases"}],\n'
                '  "overall_match_percentage": 75.5\n'
                "}\n\n"
                "Rules:\n"
                "1. matching_skills must be a list of strings\n"
                "2. partial_matches must be a list of objects with skill:score pairs\n"
                "3. missing_skills must be a list of strings\n"
                "4. skill_recommendations must be a list of objects with skill:recommendation pairs\n"
                "5. overall_match_percentage must be a number between 0 and 100"
            )
        },
        {
            "role": "user",
            "content": (
                f"Analyze these job requirements and candidate skills:\n\n"
                f"Job Requirements:\n{json.dumps(job_requirements.get('requirements', []), indent=2)}\n\n"
                f"Candidate Skills:\n{json.dumps(candidate_resume.get('skills', []), indent=2)}"
            )
        }
    ]

    try:
        response = call_llm(messages, None)
        analysis = json.loads(response)
        
        # Validate and transform the response
        gap_analysis = SkillGapAnalysis(
            matching_skills=analysis['matching_skills'],
            partial_matches=[{k: float(v)} for item in analysis['partial_matches'] for k, v in item.items()],
            missing_skills=analysis['missing_skills'],
            skill_recommendations=[{k: str(v)} for item in analysis['skill_recommendations'] for k, v in item.items()],
            overall_match_percentage=float(analysis['overall_match_percentage'])
        )
        return gap_analysis
    except Exception as e:
        print(f"Error in skill gap analysis: {str(e)}")  # For debugging
        return SkillGapAnalysis(
            matching_skills=[],
            partial_matches=[],
            missing_skills=[],
            skill_recommendations=[],
            overall_match_percentage=0.0
        )


async def score_candidates(
    parsed_requirements: Dict[str, Any], parsed_resumes: Dict[str, Any]
) -> List[Dict[str, Any]]:
    candidate_scores = []
    job_description_text = json.dumps(parsed_requirements)
    resume_list = parsed_resumes.get("parsed_resumes", [])
    
    for candidate in resume_list:
        try:
            # Get the basic scoring
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are an unbiased hiring manager. Compare the following job description "
                        "with the candidate's resume and provide scores (0-100) for relevance, "
                        "experience, and skills. Also compute an overall score that reflects the "
                        "candidate's fit and provide a comment explaining your evaluation. "
                        "Return a JSON object with the following structure:\n"
                        "{\n"
                        "  'name': string,\n"
                        "  'relevance': number,\n"
                        "  'experience': number,\n"
                        "  'skills': number,\n"
                        "  'overall': number,\n"
                        "  'comment': string,\n"
                        "  'skill_scores': {string: number}\n"
                        "}"
                    )
                },
                {
                    "role": "user",
                    "content": (
                        f"Job Description:\n{job_description_text}\n\n"
                        f"Candidate Resume:\n{json.dumps(candidate)}"
                    )
                }
            ]
            
            score_data = json.loads(call_llm(messages, None))
            
            # Add skill gap analysis
            gap_analysis = await analyze_skill_gaps(parsed_requirements, candidate)
            score_data["skill_gap_analysis"] = gap_analysis.dict()
            score_data["resume"] = candidate
            
        except Exception as e:
            score_data = {
                "name": candidate.get("name", "Unknown"),
                "relevance": 0,
                "experience": 0,
                "skills": 0,
                "overall": 0,
                "comment": f"Error during evaluation: {e}",
                "skill_gap_analysis": None
            }
        
        candidate_scores.append(score_data)
    
    return candidate_scores


def rank_candidates(candidate_scores: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Ranks candidates based on the average of their overall scores.

    For each candidate, this function calculates the average score from the keys:
    "relevance", "experience", "skills", and "overall". It adds a new key "avg_score"
    to each candidate's dictionary and then returns the sorted list in descending order.

    Parameters:
        candidate_scores (list): List of candidate score dictionaries.

    Returns:
        list: Sorted list of candidate scores in descending order based on avg_score.
    """
    for candidate in candidate_scores:
        # Compute the average of the relevant scores.
        relevance = candidate.get("relevance", 0)
        experience = candidate.get("experience", 0)
        skills = candidate.get("skills", 0)
        overall = candidate.get("overall", 0)
        candidate["avg_score"] = (relevance + experience + skills + overall) / 4.0

    # Return the sorted list of candidates based on avg_score.
    return sorted(
        candidate_scores, key=lambda candidate: candidate["avg_score"], reverse=True
    )


async def generate_email_templates(
    ranked_candidates: List[Dict[str, Any]], job_description: Dict[str, Any], top_x: int
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Generates custom email templates using an LLM for each candidate.
    Parameters:
        ranked_candidates (list): List of candidate score dictionaries.
        job_description (dict): The structured job description.
        top_x (int): Number of top candidates to invite for a call.

    Returns:
        dict: A dictionary with two keys:
              - "invitations": A list of dictionaries with candidate "name" and "email_body" for invitations.
              - "rejections": A list of dictionaries with candidate "name" and "email_body" for rejections.

    Raises:
        Exception: If the LLM call fails for any candidate.
    """
    invitations = []
    rejections = []

    for idx, candidate in enumerate(ranked_candidates):
        candidate_name = candidate.get("name", "Candidate")

        # Build the base messages for the LLM.
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an unbiased HR professional. Your task is to craft clear, concise, "
                    "and professional email responses to candidates based on the job description, "
                    "the candidate's resume details, and evaluation scores. "
                    "Return only the email body as plain text."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Job Description (structured):\n{json.dumps(job_description, indent=2)}\n\n"
                    f"Candidate Evaluation (structured):\n{json.dumps(candidate, indent=2)}\n\n"
                ),
            },
        ]

        # Append specific instructions based on candidate ranking.
        if idx < top_x:
            messages.append(
                {
                    "role": "assistant",
                    "content": (
                        "Please create an invitation email inviting the candidate for a quick call. "
                        "The email should be friendly, professional, and include a scheduling request."
                    ),
                }
            )
        else:
            messages.append(
                {
                    "role": "assistant",
                    "content": (
                        "Please create a polite rejection email. Include constructive feedback and key "
                        "suggestions for improvement based on the candidate's evaluation."
                    ),
                }
            )

        try:
            email_body = call_llm(messages, response_format=None)
        except Exception as e:
            email_body = f"Error generating email: {e}"

        email_template = {"name": candidate_name, "email_body": email_body}
        if idx < top_x:
            invitations.append(email_template)
        else:
            rejections.append(email_template)

    return {"invitations": invitations, "rejections": rejections}