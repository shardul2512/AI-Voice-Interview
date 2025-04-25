import os
import json
import sys
from typing import List, Optional
import time # For potential pauses
import logging # For better logging

# Langchain components (Ensure imports match requirements.txt)
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
    from pydantic.v1 import BaseModel, Field, validator # Use pydantic.v1
    from langchain_community.document_loaders import PyPDFLoader
except ImportError as e:
    print(f"ERROR: Langchain components not found. Please install them: pip install langchain langchain-google-genai langchain_community pydantic==1.* google-api-python-client")
    print(f"(Error details: {e})")
    sys.exit(1)

# --- Pydantic Models ---
class Skill(BaseModel):
    name: str = Field(description="Name of the skill (e.g., Python, React, SQL)")
    category: Optional[str] = Field(description="Category (e.g., Language, Framework, Tool, Database)", default="Uncategorized")

class Project(BaseModel):
    name: str = Field(description="Name of the project")
    description: str = Field(description="Brief description of the project and the candidate's role/contributions")
    technologies: Optional[List[str]] = Field(description="List of key technologies used", default=[])

class WorkExperience(BaseModel):
    company: str = Field(description="Company name")
    role: str = Field(description="Job title/role")
    duration: Optional[str] = Field(description="Dates of employment (e.g., 'Jan 2020 - Dec 2022')", default="N/A")
    responsibilities: Optional[List[str]] = Field(description="List of key responsibilities or achievements", default=[])

class ResumeData(BaseModel):
    summary: Optional[str] = Field(description="Brief professional summary if available, otherwise empty string", default="")
    skills: Optional[List[Skill]] = Field(description="List of technical skills", default=[])
    projects: Optional[List[Project]] = Field(description="List of personal or academic projects", default=[])
    work_experience: Optional[List[WorkExperience]] = Field(description="List of professional work experiences", default=[])
    candidate_name: Optional[str] = Field(description="Name of the candidate if found", default="Candidate")

    @validator('skills', 'projects', 'work_experience', pre=True, always=True)
    def ensure_list(cls, v):
        return v if v is not None else []

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# --- Core Functions ---

def initialize_llm(api_key):
    """Initializes the Gemini LLM."""
    if not api_key:
        raise ValueError("API key was not provided to initialize_llm.")
    model_name = "gemini-2.0-flash"
    logging.info(f"Initializing LLM with model: {model_name}")
    try:
        llm = ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key, request_timeout=180)
        logging.info("LLM initialized successfully.")
        return llm
    except Exception as e:
        logging.error(f"Failed to initialize LLM: {e}")
        raise RuntimeError(f"Could not initialize the AI model: {e}")

def load_resume_text(pdf_path):
    """Loads text content from a PDF file."""
    logging.info(f"Loading resume from path: {pdf_path}")
    if not os.path.exists(pdf_path):
        logging.error(f"PDF file not found at: {pdf_path}")
        raise FileNotFoundError(f"Resume PDF not found at {pdf_path}")
    try:
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        if not docs:
            logging.warning(f"PyPDFLoader returned no documents for: {pdf_path}")
            raise ValueError("No documents loaded from PDF. Is it empty or corrupted?")

        full_text = "\n".join([doc.page_content for doc in docs if doc.page_content])
        if not full_text.strip():
            logging.warning(f"Could not extract text from PDF: {pdf_path}. Ensure it's text-based.")
            raise ValueError("Could not extract text from PDF. Ensure it's text-based and not image-only.")
        logging.info(f"Successfully loaded {len(full_text)} characters from PDF.")
        return full_text
    except FileNotFoundError:
        raise
    except Exception as e:
        logging.error(f"Error loading PDF {pdf_path}: {e}")
        raise ValueError(f"Error processing the PDF resume: {e}. Check file format and permissions.")


def create_extraction_chain(llm):
    """Creates the Langchain chain for extracting structured data from resume text."""
    parser = JsonOutputParser(pydantic_object=ResumeData)
    extraction_prompt_template = """
    You are an expert resume parser. Analyze the following resume text and extract the information precisely according to the provided JSON schema.
    Focus on: Candidate's Name, Summary (if present), Work Experiences (company, role, duration, responsibilities), Projects (name, description, technologies), Technical Skills (name, category - e.g., Language, Framework, Database, Tool, Library, Other).
    If specific information (like duration, category) is clearly missing, use the defaults specified in the schema or omit the field if optional. Be accurate and concise. If a section (like Projects) is entirely missing, return an empty list for it.

    Schema: {schema}

    Resume Text:
    ---
    {resume_text}
    ---

    Extracted JSON:
    """
    extraction_prompt = ChatPromptTemplate.from_template(
        extraction_prompt_template,
        partial_variables={"schema": ResumeData.schema_json(indent=2)}
    )
    return extraction_prompt | llm | parser

def parse_resume(pdf_path, llm):
    """Loads and parses the resume PDF to extract structured data."""
    logging.info(f"Starting resume parsing for: {os.path.basename(pdf_path)}")
    try:
        resume_text = load_resume_text(pdf_path)
        logging.info("Resume text loaded, creating extraction chain...")
        extraction_chain = create_extraction_chain(llm)
        logging.info("Invoking extraction chain...")
        extracted_data_dict = extraction_chain.invoke({"resume_text": resume_text})
        logging.info("Extraction chain finished, validating data...")
        validated_data = ResumeData(**extracted_data_dict)
        logging.info("Resume parsed and validated successfully.")
        return validated_data
    except (FileNotFoundError, ValueError) as e:
        logging.error(f"Parsing failed during loading/text extraction: {e}")
        raise
    except Exception as e:
        logging.error(f"Error during resume parsing LLM call or validation: {e}")
        raise RuntimeError(f"An error occurred while the AI was analyzing the resume: {e}")

# --- Interview Session Class ---

class InterviewSession:
    """Manages the state and flow of the simulated voice interview via WebSockets."""

    MAX_QUESTIONS_PER_TYPE = 5 # Configurable number of questions

    def __init__(self, resume_data: ResumeData, llm):
        self.resume_data = resume_data
        self.llm = llm
        self.history = [] # Stores {'agent': 'type', 'question': q, 'answer': a, 'evaluation': e}
        self.candidate_name = resume_data.candidate_name or "Candidate"
        self.question_count = {'behavioral': 0, 'coding': 0}
        self.interview_stage = 'behavioral' # Start with behavioral
        self._setup_agent_chains()
        logging.info(f"InterviewSession initialized for {self.candidate_name}")

   # Replace the existing _setup_agent_chains method in interview_logic.py
    # with this version for slightly stronger prompt emphasis.

    def _setup_agent_chains(self):
        """Initializes Langchain chains for asking questions and evaluation."""
        # Refined Behavioral Prompt
        behavioral_question_prompt = ChatPromptTemplate.from_template(
             """You are a friendly AI interviewer for a behavioral interview section.
            Your goal is to understand the candidate's past behavior in specific situations.
            Based *specifically* on the candidate's resume details provided ({work_experience}, {projects}), generate ONE relevant behavioral question.
            **Try to directly link the question to a specific role, responsibility, or project listed if possible.** Use STAR-based questions ("Tell me about a time...", "Describe a situation where you...", "Give an example of how you...").
            If no direct link is obvious, ask a standard behavioral question (teamwork, challenges, strengths, weaknesses, conflict resolution, learning from mistakes) that is still relevant to the likely role.
            IMPORTANT: DO NOT ask any of the questions listed in 'Previous Questions'. Choose a *different* topic or scenario. Keep the question concise and clear.

            Candidate Name: {candidate_name}
            Resume Summary: {summary}
            Candidate Work Experience Context: {work_experience}
            Candidate Project Context: {projects}
            Previous Questions Asked in this Session:
            {previous_questions}

            Generate ONE NEW behavioral question for {candidate_name}, drawing from their resume context:"""
        )
        self.behavioral_question_chain = behavioral_question_prompt | self.llm | StrOutputParser()

        # Refined Coding Prompt
        coding_question_prompt = ChatPromptTemplate.from_template(
            """You are an AI technical interviewer assessing conceptual understanding.
            Based *specifically* on the candidate's listed technical skills ({skills}), ask ONE conceptual question.
            The question should probe understanding of algorithms, data structures, language features, system design concepts, architectural patterns, testing strategies, or problem-solving approaches *directly relevant* to one or more of their listed skills.
            DO NOT ask for live code implementation. Focus on explanation, comparison, trade-offs, or high-level design.
            **Ensure the question clearly relates to their skillset ({skills}). Avoid overly generic conceptual questions.**
            Example: If skill is 'SQL', ask about index types or transaction isolation. If 'React', ask about state management or component lifecycle.
            IMPORTANT: DO NOT ask any of the questions listed in 'Previous Questions'. Choose a *different* concept or skill area. Keep the question concise and clear.

            Candidate Name: {candidate_name}
            Candidate Skills Context: {skills}
            Previous Questions Asked in this Session:
            {previous_questions}

            Generate ONE NEW conceptual technical question for {candidate_name}, tailored to their skills:"""
        )
        self.coding_question_chain = coding_question_prompt | self.llm | StrOutputParser()

        # Evaluation and Final Feedback prompts remain the same as before
        evaluation_prompt = ChatPromptTemplate.from_template(
            """You are an AI interview coach evaluating a candidate's answer. Provide brief, constructive feedback (2-4 sentences max) directly to the candidate ({candidate_name}).
            Focus on:
            - Relevance: Did the answer directly address the question?
            - Clarity & Structure: Was the answer easy to follow? (e.g., STAR method for behavioral).
            - Depth & Detail: Did the answer provide sufficient detail/examples?
            - Technical Accuracy (for coding questions): Was the technical explanation sound?
            Be encouraging. Mention one specific positive aspect and one specific area for improvement, if applicable. Avoid generic phrases.

            Candidate Name: {candidate_name}
            Interview Stage: {question_type}
            Question Asked: {question}
            Candidate's Answer: {answer}

            Provide concise feedback:"""
        )
        self.evaluation_chain = evaluation_prompt | self.llm | StrOutputParser()

        final_feedback_prompt = ChatPromptTemplate.from_template(
            """You are an AI Hiring Manager summarizing {candidate_name}'s performance in a simulated interview.
            Review the entire interview history provided below.
            Synthesize this into comprehensive, constructive final feedback (3-5 paragraphs) addressed directly to the candidate.

            Structure the feedback:
            1.  **Overall Impression:** Start with a positive opening and brief summary.
            2.  **Behavioral Skills:** Discuss strengths (e.g., communication, STAR usage, relevant examples) and areas for improvement (e.g., more detail, clearer structure) observed in the behavioral answers. Reference specific examples if helpful.
            3.  **Technical Concepts:** Discuss strengths (e.g., clear explanations, understanding of concepts) and areas for improvement (e.g., depth of knowledge, exploring trade-offs) based on the technical answers.
            4.  **Key Recommendations:** Offer 2-3 actionable pieces of advice for future interviews (e.g., "Practice structuring answers using STAR," "Review fundamental concepts in [specific skill area]," "Consider elaborating more on project contributions").
            5.  **Closing:** End on an encouraging note.

            Be professional, balanced, and specific.

            Full Interview History:
            ---
            {interview_history}
            ---

            Generate Comprehensive Final Feedback for {candidate_name}:"""
        )
        self.final_feedback_chain = final_feedback_prompt | self.llm | StrOutputParser()
        logging.info("LLM chains for interview session created with refined prompts.")

    def _get_previous_questions(self):
        """Helper to get questions already asked in this session."""
        qs = [item['question'] for item in self.history if item.get('question')]
        # Limit context size if necessary by taking only last N questions
        # return "\n".join(f"- {q}" for q in qs[-10:]) if qs else "None"
        return "\n".join(f"- {q}" for q in qs) if qs else "None"


    def _format_resume_context(self, section):
        """Safely formats resume sections for prompts."""
        try:
            if section == 'skills':
                data = self.resume_data.skills
                return ", ".join([f"{s.name}" + (f" ({s.category})" if s.category != "Uncategorized" else "") for s in data]) if data else "Not specified"
            elif section == 'work_experience':
                data = self.resume_data.work_experience
                # Limit the length of work experience detail if it becomes too large for the context window
                # return json.dumps([exp.dict(exclude_none=True, exclude_defaults=True) for exp in data[:3]], indent=1) if data else "Not specified"
                return json.dumps([exp.dict(exclude_none=True, exclude_defaults=True) for exp in data], indent=1) if data else "Not specified"
            elif section == 'projects':
                data = self.resume_data.projects
                return json.dumps([p.dict(exclude_none=True, exclude_defaults=True) for p in data], indent=1) if data else "Not specified"
            else:
                return "N/A"
        except Exception as e:
            logging.warning(f"Error formatting resume section {section}: {e}")
            return "Error retrieving data"


    def get_next_question(self) -> Optional[str]:
        """Gets the next question based on the current stage and count, or None if finished."""
        logging.info(f"Getting next question. Stage: {self.interview_stage}, Counts: {self.question_count}")

        question_text = None
        question_type = None

        if self.interview_stage == 'behavioral' and self.question_count['behavioral'] < self.MAX_QUESTIONS_PER_TYPE:
            question_type = 'behavioral'
            chain = self.behavioral_question_chain
            context = {
                "candidate_name": self.candidate_name,
                "summary": self.resume_data.summary or "N/A",
                "work_experience": self._format_resume_context('work_experience'),
                "projects": self._format_resume_context('projects'),
                "previous_questions": self._get_previous_questions()
            }
        elif self.interview_stage == 'coding' and self.question_count['coding'] < self.MAX_QUESTIONS_PER_TYPE:
             question_type = 'coding'
             chain = self.coding_question_chain
             context = {
                "candidate_name": self.candidate_name,
                "skills": self._format_resume_context('skills'),
                "previous_questions": self._get_previous_questions()
             }
        else: # Move to the next stage or finish
            if self.interview_stage == 'behavioral':
                 self.interview_stage = 'coding'
                 logging.info("Switching to coding questions stage.")
                 # Try again to get a coding question if available
                 return self.get_next_question()
            else:
                 logging.info("All question types completed.")
                 return None # Signal interview end (before final feedback)

        try:
            logging.info(f"Invoking {question_type} question chain.")
            question_text = chain.invoke(context)
            # Basic validation
            if not question_text or len(question_text) < 10:
                 logging.warning(f"Generated {question_type} question seems too short or empty: '{question_text}'. Retrying once.")
                 time.sleep(1)
                 question_text = chain.invoke(context)
                 if not question_text or len(question_text) < 10:
                     raise ValueError(f"Failed to generate a valid {question_type} question after retry.")

            self.question_count[question_type] += 1
            self.history.append({
                "agent": question_type,
                "question": question_text,
                "answer": None,
                "evaluation": None
            })
            logging.info(f"Generated {question_type} question #{self.question_count[question_type]}: {question_text}")
            return question_text

        except Exception as e:
            logging.error(f"Error generating {question_type} question: {e}")
            error_message = f"[Error generating {question_type} question]"
            self.history.append({
                "agent": question_type,
                "question": error_message,
                "answer": None, "evaluation": None
            })
            # Advance the counter anyway to avoid getting stuck
            self.question_count[question_type] += 1
            logging.warning("Trying to skip failed question and get next one.")
            # Return the error message to be displayed to the user before trying next
            # return error_message # Option 1: Show error before next question
            return self.get_next_question() # Option 2: Silently skip and try next


    def record_answer_and_evaluate(self, answer: str) -> Optional[str]:
        """Records the user's answer to the last question and generates feedback."""
        if not self.history:
            logging.warning("Received answer but no history exists.")
            return "I wasn't expecting an answer yet. Let's start with the first question."

        last_interaction = self.history[-1]

        if last_interaction.get("answer") is not None:
            logging.warning("Attempted to record answer for a question that already has one.")
            return last_interaction.get("evaluation", "I seem to have already processed this answer.")

        if last_interaction["question"].startswith("[Error generating"):
            logging.warning("Received answer for a question that failed generation.")
            return "I couldn't generate the previous question properly, so I can't evaluate an answer for it. Let's try the next one."

        last_interaction["answer"] = answer
        question = last_interaction["question"]
        question_type = last_interaction["agent"]

        logging.info(f"Evaluating answer for {question_type} question: '{question}'")
        feedback = f"Sorry, I encountered an issue evaluating that response for '{question}'."

        if not answer or answer == "[No Answer Provided]" or len(answer.strip()) == 0:
             feedback = "It seems I didn't get an answer for that question. Make sure to click 'Start Speaking' and respond clearly. I can't provide feedback without an answer."
             logging.warning(f"Evaluation skipped for question '{question}' due to missing answer.")
        else:
             try:
                 feedback = self.evaluation_chain.invoke({
                     "candidate_name": self.candidate_name,
                     "question_type": question_type.capitalize(),
                     "question": question,
                     "answer": answer
                 })
                 logging.info(f"Generated feedback: {feedback}")
             except Exception as e:
                 logging.error(f"Error during evaluation LLM call: {e}")
                 feedback = "I encountered a technical issue while trying to evaluate your response. Let's move on for now."

        last_interaction["evaluation"] = feedback
        return feedback


    def generate_final_feedback(self) -> str:
        """Generates overall interview feedback based on the complete history."""
        logging.info("Generating final feedback.")
        if not self.history or all(item['question'].startswith("[Error generating") for item in self.history):
            logging.warning("Not enough valid interactions to generate final feedback.")
            return "There weren't enough completed questions in our session to provide overall feedback. Please try uploading the resume again."

        valid_history = [
            item for item in self.history
            if not item['question'].startswith("[Error generating") and item.get('answer') is not None
        ]

        if not valid_history:
             logging.warning("No valid Q&A pairs found in history for final feedback.")
             return "It seems we had trouble with the questions or answers. I can't generate a final summary based on this session."

        history_str = "\n\n".join([
            f"**{item['agent'].upper()} Stage**\n"
            f"Q: {item['question']}\n"
            f"A: {item['answer']}"
            # Optionally include immediate feedback in the context for final summary:
            # f"\nImmediate Feedback: {item['evaluation']}"
            for item in valid_history
        ])

        final_feedback_text = "I had trouble generating the final feedback summary."
        try:
            final_feedback_text = self.final_feedback_chain.invoke({
                "candidate_name": self.candidate_name,
                "interview_history": history_str
            })
            logging.info("Final feedback generated successfully.")
        except Exception as e:
            logging.error(f"Error generating final feedback: {e}")
            final_feedback_text = "I encountered a technical issue while trying to generate the final summary. Apologies for that."

        return final_feedback_text