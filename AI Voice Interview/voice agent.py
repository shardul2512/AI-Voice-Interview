import os
import json
import sys
from typing import List, Optional
import time # For potential pauses

# --- Voice Libraries ---
try:
    import speech_recognition as sr
except ImportError:
    print("ERROR: SpeechRecognition library not found. Please install it: pip install SpeechRecognition")
    sys.exit(1)
try:
    import pyttsx3
except ImportError:
    print("ERROR: pyttsx3 library not found. Please install it: pip install pyttsx3")
    sys.exit(1)
# PyAudio is implicitly needed by sr.Microphone, check happens later

# Langchain components
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
    from pydantic.v1 import BaseModel, Field, validator # Use pydantic.v1
    from langchain_community.document_loaders import PyPDFLoader
except ImportError as e:
    print(f"ERROR: Langchain components not found. Please install them: pip install langchain langchain-google-genai langchain_community pydantic==1.*")
    print(f"(Error details: {e})")
    sys.exit(1)


# --- Pydantic Models (Same as before) ---
class Skill(BaseModel):
    """Represents a single skill extracted from the resume."""
    name: str = Field(description="Name of the skill (e.g., Python, React, SQL)")
    category: Optional[str] = Field(description="Category (e.g., Language, Framework, Tool, Database)", default="Uncategorized")

class Project(BaseModel):
    """Represents a project described in the resume."""
    name: str = Field(description="Name of the project")
    description: str = Field(description="Brief description of the project and the candidate's role/contributions")
    technologies: Optional[List[str]] = Field(description="List of key technologies used", default=[])

class WorkExperience(BaseModel):
    """Represents a work experience entry from the resume."""
    company: str = Field(description="Company name")
    role: str = Field(description="Job title/role")
    duration: Optional[str] = Field(description="Dates of employment (e.g., 'Jan 2020 - Dec 2022')", default="N/A")
    responsibilities: Optional[List[str]] = Field(description="List of key responsibilities or achievements", default=[])

class ResumeData(BaseModel):
    """Overall structure for the parsed resume data."""
    summary: Optional[str] = Field(description="Brief professional summary if available, otherwise empty string", default="")
    skills: Optional[List[Skill]] = Field(description="List of technical skills", default=[])
    projects: Optional[List[Project]] = Field(description="List of personal or academic projects", default=[])
    work_experience: Optional[List[WorkExperience]] = Field(description="List of professional work experiences", default=[])
    candidate_name: Optional[str] = Field(description="Name of the candidate if found", default="Candidate")

    @validator('skills', 'projects', 'work_experience', pre=True, always=True)
    def ensure_list(cls, v):
        return v if v is not None else []

# --- Voice Engine Setup ---
tts_engine = None
try:
    tts_engine = pyttsx3.init()
    # Optional: Adjust voice properties
    # tts_engine.setProperty('rate', 180)
    print("TTS engine initialized successfully.")
except Exception as e:
    print(f"Warning: Error initializing TTS engine: {e}")
    print("Text-to-speech will be disabled, agent will only print text.")

recognizer = None
microphone = None
try:
    recognizer = sr.Recognizer()
    # Check for microphone availability early
    with sr.Microphone() as source_check:
         print(f"Microphone detected: {source_check.device_index}")
    microphone = sr.Microphone() # Keep instance for later use
    print("Speech recognizer and microphone initialized successfully.")
except ImportError:
    print("ERROR: PyAudio not found. Please install it (requires PortAudio system library).")
    print("Speech recognition will be disabled.")
except OSError as e:
    print(f"ERROR: Microphone OS Error: {e}")
    print("Could not find default microphone. Ensure it's connected and configured.")
    print("Speech recognition will be disabled.")
except Exception as e:
    print(f"ERROR: Error initializing microphone: {e}")
    print("Speech recognition will be disabled.")

# --- Voice Functions ---
def speak(text):
    """Uses the TTS engine to speak the given text, falls back to print."""
    print(f"Agent: {text}") # Always print to console
    if tts_engine:
        try:
            # Ensure the engine processes pending commands before speaking
            tts_engine.stop() # Stop current speech if any
            tts_engine.say(text)
            tts_engine.runAndWait()
        except Exception as e:
            print(f"Error during TTS: {e}")
    # No else needed, already printed

def listen(prompt="Listening..."):
    """Listens for audio via microphone and returns the transcribed text."""
    # Check if microphone was initialized successfully
    if not microphone or not recognizer:
        print("--- Microphone/Speech Recognition Disabled ---")
        # Fallback to text input if voice input failed initialization
        text_input = input(f"{prompt} (Voice input disabled, type your answer): ")
        return text_input.strip() if text_input else None

    print(f"\n{prompt}")
    with microphone as source:
        # Adjust for ambient noise dynamically
        try:
             # Add a timeout to ambient noise adjustment
             recognizer.adjust_for_ambient_noise(source, duration=0.7)
             print("Adjusted for ambient noise. Speak now.")
             # Listen with timeouts
             # Increased timeout slightly
             audio = recognizer.listen(source, timeout=7, phrase_time_limit=25)
        except sr.WaitTimeoutError:
             # Avoid speaking here if TTS might be speaking "didn't hear anything"
             # speak("Sorry, I didn't hear anything. Let's try that again.")
             print("Timeout waiting for phrase start.")
             return None # Indicate timeout
        except Exception as e:
             print(f"Error adjusting/listening for audio: {e}")
             # Avoid speaking here if TTS might be speaking
             # speak("Sorry, I had trouble with the microphone during listening.")
             return None

    try:
        print("Recognizing...")
        # Use Google Web Speech API for recognition (requires internet)
        text = recognizer.recognize_google(audio)
        print(f"You said: {text}")
        return text
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
        # speak("Sorry, I couldn't understand what you said. Could you please repeat?")
        return None # Indicate failure to understand
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")
        speak("Sorry, I'm having trouble connecting to the speech service. Check your internet connection.")
        return None # Indicate service error
    except Exception as e:
        print(f"An unexpected error occurred during speech recognition: {e}")
        speak("Sorry, an unexpected error occurred while trying to understand you.")
        return None

# --- Core Functions (Adapted) ---

# Removed load_api_key function

def initialize_llm(api_key):
    """Initializes the Gemini LLM."""
    # Check if API key is provided
    if not api_key:
        raise ValueError("API key was not provided to initialize_llm.")
    model_name = "gemini-2.0-flash"
    print(f"Initializing LLM with model: {model_name}")
    # Increase default timeout for potentially longer LLM calls
    return ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key, request_timeout=180)

def load_resume_text(pdf_path):
    """Loads text content from a PDF file."""
    try:
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        full_text = "\n".join([doc.page_content for doc in docs])
        if not full_text.strip():
             raise ValueError("Could not extract text from PDF. Ensure it's text-based.")
        return full_text
    except Exception as e:
        # Use speak function for user feedback if TTS is available
        speak(f"Error loading the PDF resume from {os.path.basename(pdf_path)}. Please check the file and permissions.")
        print(f"Error loading PDF {pdf_path}: {e}")
        raise

def create_extraction_chain(llm):
    """Creates the Langchain chain for extracting structured data from resume text."""
    parser = JsonOutputParser(pydantic_object=ResumeData)
    extraction_prompt_template = """
    You are an expert resume parser. Analyze the following resume text and extract the information precisely according to the provided JSON schema.
    Focus on: Candidate's Name, Summary, Work Experiences, Projects, Technical Skills (categorized).
    If information is missing, use defaults. Be accurate.
    Schema: {schema}
    Resume Text: --- {resume_text} ---
    Extracted JSON:
    """
    extraction_prompt = ChatPromptTemplate.from_template(
        extraction_prompt_template,
        partial_variables={"schema": ResumeData.schema_json(indent=2)}
    )
    return extraction_prompt | llm | parser

def parse_resume(pdf_path, llm):
    """Loads and parses the resume PDF to extract structured data."""
    speak(f"Loading and analyzing the resume from {os.path.basename(pdf_path)}...")
    print(f"Loading resume: {pdf_path}")
    resume_text = load_resume_text(pdf_path)
    print("Resume text loaded, starting extraction...")
    extraction_chain = create_extraction_chain(llm)
    try:
        print("Invoking extraction chain...")
        extracted_data_dict = extraction_chain.invoke({"resume_text": resume_text})
        print("Extraction chain finished.")
        validated_data = ResumeData(**extracted_data_dict)
        speak("Resume analysis complete.")
        print("Resume parsed successfully.")
        return validated_data
    except Exception as e:
        speak("Sorry, I encountered an error while analyzing the resume.")
        print(f"Error during resume parsing or validation: {e}")
        raise

# --- Interview Session Class (Adapted for Voice) ---

class InterviewSession:
    """Manages the state and flow of the simulated voice interview."""

    def __init__(self, resume_data: ResumeData, llm):
        self.resume_data = resume_data
        self.llm = llm
        self.history = []
        self.candidate_name = resume_data.candidate_name or "Candidate"
        self._setup_agent_chains()

    def _setup_agent_chains(self):
        """Initializes Langchain chains for each interview agent."""
        # Prompts remain the same
        behavioral_question_prompt = ChatPromptTemplate.from_template(
            """You are a professional interviewer starting a behavioral interview section.
            Based on the candidate's resume details below, generate ONE relevant behavioral question.
            Focus on their experiences and projects. For example: "Tell me about a time..." or "Describe a situation..." questions.
            Alternatively, ask a standard behavioral question (strengths, weaknesses, teamwork, conflict resolution).
            AVOID asking questions that were already asked in the 'Previous Questions' list. 

            Candidate Name: {candidate_name}
            Resume Summary: {summary}
            Work Experience: {work_experience}
            Projects: {projects}
            Previous Questions:
            {previous_questions}

            Generate ONE behavioral question for {candidate_name}:"""
        )
        self.behavioral_question_chain = behavioral_question_prompt | self.llm | StrOutputParser()

        coding_question_prompt = ChatPromptTemplate.from_template(
            """You are a technical interviewer preparing a coding-related question.
            Based on the candidate's skills, ask ONE conceptual question about algorithms, data structures, language features, or problem-solving approaches relevant to their skills.
            DO NOT ask for live code implementation. Focus on understanding and explanation.
            Example: "Considering your Python skills, explain the difference between lists and tuples and when you'd use each." or "How would you approach optimizing a database query if you noticed slow performance, given your SQL experience?"
            AVOID asking questions that were already asked in the 'Previous Questions' list.

            Candidate Name: {candidate_name}
            Skills: {skills}
            Previous Questions:
            {previous_questions}

            Generate ONE conceptual coding question for {candidate_name}:"""
        )
        self.coding_question_chain = coding_question_prompt | self.llm | StrOutputParser()

        evaluation_prompt = ChatPromptTemplate.from_template(
            """You are an interview coach evaluating a candidate's answer during an interview simulation.
            Provide brief, constructive feedback (2-3 sentences). Focus on clarity, relevance, structure (like STAR for behavioral), depth, and technical accuracy (where applicable).
            Be encouraging but also point out specific areas for improvement if needed. Speak directly to the candidate.

            Interview Stage: {question_type}
            Question Asked: {question}
            Candidate's Answer: {answer}

            Provide feedback on the answer:"""
        )
        self.evaluation_chain = evaluation_prompt | self.llm | StrOutputParser()

        final_feedback_prompt = ChatPromptTemplate.from_template(
            """You are an experienced hiring manager summarizing the performance of {candidate_name} in a simulated technical interview.
            Review the entire interview history provided below, including questions, answers, and individual evaluations.
            Synthesize this into comprehensive, constructive feedback, speaking directly to the candidate.

            Structure the feedback clearly:
            1.  Overall Summary: Brief overview of performance.
            2.  Behavioral Section:Strengths and areas for improvement.
            3.  Technical Concepts/Coding Section:** Strengths and areas for improvement
            5.  Key Recommendations: Actionable advice for the candidate.

            Be professional, balanced, and encouraging.

            Full Interview History:
            ---
            {interview_history}
            ---

            Generate Comprehensive Final Feedback for {candidate_name}:"""
        )
        self.final_feedback_chain = final_feedback_prompt | self.llm | StrOutputParser()


    def _get_previous_questions(self, agent_type=None):
        """Helper to get questions already asked, optionally filtered by agent type."""
        qs = []
        for item in self.history:
            if agent_type is None or item['agent'] == agent_type:
                 if item['question']:
                     qs.append(item['question'])
        # Limit context length if needed
        return "\n".join(f"- {q}" for q in qs[-10:]) if qs else "None" # Limit to last 10 questions

    def add_interaction(self, agent_type, question, answer, evaluation):
        """Adds a question-answer-evaluation cycle to the history."""
        self.history.append({
            "agent": agent_type,
            "question": question,
            "answer": answer,
            "evaluation": evaluation
        })

    def evaluate_answer(self, question, answer, question_type):
        """Uses the LLM chain to evaluate the candidate's answer and speaks the feedback."""
        print("\nAgent: Thinking...") # Console feedback
        feedback = "Sorry, I couldn't evaluate that response." # Default
        # Handle case where answer was not understood
        if answer == "[No Answer Understood]":
             feedback = "Since I couldn't understand the answer, I can't provide feedback for that question."
        else:
            try:
                feedback = self.evaluation_chain.invoke({
                    "question_type": question_type,
                    "question": question,
                    "answer": answer
                })
            except Exception as e:
                print(f"Error during evaluation LLM call: {e}")
                speak("I encountered an issue evaluating that response.")
                # Return default feedback on error
                feedback = "Sorry, I encountered an issue evaluating that response."

        speak(feedback) # Speak the feedback
        return feedback

    # --- Agent Interaction Methods (Using Voice) ---

    def ask_behavioral_question(self):
        """Generates and asks a behavioral question using voice."""
        print("\n--- Behavioral Question ---") # Console indicator
        previous_qs = self._get_previous_questions('behavioral')
        work_exp_list = self.resume_data.work_experience if self.resume_data.work_experience else []
        projects_list = self.resume_data.projects if self.resume_data.projects else []
        context = {
            "candidate_name": self.candidate_name,
            "summary": self.resume_data.summary or "N/A",
            "work_experience": json.dumps([exp.dict(exclude_none=True) for exp in work_exp_list], indent=2),
            "projects": json.dumps([p.dict(exclude_none=True) for p in projects_list], indent=2),
            "previous_questions": previous_qs
        }
        question = "I couldn't think of a question right now. Let's move on." # Default
        try:
            question = self.behavioral_question_chain.invoke(context)
        except Exception as e:
            print(f"Error generating behavioral question: {e}")
            speak("I'm having trouble formulating the next question.")
            self.add_interaction("behavioral", "Error generating question", "N/A", "N/A")
            return # Skip this question if generation fails

        speak(question)

        answer = None
        # Retry loop for listening
        for attempt in range(3):
            answer = listen()
            if answer is not None:
                break # Successfully heard an answer
            # If answer is None (STT failed/timeout)
            if attempt < 2: # Don't say "let's try again" on the last attempt
                 speak("Sorry, I didn't catch that. Could you please repeat your answer?")
                 time.sleep(0.5) # Short pause
            else:
                 speak("I'm still having trouble understanding. We'll have to skip the answer for this one.")
                 answer = "[No Answer Understood]" # Mark as not understood

        # Proceed to evaluation even if answer wasn't understood
        evaluation = self.evaluate_answer(question, answer, "Behavioral")
        self.add_interaction("behavioral", question, answer, evaluation)

    def ask_coding_question(self):
        """Generates and asks a conceptual coding question using voice."""
        print("\n--- Technical/Coding Concept Question ---")
        previous_qs = self._get_previous_questions('coding')
        skills_list = [f"{s.name} ({s.category})" for s in self.resume_data.skills] if self.resume_data.skills else ["General Concepts"]
        context = {
             "candidate_name": self.candidate_name,
             "skills": ", ".join(skills_list),
             "previous_questions": previous_qs
        }
        question = "I couldn't think of a technical question right now. Let's move on." # Default
        try:
             question = self.coding_question_chain.invoke(context)
        except Exception as e:
             print(f"Error generating coding question: {e}")
             speak("I'm having trouble formulating the next technical question.")
             self.add_interaction("coding", "Error generating question", "N/A", "N/A")
             return # Skip this question

        speak(question)

        answer = None
        # Retry loop for listening
        for attempt in range(3):
            answer = listen("Please explain your approach...")
            if answer is not None:
                 break
            if attempt < 2:
                 speak("Sorry, could you explain that again?")
                 time.sleep(0.5)
            else:
                 speak("I seem to be having trouble understanding the explanation. We'll skip the answer for this one.")
                 answer = "[No Answer Understood]"

        # Proceed to evaluation
        evaluation = self.evaluate_answer(question, answer, "Coding/Concepts")
        self.add_interaction("coding", question, answer, evaluation)


    # --- Final Feedback ---

    def generate_final_feedback(self):
        """Generates and speaks the overall interview feedback."""
        speak("Okay, that concludes the main questions. I'll now provide some overall feedback based on our conversation.")
        print("\n" + "="*25 + " Generating Final Feedback " + "="*25)
        if not self.history:
            speak("Actually, there were no interactions recorded, so I can't provide feedback.")
            print("No interview interactions recorded to generate feedback.")
            return

        # Filter out interactions where question generation failed
        valid_history = [item for item in self.history if item['question'] != "Error generating question"]

        if not valid_history:
             speak("It seems there were issues generating questions, so I can't provide feedback.")
             print("No valid interview interactions recorded.")
             return

        history_str = "\n\n".join([
            f"**{item['agent'].upper()} Stage**\n"
            f"Q: {item['question']}\n"
            # Display placeholder if answer wasn't understood
            f"A: {item['answer'] if item['answer'] != '[No Answer Understood]' else '<<No Answer Understood>>'}\n"
            f"Feedback: {item['evaluation']}"
            for item in valid_history
        ])

        final_feedback = "I had trouble generating the final feedback summary." # Default
        try:
            final_feedback = self.final_feedback_chain.invoke({
                "candidate_name": self.candidate_name,
                "interview_history": history_str
            })
        except Exception as e:
             print(f"Error generating final feedback: {e}")

        print("\n" + "="*20 + f" Final Interview Feedback for {self.candidate_name} " + "="*20)
        print(final_feedback) # Print final feedback to console
        print("="* (42 + len(self.candidate_name) + 1))

        speak(final_feedback) # Speak final feedback
        return final_feedback


# --- Main Execution Logic ---

def run_interview(pdf_path, api_key): # Accept api_key as argument
    """Orchestrates the entire voice-based interview process."""
    # Initial check for voice capabilities moved to __main__ block

    try:
        # 1. Setup - Initialize LLM with the provided key
        if not api_key:
             raise ValueError("API Key was not provided to run_interview.")
        llm = initialize_llm(api_key)

        # 2. Parse Resume
        resume_data = parse_resume(pdf_path, llm)

        # 3. Initialize Interview Session
        session = InterviewSession(resume_data, llm)
        speak(f"Okay {session.candidate_name}, let's start the interview simulation.")
        print(f"\nStarting Interview Simulation for {session.candidate_name}...")
        print("="*30)

        # 4. Define Interview Flow (5 Behavioral + 5 Coding)
        interview_stages = []
        num_questions_each_type = 5
        for i in range(num_questions_each_type):
            interview_stages.append(session.ask_behavioral_question)
            interview_stages.append(session.ask_coding_question)
        print(f"Interview flow set for {len(interview_stages)} questions.")

        # 5. Run Interview Stages
        for i, stage_func in enumerate(interview_stages):
            print(f"\n--- Starting Question {i+1}/{len(interview_stages)} ---")
            try:
                stage_func() # This now handles its own errors more gracefully
                time.sleep(1) # Small pause after feedback before next question
            except Exception as e:
                # Catch unexpected errors within the stage function itself if they propagate
                speak("Apologies, an unexpected error occurred processing that question. Let's try moving to the next one.")
                print(f"\n!! CRITICAL error during stage {stage_func.__name__}: {e}")
                print("Attempting to continue interview...")

        # 6. Generate Final Feedback
        session.generate_final_feedback()

        speak("Thank you for participating in the interview simulation.")
        print("\nInterview Simulation Complete.")

    except ValueError as ve: # Catch config errors like missing API key during init
        print(f"\nConfiguration Error: {ve}")
        # Attempt to speak the error if possible
        speak(f"Configuration Error: {ve}")
    except FileNotFoundError:
        speak(f"Error: I couldn't find the resume PDF file at {pdf_path}. Please check the path and restart.")
        print(f"\nError: Resume PDF file not found at {pdf_path}")
    except Exception as e: # Catch other critical errors (e.g., during parsing)
        speak("An unexpected error occurred, and I have to stop the interview simulation. Please check the console logs.")
        print(f"\nAn unexpected critical error occurred: {e}")


# --- Script Entry Point ---
if __name__ == "__main__":
    # --- Configuration ---
    # IMPORTANT: Run this script locally.
    # Set the path to your resume PDF file:
    resume_pdf_path = "D:/DATA/R project/Sresume.pdf" # <<< CHANGE THIS TO YOUR ACTUAL FILE PATH

    # --- API Key Configuration ---
    # PASTE YOUR GOOGLE API KEY HERE
    # WARNING: Hardcoding keys is generally insecure. Use environment variables for shared/production code.
    GOOGLE_API_KEY = "AIzaSyAnizf7igbcQGbQ0srTstvz4Lx_KX3hXVc" # <<< PASTE YOUR KEY HERE
    # --- End Configuration ---


    # --- Pre-run Checks ---
    print("Performing pre-run checks...")
    ready_to_run = True

    # Check if API key placeholder is still present
    if GOOGLE_API_KEY == "YOUR_API_KEY_HERE" or not GOOGLE_API_KEY:
         print("="*60)
         print("!! ERROR: Please replace 'YOUR_API_KEY_HERE' with your actual Google API Key in the script. !!")
         print("="*60)
         ready_to_run = False

    # Check if resume path placeholder is still present
    if resume_pdf_path == "Sresume.pdf":
        print("="*60)
        print("!! PLEASE UPDATE 'resume_pdf_path' variable in the script with the actual path to the PDF file. !!")
        print("="*60)
        ready_to_run = False
    # Check if the specified resume file exists only if the path is not the placeholder
    elif not os.path.exists(resume_pdf_path):
        print("="*60)
        print(f"!! ERROR: The specified resume file path does not exist: {resume_pdf_path} !!")
        print(f"!! Please ensure the file is present at that location. !!")
        print("="*60)
        ready_to_run = False

    # Check if voice components initialized
    if not recognizer or not microphone:
        print("\n" + "="*60)
        print("ERROR: Cannot start interview. Speech recognition components failed to initialize.")
        print("Please check dependencies (PyAudio, PortAudio) and microphone setup.")
        print("See errors printed during initialization above.")
        print("="*60)
        ready_to_run = False # Cannot run voice interview without mic

    # If all checks pass, run the interview
    if ready_to_run:
        print(f"\nPre-run checks passed. Attempting to run voice interview with resume: {resume_pdf_path}")
        run_interview(resume_pdf_path, GOOGLE_API_KEY) # Pass the key to the function
    else:
        print("\nInterview cannot start due to configuration errors listed above.")
