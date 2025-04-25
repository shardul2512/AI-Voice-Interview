import eventlet 
eventlet.monkey_patch() 
import os 
import logging 
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify # Removed Flask session import
from flask_socketio import SocketIO, emit, disconnect # Removed join/leave room imports
import uuid # For temporary file names
import base64 # Needed for decoding file data
import sys # For potential exit on critical error

# Import from your refactored logic
from interview_logic import (
    initialize_llm,
    parse_resume,
    ResumeData, # ResumeData needed? Only for type hint maybe
    InterviewSession
)
# Load environment variables (especially GOOGLE_API_KEY)
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Flask App Setup ---
app = Flask(__name__)
# SECRET_KEY is needed for SocketIO even if Flask sessions aren't used explicitly
app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY', os.urandom(24))
app.config['UPLOAD_FOLDER'] = 'uploads'
# Ensure upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    try:
        os.makedirs(app.config['UPLOAD_FOLDER'])
        logging.info(f"Created upload folder: {app.config['UPLOAD_FOLDER']}")
    except OSError as e:
        logging.error(f"Could not create upload folder {app.config['UPLOAD_FOLDER']}: {e}")
        # Depending on severity, you might want to exit
        # sys.exit(f"Fatal Error: Cannot create upload directory {app.config['UPLOAD_FOLDER']}")


# Use eventlet or gevent for better concurrency with WebSockets
async_mode = None
try:
    import eventlet
    async_mode = 'eventlet'
    eventlet.monkey_patch()
    logging.info("Using eventlet for async mode.")
except ImportError:
    logging.warning("eventlet not found, WebSocket performance might be limited.")
    pass

socketio = SocketIO(app, async_mode=async_mode, cors_allowed_origins="*") # Allow all origins for simplicity
# --- Global Variables / State Management ---
# WARNING: In-memory storage is lost on server restart and doesn't scale.
# Use Redis, a database, etc., for production.
active_sessions = {} # Store InterviewSession instances mapped by sid 
llm = None # Initialize LLM globally once
# --- Helper Functions ---
def get_llm():
    """Gets the initialized LLM instance, initializing if needed."""
    global llm
    if llm is None:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key or api_key == "YOUR_ACTUAL_GOOGLE_API_KEY_HERE":
             logging.error("GOOGLE_API_KEY not found or not set in .env file.")
             # This is a critical configuration error
             raise ValueError("Server configuration error: Google API Key is missing.")
        try:
            llm = initialize_llm(api_key)
        except Exception as e:
            logging.error(f"LLM Initialization failed: {e}")
            # Raise the error to prevent the server from starting incorrectly
            raise RuntimeError(f"Fatal Error: Could not initialize LLM - {e}")
    return llm

# --- Flask Routes ---
@app.route('/')
def index():
    """Serves the main HTML page."""
    return render_template('index.html')

# --- SocketIO Event Handlers ---
@socketio.on('connect')
def handle_connect():
    """Handles new client connections."""
    sid = request.sid
    logging.info(f"Client connected: {sid}")
    emit('status_update', {'message': 'Connected to interview server.'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handles client disconnections and cleans up resources."""
    sid = request.sid
    logging.info(f"Client disconnected: {sid}")
    # Clean up the session for this user
    session_data = active_sessions.pop(sid, None)
    if session_data:
        logging.info(f"Cleaned up session for {sid}")
        # Clean up temporary PDF file if it exists
        pdf_path = session_data.get('pdf_path')
        if pdf_path and os.path.exists(pdf_path):
            try:
                os.remove(pdf_path)
                logging.info(f"Removed temporary PDF: {pdf_path}")
            except OSError as e:
                logging.error(f"Error removing temporary PDF {pdf_path}: {e}")

@socketio.on('upload_resume')
def handle_resume_upload(data):
    """Handles PDF upload, parsing, and starting the interview."""
    sid = request.sid
    logging.info(f"Received resume upload request from {sid}")
    pdf_path = None # Initialize pdf_path for potential cleanup in error handlers

    # Ensure LLM is ready (called here to handle potential init issues per request)
    try:
        current_llm = get_llm()
        if not current_llm: # Should not happen if get_llm raises error, but check anyway
            emit('error_message', {'message': 'Server error: AI model not available.'}, room=sid)
            return
    except (ValueError, RuntimeError) as llm_error:
         emit('error_message', {'message': f'{llm_error}'}, room=sid)
         return


    # Clean up any existing session for this connection ID
    if sid in active_sessions:
         logging.warning(f"Session already exists for {sid}. Cleaning up old session first.")
         handle_disconnect() # Reuse disconnect logic for cleanup

    pdf_data = data.get('file_data') # Expecting base64 encoded data URI
    file_name = data.get('file_name', 'uploaded_resume.pdf')

    if not pdf_data:
        emit('error_message', {'message': 'No file data received.'}, room=sid)
        return

    try:
        # Decode base64 data URI
        try:
            header, encoded = pdf_data.split(',', 1)
            pdf_bytes = base64.b64decode(encoded)
        except (ValueError, base64.binascii.Error) as decode_error:
            logging.error(f"Invalid base64 data received from {sid}: {decode_error}")
            emit('error_message', {'message': 'Invalid file data format received.'}, room=sid)
            return

        # Sanitize filename slightly (prevent directory traversal)
        safe_filename = os.path.basename(file_name)
        temp_filename = f"{uuid.uuid4()}_{safe_filename}"
        pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)

        # Ensure upload folder still exists (paranoid check)
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])

        with open(pdf_path, 'wb') as f:
            f.write(pdf_bytes)
        logging.info(f"Temporary PDF saved to: {pdf_path}")

        # --- Start Parsing ---
        emit('status_update', {'message': f"Analyzing resume: {safe_filename}..."}, room=sid)
        resume_data = parse_resume(pdf_path, current_llm)

        # --- Initialize Interview Session ---
        interview_session = InterviewSession(resume_data, current_llm)
        active_sessions[sid] = {
            'interview_session': interview_session,
            'pdf_path': pdf_path # Store path for cleanup
            }
        logging.info(f"Interview session created for {sid} with candidate {interview_session.candidate_name}")

        # --- Send Welcome and First Question ---
        welcome_message = f"Okay {interview_session.candidate_name}, I've analyzed your resume. Let's start the interview simulation."
        emit('agent_message', {'text': welcome_message}, room=sid)
        socketio.sleep(1.5) # Pause slightly after welcome

        first_question = interview_session.get_next_question()
        if first_question and not first_question.startswith("[Error"):
            emit('agent_message', {'text': first_question, 'is_question': True}, room=sid) # Mark as question
        else:
             failed_question_msg = first_question if first_question else "I couldn't generate the first question."
             emit('agent_message', {'text': f"{failed_question_msg} Please try uploading the resume again."}, room=sid)
             logging.error(f"Failed to get a valid first question for {sid}.")
             # Clean up immediately if first question fails critically
             handle_disconnect()


    except (FileNotFoundError, ValueError, RuntimeError) as e:
         # These are expected errors during processing
         logging.error(f"Error processing resume for {sid}: {e}")
         emit('error_message', {'message': f"Error processing resume: {e}"}, room=sid)
         if pdf_path and os.path.exists(pdf_path):
             try: os.remove(pdf_path)
             except OSError: pass
         active_sessions.pop(sid, None)

    except Exception as e:
        # Catch unexpected errors
        logging.exception(f"Unexpected error during resume upload/parse for {sid}: {e}") # Log full traceback
        emit('error_message', {'message': 'An unexpected server error occurred during upload.'}, room=sid)
        if pdf_path and os.path.exists(pdf_path):
             try: os.remove(pdf_path)
             except OSError: pass
        active_sessions.pop(sid, None)

@socketio.on('user_response')
def handle_user_response(data):
    """Handles the transcribed text from the user."""
    sid = request.sid
    logging.info(f"Received user response from {sid}")

    session_data = active_sessions.get(sid)
    if not session_data or 'interview_session' not in session_data:
        logging.warning(f"Received response from {sid}, but no active session found.")
        emit('error_message', {'message': 'Your session seems to have expired. Please upload the resume again.'}, room=sid)
        return

    interview_session = session_data['interview_session']
    answer_text = data.get('text', '[No Answer Provided]')

    try:
        # --- Evaluate Answer ---
        emit('status_update', {'message': 'Evaluating your answer...'}, room=sid)
        feedback = interview_session.record_answer_and_evaluate(answer_text)
        if feedback:
            emit('agent_message', {'text': feedback}, room=sid)
            socketio.sleep(1.5) # Pause after feedback

        # --- Get Next Question or Final Feedback ---
        next_question = interview_session.get_next_question()

        if next_question and not next_question.startswith("[Error"):
            emit('agent_message', {'text': next_question, 'is_question': True}, room=sid) # Mark as question
        else:
             # Handle case where next question generation fails OR interview is done
             if next_question and next_question.startswith("[Error"):
                 logging.error(f"Failed to generate next question for {sid}. Proceeding to final feedback.")
                 emit('agent_message', {'text': "I had trouble generating the next question. Let's move to the final feedback."}, room=sid)
                 socketio.sleep(1.5)

             # Interview questions finished OR failed to get next question, generate final feedback
             emit('status_update', {'message': 'Generating final feedback...'}, room=sid)
             final_feedback = interview_session.generate_final_feedback()
             emit('agent_message', {'text': final_feedback}, room=sid)
             socketio.sleep(1) # Short pause
             emit('interview_finished', {'message': 'Interview simulation complete. Thank you!'}, room=sid)
             # Clean up the session after completion
             handle_disconnect() # Reuse disconnect logic

    except Exception as e:
        logging.exception(f"Unexpected error during user response processing for {sid}: {e}")
        emit('error_message', {'message': 'An unexpected server error occurred while processing your answer.'}, room=sid)
        # Consider ending the interview gracefully
        handle_disconnect()


# --- Main Execution ---
if __name__ == '__main__':
    print("Starting Flask-SocketIO server...")
    # Attempt to initialize LLM on startup to catch config errors early
    llm_init_error = False
    try:
        get_llm()
        print("AI Model initialized successfully.")
    except (ValueError, RuntimeError) as e:
        print(f"FATAL STARTUP ERROR: {e}")
        print("Please check your .env file and API key permissions.")
        llm_init_error = True
        # Exit if LLM is critical for the app to function
        sys.exit(f"Exiting due to LLM initialization failure: {e}")
    except Exception as e:
        # Catch other potential init errors
        print(f"WARNING: Unexpected error during LLM pre-initialization: {e}")
        llm_init_error = True # Treat as potential issue

    if llm_init_error:
         # This part might not be reached if sys.exit() was called
         print("There was an issue initializing the AI model on startup.")
         print("The application server might run, but API calls will likely fail.")

    print(f"Server starting on http://127.0.0.1:5000 (or your local IP on port 5000)")
    # Use socketio.run for development. For production, use Gunicorn/Waitress.
    # Example: gunicorn --worker-class eventlet -w 1 app:app
    socketio.run(app, debug=True, host='0.0.0.0', port=5000, use_reloader=True)