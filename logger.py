import logging
import multiprocessing
import sys
import os
import time
import uuid
from pathlib import Path
from typing import Type, Dict

# Powertools and Uvicorn must be installed
from aws_lambda_powertools import Logger
import uvicorn

from adk.tools.langchain_tool import LangChainTool
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

# --- 1. Core Logging Setup with Powertools ---

def setup_powertools_logging(service_name: str, session_id: str, log_file: Path) -> Logger:
    """
    Configures a Powertools logger for a specific service and session.
    
    For local execution, this function will redirect stdout to a file to capture all
    output in a single place.
    
    Args:
        service_name: The name of the service logging (e.g., 'AgentSystem', 'Subprocess').
        session_id: The unique ID for the current session.
        log_file: The path to the file where logs will be written.
        
    Returns:
        A configured Logger instance.
    """
    # Redirect stdout to the session log file. The Powertools Logger writes to stdout by default.
    # This is a simple way to capture all output (including from non-logger libraries) locally.
    sys.stdout.flush()
    log_file_handle = open(log_file, 'a')
    os.dup2(log_file_handle.fileno(), sys.stdout.fileno())
    
    # Redirect stderr as well for complete capture
    sys.stderr.flush()
    os.dup2(log_file_handle.fileno(), sys.stderr.fileno())
    
    # Instantiate the logger
    logger = Logger(service=service_name, level="INFO")
    
    # Use the correlation_id feature for our session_id
    logger.set_correlation_id(session_id)
    
    return logger


# --- 2. Modified Subprocess and Tool ---

class FastApiServerInput(BaseModel):
    filepath: str = Field(..., description="Path to the FastAPI file relative to /sandbox.")
    port: int = Field(default=8000, description="Port number for the server.")

def execute_server(app_string: str, port: int, sandbox_dir: str, log_file: Path, session_id: str):
    """
    Target function for the server process. It sets up its own Powertools logging.
    """
    # CRUCIAL: The new process configures its own logging, pointing to the same file.
    # It identifies itself with the 'Subprocess' service name.
    log = setup_powertools_logging(service_name="Subprocess", session_id=session_id, log_file=log_file)

    try:
        log.info(f"Process {os.getpid()} starting up.")
        os.chdir(sandbox_dir)
        sys.path.insert(0, sandbox_dir)
        log.info(f"Starting Uvicorn for '{app_string}' on port {port}.")
        
        # All of Uvicorn's stdout/stderr will now be captured via our file redirection.
        uvicorn.run(app_string, host="0.0.0.0", port=port, log_level="info")

    except Exception:
        log.exception("Failed to start server in subprocess")


class FastApiServerTool(BaseTool):
    """
    Starts a FastAPI server using Powertools for structured logging.
    """
    name: str = "fastapi_server_starter"
    description: str = (
        "Starts a FastAPI web server from a file in /sandbox. "
        "Returns a PID. All logs are streamed in a structured JSON format to a central log file."
    )
    args_schema: Type[BaseModel] = FastApiServerInput
    processes: Dict[int, multiprocessing.Process] = {}

    # These will be configured by the main script before use
    log_file: Path = None
    session_id: str = None
    logger: Logger = None

    def configure(self, logger: Logger, log_file: Path, session_id: str):
        """Configures the tool with the session's logging details."""
        self.logger = logger
        self.log_file = log_file
        self.session_id = session_id

    def _run(self, filepath: str, port: int = 8000) -> str:
        if not self.logger:
            return "❌ Error: Tool has not been configured with a logger. Call configure() first."
        
        sandbox_dir = "/sandbox"
        self.logger.info(f"Attempting to start server from file: {filepath}", extra={"port": port})
        
        try:
            path_no_ext = Path(filepath).with_suffix('')
            import_path = str(path_no_ext).replace(os.path.sep, '.')
            app_string = f"{import_path}:app"

            process = multiprocessing.Process(
                target=execute_server, 
                args=(app_string, port, sandbox_dir, self.log_file, self.session_id)
            )
            process.start()
            self.processes[process.pid] = process

            self.logger.info(f"Server process created successfully", extra={"pid": process.pid})
            return (f"✅ Server process started with PID {process.pid}. "
                    f"Logs are being streamed to '{self.log_file}'.")
        except Exception:
            self.logger.exception("Failed to create server process")
            return f"❌ Error creating server process. Check log for details."

    def stop_server(self, pid: int) -> str:
        # ... (stop_server method is unchanged) ...
        return f"✅ Server with PID {pid} has been stopped."

# --- 3. Main execution block demonstrating the system ---

if __name__ == '__main__':
    # -- A. INITIAL SETUP --
    SESSION_ID = f"{int(time.time())}-{uuid.uuid4().hex[:6]}"
    LOG_FILE_PATH = Path(f"./session_{SESSION_ID}.log")

    # Configure logging for the main agent process, identifying as 'AgentSystem'
    agent_logger = setup_powertools_logging(
        service_name="AgentSystem",
        session_id=SESSION_ID,
        log_file=LOG_FILE_PATH
    )
    
    agent_logger.info("--- Main Agent Process Started ---", extra={"location": "Bournemouth, UK"})

    # -- B. TOOL SETUP --
    server_tool = FastApiServerTool()
    server_tool.configure(logger=agent_logger, log_file=LOG_FILE_PATH, session_id=SESSION_ID)
    adk_wrapped_tool = LangChainTool(langchain_tool=server_tool)

    # -- C. CREATE DUMMY SANDBOX FILE --
    SANDBOX_PATH = Path("./sandbox").resolve()
    SANDBOX_PATH.mkdir(exist_ok=True)
    api_file = SANDBOX_PATH / "main.py"
    api_file.write_text("""
from aws_lambda_powertools import Logger
# In a real app, you might pass the service name via an env var
# Here we hardcode it for demonstration.
log = Logger(service="Subprocess")

from fastapi import FastAPI
app = FastAPI()

@app.on_event("startup")
async def startup_event():
    # This log will automatically have the correlation_id from the parent
    log.info("FastAPI app startup complete. Server is ready.")

@app.get("/")
def root():
    log.info("Root endpoint was hit.", extra={"client_ip": "127.0.0.1"})
    return {"message": "Hello from the server"}
""")
    agent_logger.info(f"Created dummy API file at '{api_file}'")

    # -- D. EXECUTE THE WORKFLOW --
    agent_logger.info("Invoking the server starter tool.")
    result = adk_wrapped_tool.invoke(input={"filepath": "main.py", "port": 8001})
    agent_logger.info(f"Tool invocation result", extra={"result": result})
    
    pid = None
    if "PID" in result:
        pid = int(result.split("PID ")[1].split(".")[0])

    agent_logger.info("Main process is waiting for 5 seconds...")
    time.sleep(5)

    if pid:
        agent_logger.info(f"Stopping server", extra={"pid": pid})
        server_tool.stop_server(pid)
        agent_logger.info(f"Stop command issued.")
    
    agent_logger.info("--- Main Agent Process Finished ---")

    # -- E. DISPLAY THE FINAL LOG FILE --
    # In a real scenario, you would inspect the JSON log file.
    # Here we print it to show the structured output.
    print("\n" + "="*80)
    print(f"COMPLETE STRUCTURED LOGS FROM: {LOG_FILE_PATH}")
    print("="*80)
    
    # Close the file handle so we can read it
    logging.shutdown()
    os.close(sys.stdout.fileno())

    # Re-open stdout to print to console again
    sys.stdout = sys.__stdout__
    
    # Print each JSON line from the log file
    with open(LOG_FILE_PATH, 'r') as f:
        for line in f:
            print(line, end='')
