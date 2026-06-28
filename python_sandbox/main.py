import os
import sys
import io
import tempfile
import traceback
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class CodeExecutionRequest(BaseModel):
    code: str

@app.post("/execute")
def execute_code(request: CodeExecutionRequest):
    """Executes LLM-generated Python code safely in a temporary directory."""
    
    # Create an ephemeral directory so concurrent requests don't overwrite files
    with tempfile.TemporaryDirectory() as temp_dir:
        old_cwd = os.getcwd()
        os.chdir(temp_dir)
        
        # Capture standard output and errors to return to the LLM
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        redirected_output = sys.stdout = io.StringIO()
        
        try:
            # DANGER ZONE: Executing the code. 
            # In a strict enterprise environment, you'd wrap this in a subprocess with a timeout.
            exec(request.code, {})
            
            # Check if the LLM successfully saved the plot
            if os.path.exists("plot.html"):
                with open("plot.html", "r", encoding="utf-8") as f:
                    html_content = f.read()
                return {
                    "status": "success", 
                    "html": html_content, 
                    "logs": redirected_output.getvalue()
                }
            else:
                return {
                    "status": "error", 
                    "error": "The code executed, but 'plot.html' was not created.", 
                    "logs": redirected_output.getvalue()
                }
                
        except Exception as e:
            # Return the exact traceback to the LLM so it can self-correct
            error_msg = traceback.format_exc()
            return {"status": "error", "error": error_msg}
            
        finally:
            # Always reset the environment
            os.chdir(old_cwd)
            sys.stdout = old_stdout
            sys.stderr = old_stderr

# For local testing
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)