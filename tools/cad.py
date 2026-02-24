import subprocess
import tempfile
import os
import sys

def execute_build123d_script(script_content: str, output_path: str = "output.step") -> dict:
    """
    Executes a build123d script and captures the output/errors.
    Ensures the script exports to the specified output_path.
    """
    
    # Wrap the user script to ensure it exports correctly and handles errors
    
    # Write the script parts separately to avoid f-string/triple-quote collision issues
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as tmp_file:
        tmp_file.write("import sys\nimport os\n")
        tmp_file.write("try:\n    from build123d import *\nexcept ImportError:\n")
        tmp_file.write("    print(\"Error: build123d not installed. Please install it via 'pip install build123d'.\", file=sys.stderr)\n")
        tmp_file.write("    sys.exit(1)\n\n")
        
        tmp_file.write("# User script content starts here\n")
        tmp_file.write(script_content)
        tmp_file.write("\n# User script content ends here\n\n")
        
        tmp_file.write(f"# specific check to see if 'part' or relevant object exists and export if the user didn't\n")
        tmp_file.write(f"if 'part' in locals() and isinstance(part, (Part, BuildPart)):\n")
        tmp_file.write(f"    try:\n")
        tmp_file.write(f"        if isinstance(part, BuildPart):\n")
        tmp_file.write(f"            part.part.export_step(\"{output_path}\")\n")
        tmp_file.write(f"        else:\n")
        tmp_file.write(f"            part.export_step(\"{output_path}\")\n")
        tmp_file.write(f"        print(f\"Successfully exported to {output_path}\")\n")
        tmp_file.write(f"    except Exception as e:\n")
        tmp_file.write(f"        print(f\"Error exporting STEP file: {{e}}\", file=sys.stderr)\n")
        tmp_file.write(f"elif not os.path.exists(\"{output_path}\"):\n")
        tmp_file.write(f"     print(\"Warning: Script finished but no '{output_path}' was created and no 'part' object found.\", file=sys.stderr)\n")
        
        tmp_script_path = tmp_file.name

    try:
        # Run the script in a subprocess
        result = subprocess.run(
            [sys.executable, tmp_script_path],
            capture_output=True,
            text=True,
            timeout=30 # Safety timeout
        )
        
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "output_file": output_path if os.path.exists(output_path) else None
        }

    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "stdout": "",
            "stderr": "Execution timed out.",
            "output_file": None
        }
    finally:
        os.remove(tmp_script_path)

if __name__ == "__main__":
    # Test stub
    test_code = """
from build123d import *
with BuildPart() as part:
    Box(10, 10, 10)
"""
    print(execute_build123d_script(test_code))
