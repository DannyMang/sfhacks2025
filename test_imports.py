import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("Python path:", sys.path)

try:
    from app.components.avatar_pipeline import AvatarPipeline
    print("Successfully imported AvatarPipeline")
except ImportError as e:
    print("Import error:", e)
    print("Current working directory:", os.getcwd())
    print("Directory contents:", os.listdir('.'))
    print("app directory contents:", os.listdir('app') if os.path.exists('app') else "app directory not found")