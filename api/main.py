import os
import sys
import logging
from dotenv import load_dotenv

load_dotenv()

from api.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

watchfiles_logger = logging.getLogger("watchfiles.main")
watchfiles_logger.setLevel(logging.DEBUG)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

is_development: bool = os.environ.get("NODE_ENV") != "production"
if is_development:
    import watchfiles
    current_dir: str = os.path.dirname(os.path.abspath(__file__))
    logs_dir: str = os.path.join(current_dir, "logs")

    original_watch = watchfiles.watch
    def patched_watch(*args, **kwargs):
        api_subdirs: list[str] = []
        for item in os.listdir(current_dir):
            item_path = os.path.join(current_dir, item)
            if os.path.isdir(item_path) and item != "logs":
                api_subdirs.append(item_path)

        api_subdirs.append(current_dir + "/*.py")

        return original_watch(*api_subdirs, **kwargs)
    watchfiles.watch = patched_watch

import uvicorn

required_env_vars: list[str] = ['GOOGLE_API_KEY', 'OPENAI_API_KEY']
missing_vars: list[str] = [var for var in required_env_vars if not os.environ.get(var)]
if missing_vars:
    logger.warning(f"Missing environment variables: {', '.join(missing_vars)}")
    logger.warning("Some functionality may not work correctly without these variables.")

import google.generativeai as genai
from api.config import GOOGLE_API_KEY

if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
else:
    logger.warning("GOOGLE_API_KEY not configured")

if __name__ == "__main__":
    port: int = int(os.environ.get("PORT", 8001))

    # Import the app here to ensure environment variables are set first
    from api.api import app

    logger.info(f"Starting Streaming API on port {port}")

    uvicorn.run(
        "api.api:app",
        host="0.0.0.0",
        port=port,
        reload=is_development,
        reload_excludes=["**/logs/*", "**/__pycache__/*", "**/*.pyc"] if is_development else None,
    )
