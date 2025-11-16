import adalflow as adal
from adalflow.core.types import Document, List, Any
from adalflow.components.data_process import TextSplitter, ToEmbeddings
import os
import subprocess
import json
import tiktoken
import logging
import base64
import glob
from adalflow.utils import get_adalflow_default_root_path
from adalflow.core.db import LocalDB
from api.config import configs, DEFAULT_EXCLUDED_DIRS, DEFAULT_EXCLUDED_FILES
from api.ollama_patch import OllamaDocumentProcessor
from urllib.parse import urlparse, urlunparse, quote
import requests
from requests.exceptions import RequestException

from api.tools.embedder import get_embedder

# Configure logging
logger = logging.getLogger(__name__)

MAX_EMBEDDING_TOKENS: int = 8192

def count_tokens(text: str, embedder_type: str = None, is_ollama_embedder: bool = None) -> int:
    try:
        if embedder_type is None and is_ollama_embedder is not None:
            embedder_type = 'ollama' if is_ollama_embedder else None
        
        if embedder_type is None:
            from api.config import get_embedder_type
            embedder_type: str = get_embedder_type()

        if embedder_type == 'ollama':
            encoding = tiktoken.get_encoding("cl100k_base")
        elif embedder_type == 'google':
            encoding = tiktoken.get_encoding("cl100k_base")
        else:
            encoding = tiktoken.encoding_for_model("text-embedding-3-small")

        return len(encoding.encode(text))
    except Exception as e:
        logger.warning(f"Error counting tokens with tiktoken: {e}")
        return len(text) // 4

def download_repo(repo_url: str, local_path: str, repo_type: str = None, access_token: str = None) -> str:
    try:
        logger.info(f"Preparing to clone repository to {local_path}")
        subprocess.run(
            ["git", "--version"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        if os.path.exists(local_path) and os.listdir(local_path):
            logger.warning(f"Repository already exists at {local_path}. Using existing repository.")
            return f"Using existing repository at {local_path}"

        os.makedirs(local_path, exist_ok=True)

        clone_url: str = repo_url
        if access_token:
            parsed = urlparse(repo_url)
            if repo_type == "github":
                # Format: https://{token}@{domain}/owner/repo.git
                # Works for both github.com and enterprise GitHub domains
                clone_url = urlunparse((parsed.scheme, f"{access_token}@{parsed.netloc}", parsed.path, '', '', ''))
            elif repo_type == "gitlab":
                # Format: https://oauth2:{token}@gitlab.com/owner/repo.git
                clone_url = urlunparse((parsed.scheme, f"oauth2:{access_token}@{parsed.netloc}", parsed.path, '', '', ''))
            elif repo_type == "bitbucket":
                # Format: https://x-token-auth:{token}@bitbucket.org/owner/repo.git
                clone_url = urlunparse((parsed.scheme, f"x-token-auth:{access_token}@{parsed.netloc}", parsed.path, '', '', ''))

            logger.info("Using access token for authentication")

        logger.info(f"Cloning repository from {repo_url} to {local_path}")
        result = subprocess.run(
            ["git", "clone", "--depth=1", "--single-branch", clone_url, local_path],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        logger.info("Repository cloned successfully")
        return result.stdout.decode("utf-8")

    except subprocess.CalledProcessError as e:
        error_msg: str = e.stderr.decode('utf-8')
        if access_token and access_token in error_msg:
            error_msg = error_msg.replace(access_token, "***TOKEN***")
        raise ValueError(f"Error during cloning: {error_msg}")
    except Exception as e:
        raise ValueError(f"An unexpected error occurred: {str(e)}")

# Alias for backward compatibility
download_github_repo = download_repo

def read_all_documents(path: str, embedder_type: str = None, is_ollama_embedder: bool = None, 
                      excluded_dirs: List[str] = None, excluded_files: List[str] = None,
                      included_dirs: List[str] = None, included_files: List[str] = None) -> List[Document]:
    if embedder_type is None and is_ollama_embedder is not None:
        embedder_type = 'ollama' if is_ollama_embedder else None
    documents: List[Document] = []

    code_extensions: list[str] = [".py", ".js", ".ts", ".java", ".cpp", ".c", ".h", ".hpp", ".go", ".rs",
                       ".jsx", ".tsx", ".html", ".css", ".php", ".swift", ".cs"]
    doc_extensions: list[str] = [".md", ".txt", ".rst", ".json", ".yaml", ".yml"]

    use_inclusion_mode: bool = (included_dirs is not None and len(included_dirs) > 0) or (included_files is not None and len(included_files) > 0)

    if use_inclusion_mode:
        final_included_dirs: set[str] = set(included_dirs) if included_dirs else set()
        final_included_files: set[str] = set(included_files) if included_files else set()

        logger.info(f"Using inclusion mode")
        logger.info(f"Included directories: {list(final_included_dirs)}")
        logger.info(f"Included files: {list(final_included_files)}")

        included_dirs: list[str] = list(final_included_dirs)
        included_files: list[str] = list(final_included_files)
        excluded_dirs: list[str] = []
        excluded_files: list[str] = []
    else:
        final_excluded_dirs: set[str] = set(DEFAULT_EXCLUDED_DIRS)
        final_excluded_files: set[str] = set(DEFAULT_EXCLUDED_FILES)

        if "file_filters" in configs and "excluded_dirs" in configs["file_filters"]:
            final_excluded_dirs.update(configs["file_filters"]["excluded_dirs"])

        if "file_filters" in configs and "excluded_files" in configs["file_filters"]:
            final_excluded_files.update(configs["file_filters"]["excluded_files"])

        if excluded_dirs is not None:
            final_excluded_dirs.update(excluded_dirs)

        if excluded_files is not None:
            final_excluded_files.update(excluded_files)

        excluded_dirs: list[str] = list(final_excluded_dirs)
        excluded_files: list[str] = list(final_excluded_files)
        included_dirs: list[str] = []
        included_files: list[str] = []

        logger.info(f"Using exclusion mode")
        logger.info(f"Excluded directories: {excluded_dirs}")
        logger.info(f"Excluded files: {excluded_files}")

    logger.info(f"Reading documents from {path}")

    def should_process_file(file_path: str, use_inclusion: bool, included_dirs: List[str], included_files: List[str],
                           excluded_dirs: List[str], excluded_files: List[str]) -> bool:
        file_path_parts: list[str] = os.path.normpath(file_path).split(os.sep)
        file_name: str = os.path.basename(file_path)

        if use_inclusion:
            is_included: bool = False

            if included_dirs:
                for included in included_dirs:
                    clean_included: str = included.strip("./").rstrip("/")
                    if clean_included in file_path_parts:
                        is_included = True
                        break

            if not is_included and included_files:
                for included_file in included_files:
                    if file_name == included_file or file_name.endswith(included_file):
                        is_included = True
                        break

            if not included_dirs and not included_files:
                is_included = True
            elif not included_dirs and included_files:
                pass
            elif included_dirs and not included_files:
                pass

            return is_included
        else:
            is_excluded: bool = False

            for excluded in excluded_dirs:
                clean_excluded: str = excluded.strip("./").rstrip("/")
                if clean_excluded in file_path_parts:
                    is_excluded = True
                    break

            if not is_excluded:
                for excluded_file in excluded_files:
                    if file_name == excluded_file:
                        is_excluded = True
                        break

            return not is_excluded

    for ext in code_extensions:
        files: list[str] = glob.glob(f"{path}/**/*{ext}", recursive=True)
        for file_path in files:
            if not should_process_file(file_path, use_inclusion_mode, included_dirs, included_files, excluded_dirs, excluded_files):
                continue

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content: str = f.read()
                    relative_path: str = os.path.relpath(file_path, path)

                    is_implementation: bool = (
                        not relative_path.startswith("test_")
                        and not relative_path.startswith("app_")
                        and "test" not in relative_path.lower()
                    )

                    token_count: int = count_tokens(content, embedder_type)
                    if token_count > MAX_EMBEDDING_TOKENS * 10:
                        logger.warning(f"Skipping large file {relative_path}: Token count ({token_count}) exceeds limit")
                        continue

                    doc: Document = Document(
                        text=content,
                        meta_data={
                            "file_path": relative_path,
                            "type": ext[1:],
                            "is_code": True,
                            "is_implementation": is_implementation,
                            "title": relative_path,
                            "token_count": token_count,
                        },
                    )
                    documents.append(doc)
            except Exception as e:
                logger.error(f"Error reading {file_path}: {e}")

    for ext in doc_extensions:
        files: list[str] = glob.glob(f"{path}/**/*{ext}", recursive=True)
        for file_path in files:
            if not should_process_file(file_path, use_inclusion_mode, included_dirs, included_files, excluded_dirs, excluded_files):
                continue

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content: str = f.read()
                    relative_path: str = os.path.relpath(file_path, path)

                    token_count: int = count_tokens(content, embedder_type)
                    if token_count > MAX_EMBEDDING_TOKENS:
                        logger.warning(f"Skipping large file {relative_path}: Token count ({token_count}) exceeds limit")
                        continue

                    doc: Document = Document(
                        text=content,
                        meta_data={
                            "file_path": relative_path,
                            "type": ext[1:],
                            "is_code": False,
                            "is_implementation": False,
                            "title": relative_path,
                            "token_count": token_count,
                        },
                    )
                    documents.append(doc)
            except Exception as e:
                logger.error(f"Error reading {file_path}: {e}")

    logger.info(f"Found {len(documents)} documents")
    return documents

def transform_documents_and_save_to_db(
    documents: List[Document], db_path: str, embedder_type: str = None, is_ollama_embedder: bool = None
) -> LocalDB:
    data_transformer: adal.Sequential = prepare_data_pipeline(embedder_type, is_ollama_embedder)

    db: LocalDB = LocalDB()
    db.register_transformer(transformer=data_transformer, key="split_and_embed")
    db.load(documents)
    db.transform(key="split_and_embed")
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    db.save_state(filepath=db_path)
    return db

def prepare_data_pipeline(embedder_type: str = None, is_ollama_embedder: bool = None) -> adal.Sequential:
    from api.config import get_embedder_config, get_embedder_type

    if embedder_type is None and is_ollama_embedder is not None:
        embedder_type = 'ollama' if is_ollama_embedder else None
    
    if embedder_type is None:
        embedder_type = get_embedder_type()

    splitter: TextSplitter = TextSplitter(**configs["text_splitter"])
    embedder_config: dict[str, Any] = get_embedder_config()

    embedder: adal.Embedder = get_embedder(embedder_type=embedder_type)

    # Choose appropriate processor based on embedder type
    if embedder_type == 'ollama':
        # Use Ollama document processor for single-document processing
        embedder_transformer = OllamaDocumentProcessor(embedder=embedder)
    else:
        batch_size: int = embedder_config.get("batch_size", 500)
        embedder_transformer: ToEmbeddings = ToEmbeddings(
            embedder=embedder, batch_size=batch_size
        )

    data_transformer: adal.Sequential = adal.Sequential(
        splitter, embedder_transformer
    )
    return data_transformer

def get_file_content(repo_url: str, file_path: str, repo_type: str = None, access_token: str = None) -> str:
    if repo_type == "github":
        return get_github_file_content(repo_url, file_path, access_token)
    elif repo_type == "gitlab":
        return get_gitlab_file_content(repo_url, file_path, access_token)
    elif repo_type == "bitbucket":
        return get_bitbucket_file_content(repo_url, file_path, access_token)
    else:
        raise ValueError("Unsupported repository type. Only GitHub, GitLab, and Bitbucket are supported.")

def get_github_file_content(repo_url: str, file_path: str, access_token: str = None) -> str:
    try:
        parsed_url = urlparse(repo_url)
        if not parsed_url.scheme or not parsed_url.netloc:
            raise ValueError("Not a valid GitHub repository URL")

        path_parts: list[str] = parsed_url.path.strip('/').split('/')
        if len(path_parts) < 2:
            raise ValueError("Invalid GitHub URL format - expected format: https://domain/owner/repo")

        owner: str = path_parts[-2]
        repo: str = path_parts[-1].replace(".git", "")

        if parsed_url.netloc == "github.com":
            api_base: str = "https://api.github.com"
        else:
            api_base: str = f"{parsed_url.scheme}://{parsed_url.netloc}/api/v3"
        
        api_url: str = f"{api_base}/repos/{owner}/{repo}/contents/{file_path}"
        headers: dict[str, str] = {}
        if access_token:
            headers["Authorization"] = f"token {access_token}"
        logger.info(f"Fetching file content from GitHub API: {api_url}")
        try:
            response = requests.get(api_url, headers=headers)
            response.raise_for_status()
        except RequestException as e:
            raise ValueError(f"Error fetching file content: {e}")
        try:
            content_data = response.json()
        except json.JSONDecodeError:
            raise ValueError("Invalid response from GitHub API")

        if "message" in content_data and "documentation_url" in content_data:
            raise ValueError(f"GitHub API error: {content_data['message']}")

        if "content" in content_data and "encoding" in content_data:
            if content_data["encoding"] == "base64":
                content_base64: str = content_data["content"].replace("\n", "")
                content: str = base64.b64decode(content_base64).decode("utf-8")
                return content
            else:
                raise ValueError(f"Unexpected encoding: {content_data['encoding']}")
        else:
            raise ValueError("File content not found in GitHub API response")

    except Exception as e:
        raise ValueError(f"Failed to get file content: {str(e)}")

def get_gitlab_file_content(repo_url: str, file_path: str, access_token: str = None) -> str:
    """
    Retrieves the content of a file from a GitLab repository (cloud or self-hosted).

    Args:
        repo_url (str): The GitLab repo URL (e.g., "https://gitlab.com/username/repo" or "http://localhost/group/project")
        file_path (str): File path within the repository (e.g., "src/main.py")
        access_token (str, optional): GitLab personal access token

    Returns:
        str: File content

    Raises:
        ValueError: If anything fails
    """
    try:
        # Parse and validate the URL
        parsed_url = urlparse(repo_url)
        if not parsed_url.scheme or not parsed_url.netloc:
            raise ValueError("Not a valid GitLab repository URL")

        gitlab_domain = f"{parsed_url.scheme}://{parsed_url.netloc}"
        if parsed_url.port not in (None, 80, 443):
            gitlab_domain += f":{parsed_url.port}"
        path_parts = parsed_url.path.strip("/").split("/")
        if len(path_parts) < 2:
            raise ValueError("Invalid GitLab URL format — expected something like https://gitlab.domain.com/group/project")

        # Build project path and encode for API
        project_path = "/".join(path_parts).replace(".git", "")
        encoded_project_path = quote(project_path, safe='')

        # Encode file path
        encoded_file_path = quote(file_path, safe='')

        # Try to get the default branch from the project info
        default_branch = None
        try:
            project_info_url = f"{gitlab_domain}/api/v4/projects/{encoded_project_path}"
            project_headers = {}
            if access_token:
                project_headers["PRIVATE-TOKEN"] = access_token
            
            project_response = requests.get(project_info_url, headers=project_headers)
            if project_response.status_code == 200:
                project_data = project_response.json()
                default_branch = project_data.get('default_branch', 'main')
                logger.info(f"Found default branch: {default_branch}")
            else:
                logger.warning(f"Could not fetch project info, using 'main' as default branch")
                default_branch = 'main'
        except Exception as e:
            logger.warning(f"Error fetching project info: {e}, using 'main' as default branch")
            default_branch = 'main'

        api_url = f"{gitlab_domain}/api/v4/projects/{encoded_project_path}/repository/files/{encoded_file_path}/raw?ref={default_branch}"
        # Fetch file content from GitLab API
        headers = {}
        if access_token:
            headers["PRIVATE-TOKEN"] = access_token
        logger.info(f"Fetching file content from GitLab API: {api_url}")
        try:
            response = requests.get(api_url, headers=headers)
            response.raise_for_status()
            content = response.text
        except RequestException as e:
            raise ValueError(f"Error fetching file content: {e}")

        # Check for GitLab error response (JSON instead of raw file)
        if content.startswith("{") and '"message":' in content:
            try:
                error_data = json.loads(content)
                if "message" in error_data:
                    raise ValueError(f"GitLab API error: {error_data['message']}")
            except json.JSONDecodeError:
                pass

        return content

    except Exception as e:
        raise ValueError(f"Failed to get file content: {str(e)}")

def get_bitbucket_file_content(repo_url: str, file_path: str, access_token: str = None) -> str:
    """
    Retrieves the content of a file from a Bitbucket repository using the Bitbucket API.

    Args:
        repo_url (str): The URL of the Bitbucket repository (e.g., "https://bitbucket.org/username/repo")
        file_path (str): The path to the file within the repository (e.g., "src/main.py")
        access_token (str, optional): Bitbucket personal access token for private repositories

    Returns:
        str: The content of the file as a string
    """
    try:
        # Extract owner and repo name from Bitbucket URL
        if not (repo_url.startswith("https://bitbucket.org/") or repo_url.startswith("http://bitbucket.org/")):
            raise ValueError("Not a valid Bitbucket repository URL")

        parts = repo_url.rstrip('/').split('/')
        if len(parts) < 5:
            raise ValueError("Invalid Bitbucket URL format")

        owner = parts[-2]
        repo = parts[-1].replace(".git", "")

        # Try to get the default branch from the repository info
        default_branch = None
        try:
            repo_info_url = f"https://api.bitbucket.org/2.0/repositories/{owner}/{repo}"
            repo_headers = {}
            if access_token:
                repo_headers["Authorization"] = f"Bearer {access_token}"
            
            repo_response = requests.get(repo_info_url, headers=repo_headers)
            if repo_response.status_code == 200:
                repo_data = repo_response.json()
                default_branch = repo_data.get('mainbranch', {}).get('name', 'main')
                logger.info(f"Found default branch: {default_branch}")
            else:
                logger.warning(f"Could not fetch repository info, using 'main' as default branch")
                default_branch = 'main'
        except Exception as e:
            logger.warning(f"Error fetching repository info: {e}, using 'main' as default branch")
            default_branch = 'main'

        # Use Bitbucket API to get file content
        # The API endpoint for getting file content is: /2.0/repositories/{owner}/{repo}/src/{branch}/{path}
        api_url = f"https://api.bitbucket.org/2.0/repositories/{owner}/{repo}/src/{default_branch}/{file_path}"

        # Fetch file content from Bitbucket API
        headers = {}
        if access_token:
            headers["Authorization"] = f"Bearer {access_token}"
        logger.info(f"Fetching file content from Bitbucket API: {api_url}")
        try:
            response = requests.get(api_url, headers=headers)
            if response.status_code == 200:
                content = response.text
            elif response.status_code == 404:
                raise ValueError("File not found on Bitbucket. Please check the file path and repository.")
            elif response.status_code == 401:
                raise ValueError("Unauthorized access to Bitbucket. Please check your access token.")
            elif response.status_code == 403:
                raise ValueError("Forbidden access to Bitbucket. You might not have permission to access this file.")
            elif response.status_code == 500:
                raise ValueError("Internal server error on Bitbucket. Please try again later.")
            else:
                response.raise_for_status()
                content = response.text
            return content
        except RequestException as e:
            raise ValueError(f"Error fetching file content: {e}")

    except Exception as e:
        raise ValueError(f"Failed to get file content: {str(e)}")

class DatabaseManager:

    def __init__(self):
        self.db = None
        self.repo_url_or_path = None
        self.repo_paths = None

    def prepare_database(self, repo_url_or_path: str, repo_type: str = None, access_token: str = None,
                         embedder_type: str = None, is_ollama_embedder: bool = None,
                         excluded_dirs: List[str] = None, excluded_files: List[str] = None,
                         included_dirs: List[str] = None, included_files: List[str] = None) -> List[Document]:
        if embedder_type is None and is_ollama_embedder is not None:
            embedder_type = 'ollama' if is_ollama_embedder else None
        
        self.reset_database()
        self._create_repo(repo_url_or_path, repo_type, access_token)
        return self.prepare_db_index(embedder_type=embedder_type, excluded_dirs=excluded_dirs, excluded_files=excluded_files,
                                   included_dirs=included_dirs, included_files=included_files)

    def reset_database(self) -> None:
        self.db: LocalDB = None
        self.repo_url_or_path: str = None
        self.repo_paths: List[str] = None

    def _create_repo(self, repo_url_or_path: str, repo_type: str = None, access_token: str = None) -> None:
        logger.info(f"Preparing repo storage for {repo_url_or_path}...")

        try:
            root_path: str = get_adalflow_default_root_path()
            os.makedirs(root_path, exist_ok=True)
            
            if repo_url_or_path.startswith("https://") or repo_url_or_path.startswith("http://"):
                repo_name: str = self._extract_repo_name_from_url(repo_url_or_path, repo_type)
                logger.info(f"Extracted repo name: {repo_name}")

                save_repo_dir: str = os.path.join(root_path, "repos", repo_name)
                if not (os.path.exists(save_repo_dir) and os.listdir(save_repo_dir)):
                    download_repo(repo_url_or_path, save_repo_dir, repo_type, access_token)
                else:
                    logger.info(f"Repository already exists at {save_repo_dir}. Using existing repository.")
            else:
                repo_name: str = os.path.basename(repo_url_or_path)
                save_repo_dir: str = repo_url_or_path

            save_db_file: str = os.path.join(root_path, "databases", f"{repo_name}.pkl")
            os.makedirs(save_repo_dir, exist_ok=True)
            os.makedirs(os.path.dirname(save_db_file), exist_ok=True)

            self.repo_paths: dict[str, str] = {
                "save_repo_dir": save_repo_dir,
                "save_db_file": save_db_file,
            }
            self.repo_url_or_path: str = repo_url_or_path
            logger.info(f"Repo paths: {self.repo_paths}")

        except Exception as e:
            logger.error(f"Failed to create repository structure: {e}")
            raise

    def _extract_repo_name_from_url(self, repo_url_or_path: str, repo_type: str) -> str:
        url_parts: List[str] = repo_url_or_path.rstrip('/').split('/')

        if repo_type in ["github", "gitlab", "bitbucket"] and len(url_parts) >= 5:
            # GitHub URL format: https://github.com/owner/repo
            # GitLab URL format: https://gitlab.com/owner/repo or https://gitlab.com/group/subgroup/repo
            # Bitbucket URL format: https://bitbucket.org/owner/repo
            owner: str = url_parts[-2]
            repo: str = url_parts[-1].replace(".git", "")
            repo_name: str = f"{owner}_{repo}"
        else:
            repo_name: str = url_parts[-1].replace(".git", "")
        return repo_name

    def prepare_db_index(self, embedder_type: str = None, is_ollama_embedder: bool = None, 
                        excluded_dirs: List[str] = None, excluded_files: List[str] = None,
                        included_dirs: List[str] = None, included_files: List[str] = None) -> List[Document]:
        if embedder_type is None and is_ollama_embedder is not None:
            embedder_type = 'ollama' if is_ollama_embedder else None
        
        if self.repo_paths and os.path.exists(self.repo_paths["save_db_file"]):
            logger.info("Loading existing database...")
            try:
                self.db = LocalDB.load_state(self.repo_paths["save_db_file"])
                documents: List[Document] = self.db.get_transformed_data(key="split_and_embed")
                if documents:
                    logger.info(f"Loaded {len(documents)} documents from existing database")
                    return documents
            except Exception as e:
                logger.error(f"Error loading existing database: {e}")

        logger.info("Creating new database...")
        documents: List[Document] = read_all_documents(
            self.repo_paths["save_repo_dir"],
            embedder_type=embedder_type,
            excluded_dirs=excluded_dirs,
            excluded_files=excluded_files,
            included_dirs=included_dirs,
            included_files=included_files
        )
        self.db: List[Document] = transform_documents_and_save_to_db(
            documents, self.repo_paths["save_db_file"], embedder_type=embedder_type
        )
        logger.info(f"Total documents: {len(documents)}")
        transformed_docs = self.db.get_transformed_data(key="split_and_embed")
        logger.info(f"Total transformed documents: {len(transformed_docs)}")
        return transformed_docs

    def prepare_retriever(self, repo_url_or_path: str, repo_type: str = None, access_token: str = None):
        """
        Prepare the retriever for a repository.
        This is a compatibility method for the isolated API.

        Args:
            repo_type(str): Type of repository
            repo_url_or_path (str): The URL or local path of the repository
            access_token (str, optional): Access token for private repositories

        Returns:
            List[Document]: List of Document objects
        """
        return self.prepare_database(repo_url_or_path, repo_type, access_token)
