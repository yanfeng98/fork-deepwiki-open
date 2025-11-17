import { NextResponse } from 'next/server';

interface ApiProcessedProject {
  id: string;
  owner: string;
  repo: string;
  name: string;
  repo_type: string;
  submittedAt: number;
  language: string;
}

interface DeleteProjectCachePayload {
  owner: string;
  repo: string;
  repo_type: string;
  language: string;
}

function isDeleteProjectCachePayload(obj: unknown): obj is DeleteProjectCachePayload {
  return (
    obj != null &&
    typeof obj === 'object' &&
    'owner' in obj && typeof (obj as Record<string, unknown>).owner === 'string' && ((obj as Record<string, unknown>).owner as string).trim() !== '' &&
    'repo' in obj && typeof (obj as Record<string, unknown>).repo === 'string' && ((obj as Record<string, unknown>).repo as string).trim() !== '' &&
    'repo_type' in obj && typeof (obj as Record<string, unknown>).repo_type === 'string' && ((obj as Record<string, unknown>).repo_type as string).trim() !== '' &&
    'language' in obj && typeof (obj as Record<string, unknown>).language === 'string' && ((obj as Record<string, unknown>).language as string).trim() !== ''
  );
}

const PYTHON_BACKEND_URL = process.env.PYTHON_BACKEND_HOST || 'http://localhost:8001';
const PROJECTS_API_ENDPOINT = `${PYTHON_BACKEND_URL}/api/processed_projects`;
const CACHE_API_ENDPOINT = `${PYTHON_BACKEND_URL}/api/wiki_cache`;

export async function GET() {
  try {
    const response = await fetch(PROJECTS_API_ENDPOINT, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
      cache: 'no-store',
    });

    if (!response.ok) {
      let errorBody = { error: `Failed to fetch from Python backend: ${response.statusText}` };
      try {
        errorBody = await response.json();
      } catch {
        // If parsing JSON fails, errorBody will retain its default value
        // The error from backend is logged in the next line anyway
      }
      console.error(`Error from Python backend (${PROJECTS_API_ENDPOINT}): ${response.status} - ${JSON.stringify(errorBody)}`);
      return NextResponse.json(errorBody, { status: response.status });
    }

    const projects: ApiProcessedProject[] = await response.json();
    return NextResponse.json(projects);

  } catch (error: unknown) {
    console.error(`Network or other error when fetching from ${PROJECTS_API_ENDPOINT}:`, error);
    const message = error instanceof Error ? error.message : 'An unknown error occurred';
    return NextResponse.json(
      { error: `Failed to connect to the Python backend. ${message}` },
      { status: 503 }
    );
  }
}

export async function DELETE(request: Request) {
  try {
    const body: unknown = await request.json();
    if (!isDeleteProjectCachePayload(body)) {
      return NextResponse.json(
        { error: 'Invalid request body: owner, repo, repo_type, and language are required and must be non-empty strings.' },
        { status: 400 }
      );
    }
    const { owner, repo, repo_type, language } = body;
    const params = new URLSearchParams({ owner, repo, repo_type, language });
    const response = await fetch(`${CACHE_API_ENDPOINT}?${params}`, {
      method: 'DELETE',
      headers: { 'Content-Type': 'application/json' },
    });
    if (!response.ok) {
      let errorBody = { error: response.statusText };
      try {
        errorBody = await response.json();
      } catch {}
      console.error(`Error deleting project cache (${CACHE_API_ENDPOINT}): ${response.status} - ${JSON.stringify(errorBody)}`);
      return NextResponse.json(errorBody, { status: response.status });
    }
    return NextResponse.json({ message: 'Project deleted successfully' });
  } catch (error: unknown) {
    console.error('Error in DELETE /api/wiki/projects:', error);
    const message = error instanceof Error ? error.message : 'An unknown error occurred';
    return NextResponse.json({ error: `Failed to delete project: ${message}` }, { status: 500 });
  }
}