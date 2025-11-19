export function extractUrlDomain(input: string): string | null {
    try {
        const normalizedInput = input.startsWith('http') ? input : `https://${input}`;
        const url = new URL(normalizedInput);
        return `${url.protocol}//${url.hostname}${url.port ? ':' + url.port : ''}`;
    } catch {
        return null;
    }
}

export function extractUrlPath(input: string): string | null {
    try {
        const normalizedInput = input.startsWith('http') ? input : `https://${input}`;
        const url = new URL(normalizedInput);
        return url.pathname.replace(/^\/|\/$/g, '');
    } catch {
        return null;
    }
}