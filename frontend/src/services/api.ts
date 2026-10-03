import type {
  AnalysisResultResponse,
  DashboardStats,
  EvidenceUploadResponse,
  SystemHealth,
} from '../types/forensic';

const API_BASE = '/api/v1';

export class ApiError extends Error {
  status: number;
  data: any;

  constructor(message: string, status: number, data?: any) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.data = data;
  }
}

async function request<T>(endpoint: string, options?: RequestInit): Promise<T> {
  const url = `${API_BASE}${endpoint}`;
  try {
    const res = await fetch(url, {
      ...options,
      headers: {
        'Accept': 'application/json',
        ...(options?.headers || {}),
      },
    });

    if (!res.ok) {
      let errorBody: any = null;
      try {
        errorBody = await res.json();
      } catch {
        errorBody = await res.text();
      }
      const msg = errorBody?.detail || errorBody?.error?.message || `HTTP ${res.status}: ${res.statusText}`;
      throw new ApiError(msg, res.status, errorBody);
    }

    return (await res.json()) as T;
  } catch (err) {
    if (err instanceof ApiError) {
      throw err;
    }
    throw new ApiError(
      `Network connection failed to backend service: ${(err as Error).message}`,
      0
    );
  }
}

export const api = {
  getHealth: async (): Promise<SystemHealth> => {
    return request<SystemHealth>('/health');
  },

  getDashboardStats: async (): Promise<DashboardStats> => {
    return request<DashboardStats>('/evidence/dashboard/stats');
  },

  listEvidence: async (): Promise<EvidenceUploadResponse[]> => {
    return request<EvidenceUploadResponse[]>('/evidence');
  },

  getEvidence: async (evidenceId: string): Promise<EvidenceUploadResponse> => {
    return request<EvidenceUploadResponse>(`/evidence/${evidenceId}`);
  },

  getEvidenceFileUrl: (evidenceId: string): string => {
    return `${API_BASE}/evidence/${evidenceId}/file`;
  },

  uploadEvidence: async (
    file: File,
    autoAnalyze: boolean = true
  ): Promise<AnalysisResultResponse | EvidenceUploadResponse> => {
    const formData = new FormData();
    formData.append('file', file);

    const res = await fetch(`${API_BASE}/evidence/upload?auto_analyze=${autoAnalyze}`, {
      method: 'POST',
      body: formData,
    });

    if (!res.ok) {
      let errorBody: any = null;
      try {
        errorBody = await res.json();
      } catch {
        errorBody = await res.text();
      }
      const msg = errorBody?.detail || `Upload failed (HTTP ${res.status})`;
      throw new ApiError(msg, res.status, errorBody);
    }

    return await res.json();
  },

  getAnalysis: async (evidenceId: string): Promise<AnalysisResultResponse> => {
    return request<AnalysisResultResponse>(`/evidence/${evidenceId}/analysis`);
  },

  getReport: async (evidenceId: string): Promise<any> => {
    return request<any>(`/evidence/${evidenceId}/report`);
  },

  getReportPdfUrl: (evidenceId: string): string => {
    return `${API_BASE}/evidence/${evidenceId}/report/pdf`;
  },

  downloadReportPdf: async (evidenceId: string, filename?: string): Promise<void> => {
    const res = await fetch(`${API_BASE}/evidence/${evidenceId}/report/pdf`);
    if (!res.ok) {
      throw new Error(`Failed to download official PDF report (HTTP ${res.status})`);
    }
    const blob = await res.blob();
    const url = window.URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = filename || `TRUSTTRACE_FORENSIC_REPORT_${evidenceId.slice(0, 8).toUpperCase()}.pdf`;
    document.body.appendChild(link);
    link.click();
    window.URL.revokeObjectURL(url);
    document.body.removeChild(link);
  },
};
