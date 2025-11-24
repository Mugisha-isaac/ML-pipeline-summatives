import axios from 'axios'

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

const apiClient = axios.create({
  baseURL: `${API_BASE_URL}/api/v1`,
  headers: {
    'Content-Type': 'application/json',
  },
})

export const healthApi = {
  check: () => apiClient.get('/health'),
  getModelInfo: () => apiClient.get('/model-info'),
}

export const predictionsApi = {
  single: (file: File) => {
    const formData = new FormData()
    formData.append('file', file)
    return apiClient.post('/predictions/single', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    })
  },
  batch: (files: File[]) => {
    const formData = new FormData()
    files.forEach(file => formData.append('files', file))
    return apiClient.post('/predictions/batch', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    })
  },
}

export const trainingApi = {
  uploadData: (files: File[]) => {
    const formData = new FormData()
    files.forEach(file => formData.append('files', file))
    return apiClient.post('/upload-data', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    })
  },
  retrain: (epochs: number, batchSize: number) =>
    apiClient.post('/retrain', { epochs, batch_size: batchSize }),
  getStatus: () => apiClient.get('/train-status'),
  getMetrics: () => apiClient.get('/model-metrics'),
}

export const visualizationsApi = {
  getMfcc: () => apiClient.get('/visualizations/mfcc'),
  getSpectral: () => apiClient.get('/visualizations/spectral'),
  getFeatureInfo: () => apiClient.get('/visualizations/feature-info'),
}

export default apiClient
