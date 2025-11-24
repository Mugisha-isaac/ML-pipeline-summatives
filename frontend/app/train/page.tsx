'use client'

import { useState } from 'react'
import Link from 'next/link'

export default function TrainPage() {
  const [files, setFiles] = useState<File[]>([])
  const [epochs, setEpochs] = useState(50)
  const [batchSize, setBatchSize] = useState(32)
  const [loading, setLoading] = useState(false)
  const [status, setStatus] = useState<any>(null)
  const [error, setError] = useState<string | null>(null)

  const handleFilesChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      setFiles(Array.from(e.target.files))
      setError(null)
    }
  }

  const handleUpload = async () => {
    if (files.length === 0) {
      setError('Please select audio files')
      return
    }

    setLoading(true)
    setError(null)

    try {
      const formData = new FormData()
      files.forEach(file => formData.append('files', file))

      const response = await fetch(
        `${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/api/v1/upload-data`,
        {
          method: 'POST',
          body: formData,
        }
      )

      if (!response.ok) throw new Error('Upload failed')
      setStatus({ message: 'Files uploaded successfully' })
      setFiles([])
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Upload failed')
    } finally {
      setLoading(false)
    }
  }

  const handleRetrain = async () => {
    setLoading(true)
    setError(null)

    try {
      const response = await fetch(
        `${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/api/v1/retrain`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ epochs, batch_size: batchSize }),
        }
      )

      if (!response.ok) throw new Error('Retraining failed')
      const data = await response.json()
      setStatus(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Retraining failed')
    } finally {
      setLoading(false)
    }
  }

  const checkTrainingStatus = async () => {
    try {
      const response = await fetch(
        `${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/api/v1/train-status`
      )
      const data = await response.json()
      setStatus(data)
    } catch (err) {
      setError('Failed to check status')
    }
  }

  return (
    <main className="min-h-screen bg-gradient-to-b from-slate-900 to-slate-800">
      <div className="container mx-auto px-4 py-20">
        <Link href="/" className="text-blue-400 hover:text-blue-300 mb-8 inline-block">
          Back to Home
        </Link>

        <div className="max-w-3xl mx-auto">
          <h1 className="text-4xl font-bold text-white mb-8">Train Model</h1>

          <div className="space-y-6">
            <div className="bg-slate-700 rounded-lg p-8">
              <h2 className="text-2xl font-semibold text-white mb-4">Step 1: Upload Data</h2>
              <div className="mb-4">
                <input
                  type="file"
                  multiple
                  accept="audio/*"
                  onChange={handleFilesChange}
                  className="block w-full text-sm text-slate-300
                    file:mr-4 file:py-2 file:px-4
                    file:rounded-md file:border-0
                    file:text-sm file:font-semibold
                    file:bg-blue-600 file:text-white
                    hover:file:bg-blue-700"
                />
                {files.length > 0 && (
                  <p className="text-slate-300 mt-2">Selected: {files.length} files</p>
                )}
              </div>
              <button
                onClick={handleUpload}
                disabled={loading || files.length === 0}
                className="w-full bg-green-600 hover:bg-green-700 disabled:bg-slate-500
                  text-white font-semibold py-2 rounded-md transition"
              >
                {loading ? 'Uploading...' : 'Upload Files'}
              </button>
            </div>

            <div className="bg-slate-700 rounded-lg p-8">
              <h2 className="text-2xl font-semibold text-white mb-4">Step 2: Configure Training</h2>
              <div className="space-y-4">
                <div>
                  <label className="block text-white mb-2">Epochs: {epochs}</label>
                  <input
                    type="range"
                    min="10"
                    max="200"
                    value={epochs}
                    onChange={(e) => setEpochs(Number(e.target.value))}
                    className="w-full"
                  />
                </div>
                <div>
                  <label className="block text-white mb-2">Batch Size: {batchSize}</label>
                  <input
                    type="range"
                    min="8"
                    max="128"
                    step="8"
                    value={batchSize}
                    onChange={(e) => setBatchSize(Number(e.target.value))}
                    className="w-full"
                  />
                </div>
              </div>
            </div>

            <div className="bg-slate-700 rounded-lg p-8">
              <h2 className="text-2xl font-semibold text-white mb-4">Step 3: Retrain</h2>
              <button
                onClick={handleRetrain}
                disabled={loading}
                className="w-full bg-purple-600 hover:bg-purple-700 disabled:bg-slate-500
                  text-white font-semibold py-3 rounded-md transition"
              >
                {loading ? 'Training...' : 'Start Training'}
              </button>
            </div>

            {status && (
              <div className="bg-blue-900 rounded-lg p-8">
                <h2 className="text-2xl font-bold text-blue-200 mb-4">Status</h2>
                <pre className="text-blue-100 overflow-auto bg-slate-800 p-4 rounded text-sm">
                  {JSON.stringify(status, null, 2)}
                </pre>
                <button
                  onClick={checkTrainingStatus}
                  className="mt-4 bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-4 rounded"
                >
                  Refresh Status
                </button>
              </div>
            )}

            {error && (
              <div className="p-4 bg-red-900 text-red-200 rounded-md">
                {error}
              </div>
            )}
          </div>
        </div>
      </div>
    </main>
  )
}
