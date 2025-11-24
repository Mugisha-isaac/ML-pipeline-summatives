'use client'

import { useState } from 'react'
import Link from 'next/link'

export default function PredictPage() {
  const [files, setFiles] = useState<File[]>([])
  const [mode, setMode] = useState<'single' | 'batch'>('single')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<any>(null)
  const [error, setError] = useState<string | null>(null)

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      const newFiles = Array.from(e.target.files)
      
      if (mode === 'single') {
        setFiles(newFiles.slice(0, 1))
      } else {
        // In batch mode, append new files to existing ones
        setFiles(prevFiles => [...prevFiles, ...newFiles])
      }
      setError(null)
      
      // Reset the input so same files can be selected again
      e.target.value = ''
    }
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (files.length === 0) {
      setError('Please select at least one audio file')
      return
    }

    if (mode === 'single' && files.length > 1) {
      setError('Single prediction mode allows only one file')
      return
    }

    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const formData = new FormData()
      const endpoint = mode === 'single' 
        ? '/predictions/single'
        : '/predictions/batch'

      if (mode === 'single') {
        formData.append('file', files[0])
      } else {
        files.forEach(file => formData.append('files', file))
      }

      const response = await fetch(
        `${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/api/v1${endpoint}`,
        {
          method: 'POST',
          body: formData,
        }
      )

      if (!response.ok) {
        throw new Error('Prediction failed')
      }

      const data = await response.json()
      setResult(data)
      setFiles([])
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred')
    } finally {
      setLoading(false)
    }
  }

  return (
    <main className="min-h-screen bg-gradient-to-b from-slate-900 to-slate-800">
      <div className="container mx-auto px-4 py-20">
        <Link href="/" className="text-blue-400 hover:text-blue-300 mb-8 inline-block">
          Back to Home
        </Link>

        <div className="max-w-2xl mx-auto">
          <h1 className="text-4xl font-bold text-white mb-8">Make a Prediction</h1>

          <form onSubmit={handleSubmit} className="bg-slate-700 rounded-lg p-8">
            <div className="mb-6">
              <label className="block text-white font-semibold mb-4">Prediction Mode</label>
              <div className="flex gap-4">
                <label className="flex items-center text-slate-300 cursor-pointer">
                  <input
                    type="radio"
                    value="single"
                    checked={mode === 'single'}
                    onChange={(e) => setMode(e.target.value as 'single' | 'batch')}
                    className="mr-2"
                  />
                  Single File
                </label>
                <label className="flex items-center text-slate-300 cursor-pointer">
                  <input
                    type="radio"
                    value="batch"
                    checked={mode === 'batch'}
                    onChange={(e) => setMode(e.target.value as 'single' | 'batch')}
                    className="mr-2"
                  />
                  Batch (Multiple Files)
                </label>
              </div>
            </div>

            <div className="mb-6">
              <label className="block text-white font-semibold mb-4">
                Upload Audio File{mode === 'batch' ? 's' : ''}
                {mode === 'batch' && <span className="text-slate-400 text-sm font-normal"> (hold Ctrl/Cmd to select multiple)</span>}
              </label>
              <input
                type="file"
                accept="audio/*"
                multiple
                onChange={handleFileChange}
                className="block w-full text-sm text-slate-300
                  file:mr-4 file:py-2 file:px-4
                  file:rounded-md file:border-0
                  file:text-sm file:font-semibold
                  file:bg-blue-600 file:text-white
                  hover:file:bg-blue-700"
              />
              {files.length > 0 && (
                <div className="text-slate-300 mt-4 bg-slate-800 p-4 rounded-md">
                  <div className="flex justify-between items-center mb-3">
                    <p className="font-semibold text-blue-300">
                      Selected: {files.length} file{files.length > 1 ? 's' : ''} 
                      <span className="text-slate-400 text-sm ml-2">
                        ({(files.reduce((sum, f) => sum + f.size, 0) / 1024 / 1024).toFixed(2)} MB)
                      </span>
                    </p>
                    <button
                      type="button"
                      onClick={() => setFiles([])}
                      className="text-sm text-red-400 hover:text-red-300"
                    >
                      Clear All
                    </button>
                  </div>
                  <ul className="mt-3 space-y-2 max-h-48 overflow-y-auto">
                    {files.map((f, i) => (
                      <li key={i} className="flex justify-between items-center text-sm text-slate-300 bg-slate-700 p-2 rounded">
                        <span>• {f.name}</span>
                        <button
                          type="button"
                          onClick={() => setFiles(files.filter((_, idx) => idx !== i))}
                          className="text-red-400 hover:text-red-300 text-xs"
                        >
                          Remove
                        </button>
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>

            {error && (
              <div className="mb-6 p-4 bg-red-900 text-red-200 rounded-md">
                {error}
              </div>
            )}

            <button
              type="submit"
              disabled={loading}
              className="w-full bg-blue-600 hover:bg-blue-700 disabled:bg-slate-500 
                text-white font-semibold py-3 rounded-md transition"
            >
              {loading ? 'Processing...' : mode === 'single' ? 'Predict' : 'Predict All'}
            </button>
          </form>

          {result && (
            <div className="mt-8 bg-green-900 rounded-lg p-8">
              <h2 className="text-2xl font-bold text-green-200 mb-4">
                Result{Array.isArray(result) ? 's' : ''}
              </h2>
              {Array.isArray(result) ? (
                <div className="space-y-4">
                  {result.map((item: any, index: number) => (
                    <div key={index} className="bg-slate-800 p-4 rounded">
                      <p className="text-green-300 font-semibold mb-2">File {index + 1}</p>
                      <pre className="text-green-100 overflow-auto text-sm">
                        {JSON.stringify(item, null, 2)}
                      </pre>
                    </div>
                  ))}
                </div>
              ) : (
                <pre className="text-green-100 overflow-auto bg-slate-800 p-4 rounded">
                  {JSON.stringify(result, null, 2)}
                </pre>
              )}
            </div>
          )}
        </div>
      </div>
    </main>
  )
}
