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

  const closeModal = () => {
    setResult(null)
  }

  const renderResultModal = () => {
    if (!result || Array.isArray(result)) return null

    const isGood = result.label === 'good'
    const confidence = (result.confidence * 100).toFixed(2)
    
    return (
      <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
        <div className={`bg-white rounded-lg shadow-xl max-w-md w-full overflow-hidden`}>
          {/* Header */}
          <div className={`${isGood ? 'bg-green-600' : 'bg-red-600'} px-6 py-8 text-center`}>
            <div className="text-5xl font-bold text-white mb-2">
              {isGood ? 'QUALIFIED' : 'REJECTED'}
            </div>
            <p className="text-white text-sm opacity-90">
              {isGood ? 'Singer Talent Assessment' : 'Assessment Result'}
            </p>
          </div>

          {/* Content */}
          <div className="px-6 py-8">
            <div className="text-center mb-6">
              <p className="text-gray-600 text-sm mb-2">Prediction Result</p>
              <p className="text-2xl font-bold text-gray-800 capitalize">
                {result.label}
              </p>
            </div>

            {/* Confidence Score */}
            <div className="mb-6">
              <div className="flex justify-between items-center mb-2">
                <span className="text-gray-600 font-medium">Confidence</span>
                <span className="text-lg font-bold text-gray-800">{confidence}%</span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div
                  className={`h-2 rounded-full ${isGood ? 'bg-green-600' : 'bg-red-600'}`}
                  style={{ width: `${confidence}%` }}
                ></div>
              </div>
            </div>

            {/* Probability Breakdown */}
            <div className="bg-gray-50 rounded-lg p-4 mb-6 space-y-3">
              <div className="flex justify-between items-center">
                <span className="text-gray-600 text-sm">Probability Good</span>
                <span className="font-mono font-semibold text-gray-800">
                  {(result.probability_good * 100).toFixed(4)}%
                </span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-gray-600 text-sm">Probability Bad</span>
                <span className="font-mono font-semibold text-gray-800">
                  {(result.probability_bad * 100).toFixed(4)}%
                </span>
              </div>
            </div>

            {/* File Info */}
            <div className="bg-gray-50 rounded-lg p-4 mb-6">
              <p className="text-gray-600 text-sm mb-1">File</p>
              <p className="text-gray-800 font-semibold truncate">{result.filename}</p>
            </div>

            {/* Close Button */}
            <button
              onClick={closeModal}
              className={`w-full py-3 rounded-lg font-semibold text-white transition ${
                isGood
                  ? 'bg-green-600 hover:bg-green-700'
                  : 'bg-red-600 hover:bg-red-700'
              }`}
            >
              Close
            </button>
          </div>
        </div>
      </div>
    )
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
                disabled={loading}
                className="block w-full text-sm text-slate-300
                  file:mr-4 file:py-2 file:px-4
                  file:rounded-md file:border-0
                  file:text-sm file:font-semibold
                  file:bg-blue-600 file:text-white
                  hover:file:bg-blue-700
                  disabled:opacity-50 disabled:cursor-not-allowed"
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
                      disabled={loading}
                      className="text-sm text-red-400 hover:text-red-300 disabled:opacity-50"
                    >
                      Clear All
                    </button>
                  </div>
                  <ul className="mt-3 space-y-2 max-h-48 overflow-y-auto">
                    {files.map((f, i) => (
                      <li key={i} className="flex justify-between items-center text-sm text-slate-300 bg-slate-700 p-2 rounded">
                        <span>* {f.name}</span>
                        <button
                          type="button"
                          onClick={() => setFiles(files.filter((_, idx) => idx !== i))}
                          disabled={loading}
                          className="text-red-400 hover:text-red-300 text-xs disabled:opacity-50"
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

          {/* Loading Spinner */}
          {loading && (
            <div className="mt-8 flex justify-center">
              <div className="flex flex-col items-center">
                <div className="w-16 h-16 border-4 border-blue-200 border-t-blue-600 rounded-full animate-spin mb-4"></div>
                <p className="text-white text-lg font-semibold">Processing prediction...</p>
                <p className="text-slate-300 text-sm mt-2">Please wait while we analyze the audio file</p>
              </div>
            </div>
          )}

          {/* Batch Results */}
          {result && Array.isArray(result) && (
            <div className="mt-8 bg-blue-900 rounded-lg p-8">
              <h2 className="text-2xl font-bold text-blue-200 mb-6">Batch Results</h2>
              <div className="space-y-4">
                {result.map((item: any, index: number) => (
                  <div 
                    key={index} 
                    className={`rounded-lg p-4 ${
                      item.label === 'good' 
                        ? 'bg-green-900 border-l-4 border-green-500' 
                        : 'bg-red-900 border-l-4 border-red-500'
                    }`}
                  >
                    <div className="flex justify-between items-start">
                      <div>
                        <p className="font-semibold text-white mb-1">File {index + 1}</p>
                        <p className="text-slate-300 text-sm truncate">{item.filename}</p>
                      </div>
                      <div className="text-right">
                        <p className={`font-bold text-lg capitalize ${
                          item.label === 'good' ? 'text-green-300' : 'text-red-300'
                        }`}>
                          {item.label === 'good' ? 'QUALIFIED' : 'REJECTED'}
                        </p>
                        <p className="text-slate-300 text-xs">
                          {(item.confidence * 100).toFixed(1)}% confidence
                        </p>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Result Modal */}
      {renderResultModal()}
    </main>
  )
}
