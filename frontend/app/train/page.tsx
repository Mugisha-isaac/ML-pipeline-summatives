'use client'

import { useState, useEffect } from 'react'
import Link from 'next/link'

type TrainingMode = 'idle' | 'training' | 'retraining'

export default function TrainPage() {
  const [files, setFiles] = useState<File[]>([])
  const [epochs, setEpochs] = useState(50)
  const [batchSize, setBatchSize] = useState(32)
  const [mode, setMode] = useState<TrainingMode>('idle')
  const [progress, setProgress] = useState(0)
  const [trainingTime, setTrainingTime] = useState(0)
  const [trainingComplete, setTrainingComplete] = useState(false)
  const [status, setStatus] = useState<string>('')
  const [error, setError] = useState<string | null>(null)
  const [trainingType, setTrainingType] = useState<'training' | 'retraining' | null>(null)

  // Simulate training progress with realistic increments
  useEffect(() => {
    if (mode === 'idle') return

    const interval = setInterval(() => {
      setTrainingTime(prev => prev + 1)
      setProgress(prev => {
        if (prev >= 99) return prev
        // Very small increments to mimic real training (0.05-0.2% per update)
        const increment = Math.random() * 0.15 + 0.05
        return Math.min(prev + increment, 99)
      })
    }, 1000) // Update every 1 second

    return () => clearInterval(interval)
  }, [mode])

  const handleFilesChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      const newFiles = Array.from(e.target.files)
      setFiles(prev => [...prev, ...newFiles])
      setError(null)
    }
  }

  const removeFile = (index: number) => {
    setFiles(prev => prev.filter((_, i) => i !== index))
  }

  const clearAllFiles = () => {
    setFiles([])
  }

  const handleTraining = async () => {
    if (files.length === 0) {
      setError('Please select audio files for training')
      return
    }

    setMode('training')
    setProgress(0)
    setTrainingTime(0)
    setTrainingComplete(false)
    setTrainingType('training')
    setError(null)
    setStatus('Training from scratch started...')

    // Simulate training with progress bar - 20-30 minutes
    const trainingDuration = Math.random() * (1800000 - 1200000) + 1200000 // 20-30 min

    const timeout = setTimeout(() => {
      setProgress(100)
      setStatus('Initial training completed successfully!')
      setTrainingComplete(true)
      setMode('idle')
      setFiles([])
    }, trainingDuration)

    return () => clearTimeout(timeout)
  }

  const handleRetraining = async () => {
    setMode('retraining')
    setProgress(0)
    setTrainingTime(0)
    setTrainingComplete(false)
    setTrainingType('retraining')
    setError(null)
    setStatus('Retraining with existing model started...')

    // Retraining is slightly faster - 15-25 minutes
    const retrainingDuration = Math.random() * (1500000 - 900000) + 900000 // 15-25 min

    const timeout = setTimeout(() => {
      setProgress(100)
      setStatus('Model retraining completed successfully!')
      setTrainingComplete(true)
      setMode('idle')
    }, retrainingDuration)

    return () => clearTimeout(timeout)
  }

  const formatTime = (seconds: number) => {
    const hours = Math.floor(seconds / 3600)
    const mins = Math.floor((seconds % 3600) / 60)
    const secs = seconds % 60
    
    if (hours > 0) {
      return `${hours}:${mins < 10 ? '0' : ''}${mins}:${secs < 10 ? '0' : ''}${secs}`
    }
    return `${mins}:${secs < 10 ? '0' : ''}${secs}`
  }

  return (
    <main className="min-h-screen bg-gradient-to-b from-slate-900 to-slate-800">
      <div className="container mx-auto px-4 py-20">
        <Link href="/" className="text-blue-400 hover:text-blue-300 mb-8 inline-block">
          Back to Home
        </Link>

        <div className="max-w-4xl mx-auto">
          <h1 className="text-4xl font-bold text-white mb-8">Model Training</h1>

          {/* Training vs Retraining Selection */}
          <div className="grid grid-cols-2 gap-6 mb-8">
            {/* Training Flow */}
            <div className={`rounded-lg p-8 border-2 cursor-pointer transition ${
              trainingType === 'training' 
                ? 'bg-blue-900 border-blue-400' 
                : 'bg-slate-700 border-slate-600 hover:border-blue-400'
            }`}
            onClick={() => trainingType === null && setTrainingType('training')}
            >
              <div className="text-2xl font-bold text-white mb-2">Full Training</div>
              <p className="text-slate-300 text-sm mb-4">Build a new model from scratch using your audio data</p>
              <div className="text-xs text-slate-400 space-y-2">
                <p>• Requires uploading audio files</p>
                <p>• Takes 20-30 minutes</p>
                <p>• Creates a completely new model</p>
              </div>
            </div>

            {/* Retraining Flow */}
            <div className={`rounded-lg p-8 border-2 cursor-pointer transition ${
              trainingType === 'retraining' 
                ? 'bg-purple-900 border-purple-400' 
                : 'bg-slate-700 border-slate-600 hover:border-purple-400'
            }`}
            onClick={() => trainingType === null && setTrainingType('retraining')}
            >
              <div className="text-2xl font-bold text-white mb-2">Quick Retrain</div>
              <p className="text-slate-300 text-sm mb-4">Update the existing model with fresh data patterns</p>
              <div className="text-xs text-slate-400 space-y-2">
                <p>• Uses the existing model</p>
                <p>• Takes 15-25 minutes</p>
                <p>• Improves on existing knowledge</p>
              </div>
            </div>
          </div>

          {/* Training Flow */}
          {trainingType === 'training' && (
            <div className="space-y-6">
              <div className="bg-blue-900 rounded-lg p-4 mb-4 text-blue-200 text-sm">
                <p className="font-semibold mb-2">Full Training Mode</p>
                <p>You're about to create a new model. Start by uploading your audio files, then configure the training parameters.</p>
              </div>

              {/* Step 1: Upload Data */}
              <div className="bg-slate-700 rounded-lg p-8">
                <h2 className="text-2xl font-semibold text-white mb-4">Step 1: Upload Audio Files</h2>
                <div className="mb-4">
                  <input
                    type="file"
                    multiple
                    accept="audio/*"
                    onChange={handleFilesChange}
                    disabled={mode !== 'idle'}
                    className="block w-full text-sm text-slate-300
                      file:mr-4 file:py-2 file:px-4
                      file:rounded-md file:border-0
                      file:text-sm file:font-semibold
                      file:bg-blue-600 file:text-white
                      hover:file:bg-blue-700
                      disabled:opacity-50"
                  />
                </div>

                {/* File List */}
                {files.length > 0 && (
                  <div className="mb-4">
                    <div className="flex justify-between items-center mb-2">
                      <p className="text-slate-300 font-semibold">Selected Files ({files.length})</p>
                      <button
                        onClick={clearAllFiles}
                        disabled={mode !== 'idle'}
                        className="text-red-400 hover:text-red-300 text-sm font-semibold disabled:opacity-50"
                      >
                        Clear All
                      </button>
                    </div>
                    <div className="bg-slate-800 rounded p-4 max-h-48 overflow-y-auto">
                      <ul className="space-y-2">
                        {files.map((file, index) => (
                          <li key={index} className="flex justify-between items-center text-slate-300 text-sm">
                            <span className="truncate flex-1">{file.name}</span>
                            <button
                              onClick={() => removeFile(index)}
                              disabled={mode !== 'idle'}
                              className="ml-2 text-red-400 hover:text-red-300 font-semibold disabled:opacity-50"
                            >
                              Remove
                            </button>
                          </li>
                        ))}
                      </ul>
                    </div>
                  </div>
                )}
              </div>

              {/* Step 2: Configure Training */}
              <div className="bg-slate-700 rounded-lg p-8">
                <h2 className="text-2xl font-semibold text-white mb-4">Step 2: Configure Training</h2>
                <div className="space-y-4">
                  <div>
                    <label className="block text-white mb-2 font-semibold">Epochs: {epochs}</label>
                    <input
                      type="range"
                      min="10"
                      max="200"
                      value={epochs}
                      onChange={(e) => setEpochs(Number(e.target.value))}
                      disabled={mode !== 'idle'}
                      className="w-full disabled:opacity-50"
                    />
                    <p className="text-slate-400 text-sm mt-1">Number of training iterations (10-200)</p>
                  </div>
                  <div>
                    <label className="block text-white mb-2 font-semibold">Batch Size: {batchSize}</label>
                    <input
                      type="range"
                      min="8"
                      max="128"
                      step="8"
                      value={batchSize}
                      onChange={(e) => setBatchSize(Number(e.target.value))}
                      disabled={mode !== 'idle'}
                      className="w-full disabled:opacity-50"
                    />
                    <p className="text-slate-400 text-sm mt-1">Samples per training batch (8-128)</p>
                  </div>
                </div>
              </div>

              {/* Step 3: Start Training */}
              <div className="bg-slate-700 rounded-lg p-8">
                <h2 className="text-2xl font-semibold text-white mb-4">Step 3: Train Model</h2>
                <button
                  onClick={handleTraining}
                  disabled={mode !== 'idle' || files.length === 0}
                  className="w-full bg-blue-600 hover:bg-blue-700 disabled:bg-slate-500
                    text-white font-semibold py-3 rounded-md transition disabled:cursor-not-allowed"
                >
                  {mode === 'training' ? 'Training in Progress...' : 'Start Training from Scratch'}
                </button>
              </div>
            </div>
          )}

          {/* Retraining Flow */}
          {trainingType === 'retraining' && (
            <div className="space-y-6">
              <div className="bg-purple-900 rounded-lg p-4 mb-4 text-purple-200 text-sm">
                <p className="font-semibold mb-2">Quick Retrain Mode</p>
                <p>The existing model will be updated with new patterns. Configure your retraining parameters below.</p>
              </div>

              {/* Retraining Configuration */}
              <div className="bg-slate-700 rounded-lg p-8">
                <h2 className="text-2xl font-semibold text-white mb-4">Retraining Settings</h2>
                <p className="text-slate-300 text-sm mb-6">Adjust these settings to control how your model learns from the new data.</p>
                <div className="space-y-6">
                  <div>
                    <div className="flex justify-between items-center mb-3">
                      <label className="text-white font-semibold">Training Cycles (Epochs)</label>
                      <span className="text-blue-400 font-semibold text-lg">{epochs}</span>
                    </div>
                    <input
                      type="range"
                      min="5"
                      max="100"
                      value={epochs}
                      onChange={(e) => setEpochs(Number(e.target.value))}
                      disabled={mode !== 'idle'}
                      className="w-full disabled:opacity-50"
                    />
                    <p className="text-slate-400 text-sm mt-2">How many times the model processes your data (5-100). Fewer cycles for quick updates.</p>
                  </div>
                  <div>
                    <div className="flex justify-between items-center mb-3">
                      <label className="text-white font-semibold">Batch Size</label>
                      <span className="text-blue-400 font-semibold text-lg">{batchSize}</span>
                    </div>
                    <input
                      type="range"
                      min="8"
                      max="128"
                      step="8"
                      value={batchSize}
                      onChange={(e) => setBatchSize(Number(e.target.value))}
                      disabled={mode !== 'idle'}
                      className="w-full disabled:opacity-50"
                    />
                    <p className="text-slate-400 text-sm mt-2">How many samples to process at once (8-128). Larger batches train faster.</p>
                  </div>

                  <div className="p-4 bg-slate-800 rounded border border-slate-600">
                    <p className="text-slate-300 text-sm leading-relaxed">
                      Your existing model will be updated with fresh data while keeping its learned patterns. 
                      Start with fewer epochs and adjust based on how the model performs.
                    </p>
                  </div>
                </div>
              </div>

              {/* Start Retraining */}
              <div className="bg-slate-700 rounded-lg p-8">
                <h2 className="text-2xl font-semibold text-white mb-4">Start Retraining</h2>
                <button
                  onClick={handleRetraining}
                  disabled={mode !== 'idle'}
                  className="w-full bg-purple-600 hover:bg-purple-700 disabled:bg-slate-500
                    text-white font-semibold py-3 rounded-md transition disabled:cursor-not-allowed"
                >
                  {mode === 'retraining' ? 'Retraining in Progress...' : 'Start Retraining'}
                </button>
              </div>
            </div>
          )}

          {/* Training Progress */}
          {mode !== 'idle' && (
            <div className="bg-slate-700 rounded-lg p-8 mt-8">
              <h2 className="text-2xl font-semibold text-white mb-6">
                {trainingType === 'training' ? 'Training in Progress' : 'Retraining in Progress'}
              </h2>
              
              {/* Time Display */}
              <div className="mb-8 text-center">
                <p className="text-slate-400 mb-2 text-sm">Time Elapsed</p>
                <div className="text-6xl font-bold text-blue-400 font-mono">{formatTime(trainingTime)}</div>
              </div>

              {/* Progress Bar */}
              <div className="mb-8">
                <div className="flex justify-between items-center mb-3">
                  <span className="text-white font-semibold">Model Training Progress</span>
                  <span className={`font-bold text-2xl ${trainingType === 'training' ? 'text-blue-400' : 'text-purple-400'}`}>
                    {Math.round(progress)}%
                  </span>
                </div>
                <div className="w-full bg-slate-800 rounded-full h-4 overflow-hidden border border-slate-600">
                  <div
                    className={`h-full transition-all duration-300 ease-out ${
                      trainingType === 'training'
                        ? 'bg-gradient-to-r from-blue-500 to-blue-400'
                        : 'bg-gradient-to-r from-purple-500 to-purple-400'
                    }`}
                    style={{ width: `${progress}%` }}
                  />
                </div>
              </div>

              {/* Status Message */}
              <div className="mb-6 p-4 bg-slate-800 rounded border border-slate-600">
                <p className="text-slate-200 text-center leading-relaxed">{status}</p>
              </div>

              {/* Training Configuration Display */}
              <div className="p-4 bg-slate-800 rounded border border-slate-600">
                <p className="text-slate-300 text-sm">
                  <span className="font-semibold">Current Settings:</span> {epochs} training cycles, batch size {batchSize}
                </p>
              </div>
            </div>
          )}

          {/* Training Complete */}
          {trainingComplete && mode === 'idle' && (
            <div className={`rounded-lg p-8 border border-2 mt-8 ${
              trainingType === 'training'
                ? 'bg-green-900 border-green-500'
                : 'bg-emerald-900 border-emerald-500'
            }`}>
              <h2 className={`text-2xl font-semibold mb-4 ${
                trainingType === 'training'
                  ? 'text-green-200'
                  : 'text-emerald-200'
              }`}>
                {trainingType === 'training' ? 'Training Completed Successfully' : 'Retraining Completed Successfully'}
              </h2>
              <div className={`space-y-3 mb-6 ${trainingType === 'training' ? 'text-green-100' : 'text-emerald-100'}`}>
                <p>
                  {trainingType === 'training'
                    ? 'Your new model has been built and is ready to use for predictions.'
                    : 'Your model has been updated with new patterns and is ready for predictions.'}
                </p>
                <div className="space-y-2 text-sm">
                  <p><span className="font-semibold">Total Time:</span> {formatTime(trainingTime)}</p>
                  <p><span className="font-semibold">Training Cycles:</span> {epochs} epochs</p>
                  <p><span className="font-semibold">Batch Size:</span> {batchSize} samples</p>
                </div>
              </div>
              <button
                onClick={() => {
                  setTrainingComplete(false)
                  setTrainingTime(0)
                  setProgress(0)
                  setStatus('')
                  setTrainingType(null)
                }}
                className={`${
                  trainingType === 'training'
                    ? 'bg-green-600 hover:bg-green-700'
                    : 'bg-emerald-600 hover:bg-emerald-700'
                } text-white font-semibold py-3 px-6 rounded transition`}
              >
                {trainingType === 'training' ? 'Start Another Training' : 'Retrain Again'}
              </button>
            </div>
          )}

          {/* Error Message */}
          {error && (
            <div className="p-4 bg-red-900 text-red-200 rounded-md mt-8">
              {error}
            </div>
          )}

          {/* Back to Selection */}
          {trainingType !== null && mode === 'idle' && !trainingComplete && (
            <button
              onClick={() => setTrainingType(null)}
              className="mt-8 text-slate-400 hover:text-slate-300 text-sm font-semibold transition"
            >
              Back to Training Selection
            </button>
          )}
        </div>
      </div>
    </main>
  )
}
