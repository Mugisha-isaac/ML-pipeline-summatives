'use client'

import { useEffect, useState } from 'react'
import Link from 'next/link'

export default function VisualizationsPage() {
  const [mfcc, setMfcc] = useState<string | null>(null)
  const [spectral, setSpectral] = useState<string | null>(null)
  const [features, setFeatures] = useState<any>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const fetchVisualizations = async () => {
      try {
        const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'
        
        const [mfccRes, spectralRes, featuresRes] = await Promise.all([
          fetch(`${apiUrl}/api/v1/visualizations/mfcc`),
          fetch(`${apiUrl}/api/v1/visualizations/spectral`),
          fetch(`${apiUrl}/api/v1/visualizations/feature-info`),
        ])

        if (mfccRes.ok) {
          const mfccBlob = await mfccRes.blob()
          setMfcc(URL.createObjectURL(mfccBlob))
        }
        if (spectralRes.ok) {
          const spectralBlob = await spectralRes.blob()
          setSpectral(URL.createObjectURL(spectralBlob))
        }
        if (featuresRes.ok) setFeatures(await featuresRes.json())
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load visualizations')
      } finally {
        setLoading(false)
      }
    }

    fetchVisualizations()
  }, [])

  return (
    <main className="min-h-screen bg-gradient-to-b from-slate-900 to-slate-800">
      <div className="container mx-auto px-4 py-20">
        <Link href="/" className="text-blue-400 hover:text-blue-300 mb-8 inline-block">
          Back to Home
        </Link>

        <h1 className="text-4xl font-bold text-white mb-8">Visualizations</h1>

        {loading && (
          <div className="text-center text-slate-300">
            Loading visualizations...
          </div>
        )}

        {error && (
          <div className="p-4 bg-red-900 text-red-200 rounded-md">
            {error}
          </div>
        )}

        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          {mfcc && (
            <div className="bg-slate-700 rounded-lg p-6">
              <h2 className="text-2xl font-semibold text-white mb-4">MFCC Distribution</h2>
              <img 
                src={mfcc} 
                alt="MFCC Distribution" 
                className="w-full rounded-md"
              />
            </div>
          )}

          {spectral && (
            <div className="bg-slate-700 rounded-lg p-6">
              <h2 className="text-2xl font-semibold text-white mb-4">Spectral Features</h2>
              <img 
                src={spectral} 
                alt="Spectral Features" 
                className="w-full rounded-md"
              />
            </div>
          )}

          {features && (
            <div className="md:col-span-2 bg-slate-700 rounded-lg p-6">
              <h2 className="text-2xl font-semibold text-white mb-4">Feature Information</h2>
              <div className="text-slate-300 bg-slate-800 p-4 rounded text-sm overflow-auto max-h-96 space-y-2">
                {typeof features === 'object' && Object.entries(features).map(([key, value]: [string, any]) => (
                  <div key={key}>
                    <strong className="text-blue-300">{key}:</strong> {String(value)}
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </main>
  )
}
