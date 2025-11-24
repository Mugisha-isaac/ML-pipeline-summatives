import Link from 'next/link'

export default function Home() {
  return (
    <main className="min-h-screen bg-gradient-to-b from-slate-900 to-slate-800">
      <div className="container mx-auto px-4 py-20">
        <div className="text-center">
          <h1 className="text-5xl font-bold text-white mb-4">
            Audio Talent Classification
          </h1>
          <p className="text-xl text-slate-300 mb-8">
            ML-powered system for analyzing and classifying audio talent
          </p>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mt-12">
            <Link 
              href="/predict"
              className="p-6 bg-blue-600 hover:bg-blue-700 rounded-lg text-white transition"
            >
              <h2 className="text-2xl font-semibold mb-2">Predict</h2>
              <p className="text-slate-200">Upload audio and get predictions</p>
            </Link>
            
            <Link 
              href="/train"
              className="p-6 bg-green-600 hover:bg-green-700 rounded-lg text-white transition"
            >
              <h2 className="text-2xl font-semibold mb-2">Train</h2>
              <p className="text-slate-200">Retrain model with new data</p>
            </Link>
            
            <Link 
              href="/visualizations"
              className="p-6 bg-purple-600 hover:bg-purple-700 rounded-lg text-white transition"
            >
              <h2 className="text-2xl font-semibold mb-2">Visualizations</h2>
              <p className="text-slate-200">View feature distributions</p>
            </Link>
          </div>
        </div>
      </div>
    </main>
  )
}
