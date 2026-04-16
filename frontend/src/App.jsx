import './App.css'
import Navbar from './components/Navbar'
import UploadImage from './components/UploadImage'
import { ToastContainer } from 'react-toastify'
import 'react-toastify/dist/ReactToastify.css'

function App() {
  return (
    <>
      <Navbar />

      {/* Hero */}
      <section className="relative overflow-hidden">
        <div className="absolute inset-0 -z-10">
          <div className="absolute top-0 left-1/4 w-96 h-96 bg-red-300/30 rounded-full blur-3xl" />
          <div className="absolute top-20 right-1/4 w-80 h-80 bg-rose-300/20 rounded-full blur-3xl" />
        </div>

        <div className="max-w-6xl mx-auto px-4 pt-16 pb-8 text-center">
          {/* <div className="inline-flex items-center gap-2 bg-white/70 backdrop-blur-sm border border-violet-200 rounded-full px-4 py-1.5 mb-6 shadow-sm">
            <Sparkles className="h-3.5 w-3.5 text-violet-600" />
            <span className="text-xs font-semibold text-violet-700">AI-Powered Radiology Assistant</span>
          </div> */}

          <h1 className="text-4xl sm:text-5xl lg:text-6xl font-extrabold tracking-tight">
            Detect brain tumors in {' '}
            <span className="bg-gradient-to-r from-red-600 to-rose-500 bg-clip-text text-transparent">
              seconds
            </span>
          </h1>
          <p className="mt-5 text-lg text-zinc-600 max-w-2xl mx-auto font-medium">
            Upload an MRI scan and get class probabilities with confidence,
            plus a Grad-CAM annotated image showing detected tumor bounding boxes.
          </p>

          {/* <div className="mt-8 flex flex-wrap justify-center gap-6 text-sm text-zinc-600">
            <FeaturePill icon={<Brain className="h-4 w-4" />} text="4 tumor classes" />
            <FeaturePill icon={<Zap className="h-4 w-4" />} text="Results in &lt;2s" />
            <FeaturePill icon={<ShieldCheck className="h-4 w-4" />} text="Research-grade model" />
          </div> */}
        </div>
      </section>

      {/* Upload + results */}
      <main className="flex-1">
        <UploadImage />
      </main>

      {/* Footer */}
      <footer className="border-t border-red-100 mt-12 bg-red-50">
        <div className="max-w-6xl mx-auto px-4 py-6 flex flex-col sm:flex-row items-center justify-between gap-2 text-sm text-zinc-500">
          <span>© {new Date().getFullYear()} DuskerAi · Research preview</span>
          <span>Built with React + Vite · Tailwind CSS</span>
        </div>
      </footer>

      <ToastContainer position="top-right" autoClose={3000} theme="light" />
    </>
  )
}

export default App
