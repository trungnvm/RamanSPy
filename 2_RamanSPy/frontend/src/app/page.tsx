import { UploadCloud, Layers, Activity } from "lucide-react"

export default function DashboardPage() {
  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-bold tracking-tight text-white mb-4">Dashboard</h1>
      </div>

      {/* Stats Grid */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
        <div className="p-6 rounded-2xl bg-white/5 border border-white/10 backdrop-blur-md hover:bg-white/10 transition duration-300">
          <div className="flex items-center space-x-4">
            <div className="p-3 bg-violet-500/20 rounded-xl">
              <UploadCloud className="h-6 w-6 text-violet-400" />
            </div>
            <div>
              <p className="text-sm font-medium text-slate-400">Total Spectra</p>
              <h3 className="text-2xl font-bold text-white">—</h3>
            </div>
          </div>
        </div>
        <div className="p-6 rounded-2xl bg-white/5 border border-white/10 backdrop-blur-md hover:bg-white/10 transition duration-300">
          <div className="flex items-center space-x-4">
            <div className="p-3 bg-blue-500/20 rounded-xl">
              <Layers className="h-6 w-6 text-blue-400" />
            </div>
            <div>
              <p className="text-sm font-medium text-slate-400">Active Sessions</p>
              <h3 className="text-2xl font-bold text-white">1</h3>
            </div>
          </div>
        </div>
        <div className="p-6 rounded-2xl bg-white/5 border border-white/10 backdrop-blur-md hover:bg-white/10 transition duration-300">
          <div className="flex items-center space-x-4">
            <div className="p-3 bg-emerald-500/20 rounded-xl">
              <Activity className="h-6 w-6 text-emerald-400" />
            </div>
            <div>
              <p className="text-sm font-medium text-slate-400">Processing Jobs</p>
              <h3 className="text-2xl font-bold text-white">Idle</h3>
            </div>
          </div>
        </div>
      </div>

      {/* Recent Activity / Empty State */}
      <div className="mt-8">
        <h2 className="text-xl font-semibold mb-4 text-slate-200">Get Started</h2>
        <div className="grid gap-6 md:grid-cols-2">
          <a href="/upload" className="group relative block overflow-hidden rounded-2xl bg-gradient-to-br from-violet-900/50 to-slate-900 border border-white/10 p-8 hover:border-violet-500/50 transition-all duration-300">
            <div className="absolute inset-0 bg-gradient-to-r from-violet-600/10 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
            <UploadCloud className="h-10 w-10 text-violet-400 mb-4" />
            <h3 className="text-lg font-semibold text-white mb-2">Upload Data</h3>
            <p className="text-slate-400">Import your spectral data (Witec, Renishaw, CSV, NumPy) to begin analysis.</p>
          </a>

          <a href="/process" className="group relative block overflow-hidden rounded-2xl bg-gradient-to-br from-cyan-900/50 to-slate-900 border border-white/10 p-8 hover:border-cyan-500/50 transition-all duration-300">
            <div className="absolute inset-0 bg-gradient-to-r from-cyan-600/10 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
            <Layers className="h-10 w-10 text-cyan-400 mb-4" />
            <h3 className="text-lg font-semibold text-white mb-2">Preprocessing Pipeline</h3>
            <p className="text-slate-400">Clean your data with advanced algorithms: Denoising, Baseline Correction, Normalization.</p>
          </a>
        </div>
      </div>
    </div>
  )
}
