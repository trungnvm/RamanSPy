"use client"

import { useState } from "react"
import { motion, AnimatePresence } from "framer-motion"
import { UploadCloud, FileType, CheckCircle, AlertCircle, Loader2 } from "lucide-react"
import { uploadFile } from "@/lib/api"
import { useRouter } from "next/navigation"

const formats = [
    { id: "csv", name: "CSV / Text", desc: "Common delimited text format" },
    { id: "witec", name: "WITec Project", desc: "WITec Scan Data (.wip, .wjp)" },
    { id: "renishaw", name: "Renishaw", desc: "Renishaw WiRE files (.wdf)" },
    { id: "numpy", name: "NumPy Array", desc: "Python stored array (.npy)" },
]

export default function UploadPage() {
    const [selectedFile, setSelectedFile] = useState<File | null>(null)
    const [format, setFormat] = useState("csv")
    const [loading, setLoading] = useState(false)
    const [error, setError] = useState<string | null>(null)
    const [success, setSuccess] = useState(false)
    const router = useRouter()

    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        if (e.target.files && e.target.files[0]) {
            setSelectedFile(e.target.files[0])
            setError(null)
        }
    }

    const handleUpload = async () => {
        if (!selectedFile) return
        setLoading(true)
        setError(null)

        try {
            const res = await uploadFile(selectedFile, format)
            if (res.status === "success") {
                setSuccess(true)
                setTimeout(() => {
                    // Redirect to preprocess page with spectrum ID?
                    // Or store in context? For now just log
                    console.log("Uploaded:", res.spectrum_id)
                }, 1000)
            }
        } catch (err: any) {
            console.error(err)
            setError("Upload failed. Please check the file format and try again.")
        } finally {
            setLoading(false)
        }
    }

    return (
        <div className="max-w-4xl mx-auto space-y-8">
            <div>
                <h1 className="text-3xl font-bold text-white mb-2">Upload Data</h1>
                <p className="text-slate-400">Select your spectral data file to begin analysis.</p>
            </div>

            <div className="grid md:grid-cols-3 gap-8">
                {/* Left: Format Selection */}
                <div className="col-span-1 space-y-4">
                    <label className="text-sm font-medium text-slate-300">Format Type</label>
                    <div className="space-y-3">
                        {formats.map((fmt) => (
                            <div
                                key={fmt.id}
                                onClick={() => setFormat(fmt.id)}
                                className={`p-4 rounded-xl border cursor-pointer transition-all duration-200 ${format === fmt.id
                                        ? "bg-violet-500/20 border-violet-500/50 ring-1 ring-violet-500/50"
                                        : "bg-white/5 border-white/10 hover:bg-white/10"
                                    }`}
                            >
                                <div className="flex items-center justify-between mb-1">
                                    <span className={`font-medium ${format === fmt.id ? "text-violet-200" : "text-slate-300"}`}>
                                        {fmt.name}
                                    </span>
                                    {format === fmt.id && <CheckCircle className="w-4 h-4 text-violet-400" />}
                                </div>
                                <p className="text-xs text-slate-500">{fmt.desc}</p>
                            </div>
                        ))}
                    </div>
                </div>

                {/* Right: Upload Area */}
                <div className="col-span-2">
                    <div className="bg-white/5 backdrop-blur-sm border border-white/10 rounded-2xl p-8 text-center min-h-[400px] flex flex-col justify-center items-center relative overflow-hidden group">
                        <AnimatePresence mode="wait">
                            {!selectedFile ? (
                                <motion.div
                                    key="empty"
                                    initial={{ opacity: 0, y: 10 }}
                                    animate={{ opacity: 1, y: 0 }}
                                    exit={{ opacity: 0, y: -10 }}
                                    className="flex flex-col items-center"
                                >
                                    <div className="w-20 h-20 mb-6 rounded-full bg-slate-800/50 flex items-center justify-center group-hover:scale-110 transition-transform duration-300 border border-white/5">
                                        <UploadCloud className="w-10 h-10 text-slate-400 group-hover:text-violet-400 transition-colors" />
                                    </div>
                                    <h3 className="text-xl font-medium text-slate-200 mb-2">
                                        Drag & Drop or Click to Upload
                                    </h3>
                                    <p className="text-sm text-slate-500 max-w-xs mx-auto mb-8">
                                        Supports text files, CSV, WiRE data, and NumPy arrays up to 50MB.
                                    </p>

                                    <label className="relative">
                                        <span className="px-6 py-3 rounded-lg bg-violet-600 hover:bg-violet-500 text-white font-medium cursor-pointer transition-all shadow-lg shadow-violet-900/20">
                                            Browse Files
                                        </span>
                                        <input
                                            type="file"
                                            className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                                            onChange={handleFileChange}
                                        />
                                    </label>
                                </motion.div>
                            ) : (
                                <motion.div
                                    key="selected"
                                    initial={{ opacity: 0, scale: 0.95 }}
                                    animate={{ opacity: 1, scale: 1 }}
                                    exit={{ opacity: 0, scale: 0.95 }}
                                    className="w-full max-w-md"
                                >
                                    <div className="flex items-center p-4 bg-slate-800/50 rounded-xl border border-white/10 mb-6">
                                        <div className="p-3 bg-violet-500/20 rounded-lg mr-4">
                                            <FileType className="w-6 h-6 text-violet-400" />
                                        </div>
                                        <div className="flex-1 text-left overflow-hidden">
                                            <h4 className="font-medium text-slate-200 truncate">{selectedFile.name}</h4>
                                            <p className="text-xs text-slate-500">{(selectedFile.size / 1024).toFixed(2)} KB • {format.toUpperCase()}</p>
                                        </div>
                                        <button
                                            onClick={() => setSelectedFile(null)}
                                            className="p-2 hover:bg-white/10 rounded-full text-slate-400 hover:text-red-400 transition-colors"
                                        >
                                            ✕
                                        </button>
                                    </div>

                                    {error && (
                                        <div className="mb-6 p-4 bg-red-500/10 border border-red-500/20 rounded-xl flex items-center gap-3 text-red-200 text-sm">
                                            <AlertCircle className="w-5 h-5 flex-shrink-0" />
                                            {error}
                                        </div>
                                    )}

                                    {success ? (
                                        <div className="flex flex-col items-center text-emerald-400">
                                            <CheckCircle className="w-12 h-12 mb-2" />
                                            <p className="font-medium">Upload Complete!</p>
                                        </div>
                                    ) : (
                                        <button
                                            onClick={handleUpload}
                                            disabled={loading}
                                            className="w-full py-3 rounded-xl bg-gradient-to-r from-violet-600 to-indigo-600 hover:from-violet-500 hover:to-indigo-500 text-white font-medium shadow-lg shadow-indigo-900/20 transition-all flex items-center justify-center disabled:opacity-50 disabled:cursor-not-allowed"
                                        >
                                            {loading ? (
                                                <>
                                                    <Loader2 className="w-5 h-5 animate-spin mr-2" />
                                                    Uploading...
                                                </>
                                            ) : (
                                                "Start Upload"
                                            )}
                                        </button>
                                    )}
                                </motion.div>
                            )}
                        </AnimatePresence>
                    </div>
                </div>
            </div>
        </div>
    )
}
