"use client"

import { useState } from "react"
import { motion } from "framer-motion"
import { Layers, Zap, X, Plus, Filter, Activity } from "lucide-react"

// Mock data for preview
const MOCK_DATA = Array.from({ length: 100 }, (_, i) => ({
    x: 400 + i * 16,
    y: Math.sin(i * 0.1) * 50 + Math.random() * 10 + 100
}))

import { SpectrumChart } from "@/components/visualize/spectrum-chart"

const PIPELINE_STEPS = [
    { id: "denoise", name: "Gaussian Denoising", icon: Filter, color: "text-blue-400", bg: "bg-blue-500/10" },
    { id: "baseline", name: "Baseline Correction (ALS)", icon: Activity, color: "text-emerald-400", bg: "bg-emerald-500/10" },
    { id: "normalize", name: "Min-Max Normalization", icon: Layers, color: "text-purple-400", bg: "bg-purple-500/10" },
]

export default function ProcessPage() {
    const [pipeline, setPipeline] = useState<any[]>([])
    const [isProcessing, setIsProcessing] = useState(false)

    const addStep = (step: any) => {
        setPipeline([...pipeline, { ...step, uuid: Math.random().toString() }])
    }

    const removeStep = (index: number) => {
        const newPipeline = [...pipeline]
        newPipeline.splice(index, 1)
        setPipeline(newPipeline)
    }

    const runPipeline = () => {
        setIsProcessing(true)
        setTimeout(() => setIsProcessing(false), 2000)
    }

    return (
        <div className="h-[calc(100vh-8rem)] flex flex-col lg:flex-row gap-6">
            {/* Left: Pipeline Builder */}
            <div className="w-full lg:w-1/3 flex flex-col space-y-4">
                <div className="flex items-center justify-between">
                    <h2 className="text-xl font-bold text-white flex items-center">
                        <Zap className="w-5 h-5 mr-2 text-yellow-400" />
                        Processing Pipeline
                    </h2>
                    <button
                        onClick={runPipeline}
                        disabled={isProcessing || pipeline.length === 0}
                        className="px-4 py-2 bg-gradient-to-r from-violet-600 to-indigo-600 hover:from-violet-500 hover:to-indigo-500 text-white font-medium rounded-lg text-sm shadow-lg shadow-indigo-900/20 disabled:opacity-50 transition-all"
                    >
                        {isProcessing ? "Running..." : "Run Pipeline"}
                    </button>
                </div>

                <div className="flex-1 bg-slate-900/50 border border-white/10 rounded-2xl p-4 overflow-y-auto space-y-3 relative">
                    {pipeline.length === 0 && (
                        <div className="absolute inset-0 flex flex-col items-center justify-center text-slate-500 pointer-events-none">
                            <Layers className="w-12 h-12 mb-2 opacity-20" />
                            <p>Add steps to build pipeline</p>
                        </div>
                    )}

                    {pipeline.map((step, idx) => (
                        <motion.div
                            key={step.uuid}
                            layout
                            initial={{ opacity: 0, x: -20 }}
                            animate={{ opacity: 1, x: 0 }}
                            exit={{ opacity: 0, x: 20 }}
                            className="group relative flex items-center p-3 bg-slate-800/80 border border-white/5 rounded-xl hover:border-violet-500/30 transition-colors"
                        >
                            <div className={`p-2 rounded-lg ${step.bg} mr-3`}>
                                <step.icon className={`w-4 h-4 ${step.color}`} />
                            </div>
                            <span className="text-sm font-medium text-slate-200">{step.name}</span>
                            <button
                                onClick={() => removeStep(idx)}
                                className="ml-auto p-1 text-slate-500 hover:text-red-400 hover:bg-white/5 rounded-md opacity-0 group-hover:opacity-100 transition-all"
                            >
                                <X className="w-4 h-4" />
                            </button>
                        </motion.div>
                    ))}
                </div>

                <div className="grid grid-cols-2 gap-3">
                    {PIPELINE_STEPS.map((step) => (
                        <button
                            key={step.id}
                            onClick={() => addStep(step)}
                            className="flex items-center p-3 text-left bg-white/5 hover:bg-white/10 border border-white/5 hover:border-white/20 rounded-xl transition-all group"
                        >
                            <Plus className="w-4 h-4 mr-2 text-slate-400 group-hover:text-white" />
                            <span className="text-xs font-medium text-slate-300">{step.name}</span>
                        </button>
                    ))}
                </div>
            </div>

            {/* Right: Visualization Preview */}
            <div className="w-full lg:w-2/3 flex flex-col">
                <div className="bg-slate-900/80 border border-white/10 rounded-2xl p-1 flex-1 flex flex-col">
                    <div className="px-4 py-3 border-b border-white/5 flex justify-between items-center">
                        <h3 className="font-medium text-slate-300 text-sm">Preview Result</h3>
                        <span className="px-2 py-1 rounded bg-violet-500/10 text-violet-300 text-xs border border-violet-500/20">Live</span>
                    </div>
                    <div className="flex-1 p-4">
                        <SpectrumChart data={MOCK_DATA} height="100%" />
                    </div>
                </div>
            </div>
        </div>
    )
}
