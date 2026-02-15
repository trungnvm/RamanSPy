"use client"

import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid } from 'recharts';

interface SpectrumChartProps {
    data: { x: number; y: number }[];
    color?: string;
    height?: number | string;
}

export function SpectrumChart({ data, color = "#8b5cf6", height = 400 }: SpectrumChartProps) {
    if (!data || data.length === 0) {
        return (
            <div className="flex items-center justify-center bg-white/5 border border-white/10 rounded-xl h-[400px] text-slate-500">
                No data to display
            </div>
        )
    }

    // Prepare data for Recharts (array of objects)
    const formattedData = data.map(d => ({ wavenumber: d.x, intensity: d.y }));

    return (
        <div style={{ height }} className="bg-slate-900/50 border border-white/10 rounded-xl p-4 shadow-inner relative overflow-hidden">
            {/* Glossy overlay */}
            <div className="absolute top-0 left-0 right-0 h-1/3 bg-gradient-to-b from-white/5 to-transparent pointer-events-none" />

            <ResponsiveContainer width="100%" height="100%">
                <LineChart data={formattedData} margin={{ top: 10, right: 10, left: 10, bottom: 20 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.4} vertical={false} />
                    <XAxis
                        dataKey="wavenumber"
                        stroke="#94a3b8"
                        tick={{ fill: '#94a3b8', fontSize: 11 }}
                        label={{ value: 'Wavenumber (cm⁻¹)', position: 'insideBottom', offset: -10, fill: '#64748b' }}
                    />
                    <YAxis
                        stroke="#94a3b8"
                        tick={{ fill: '#94a3b8', fontSize: 11 }}
                        width={40}
                        domain={['auto', 'auto']}
                    />
                    <Tooltip
                        contentStyle={{
                            backgroundColor: '#0f172a',
                            borderColor: '#1e293b',
                            borderRadius: '8px',
                            color: '#e2e8f0',
                            boxShadow: '0 10px 15px -3px rgba(0, 0, 0, 0.5)'
                        }}
                        itemStyle={{ color: color }}
                        formatter={(value: number) => [value.toFixed(2), 'Intensity']}
                        labelFormatter={(label) => `Wavenumber: ${Number(label).toFixed(1)}`}
                    />
                    <Line
                        type="monotone"
                        dataKey="intensity"
                        stroke={color}
                        strokeWidth={2}
                        dot={false}
                        activeDot={{ r: 6, strokeWidth: 0, fill: '#fff' }}
                        animationDuration={1500}
                    />
                </LineChart>
            </ResponsiveContainer>
        </div>
    )
}
