import './globals.css'
import type { Metadata } from 'next'
import { Inter } from 'next/font/google'
import { Sidebar } from '@/components/layout/sidebar' // Check alias

const inter = Inter({ subsets: ['latin'] })

export const metadata: Metadata = {
    title: 'RamanSPy | Advanced Spectral Analysis',
    description: 'AI-Enhanced Raman Spectroscopy Analysis Platform',
}

export default function RootLayout({
    children,
}: {
    children: React.ReactNode
}) {
    return (
        <html lang="en" className="dark h-full">
            <body className={`${inter.className} h-full bg-[#020617] text-slate-200 antialiased selection:bg-indigo-500/30 selection:text-indigo-200`}>
                <div className="flex h-screen w-full overflow-hidden">
                    {/* Sidebar Fixed */}
                    <Sidebar />

                    {/* Main Content Area */}
                    <main className="flex-1 ml-64 h-full overflow-y-auto relative bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950">
                        {/* Subtle mesh gradient background effect */}
                        <div
                            className="absolute inset-0 z-0 opacity-20 pointer-events-none"
                            style={{
                                backgroundImage: `radial-gradient(circle at 15% 50%, rgba(76, 29, 149, 0.15), transparent 25%), radial-gradient(circle at 85% 30%, rgba(14, 165, 233, 0.15), transparent 25%)`
                            }}
                        />

                        <div className="relative z-10 p-8 max-w-7xl mx-auto w-full">
                            {children}
                        </div>
                    </main>
                </div>
            </body>
        </html>
    )
}
