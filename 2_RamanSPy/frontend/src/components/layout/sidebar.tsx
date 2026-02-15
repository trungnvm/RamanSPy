"use client"

import Link from "next/link"
import { usePathname } from "next/navigation"
import { cn } from "@/lib/utils"
import { LayoutDashboard, UploadCloud, Microscope, SlidersHorizontal, Settings, Menu } from "lucide-react"

const routes = [
    {
        label: "Dashboard",
        icon: LayoutDashboard,
        href: "/",
        color: "text-sky-500",
    },
    {
        label: "Data Upload",
        icon: UploadCloud,
        href: "/upload",
        color: "text-violet-500",
    },
    {
        label: "Preprocessing",
        icon: SlidersHorizontal,
        href: "/process",
        color: "text-pink-700",
    },
    {
        label: "Analysis",
        icon: Microscope,
        href: "/analyze",
        color: "text-emerald-500",
    },
    {
        label: "Settings",
        icon: Settings,
        href: "/settings",
        color: "text-gray-400",
    },
]

export function Sidebar() {
    const pathname = usePathname()

    return (
        <div className="space-y-4 py-4 flex flex-col h-full bg-[#111827] text-white overflow-y-auto border-r border-white/10 shadow-xl w-64 fixed left-0 top-0 bottom-0 z-50">
            <div className="px-3 py-2 flex-1">
                <Link href="/" className="flex items-center pl-3 mb-14">
                    <div className="relative w-8 h-8 mr-4 bg-gradient-to-br from-purple-600 to-blue-600 rounded-lg shadow-lg shadow-purple-500/20 flex items-center justify-center">
                        <Microscope className="w-5 h-5 text-white" />
                    </div>
                    <h1 className="text-2xl font-bold bg-gradient-to-r from-white to-gray-400 bg-clip-text text-transparent">
                        RamanSPy
                    </h1>
                </Link>
                <div className="space-y-1">
                    {routes.map((route) => (
                        <Link
                            key={route.href}
                            href={route.href}
                            className={cn(
                                "text-sm group flex p-3 w-full justify-start font-medium cursor-pointer hover:text-white hover:bg-white/10 rounded-lg transition py-3 relative overflow-hidden",
                                pathname === route.href ? "text-white bg-white/10 shadow-[0_0_15px_rgba(255,255,255,0.1)]" : "text-zinc-400"
                            )}
                        >
                            {pathname === route.href && (
                                <div className="absolute left-0 top-0 bottom-0 w-1 bg-purple-500 rounded-r-full" />
                            )}
                            <div className="flex items-center flex-1">
                                <route.icon className={cn("h-5 w-5 mr-3 transition-colors", route.color)} />
                                {route.label}
                            </div>
                        </Link>
                    ))}
                </div>
            </div>
            <div className="px-3 py-2 text-xs text-center text-zinc-500">
                v1.0.0 (Beta)
            </div>
        </div>
    )
}
