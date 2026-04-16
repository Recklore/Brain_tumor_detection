"use client"
import { Activity } from 'lucide-react'

function Navbar() {
  return (
    <nav className="sticky top-0 z-[100] w-full border-b border-red-700 bg-gradient-to-r from-red-700 to-rose-700 shadow-md">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          <div className="flex items-center gap-10">
            <a href="/" className="flex items-center gap-2 group">
              <div className="w-9 h-9 bg-white/20 rounded-xl flex items-center justify-center shadow-lg shadow-red-900/30 group-hover:rotate-6 transition-transform">
                <Activity className="text-white w-5 h-5" />
              </div>
              <span className="text-xl font-black tracking-tight text-white">
                Dusker
              </span>
            </a>
          </div>

          <div className="hidden md:flex items-center gap-8">
            <NavLink href="/all-work">All Work</NavLink>
            <NavLink href="/team">Team</NavLink>
            <NavLink href="/ref">References</NavLink>
          </div>

        </div>
      </div>
    </nav>
  )
}

function NavLink({ href, children }) {
  return (
    <a
      href={href}
      className="text-sm font-bold text-red-100 hover:text-white transition-colors relative group"
    >
      {children}
      <span className="absolute -bottom-1 left-0 w-0 h-0.5 bg-white transition-all group-hover:w-full" />
    </a>
  )
}

export default Navbar