"use client"

import { Github } from "lucide-react"

export function Footer() {
  const currentYear = new Date().getFullYear()

  const handleHomeClick = (e: React.MouseEvent<HTMLAnchorElement>) => {
    e.preventDefault()
    const element = document.querySelector("#home")
    if (element) {
      element.scrollIntoView({ behavior: "smooth", block: "start" })
    }
  }

  return (
    <footer className="border-t border-border/50 bg-gradient-to-b from-background to-muted/20">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-6">
        <div className="flex flex-col sm:flex-row items-center justify-between gap-4">
          {/* Logo - Different colors from hero (using cyan/emerald like navbar) */}
          <a
            href="#home"
            onClick={handleHomeClick}
            className="flex items-center gap-0 text-2xl font-extrabold transition-all duration-300 hover:scale-110 active:scale-95 cursor-pointer"
            style={{ fontFamily: "var(--font-poppins)" }}
          >
            <span className="bg-gradient-to-r from-cyan-500 via-teal-500 to-emerald-600 bg-clip-text text-transparent">
              Pulse
            </span>
            <span className="bg-gradient-to-r from-emerald-500 via-green-500 to-teal-600 bg-clip-text text-transparent">
              AI
            </span>
          </a>

          {/* GitHub Link */}
          <a
            href="https://github.com/nischal0x01/pulseAI"
            target="_blank"
            rel="noopener noreferrer"
            className="group p-3 rounded-xl bg-muted/50 hover:bg-primary/10 text-muted-foreground hover:text-foreground transition-all duration-300 hover:scale-110 hover:rotate-6 hover:shadow-lg hover:shadow-primary/20"
            aria-label="GitHub Repository"
          >
            <Github className="h-6 w-6 transition-transform duration-300 group-hover:rotate-12" />
          </a>

          {/* Copyright */}
          <p className="text-base text-muted-foreground font-medium">
            © {currentYear} PulseAI. All rights reserved.
          </p>
        </div>
      </div>
    </footer>
  )
}
