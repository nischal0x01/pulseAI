"use client"

export function Footer() {
  const currentYear = new Date().getFullYear()

  return (
    <footer className="bg-gradient-to-b from-muted/30 to-background border-t border-border">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-12">
        <div className="flex flex-col items-center justify-center space-y-6">
          {/* Logo with Gradient */}
          <div className="flex items-center gap-0 text-3xl font-bold">
            <span className="bg-gradient-to-r from-blue-600 via-purple-600 to-blue-600 bg-clip-text text-transparent">
              Pulse
            </span>
            <span className="bg-gradient-to-r from-pink-600 via-red-500 to-orange-500 bg-clip-text text-transparent">
              AI
            </span>
          </div>

          <p className="text-sm text-muted-foreground text-center max-w-2xl">
            Advanced cuffless blood pressure monitoring using PPG and ECG signals through machine learning.
          </p>

          {/* Bottom Bar */}
          <div className="border-t border-border pt-6 w-full">
            <p className="text-sm text-muted-foreground text-center">
              © {currentYear} PulseAI. All rights reserved.
            </p>
          </div>
        </div>
      </div>
    </footer>
  )
}
