import "./globals.css";
import type { ReactNode } from "react";
import { Inter, JetBrains_Mono } from "next/font/google";
import { ThemeProvider } from "@/contexts/ThemeContext";
import { ToastProvider } from "@/contexts/ToastContext";
import { WebSocketStatusProvider } from "@/contexts/WebSocketStatusContext";
import Sidebar from "@/components/layout/Sidebar";
import Topbar from "@/components/layout/Topbar";
import WebSocketMonitor from "@/components/monitors/WebSocketMonitor";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-body",
  display: "swap",
});

const jetbrains = JetBrains_Mono({
  subsets: ["latin"],
  variable: "--font-mono",
  display: "swap",
});

export const metadata = {
  title: "TA Ops",
  description: "Trading operations dashboard",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html
      lang="en"
      className={`${inter.variable} ${jetbrains.variable}`}
      suppressHydrationWarning
    >
      <head>
        {/* Prevent flash of wrong theme */}
        <script
          dangerouslySetInnerHTML={{
            __html: `
              (function() {
                const stored = localStorage.getItem('ta-theme');
                const theme = stored === 'dark' || stored === 'light' ? stored :
                  (stored === 'system' || !stored) && window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
                document.documentElement.classList.add(theme);
              })();
            `,
          }}
        />
      </head>
      <body className="font-body">
        <ThemeProvider>
          <ToastProvider>
            <WebSocketStatusProvider>
              <WebSocketMonitor />
              <div className="flex min-h-screen">
                <Sidebar />
                <div className="flex-1 flex flex-col min-w-0">
                  <Topbar />
                  <main className="flex-1 p-6 overflow-auto">{children}</main>
                </div>
              </div>
            </WebSocketStatusProvider>
          </ToastProvider>
        </ThemeProvider>
      </body>
    </html>
  );
}
