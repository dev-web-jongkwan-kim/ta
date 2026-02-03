import "./globals.css";
import type { ReactNode } from "react";
import { Space_Grotesk, Sora } from "next/font/google";
import Sidebar from "@/components/layout/Sidebar";
import Topbar from "@/components/layout/Topbar";

const display = Space_Grotesk({ subsets: ["latin"], variable: "--font-display" });
const body = Sora({ subsets: ["latin"], variable: "--font-body" });

export const metadata = {
  title: "TA Ops",
  description: "Trading ops dashboard",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en" className={`${display.variable} ${body.variable}`}>
      <body>
        <div className="flex min-h-screen">
          <Sidebar />
          <div className="flex-1">
            <Topbar />
            <main className="px-8 py-6">{children}</main>
          </div>
        </div>
      </body>
    </html>
  );
}
