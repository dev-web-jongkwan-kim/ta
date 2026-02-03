import type { Config } from "tailwindcss";

export default {
  content: ["./src/**/*.{ts,tsx}"],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        // Semantic colors using CSS variables
        background: {
          DEFAULT: "var(--bg-primary)",
          secondary: "var(--bg-secondary)",
          tertiary: "var(--bg-tertiary)",
        },
        foreground: {
          DEFAULT: "var(--text-primary)",
          secondary: "var(--text-secondary)",
          muted: "var(--text-muted)",
        },
        border: "var(--border)",
        accent: {
          DEFAULT: "var(--accent)",
          hover: "var(--accent-hover)",
        },
        success: {
          DEFAULT: "var(--success)",
          muted: "var(--success-muted)",
        },
        danger: {
          DEFAULT: "var(--danger)",
          muted: "var(--danger-muted)",
        },
        warning: {
          DEFAULT: "var(--warning)",
          muted: "var(--warning-muted)",
        },
        // Legacy colors (for gradual migration)
        ink: "#10131a",
        mist: "#f5f3ef",
        tide: "#1f6f8b",
        ember: "#e07a5f",
        moss: "#88b04b",
        slate: "#4b5563",
      },
      fontFamily: {
        display: ["var(--font-display)", "system-ui", "sans-serif"],
        body: ["var(--font-body)", "system-ui", "sans-serif"],
        mono: ["var(--font-mono)", "ui-monospace", "monospace"],
      },
      fontSize: {
        "2xs": ["0.625rem", { lineHeight: "0.875rem" }],
      },
      animation: {
        "pulse-slow": "pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite",
        "flash": "flash 0.5s ease-out",
        "shimmer": "shimmer 2s infinite",
      },
      keyframes: {
        flash: {
          "0%": { backgroundColor: "var(--accent-muted)" },
          "100%": { backgroundColor: "transparent" },
        },
        shimmer: {
          "0%": { backgroundPosition: "-200% 0" },
          "100%": { backgroundPosition: "200% 0" },
        },
      },
    },
  },
  plugins: [],
} satisfies Config;
