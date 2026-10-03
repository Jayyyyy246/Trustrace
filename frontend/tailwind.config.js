/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        lab: {
          950: '#070b12',
          900: '#0c121e',
          850: '#111927',
          800: '#172234',
          750: '#1e2b40',
          700: '#26364f',
          600: '#384d6c',
          border: '#1e2c42',
          borderLight: '#2b3d5b',
        },
        forensic: {
          blue: '#0284c7',
          blueLight: '#38bdf8',
          teal: '#0d9488',
          tealLight: '#2dd4bf',
          amber: '#d97706',
          amberLight: '#fbbf24',
          rose: '#e11d48',
          roseLight: '#fb7185',
          emerald: '#059669',
          emeraldLight: '#34d399',
          slate: '#64748b',
          slateLight: '#94a3b8',
        }
      },
      fontFamily: {
        mono: ['"JetBrains Mono"', '"Fira Code"', 'ui-monospace', 'SFMono-Regular', 'Menlo', 'Monaco', 'Consolas', 'monospace'],
        sans: ['"Inter"', 'system-ui', '-apple-system', 'BlinkMacSystemFont', '"Segoe UI"', 'Roboto', 'sans-serif'],
      },
      boxShadow: {
        'forensic-card': '0 4px 20px -2px rgba(3, 7, 18, 0.7), 0 0 0 1px rgba(30, 44, 66, 0.8)',
        'forensic-glow': '0 0 25px -5px rgba(14, 165, 233, 0.15)',
        'forensic-danger': '0 0 25px -5px rgba(225, 29, 72, 0.2)',
      }
    },
  },
  plugins: [],
}
