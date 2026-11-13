import type { Metadata, Viewport } from 'next'
import { Inter, JetBrains_Mono, Instrument_Serif } from 'next/font/google'
import './globals.css'

const sans = Inter({ subsets: ['latin'], variable: '--font-sans' })
const mono = JetBrains_Mono({ subsets: ['latin'], variable: '--font-mono' })
const serif = Instrument_Serif({ subsets: ['latin'], weight: '400', style: ['normal', 'italic'], variable: '--font-serif' })

export const metadata: Metadata = {
  title: 'GiftxAI · Grounded gift recommendations from your catalogues',
  description:
    'Retrieval-augmented gift recommendation: PDFs are parsed and embedded in the browser, Llama 3.3 70B on Groq drafts and validates, and every price is checked against the sources.',
}

export const viewport: Viewport = { themeColor: '#0D0B0A', width: 'device-width', initialScale: 1 }

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${sans.variable} ${mono.variable} ${serif.variable}`}>
      <body>{children}</body>
    </html>
  )
}
