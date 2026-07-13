import { useState } from 'react'
import Head from 'next/head'
import Dropzone from '../components/Dropzone'
import ColorSwatches from '../components/ColorSwatches'
import AdjustPanel from '../components/AdjustPanel'
import ThumbnailGrid from '../components/ThumbnailGrid'

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'
const LAST_DEPLOY = '2026-07-13 — forced-white composite background + paper-color guard'

type AppState = 'upload' | 'processing' | 'results'

interface ColorInfo {
  hex: string
  pixels: number
}

interface SeparateResponse {
  job_id: string
  color_count: number
  colors: ColorInfo[]
  page_count: number
}

interface PreviewResponse {
  thumbnails: string[]
  colors: ColorInfo[]
}

interface Params {
  max_colors: number
  page_size: string
  orientation: string
}

const DEFAULT_PARAMS: Params = {
  max_colors: 4,
  page_size: 'auto',
  orientation: 'auto',
}

export default function Home() {
  const [appState, setAppState] = useState<AppState>('upload')
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [params, setParams] = useState<Params>(DEFAULT_PARAMS)
  const [error, setError] = useState<string | null>(null)

  // Results state
  const [currentJobId, setCurrentJobId] = useState<string | null>(null)
  const [originalFilename, setOriginalFilename] = useState<string>('')
  const [thumbnailPaths, setThumbnailPaths] = useState<string[]>([])
  const [colors, setColors] = useState<ColorInfo[]>([])
  const [isRegenerating, setIsRegenerating] = useState(false)

  async function fetchPreview(jobId: string): Promise<PreviewResponse> {
    const res = await fetch(`${API_BASE}/api/preview/${jobId}`)
    if (!res.ok) {
      const body = await res.json().catch(() => ({}))
      throw new Error(body.detail ?? `Preview failed (${res.status})`)
    }
    return res.json()
  }

  async function handleSeparate() {
    if (!selectedFile) return
    setError(null)
    setAppState('processing')
    setIsRegenerating(false)

    try {
      const formData = new FormData()
      formData.append('file', selectedFile)
      formData.append('max_colors', String(params.max_colors))
      formData.append('page_size', params.page_size)
      formData.append('orientation', params.orientation)

      const res = await fetch(`${API_BASE}/api/separate`, {
        method: 'POST',
        body: formData,
      })

      if (!res.ok) {
        const body = await res.json().catch(() => ({}))
        throw new Error(body.detail ?? `Separation failed (${res.status})`)
      }

      const data: SeparateResponse = await res.json()
      const preview = await fetchPreview(data.job_id)

      setCurrentJobId(data.job_id)
      setOriginalFilename(selectedFile.name)
      setColors(preview.colors)
      setThumbnailPaths(preview.thumbnails)
      setAppState('results')
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Something went wrong')
      setAppState('upload')
    }
  }

  async function handleRegenerate() {
    if (!currentJobId) return
    setError(null)
    setIsRegenerating(true)
    setAppState('processing')

    try {
      const res = await fetch(`${API_BASE}/api/regenerate/${currentJobId}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(params),
      })

      if (!res.ok) {
        const body = await res.json().catch(() => ({}))
        throw new Error(body.detail ?? `Regeneration failed (${res.status})`)
      }

      const data: SeparateResponse = await res.json()
      const preview = await fetchPreview(data.job_id)

      setCurrentJobId(data.job_id)
      setColors(preview.colors)
      setThumbnailPaths(preview.thumbnails)
      setAppState('results')
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Something went wrong')
      setAppState('results')
    } finally {
      setIsRegenerating(false)
    }
  }

  function handleStartOver() {
    setAppState('upload')
    setSelectedFile(null)
    setParams(DEFAULT_PARAMS)
    setError(null)
    setCurrentJobId(null)
    setOriginalFilename('')
    setThumbnailPaths([])
    setColors([])
    setIsRegenerating(false)
  }

  function handleDownload() {
    if (!currentJobId) return
    window.location.href = `${API_BASE}/api/download/${currentJobId}`
  }

  return (
    <>
      <Head>
        <title>Position Print Separator</title>
      </Head>

      {error && (
        <div className="error-banner" role="alert">
          <ErrorIcon />
          {error}
          <button
            className="error-banner-dismiss"
            onClick={() => setError(null)}
            aria-label="Dismiss error"
            type="button"
          >
            ×
          </button>
        </div>
      )}

      {appState === 'upload' && (
        <main className="upload-root">
          <div className="upload-inner">
            <h1 className="upload-title">Color Separator</h1>

            <Dropzone onFileSelect={setSelectedFile} selectedFile={selectedFile} />

            <div className="controls-row">
              <div className="control-group">
                <label className="control-label" htmlFor="up-colors">
                  Colors
                </label>
                <select
                  id="up-colors"
                  className="control-select"
                  value={params.max_colors}
                  onChange={(e) =>
                    setParams((p) => ({ ...p, max_colors: Number(e.target.value) }))
                  }
                >
                  <option value={2}>2</option>
                  <option value={3}>3</option>
                  <option value={4}>4</option>
                  <option value={5}>5</option>
                </select>
              </div>

              <div className="control-group">
                <label className="control-label" htmlFor="up-size">
                  Size
                </label>
                <select
                  id="up-size"
                  className="control-select"
                  value={params.page_size}
                  onChange={(e) => setParams((p) => ({ ...p, page_size: e.target.value }))}
                >
                  <option value="auto">Auto</option>
                  <option value="a3">A3</option>
                  <option value="a4">A4</option>
                  <option value="a5">A5</option>
                  <option value="letter">Letter</option>
                  <option value="legal">Legal</option>
                  <option value="tabloid">Tabloid</option>
                </select>
              </div>

              <div className="control-group">
                <label className="control-label" htmlFor="up-orient">
                  Orientation
                </label>
                <select
                  id="up-orient"
                  className="control-select"
                  value={params.orientation}
                  onChange={(e) => setParams((p) => ({ ...p, orientation: e.target.value }))}
                >
                  <option value="auto">Auto</option>
                  <option value="portrait">Portrait</option>
                  <option value="landscape">Landscape</option>
                </select>
              </div>
            </div>

            <button
              className="btn-primary"
              onClick={handleSeparate}
              disabled={!selectedFile}
              type="button"
            >
              Separate
            </button>
          </div>
        </main>
      )}

      {appState === 'processing' && (
        <main className="processing-root" aria-live="polite">
          <div className="spinner" role="status" aria-label="Processing" />
          <div className="processing-label">
            {isRegenerating ? 'Regenerating…' : 'Separating colors…'}
          </div>
          <div className="processing-sub">This usually takes 10–30 seconds</div>
        </main>
      )}

      <div className="deploy-footer" aria-hidden="true">
        Last deployment update: {LAST_DEPLOY}
      </div>

      {appState === 'results' && currentJobId && (
        <main className="results-root">
          <header className="results-topbar">
            <div className="results-filename" title={originalFilename}>
              {originalFilename}
            </div>
            <button className="btn-download" onClick={handleDownload} type="button">
              <DownloadIcon />
              Download PDF
            </button>
          </header>

          <div className="results-body">
            <ColorSwatches colors={colors} />

            <AdjustPanel
              params={params}
              onChange={setParams}
              onRegenerate={handleRegenerate}
              isLoading={isRegenerating}
            />

            <ThumbnailGrid
              thumbnailPaths={thumbnailPaths}
              colorCount={colors.length}
            />

            <div className="start-over-wrap">
              <button className="start-over-btn" onClick={handleStartOver} type="button">
                Start over
              </button>
            </div>
          </div>
        </main>
      )}
    </>
  )
}

function ErrorIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 16 16" fill="none" aria-hidden="true">
      <circle cx="8" cy="8" r="7" stroke="currentColor" strokeWidth="1.5" />
      <path d="M8 4.5v4M8 11h.01" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
    </svg>
  )
}

function DownloadIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 14 14" fill="none" aria-hidden="true">
      <path
        d="M7 1v8M4 6l3 3 3-3M2 11h10"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  )
}
