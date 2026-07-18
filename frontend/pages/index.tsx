import { useState } from 'react'
import type { GetStaticProps } from 'next'
import Head from 'next/head'
import Dropzone from '../components/Dropzone'
import ColorSwatches from '../components/ColorSwatches'
import AdjustPanel from '../components/AdjustPanel'
import ThumbnailGrid from '../components/ThumbnailGrid'

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

type AppState = 'upload' | 'processing' | 'results'
type Mode = 'single' | 'gang'

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

const SPACING_MAP: Record<string, number> = {
  tight:  0.15,
  normal: 0.25,
  loose:  0.40,
}

interface HomeProps {
  buildTime: string
}

export const getStaticProps: GetStaticProps<HomeProps> = async () => {
  return {
    props: {
      buildTime: new Date().toISOString(),
    },
  }
}

function formatBuildTime(iso: string): string {
  const d = new Date(iso)
  const pad = (n: number) => String(n).padStart(2, '0')
  return (
    `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ` +
    `${pad(d.getHours())}:${pad(d.getMinutes())}`
  )
}

export default function Home({ buildTime }: HomeProps) {
  const [appState, setAppState] = useState<AppState>('upload')
  const [mode, setMode] = useState<Mode>('single')

  // Single-design state
  const [selectedFile, setSelectedFile] = useState<File | null>(null)

  // Gang-sheet state
  const [gangFiles, setGangFiles] = useState<File[]>([])
  const [gangPreviewUrls, setGangPreviewUrls] = useState<string[]>([])
  const [gangSpacing, setGangSpacing] = useState<string>('normal')
  const [gangIsDragOver, setGangIsDragOver] = useState(false)

  // Shared state
  const [params, setParams] = useState<Params>(DEFAULT_PARAMS)
  const [error, setError] = useState<string | null>(null)

  // Results state
  const [currentJobId, setCurrentJobId] = useState<string | null>(null)
  const [originalFilename, setOriginalFilename] = useState<string>('')
  const [thumbnailPaths, setThumbnailPaths] = useState<string[]>([])
  const [colors, setColors] = useState<ColorInfo[]>([])
  const [isRegenerating, setIsRegenerating] = useState(false)

  // --- Gang file helpers ---

  function addGangFiles(newFiles: File[]) {
    const urls = newFiles.map(f =>
      f.type.startsWith('image/') ? URL.createObjectURL(f) : ''
    )
    setGangFiles(prev => [...prev, ...newFiles])
    setGangPreviewUrls(prev => [...prev, ...urls])
  }

  function removeGangFile(idx: number) {
    const url = gangPreviewUrls[idx]
    if (url) URL.revokeObjectURL(url)
    setGangFiles(prev => prev.filter((_, i) => i !== idx))
    setGangPreviewUrls(prev => prev.filter((_, i) => i !== idx))
  }

  function handleGangDrop(e: React.DragEvent) {
    e.preventDefault()
    setGangIsDragOver(false)
    const dropped = Array.from(e.dataTransfer.files)
    if (dropped.length > 0) addGangFiles(dropped)
  }

  // --- API helpers ---

  async function fetchPreview(jobId: string): Promise<PreviewResponse> {
    const res = await fetch(`${API_BASE}/api/preview/${jobId}`)
    if (!res.ok) {
      const body = await res.json().catch(() => ({}))
      throw new Error(body.detail ?? `Preview failed (${res.status})`)
    }
    return res.json()
  }

  // --- Handlers ---

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

  async function handleGangSheet() {
    if (gangFiles.length < 1) return
    setError(null)
    setAppState('processing')
    setIsRegenerating(false)

    try {
      const fd = new FormData()
      gangFiles.forEach(f => fd.append('files', f))
      fd.append('max_colors', String(params.max_colors))
      fd.append('sheet_size', params.page_size)
      fd.append('orientation', params.orientation)
      fd.append('spacing_in', String(SPACING_MAP[gangSpacing] ?? 0.25))

      const res = await fetch(`${API_BASE}/api/gang-sheet`, {
        method: 'POST',
        body: fd,
      })

      if (!res.ok) {
        const body = await res.json().catch(() => ({}))
        throw new Error(body.detail ?? `Gang sheet failed (${res.status})`)
      }

      const data: SeparateResponse = await res.json()
      const preview = await fetchPreview(data.job_id)

      setCurrentJobId(data.job_id)
      setOriginalFilename(`${gangFiles.length} designs`)
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
    gangPreviewUrls.forEach(url => { if (url) URL.revokeObjectURL(url) })
    setAppState('upload')
    setSelectedFile(null)
    setGangFiles([])
    setGangPreviewUrls([])
    setGangSpacing('normal')
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

  // Shared controls row (Colors / Size / Orientation)
  const sharedControls = (
    <div className="controls-row">
      <div className="control-group">
        <label className="control-label" htmlFor="ctrl-colors">
          Colors
        </label>
        <select
          id="ctrl-colors"
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
        <label className="control-label" htmlFor="ctrl-size">
          Size
        </label>
        <select
          id="ctrl-size"
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
        <label className="control-label" htmlFor="ctrl-orient">
          Orientation
        </label>
        <select
          id="ctrl-orient"
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
  )

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

            {/* Mode toggle */}
            <div className="mode-toggle" role="group" aria-label="Separation mode">
              <button
                type="button"
                className={`mode-btn${mode === 'single' ? ' mode-btn-active' : ''}`}
                onClick={() => setMode('single')}
              >
                Single Design
              </button>
              <button
                type="button"
                className={`mode-btn${mode === 'gang' ? ' mode-btn-active' : ''}`}
                onClick={() => setMode('gang')}
              >
                Gang Sheet
              </button>
            </div>

            {mode === 'single' ? (
              <>
                <Dropzone onFileSelect={setSelectedFile} selectedFile={selectedFile} />
                {sharedControls}
                <button
                  className="btn-primary"
                  onClick={handleSeparate}
                  disabled={!selectedFile}
                  type="button"
                >
                  Separate
                </button>
              </>
            ) : (
              <>
                {/* Gang dropzone — accepts drops at all times */}
                <div
                  className={[
                    'gang-dropzone',
                    gangIsDragOver ? 'drag-over' : '',
                    gangFiles.length > 0 ? 'has-files' : '',
                  ].filter(Boolean).join(' ')}
                  onDrop={handleGangDrop}
                  onDragOver={(e) => { e.preventDefault(); setGangIsDragOver(true) }}
                  onDragLeave={() => setGangIsDragOver(false)}
                >
                  {gangFiles.length === 0 ? (
                    <>
                      <div className="dropzone-label">Drop designs here</div>
                      <div className="dropzone-sub">
                        PNG, JPG, PDF, SVG · max 24 MB each · or use the button below
                      </div>
                    </>
                  ) : (
                    <div className="gang-file-list">
                      {gangFiles.map((file, idx) => (
                        <div key={idx} className="gang-file-item">
                          {gangPreviewUrls[idx] ? (
                            <img
                              src={gangPreviewUrls[idx]}
                              alt=""
                              className="gang-thumb"
                            />
                          ) : (
                            <SmallFileIcon />
                          )}
                          <span className="gang-file-name">{file.name}</span>
                          <button
                            type="button"
                            className="gang-remove-btn"
                            onClick={() => removeGangFile(idx)}
                            aria-label={`Remove ${file.name}`}
                          >
                            ×
                          </button>
                        </div>
                      ))}
                    </div>
                  )}
                </div>

                {/* Gang secondary controls: add button + spacing */}
                <div className="gang-controls-row">
                  <label className="btn-add-design">
                    + Add design
                    <input
                      type="file"
                      multiple
                      accept=".png,.jpg,.jpeg,.pdf,.svg"
                      className="sr-only"
                      onChange={(e) => {
                        if (e.target.files) addGangFiles(Array.from(e.target.files))
                        e.target.value = ''
                      }}
                    />
                  </label>

                  <div className="control-group" style={{ maxWidth: 140 }}>
                    <label className="control-label" htmlFor="gang-spacing">
                      Spacing
                    </label>
                    <select
                      id="gang-spacing"
                      className="control-select"
                      value={gangSpacing}
                      onChange={(e) => setGangSpacing(e.target.value)}
                    >
                      <option value="tight">Tight</option>
                      <option value="normal">Normal</option>
                      <option value="loose">Loose</option>
                    </select>
                  </div>
                </div>

                {sharedControls}

                {gangFiles.length === 0 && (
                  <p className="gang-hint">
                    Add at least 1 design — drag & drop or use the button above
                  </p>
                )}

                <button
                  className="btn-primary"
                  onClick={handleGangSheet}
                  disabled={gangFiles.length < 1}
                  type="button"
                >
                  Generate Gang Sheet
                </button>
              </>
            )}
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
        Last deployed: {formatBuildTime(buildTime)}
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

function SmallFileIcon() {
  return (
    <div className="gang-file-icon">
      <svg viewBox="0 0 24 24" width="20" height="20" fill="none" aria-hidden="true">
        <path
          d="M13 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9z"
          stroke="#888"
          strokeWidth="1.5"
          fill="none"
          strokeLinejoin="round"
        />
        <path d="M13 2v7h7" stroke="#aaa" strokeWidth="1.5" strokeLinejoin="round" />
      </svg>
    </div>
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
