import React, { useRef, useState } from 'react'

interface DropzoneProps {
  onFileSelect: (file: File) => void
  selectedFile: File | null
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

export default function Dropzone({ onFileSelect, selectedFile }: DropzoneProps) {
  const inputRef = useRef<HTMLInputElement>(null)
  const [isDragOver, setIsDragOver] = useState(false)

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragOver(false)
    const file = e.dataTransfer.files[0]
    if (file) onFileSelect(file)
  }

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragOver(true)
  }

  const handleDragLeave = () => setIsDragOver(false)

  const handleClick = () => inputRef.current?.click()

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' || e.key === ' ') handleClick()
  }

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) onFileSelect(file)
    // Reset so the same file can be re-selected
    e.target.value = ''
  }

  const className = [
    'dropzone',
    isDragOver ? 'drag-over' : '',
    selectedFile ? 'has-file' : '',
  ]
    .filter(Boolean)
    .join(' ')

  return (
    <div
      className={className}
      onDrop={handleDrop}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onClick={handleClick}
      onKeyDown={handleKeyDown}
      tabIndex={0}
      role="button"
      aria-label="Upload design file"
    >
      <input
        ref={inputRef}
        type="file"
        accept=".png,.jpg,.jpeg,.pdf,.svg"
        className="sr-only"
        onChange={handleInputChange}
        aria-hidden="true"
      />

      {selectedFile ? (
        <>
          <FileIcon />
          <div className="dropzone-file-name">{selectedFile.name}</div>
          <div className="dropzone-file-size">{formatBytes(selectedFile.size)}</div>
          <button
            className="dropzone-change"
            onClick={(e) => {
              e.stopPropagation()
              inputRef.current?.click()
            }}
            type="button"
          >
            Change file
          </button>
        </>
      ) : (
        <>
          <UploadIcon />
          <div className="dropzone-label">Drop your design here</div>
          <div className="dropzone-sub">or click to browse — PNG, JPG, PDF, SVG · max 24 MB</div>
        </>
      )}
    </div>
  )
}

function UploadIcon() {
  return (
    <svg
      className="dropzone-icon"
      viewBox="0 0 40 40"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      aria-hidden="true"
    >
      <rect width="40" height="40" rx="10" fill="#f0f0f0" />
      <path
        d="M20 26V16M16 20l4-4 4 4"
        stroke="#999"
        strokeWidth="1.8"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <path
        d="M14 28h12"
        stroke="#bbb"
        strokeWidth="1.5"
        strokeLinecap="round"
      />
    </svg>
  )
}

function FileIcon() {
  return (
    <svg
      className="dropzone-icon"
      viewBox="0 0 40 40"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      aria-hidden="true"
    >
      <rect width="40" height="40" rx="10" fill="#f0f0f0" />
      <path
        d="M13 10h9l7 7v14a1 1 0 01-1 1H13a1 1 0 01-1-1V11a1 1 0 011-1z"
        stroke="#888"
        strokeWidth="1.5"
        fill="none"
        strokeLinejoin="round"
      />
      <path d="M22 10v7h7" stroke="#aaa" strokeWidth="1.5" strokeLinejoin="round" />
    </svg>
  )
}
