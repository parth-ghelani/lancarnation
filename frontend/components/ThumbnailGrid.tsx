const API_BASE = ''  // rewrites in next.config.js proxy /api/* to the backend

interface ThumbnailGridProps {
  thumbnailPaths: string[]  // paths like "/api/thumb/{job_id}/{n}"
  colorCount: number
}

export default function ThumbnailGrid({ thumbnailPaths, colorCount }: ThumbnailGridProps) {
  const total = thumbnailPaths.length

  return (
    <div className="thumb-section">
      <div className="thumb-heading">Layer previews</div>
      <div className="thumb-grid">
        {thumbnailPaths.map((path, i) => {
          const isComposite = i === total - 1
          const label = isComposite ? 'Composite preview' : `Layer ${i + 1} of ${colorCount}`
          return (
            <div
              key={path}
              className={`thumb-card${isComposite ? ' thumb-composite' : ''}`}
            >
              <div className="thumb-img-wrap">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={`${API_BASE}${path}`}
                  alt={label}
                  className="thumb-img"
                  loading="lazy"
                />
              </div>
              <div className="thumb-label">{label}</div>
            </div>
          )
        })}
      </div>
    </div>
  )
}
