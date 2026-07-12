interface Params {
  max_colors: number
  page_size: string
  orientation: string
}

interface AdjustPanelProps {
  params: Params
  onChange: (next: Params) => void
  onRegenerate: () => void
  isLoading: boolean
}

export default function AdjustPanel({
  params,
  onChange,
  onRegenerate,
  isLoading,
}: AdjustPanelProps) {
  const set = (key: keyof Params, value: string | number) =>
    onChange({ ...params, [key]: value })

  return (
    <div className="adjust-panel">
      <div className="adjust-heading">Not right? Adjust:</div>
      <div className="adjust-row">
        <div className="control-group">
          <label className="control-label" htmlFor="adj-colors">
            Colors
          </label>
          <select
            id="adj-colors"
            className="control-select"
            value={params.max_colors}
            onChange={(e) => set('max_colors', Number(e.target.value))}
            disabled={isLoading}
          >
            <option value={2}>2</option>
            <option value={3}>3</option>
            <option value={4}>4</option>
            <option value={5}>5</option>
          </select>
        </div>

        <div className="control-group">
          <label className="control-label" htmlFor="adj-size">
            Size
          </label>
          <select
            id="adj-size"
            className="control-select"
            value={params.page_size}
            onChange={(e) => set('page_size', e.target.value)}
            disabled={isLoading}
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
          <label className="control-label" htmlFor="adj-orient">
            Orientation
          </label>
          <select
            id="adj-orient"
            className="control-select"
            value={params.orientation}
            onChange={(e) => set('orientation', e.target.value)}
            disabled={isLoading}
          >
            <option value="auto">Auto</option>
            <option value="portrait">Portrait</option>
            <option value="landscape">Landscape</option>
          </select>
        </div>

        <button
          className="btn-regenerate"
          onClick={onRegenerate}
          disabled={isLoading}
          type="button"
        >
          {isLoading ? (
            <>
              <SmallSpinner />
              Regenerating…
            </>
          ) : (
            'Regenerate'
          )}
        </button>
      </div>
    </div>
  )
}

function SmallSpinner() {
  return (
    <span
      style={{
        display: 'inline-block',
        width: 12,
        height: 12,
        border: '2px solid #ccc',
        borderTopColor: '#111',
        borderRadius: '50%',
        animation: 'spin 0.7s linear infinite',
        flexShrink: 0,
      }}
      aria-hidden="true"
    />
  )
}
