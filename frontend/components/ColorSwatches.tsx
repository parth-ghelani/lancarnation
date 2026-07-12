interface ColorInfo {
  hex: string
  pixels: number
}

interface ColorSwatchesProps {
  colors: ColorInfo[]
}

export default function ColorSwatches({ colors }: ColorSwatchesProps) {
  return (
    <div className="swatches-section">
      <div className="swatches-heading">Detected ink colors</div>
      <div className="swatches-row">
        {colors.map((color) => (
          <div key={color.hex} className="swatch-item">
            <div
              className="swatch-circle"
              style={{ backgroundColor: color.hex }}
              title={`${color.hex} — ${color.pixels.toLocaleString()} px`}
              aria-label={`Color ${color.hex}`}
            />
            <span className="swatch-hex">{color.hex}</span>
          </div>
        ))}
      </div>
    </div>
  )
}
