interface Split {
  parent: HTMLElement
  property: string
  key: string
  initial: number
  min: number
  max: number
}

// draggable divider for the next pane's width
export function splitPane(handle: HTMLElement, settings: () => Split) {
  const key = `jym-pane-${settings().key}`
  let value = settings().initial
  try {
    const stored = localStorage.getItem(key)
    if (stored !== null && Number.isFinite(Number(stored))) value = Number(stored)
  } catch { /* storage can be disabled */ }
  const set = (next: number, persist = false) => {
    const spec = settings()
    value = Math.min(spec.max, Math.max(spec.min, next))
    spec.parent.style.setProperty(spec.property, `${value}px`)
    handle.setAttribute('aria-valuemin', String(Math.round(spec.min)))
    handle.setAttribute('aria-valuemax', String(Math.round(spec.max)))
    handle.setAttribute('aria-valuenow', String(Math.round(value)))
    if (persist) {
      try { localStorage.setItem(key, String(value)) } catch { /* storage can be disabled */ }
    }
  }
  handle.addEventListener('pointerdown', (event) => {
    if (event.button !== 0) return
    event.preventDefault()
    handle.focus()
    handle.setPointerCapture(event.pointerId)
    handle.dataset.dragging = 'true'
  })
  handle.addEventListener('pointermove', (event) => {
    if (!handle.hasPointerCapture(event.pointerId)) return
    const bounds = settings().parent.getBoundingClientRect()
    set(bounds.width - (event.clientX - bounds.left))
  })
  const release = () => {
    delete handle.dataset.dragging
    set(value, true)
  }
  handle.addEventListener('pointerup', release)
  handle.addEventListener('pointercancel', release)
  handle.addEventListener('keydown', (event) => {
    if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return
    event.preventDefault()
    const spec = settings()
    set(event.key === 'Home' ? spec.min : event.key === 'End' ? spec.max : value + (event.key === 'ArrowRight' ? -24 : 24), true)
  })
  handle.addEventListener('dblclick', () => set(settings().initial, true))
  new ResizeObserver(() => set(value)).observe(settings().parent)
  set(value)
}
