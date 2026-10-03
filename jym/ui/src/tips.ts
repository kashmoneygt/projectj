// one shared tooltip for data-tip elements, on the body so nothing clips it
export function installTips() {
  const tip = document.createElement('div')
  tip.id = 'tip'
  tip.className = 'tip'
  tip.setAttribute('role', 'tooltip')
  tip.hidden = true
  document.body.append(tip)
  let shownFor: HTMLElement | null = null

  const tipped = (target: EventTarget | null) => (target instanceof Element ? target.closest<HTMLElement>('[data-tip]') : null)

  function show(element: HTMLElement) {
    const text = element.dataset.tip
    if (!text) return hide()
    shownFor?.removeAttribute('aria-describedby')
    shownFor = element
    const title = element.dataset.tipTitle
    const heading = title ? [Object.assign(document.createElement('strong'), { textContent: title })] : []
    const paragraphs = text.split('\n\n').map((part) => Object.assign(document.createElement('p'), { textContent: part }))
    tip.replaceChildren(...heading, ...paragraphs)
    tip.hidden = false
    element.setAttribute('aria-describedby', tip.id)
    const box = element.getBoundingClientRect()
    tip.style.left = `${Math.max(8, Math.min(box.left + box.width / 2 - tip.offsetWidth / 2, innerWidth - tip.offsetWidth - 8))}px`
    tip.style.top = `${box.bottom + 8}px`
  }

  function hide() {
    shownFor?.removeAttribute('aria-describedby')
    shownFor = null
    tip.hidden = true
  }

  document.addEventListener('pointerover', (event) => {
    const element = tipped(event.target)
    if (element && element !== shownFor) show(element)
  })
  document.addEventListener('pointerout', (event) => {
    const element = tipped(event.target)
    if (element && element === shownFor && !element.contains(event.relatedTarget as Node | null)) hide()
  })
  document.addEventListener('focusin', (event) => {
    const element = tipped(event.target)
    if (element) show(element)
  })
  document.addEventListener('focusout', (event) => {
    if (tipped(event.target) === shownFor) hide()
  })
  document.addEventListener('click', (event) => {
    const element = tipped(event.target)
    if (element) show(element)
    else if (shownFor) hide()
  })
  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape') hide()
  })
}

// the stylesheet draws the "i"
export function infoButton(label: string) {
  const button = document.createElement('button')
  button.type = 'button'
  button.className = 'info'
  button.setAttribute('aria-label', label)
  return button
}

export function setTip(element: HTMLElement, title: string, text: string) {
  if (element.dataset.tipTitle !== title) element.dataset.tipTitle = title
  if (element.dataset.tip !== text) element.dataset.tip = text
}
