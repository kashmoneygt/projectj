import type { AircraftId } from './scene'
import * as views from './views'

type Values = Record<string, number>

const SPRING = { elevator: 0.5, aileron: 0.5, pedals: 0.6 }
// actuators only (scenes also report derived values like the rudder)
const FLIGHT_CONTROLS = new Set(['throttle', 'elevator', 'aileron', 'pedals', 'brake', 'flaps', 'trim', 'roll_trim'])
const actuators = (controls: Values | null) =>
  Object.fromEntries(Object.entries(controls ?? {}).filter(([name]) => FLIGHT_CONTROLS.has(name)))
const SLIDERS: [string, string, number, number, number][] = [
  ['throttle', 'Throttle', 0, 1, 0.01], ['flaps', 'Flaps', 0, 1, 1 / 3], ['trim', 'Pitch trim', -1, 1, 0.01],
  ['pedals', 'Rudder', -1, 1, 0.01], ['elevator', 'Elevator', -1, 1, 0.01], ['aileron', 'Aileron', -1, 1, 0.01],
]
export const FLIGHT_KEYS = [
  ['W / S or ↑ / ↓', 'push / pull: nose down / up'], ['A / D or ← / →', 'roll left / right'],
  ['Q / E', 'rudder and nosewheel left / right'], ['R / F', 'throttle up / down'], ['Space', 'brakes on / off'],
  [']  /  [', 'flaps down / up'], ['.  /  ,', 'trim nose up / down'], ['C', 'cockpit or chase camera'],
]
// drone sticks in stabilized mode: let go and it hovers
const DRONE_SLIDERS: typeof SLIDERS = [['throttle', 'Throttle', 0, 1, 0.01], ['elevator', 'Forward', -1, 1, 0.01], ['aileron', 'Turn', -1, 1, 0.01]]
export const DRONE_KEYS = [
  ['R / F', 'throttle up / down: above half climbs, below half descends'], ['W / S or ↑ / ↓', 'fly forward / slow down'],
  ['A / D or Q / E', 'turn left / right'], ['C', 'cockpit or chase camera'],
]

// keyboard, gamepad and slider controls, sent to the server 20 times a second
export class Controls {
  readonly root = views.el('div', 'controls')
  aircraft: AircraftId | '' = ''
  active = false
  values: Values = {}
  readonly held = new Set<string>()
  private sliders = new Map<string, HTMLInputElement>()
  private touched: Record<string, number> = {}
  private last = performance.now()
  private sent = 0
  private readonly send: (values: Values) => void
  private readonly camera: () => void

  constructor(send: (values: Values) => void, camera: () => void) {
    this.send = send
    this.camera = camera
    addEventListener('keydown', (event) => this.key(event, true))
    addEventListener('keyup', (event) => this.key(event, false))
    addEventListener('blur', () => this.release())
  }

  start(controls: Values | null, aircraft: AircraftId | '') {
    this.aircraft = aircraft
    this.active = true
    this.held.clear()
    this.values = actuators(controls)
    this.render()
  }

  stop() {
    this.release()
    this.active = false
  }

  // match the aircraft's controls, except for a slider being dragged
  sync(controls: Values) {
    for (const [name, value] of Object.entries(actuators(controls))) if (!(name in this.values)) this.values[name] = value
    for (const [name, input] of this.sliders) {
      if (performance.now() - (this.touched[name] ?? 0) > 500 && typeof controls[name] === 'number') input.value = String(controls[name])
    }
  }

  // runs every frame, turning held keys and the gamepad into control positions
  tick() {
    const now = performance.now()
    const dt = Math.min(0.1, (now - this.last) / 1000)
    this.last = now
    if (!this.active) return
    const v = this.values
    const key = (...codes: string[]) => codes.some((code) => this.held.has(code))
    const pad = navigator.getGamepads?.().find((gamepad) => gamepad) ?? null
    const axis = (index: number) => (pad && Math.abs(pad.axes[index]) > 0.1 ? pad.axes[index] : 0)
    const button = (index: number) => pad?.buttons[index]?.value ?? 0
    const aim = {
      elevator: (key('KeyW', 'ArrowUp') ? SPRING.elevator : 0) - (key('KeyS', 'ArrowDown') ? SPRING.elevator : 0) + axis(1),
      aileron: (key('KeyD', 'ArrowRight') ? SPRING.aileron : 0) - (key('KeyA', 'ArrowLeft') ? SPRING.aileron : 0) + axis(0),
      pedals: (key('KeyE') ? SPRING.pedals : 0) - (key('KeyQ') ? SPRING.pedals : 0) + axis(2),
    }
    for (const [name, target] of Object.entries(aim)) {
      if (now - (this.touched[name] ?? 0) < 500) continue
      const current = v[name] ?? 0
      v[name] = current + Math.sign(target - current) * Math.min(Math.abs(target - current), 3 * dt)
    }
    const throttle = (key('KeyR') ? 1 : 0) - (key('KeyF') ? 1 : 0) + button(7) - button(6)
    v.throttle = clamp((v.throttle ?? 0) + throttle * 0.5 * dt, 0, 1)
    v.trim = clamp((v.trim ?? 0) + ((key('Comma') ? 1 : 0) - (key('Period') ? 1 : 0) + button(13) - button(12)) * 0.2 * dt, -1, 1)
    if (now - this.sent > 50) {
      this.sent = now
      this.send(Object.fromEntries(Object.entries(v).map(([name, value]) => [name, Math.round(value * 1000) / 1000])))
      this.sync(v)
    }
  }

  private key(event: KeyboardEvent, down: boolean) {
    if (!this.active || (event.target as HTMLElement).closest('input, select, textarea')) return
    if (down && !event.repeat) {
      if (event.code === 'KeyC') this.camera()
      if (event.code === 'Space') this.values.brake = (this.values.brake ?? 0) > 0.5 ? 0 : 1
      if (event.code === 'BracketRight') this.values.flaps = clamp(Math.round((this.values.flaps ?? 0) * 3 + 1) / 3, 0, 1)
      if (event.code === 'BracketLeft') this.values.flaps = clamp(Math.round((this.values.flaps ?? 0) * 3 - 1) / 3, 0, 1)
      this.render()
    }
    if (event.code === 'Space') event.preventDefault()
    if (/^(Key[WASDQERF]|Arrow|Comma|Period)/.test(event.code)) {
      event.preventDefault()
      if (down) this.held.add(event.code)
      else this.held.delete(event.code)
    }
  }

  private release() {
    if (!this.active) return
    this.held.clear()
    Object.assign(this.values, { elevator: 0, aileron: 0, pedals: 0 })
    this.send({ elevator: 0, aileron: 0, pedals: 0 })
  }

  private render() {
    this.sliders.clear()
    const drone = this.aircraft === 'drone'
    const brake = views.el('button', (this.values.brake ?? 0) > 0.5 ? 'toggle on' : 'toggle', `Brakes ${(this.values.brake ?? 0) > 0.5 ? 'on' : 'off'} (Space)`)
    brake.addEventListener('click', () => {
      this.values.brake = (this.values.brake ?? 0) > 0.5 ? 0 : 1
      this.render()
    })
    const rows = (drone ? DRONE_SLIDERS : SLIDERS).map(([name, label, min, max, step]) => {
      const input = views.el('input')
      Object.assign(input, { type: 'range', min: String(min), max: String(max), step: String(step), value: String(this.values[name] ?? 0) })
      input.addEventListener('input', () => {
        this.values[name] = Number(input.value)
        this.touched[name] = performance.now()
      })
      this.sliders.set(name, input)
      return views.el('label', 'slider', [views.el('span', '', label), input])
    })
    const note = drone ? 'With the sticks centred it hovers in place. A gamepad works too: sticks fly, triggers set the throttle.'
      : 'The aircraft starts with the brakes on. A gamepad works too: sticks fly, triggers set power.'
    this.root.replaceChildren(views.el('h3', '', 'Your controls'), ...(drone ? [] : [brake]), ...rows,
      keys(drone ? DRONE_KEYS : FLIGHT_KEYS), views.el('p', 'muted', note))
  }
}

function keys(list: string[][]) {
  return views.el('table', 'keys', list.map(([key, what]) => views.el('tr', '', [views.el('th', 'mono', key), views.el('td', '', what)])))
}

function clamp(value: number, low: number, high: number) {
  return Math.min(high, Math.max(low, value))
}
