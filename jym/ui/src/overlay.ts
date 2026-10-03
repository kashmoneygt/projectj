import type { Decision } from './decision'
import type { AircraftId } from './scene'
import * as views from './views'

type Layout = AircraftId
type Presses = Record<string, number>
interface Input {
  who: string; detail: string; pressed: Presses; heat: Presses; throttle: number | null; flaps: string | null
}
interface Key { label: string; hint?: string; col: number; row: number; span?: number }

const KEYBOARDS: Record<Layout, Record<string, Key>> = {
  cessna: {
    Q: { label: 'Q', hint: 'steer ←', col: 0, row: 0 }, W: { label: 'W', hint: 'nose down', col: 1, row: 0 },
    E: { label: 'E', hint: 'steer →', col: 2, row: 0 }, R: { label: 'R', hint: 'power +', col: 3, row: 0 },
    A: { label: 'A', hint: 'turn ←', col: 0, row: 1 }, S: { label: 'S', hint: 'nose up', col: 1, row: 1 },
    D: { label: 'D', hint: 'turn →', col: 2, row: 1 }, F: { label: 'F', hint: 'power −', col: 3, row: 1 },
    Space: { label: 'SPACE', hint: 'brakes', col: 0, row: 2, span: 3 },
  },
  // stabilized mode: above half throttle climbs, stick forward flies forward
  drone: {
    Q: { label: 'Q', hint: 'turn ←', col: 0, row: 0 }, W: { label: 'W', hint: 'forward', col: 1, row: 0 },
    E: { label: 'E', hint: 'turn →', col: 2, row: 0 }, R: { label: 'R', hint: 'climb', col: 3, row: 0 },
    A: { label: 'A', hint: 'turn ←', col: 0, row: 1 }, S: { label: 'S', hint: 'slow', col: 1, row: 1 },
    D: { label: 'D', hint: 'turn →', col: 2, row: 1 }, F: { label: 'F', hint: 'descend', col: 3, row: 1 },
  },
}
// how hard each answer presses each key
const PRESSES: Record<Layout, Record<string, Record<string, Presses>>> = {
  cessna: {
    brakes: { on: { Space: 1 } },
    lift_off: { 'lift off': { S: 1 } },
    turn: { 'left 60°': { A: 1 }, 'left 20°': { A: 0.5 }, 'left 5°': { A: 0.2 }, 'right 5°': { D: 0.2 },
      'right 20°': { D: 0.5 }, 'right 60°': { D: 1 } },
    climb: { 'climb fast': { S: 1 }, climb: { S: 0.5 }, flare: { S: 0.25 }, 'descend slowly': { W: 0.25 }, descend: { W: 0.5 }, 'descend fast': { W: 1 } },
  },
  drone: {
    turn: { 'left 60°': { A: 1 }, 'left 20°': { A: 0.5 }, 'left 5°': { A: 0.2 }, 'right 5°': { D: 0.2 },
      'right 20°': { D: 0.5 }, 'right 60°': { D: 1 } },
    climb: { 'climb fast': { R: 1 }, climb: { R: 0.5 }, flare: { F: 0.1 }, 'descend slowly': { F: 0.25 }, descend: { F: 0.5 }, 'descend fast': { F: 1 } },
    speed: { '15 kt': { W: 0.33 }, '45 kt': { W: 1 } },
  },
}
// lever position (0 to 1) for each power or speed answer
const LEVER: Record<string, number> = { idle: 0, half: 0.5, full: 1, '65 kt': 0.55, '80 kt': 0.75, '100 kt': 1 }
// match the simulator's drone lever: half is level and full is 1000 ft/min up
const DRONE_LEVER: Record<string, number> = {
  'climb fast': 1, climb: 0.75, level: 0.5, flare: 0.45, 'descend slowly': 0.375, descend: 0.25, 'descend fast': 0,
}
// the key each held KeyboardEvent.code lights up
const HELD: Record<string, string> = {
  KeyW: 'W', ArrowUp: 'W', KeyS: 'S', ArrowDown: 'S', KeyA: 'A', ArrowLeft: 'A', KeyD: 'D', ArrowRight: 'D',
  KeyQ: 'Q', KeyE: 'E', KeyR: 'R', KeyF: 'F',
}

// streamer-style input overlay: each choice is drawn as the keys a person would press
export class InputOverlay {
  readonly root = views.el('div', 'overlay')
  private head = views.el('div', 'who')
  private board = views.el('div', 'keys')
  private keys = new Map<string, HTMLElement>()
  private layout: Layout | null = null
  private shown = ''

  constructor() {
    this.root.append(this.head, this.board)
    this.root.hidden = true
  }

  // chosen options press their keys; each option's probability tints its keys
  static fromDecision(layout: Layout, who: string, decision: Decision, previous: Decision | null): Input {
    const pressed: Presses = {}
    const heat: Presses = {}
    for (const [question, answer] of Object.entries(decision.answers)) {
      for (const [option, keys] of Object.entries(PRESSES[layout][question] ?? {})) {
        const probability = answer.probabilities?.[option] ?? (answer.choice === option ? 1 : 0)
        for (const [key, strength] of Object.entries(keys)) {
          heat[key] = (heat[key] ?? 0) + probability * strength
          if (answer.choice === option) pressed[key] = Math.max(pressed[key] ?? 0, strength)
        }
      }
    }
    const lever = (answers: Decision['answers'] | undefined) => {
      if (layout === 'drone') return answers?.climb?.choice !== undefined ? DRONE_LEVER[answers.climb.choice] ?? null : null
      const choice = answers?.throttle?.choice ?? answers?.speed?.choice
      return choice !== undefined ? LEVER[choice] ?? null : null
    }
    const throttle = lever(decision.answers)
    const before = lever(previous?.answers)
    if (layout === 'cessna' && throttle !== null && before !== null && throttle !== before) pressed[throttle > before ? 'R' : 'F'] = 1
    return {
      who, detail: `step ${decision.step} · ${(decision.seconds * 1000).toFixed(0)} ms`, pressed, heat,
      throttle, flaps: decision.answers.flaps?.choice ?? null,
    }
  }

  // show a keyboard player's held keys, brakes and throttle lever
  static fromKeys(layout: Layout, held: Set<string>, values: Record<string, number>): Input {
    const pressed: Presses = {}
    for (const code of held) if (HELD[code]) pressed[HELD[code]] = 1
    if (layout === 'cessna' && (values.brake ?? 0) > 0.5) pressed.Space = 1
    const flaps = layout === 'cessna' ? `${Math.round((values.flaps ?? 0) * 30)}°` : null
    return { who: 'You', detail: 'keyboard', pressed, heat: {}, throttle: values.throttle ?? 0, flaps }
  }

  show(layout: Layout, input: Input | null) {
    this.root.hidden = !input
    if (!input) return
    if (layout !== this.layout) this.build(layout)
    this.head.replaceChildren(views.el('strong', '', input.who), views.el('span', '', input.detail))
    const changed = input.detail !== this.shown
    this.shown = input.detail
    for (const [id, key] of this.keys) {
      const pressed = input.pressed[id] ?? 0
      key.style.setProperty('--press', String(pressed))
      key.classList.toggle('pressed', pressed > 0)
      ;(key.querySelector('.heat') as HTMLElement).style.height = `${Math.round(Math.min(1, input.heat[id] ?? 0) * 100)}%`
      if (changed && pressed > 0) {
        key.classList.remove('pulse')
        void key.offsetWidth
        key.classList.add('pulse')
      }
    }
    this.board.querySelector<HTMLElement>('.gauge i')!.style.height = `${Math.round((input.throttle ?? 0) * 100)}%`
    const flaps = this.board.querySelector('.flaps')
    if (flaps) flaps.textContent = `FLAPS ${input.flaps ?? '0°'}`
  }

  private build(layout: Layout) {
    this.layout = layout
    this.keys.clear()
    this.board.replaceChildren()
    this.board.className = `keys ${layout}`
    for (const [id, key] of Object.entries(KEYBOARDS[layout])) {
      const node = views.el('div', 'key', [views.el('span', 'heat'), views.el('b', '', key.label),
        ...(key.hint ? [views.el('small', '', key.hint)] : [])])
      node.style.gridColumn = `${key.col + 1} / span ${key.span ?? 1}`
      node.style.gridRow = String(key.row + 1)
      this.board.append(node)
      this.keys.set(id, node)
    }
    this.board.append(views.el('div', 'gauge', [views.el('i'), views.el('small', '', 'THR')]))
    if (layout === 'cessna') {
      const flaps = views.el('div', 'flaps', 'FLAPS 0°')
      flaps.style.gridColumn = '4 / span 2'
      this.board.append(flaps)
    }
  }
}
