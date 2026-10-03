import * as views from './views'

interface Question { instructions: string; criteria: Record<string, string | null> }
interface Answer { choice?: string; probabilities?: Record<string, number> }
export interface Decision {
  step: number; state: Record<string, unknown>; questions: Record<string, Question>; answers: Record<string, Answer>
  seconds: number; model?: string; warnings?: string[]; action?: Record<string, unknown>
  asked_at?: number; applied_at?: number | null; clock?: number
}
export interface Pending { step: number; state: Record<string, unknown>; questions: Record<string, Question> }

const ROWS = 400

// each decision's state, questions, answers and the step that applied them
export class DecisionPanel {
  readonly root = views.el('div', 'decision-panel')
  private header = views.el('div', 'decision-head')
  private questions = views.el('div', 'questions')
  private action = views.el('p', 'action muted')
  private state = views.el('div')
  private list = views.el('ol', 'trajectory')
  private follow = views.el('button', 'follow', 'Back to latest')
  private decisions: Decision[] = []
  private selected: number | null = null
  private pending: Pending | null = null
  private picks: Record<string, string> = {}
  private readonly answer: (step: number, choices: Record<string, string>) => void
  // state and trajectory blocks, shown once there's a step
  private blocks: HTMLElement[] = []

  constructor(answer: (step: number, choices: Record<string, string>) => void) {
    this.answer = answer
    const stateBlock = views.el('details', 'block', [views.el('summary', '', 'State sent to the model'), this.state])
    stateBlock.open = true
    this.follow.addEventListener('click', () => this.select(null))
    this.follow.hidden = true
    const trajectoryBlock = views.el('div', 'block', [views.el('div', 'block-head', [views.el('h3', '', 'Trajectory'), this.follow]), this.list])
    this.blocks = [stateBlock, trajectoryBlock]
    this.root.append(this.header, this.questions, this.action, stateBlock, trajectoryBlock)
    document.addEventListener('keydown', (event) => {
      if (event.key === 'Enter' && !(event.target as HTMLElement).closest('input, select, textarea')) this.submit()
    })
    this.clear()
  }

  clear(message = '') {
    this.decisions = []
    this.selected = null
    this.pending = null
    this.list.replaceChildren()
    this.header.replaceChildren(views.el('span', 'muted', message))
    this.questions.replaceChildren()
    this.action.textContent = ''
    this.state.replaceChildren()
    this.reveal(false)
  }

  private reveal(shown: boolean) {
    for (const block of this.blocks) block.hidden = !shown
  }

  // add decisions and their steps, once each
  add(events: Record<string, unknown>[]) {
    for (const event of events) {
      const last = this.decisions.at(-1)
      if (event.type === 'decision' && (!last || (event.step as number) > last.step)) this.decisions.push(event as unknown as Decision)
      else if (event.type === 'step' && last && last.step === event.step) last.action = event.action as Record<string, unknown>
    }
    this.reveal(this.decisions.length > 0 || !!this.pending)
    this.renderList()
    if (!this.pending) this.render()
  }

  // let a person answer, with the current settings preselected
  ask(pending: Pending | null) {
    if (pending?.step === this.pending?.step) return
    this.pending = pending
    const current = (pending?.state.current_controls ?? {}) as Record<string, string>
    this.picks = pending ? Object.fromEntries(Object.keys(pending.questions).filter((q) => q in current).map((q) => [q, current[q]])) : {}
    this.reveal(this.decisions.length > 0 || !!pending)
    this.render()
  }

  // during replay, show the latest decision at or before the clock
  at(clock: number) {
    let index = -1
    while (index + 1 < this.decisions.length && (this.decisions[index + 1].clock ?? Infinity) <= clock) index++
    if (index !== this.selected) this.select(index < 0 ? null : index, false)
  }

  private select(index: number | null, scroll = true) {
    this.selected = index
    this.follow.hidden = index === null
    this.render()
    this.renderList()
    if (scroll && index !== null) this.list.children[index]?.scrollIntoView({ block: 'nearest' })
  }

  shown(): Decision | null {
    return this.selected === null ? this.decisions.at(-1) ?? null : this.decisions[this.selected] ?? null
  }

  before(decision: Decision): Decision | null {
    const index = this.decisions.indexOf(decision)
    return index > 0 ? this.decisions[index - 1] : null
  }

  private render() {
    if (this.pending) return this.renderPending(this.pending)
    const decision = this.shown()
    if (!decision) return
    const model = decision.model ? Object.assign(views.el('span', 'muted', ` · ${shortModel(decision.model)}`), { title: decision.model }) : null
    this.header.replaceChildren(views.el('strong', '', `Step ${decision.step}`), views.el('span', 'muted',
      ` · decided in ${(decision.seconds * 1000).toFixed(0)} ms`), ...(model ? [model] : []))
    this.questions.replaceChildren(...Object.entries(decision.questions).map(([id, question]) =>
      this.question(id, question, decision.answers[id], false)))
    for (const warning of decision.warnings ?? []) this.questions.append(views.el('p', 'warning', warning))
    this.action.textContent = decision.action ? `Applied: ${describe(decision.action)}`
      : decision.applied_at === null ? 'Not applied: the situation changed while it decided' : 'Applying…'
    this.state.replaceChildren(tree(decision.state))
  }

  private renderPending(pending: Pending) {
    const ids = Object.keys(pending.questions)
    const submit = views.el('button', 'primary', 'Submit (Enter)')
    submit.disabled = !ids.every((id) => id in this.picks)
    submit.addEventListener('click', () => this.submit())
    this.header.replaceChildren(views.el('strong', '', `Your turn · step ${pending.step}`),
      views.el('span', 'muted', ' · current settings are preselected'))
    this.questions.replaceChildren(...ids.map((id) => this.question(id, pending.questions[id], undefined, true)), submit)
    this.action.textContent = ''
    this.state.replaceChildren(tree(pending.state))
  }

  private question(id: string, question: Question, answer: Answer | undefined, open: boolean) {
    const card = views.el('section', 'question')
    card.append(views.el('p', 'instructions', question.instructions))
    for (const [option, about] of Object.entries(question.criteria)) {
      const p = answer?.probabilities?.[option]
      const chosen = open ? this.picks[id] === option : answer?.choice === option
      const row = views.el(open ? 'button' : 'div', `option${chosen ? ' chosen' : ''}`, [
        views.el('span', 'name', option),
        views.el('span', 'about', about ?? ''),
        views.el('span', 'meter', views.el('i')),
        views.el('span', 'pct mono', p === undefined ? '' : `${Math.round(p * 100)}%`),
      ])
      ;(row.querySelector('.meter i') as HTMLElement).style.width = `${Math.round((p ?? (chosen ? 1 : 0)) * 100)}%`
      if (open) {
        row.addEventListener('click', () => {
          this.picks[id] = option
          this.render()
        })
      }
      card.append(row)
    }
    return card
  }

  private submit() {
    const pending = this.pending
    if (!pending || !Object.keys(pending.questions).every((id) => id in this.picks)) return
    this.pending = null
    this.answer(pending.step, { ...this.picks })
    this.header.replaceChildren(views.el('span', 'muted', 'Sent'))
  }

  private renderList() {
    const start = Math.max(0, this.decisions.length - ROWS)
    const rows = this.decisions.slice(start).map((decision, offset) => {
      const index = start + offset
      const choices = Object.values(decision.answers).map((answer) => answer.choice).join(' · ')
      const row = views.el('li', index === this.selected ? 'selected' : '', [views.el('span', 'mono', `#${decision.step}`),
        views.el('span', 'choices', choices), views.el('span', 'muted mono', `${(decision.seconds * 1000).toFixed(0)} ms`)])
      row.addEventListener('click', () => this.select(index))
      return row
    })
    const pinned = this.list.scrollHeight - this.list.scrollTop - this.list.clientHeight < 24
    this.list.replaceChildren(...rows)
    if (pinned && this.selected === null) this.list.scrollTop = this.list.scrollHeight
  }
}

// "Flight-Autopilot-Decision-Model-Maverick1.0-Q8_0.gguf" -> "Flight-Autopilot-Decision-Model-Maverick1.0 Q8_0"
function shortModel(model: string) {
  const file = model.split(/[\\/]/).at(-1) ?? model
  const stem = file.replace(/\.(gguf|safetensors|bin|onnx)$/i, '')
  return stem.replace(/(Qwen3-0\.6B)-/i, '$1 ').replace(/[-_]+(Q\d(?:_\d)?)$/i, ' $1')
}

// show tick and sim_time as the time applied, not as choices
function describe(action: Record<string, unknown>) {
  const { tick: _tick, sim_time: at, ...choices } = action
  const applied = Object.entries(choices).map(([key, value]) => `${key} ${typeof value === 'number' ? value : String(value)}`).join(', ')
  return typeof at === 'number' ? `${applied} (at ${at.toFixed(1)} s)` : applied
}

function tree(value: unknown): Node {
  if (value === null || typeof value !== 'object') return views.el('span', 'value mono', value === null ? '—' : String(value))
  if (Array.isArray(value) && value.every((item) => item === null || typeof item !== 'object')) {
    return views.el('span', 'value mono', `[${value.join(', ')}]`)
  }
  const rows = Object.entries(value as Record<string, unknown>).map(([key, item]) =>
    views.el('div', 'row', [views.el('span', 'key mono', key), tree(item)]))
  return views.el('div', 'tree', rows)
}
