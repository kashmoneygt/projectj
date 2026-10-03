import '@fontsource/ibm-plex-sans/latin-400.css'
import '@fontsource/ibm-plex-sans/latin-500.css'
import '@fontsource/ibm-plex-sans/latin-600.css'
import '@fontsource/ibm-plex-mono/latin-400.css'
import './style.css'
import { Controls, DRONE_KEYS, FLIGHT_KEYS } from './controls'
import { DecisionPanel, type Pending } from './decision'
import { Track, type HeatInfo, type HeatPlayer, type LiveInfo } from './heat'
import { ghostColor, ghostName, type Ghost } from './ghosts'
import { Minimap, type Waypoint } from './minimap'
import { InputOverlay } from './overlay'
import * as panes from './panes'
import { PlaybackClock, frameTime } from './playback'
import { TabRecorder } from './recording'
import { FlightView, type AircraftId, type FlightScene } from './scene'
import { infoButton, installTips, setTip } from './tips'
import * as views from './views'

interface Task { env: string; id: string; title: string; goal: string; version: string; settings?: { aircraft?: AircraftId } }
interface Aircraft { id: AircraftId; title: string }
interface Environment { id: string; title: string; aircraft?: Aircraft[]; tasks: Task[] }
interface PlayerInfo { id: string; title: string; role: string | null; trained: string; deployed: string }
interface Catalog {
  play: { env: string; task: string }; hosted: boolean; environments: Environment[]; players: PlayerInfo[]
  leaderboard_seeds: number[]; stream: { env: string; task: string }[]
}
type Mode = 'controls' | 'questions'
interface Session {
  mode: Mode; env: string; task: string; state: string; run_id: string | null; message: string | null
  result: { status: string; score: number; reason: string | null } | null
}
type RunEvent = Record<string, unknown>
// a room's state; messages carry only the parts that changed
interface Snapshot {
  session: Session | null; scene: unknown; ghosts: Ghost[]; decision: RunEvent[] | null; pending: Pending | null
  live: LiveInfo | null; scenes: Record<string, FlightScene>; decisions: Record<string, RunEvent[]>
}
const EMPTY: Snapshot = { session: null, scene: null, ghosts: [], decision: null, pending: null, live: null, scenes: {}, decisions: {} }
interface RunRow {
  run_id: string; created: number; env: string; task: string; task_version?: string; player: string; seed: number
  status: string | null; score: number | null
}
type Event = RunEvent & { clock: number; scene?: unknown }
// show heats this far behind, so there's always a next scene to blend to
const LAG_SECONDS = 0.35
// read a saved value, or save and return the fallback
function remember(storage: 'localStorage' | 'sessionStorage', key: string, fallback: string) {
  try {
    const saved = window[storage].getItem(key)
    if (saved) return saved
    window[storage].setItem(key, fallback)
  } catch { /* storage can be disabled */ }
  return fallback
}
// this tab's id, for its own room
const CLIENT = remember('sessionStorage', 'jym-client',
  globalThis.crypto?.randomUUID?.() ?? `${Date.now().toString(16)}-${Math.random().toString(16).slice(2, 14)}`)
// this browser's id, used only to count visitors
const VISITOR = remember('localStorage', 'jym-visitor', CLIENT)
const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T
const ui = {
  env: $<HTMLSelectElement>('env'), task: $<HTMLSelectElement>('task'), player: $<HTMLSelectElement>('player'),
  aircraft: $<HTMLSelectElement>('aircraft'), aircraftField: $('aircraft-field'), envField: $('env-field'),
  playerField: $('player-field'), seed: $<HTMLInputElement>('seed'), restart: $<HTMLButtonElement>('restart'),
  status: $('status'), goalInfo: $('goal-info'), upNext: $('up-next'), runsButton: $('runs-button'), notice: $('notice'),
  flight: $('flight'),
  canvas: $<HTMLCanvasElement>('view'), minimap: $<HTMLCanvasElement>('minimap'), camera: $<HTMLButtonElement>('camera'),
  instruments: $('instruments'), rail: $('rail'), panel: $('panel'), panelBody: $('panel-body'),
  replay: $('replay'), replayPlay: $<HTMLButtonElement>('replay-play'), replayTime: $<HTMLInputElement>('replay-time'),
  replayClock: $('replay-clock'), replaySpeed: $<HTMLSelectElement>('replay-speed'), replayExit: $<HTMLButtonElement>('replay-exit'),
  record: $<HTMLButtonElement>('record'), main: $('main'), ghosts: $<HTMLButtonElement>('ghosts'),
  modeWatch: $<HTMLButtonElement>('mode-watch'), modePlay: $<HTMLButtonElement>('mode-play'), playControls: $('play-controls'),
  heatCard: $('heat-card'), now: $('now'), liveBadge: $('live-badge'), nowModels: $('now-models'), here: $('here'),
  queue: $('queue'),
}

const flight = new FlightView(ui.canvas)
// observe the view's size instead of reading it every frame
new ResizeObserver(([entry]) => {
  const { width, height } = entry.contentRect
  if (width && height) flight.resize(Math.round(width), Math.round(height))
}).observe(ui.canvas)
const minimap = new Minimap(ui.minimap)
const decisions = new DecisionPanel((step, choices) => post('/api/answer', { step, choices }).catch(toast))
const controls = new Controls((values) => send({ type: 'controls', values }), toggleCamera)
const overlay = new InputOverlay()
const recorder = new TabRecorder(showRecording)
setInterval(showRecording, 1000)
ui.flight.append(overlay.root)
ui.rail.append(decisions.root, controls.root)
panes.splitPane($('divider'), () => ({ parent: ui.main, property: '--rail', key: 'rail', initial: 440, min: 320,
  max: Math.max(320, ui.main.clientWidth - 360) }))

let catalog: Catalog | null = null
let snapshot: Snapshot = EMPTY
let socket: WebSocket | null = null
let liveRun: string | null = null
let replay: { run: RunRow; frames: Event[]; time: number; duration: number; playing: boolean } | null = null
let shownScene: unknown = null
// Watch uses the shared room, Play uses your own
let view: 'watch' | 'play' = new URLSearchParams(location.search).get('play') !== null ? 'play' : 'watch'
let room: 'live' | 'mine' = view === 'watch' ? 'live' : 'mine'
let replayGhosts: Ghost[] = []
let ghostsShown = remember('localStorage', 'jym-ghosts', 'on') !== 'off'
// the heat on screen and who the camera follows
let watching: {
  heat: HeatInfo; tracks: Map<string, Track>; selected: string | null; offset: number; shown: string | null
  clock: PlaybackClock<null>
} | null = null
// the player the viewer clicked, followed in every heat they race
let picked: string | null = null
const modelLinks = new Map<string, { button: HTMLButtonElement; dot: HTMLElement; name: HTMLElement }>()
// buttons that queue the stream's next heat, one per aircraft
const queueButtons = new Map<AircraftId | '', HTMLButtonElement>()
let lostTimer: number | null = null
let lostVisible = false
let panelFocus: HTMLElement | null = null

// ---- server

async function get<T>(path: string): Promise<T> {
  const response = await fetch(path, { headers: { 'X-Jym-Client': CLIENT } })
  if (!response.ok) throw new Error((await response.json().catch(() => ({}))).detail ?? response.statusText)
  return response.json()
}

async function post(path: string, body: unknown = {}) {
  const response = await fetch(path, { method: 'POST', headers: { 'Content-Type': 'application/json', 'X-Jym-Client': CLIENT },
    body: JSON.stringify(body) })
  const data = await response.json().catch(() => ({}))
  if (!response.ok) throw new Error(data.detail ?? response.statusText)
  return data
}

function send(message: unknown) {
  if (socket?.readyState === WebSocket.OPEN) socket.send(JSON.stringify(message))
}

function connect() {
  const scheme = location.protocol === 'https:' ? 'wss' : 'ws'
  const query = `client=${encodeURIComponent(CLIENT)}&visitor=${encodeURIComponent(VISITOR)}&room=${room}`
  const current = new WebSocket(`${scheme}://${location.host}/ws?${query}`)
  socket = current
  current.onopen = () => {
    if (socket !== current) return
    sendVisibility()
    if (lostTimer !== null) clearTimeout(lostTimer)
    lostTimer = null
    if (lostVisible) hideNotice()
    lostVisible = false
  }
  current.onmessage = (message) => socket === current && apply(JSON.parse(message.data))
  current.onclose = () => {
    if (socket !== current) return // a room switch replaced this socket
    if (lostTimer !== null) clearTimeout(lostTimer)
    lostTimer = window.setTimeout(() => {
      if (socket === current) {
        showNotice('Connection lost. Reconnecting…', 'info')
        lostVisible = true
      }
    }, 1500)
    setTimeout(connect, 1000)
  }
}

function enterRoom(next: 'live' | 'mine') {
  if (room === next) return
  room = next
  snapshot = EMPTY
  shownScene = null
  liveRun = null
  flight.setScene(null)
  minimap.clear()
  decisions.clear()
  const old = socket
  socket = null
  old?.close()
  connect()
  refresh()
}

function showNotice(message: string, tone: 'error' | 'info') {
  ui.notice.textContent = message
  ui.notice.dataset.tone = tone
  ui.notice.hidden = false
}

function hideNotice() {
  ui.notice.hidden = true
}

function toast(error: unknown) {
  showNotice(error instanceof Error ? error.message : String(error), 'error')
}

const envOf = (id: string) => catalog?.environments.find((env) => env.id === id)
const taskOf = (env: string, id: string) => envOf(env)?.tasks.find((spec) => spec.id === id)
const aircraftOf = (spec?: Task): AircraftId | '' => spec?.settings?.aircraft ?? ''
// objective name, prefixed with its aircraft when there are several
function taskTitle(env: string, id: string) {
  const spec = taskOf(env, id)
  if (!spec) return id
  const fleet = envOf(env)?.aircraft ?? []
  const craft = fleet.find((plane) => plane.id === aircraftOf(spec))
  return fleet.length > 1 && craft ? `${craft.title} · ${spec.title}` : spec.title
}
const playerOf = (id: string | null) => catalog?.players.find((player) => player.id === id)
const playerTitle = (id: string | null) => playerOf(id)?.title ?? id ?? ''
const playerRole = (id: string | null) => playerOf(id)?.role ?? null
function playerTip(id: string | null) {
  const player = playerOf(id)
  return [player?.trained && `How trained: ${player.trained}`, player?.deployed && `How deployed: ${player.deployed}`]
    .filter(Boolean).join('\n\n')
}
const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms))

// ---- watch or play

function setView(next: 'watch' | 'play', force = false) {
  if (next === view && !force) return refresh()
  exitReplay()
  if (view === 'play' && active()) post('/api/stop').catch(() => {})
  controls.stop()
  view = next
  syncHeat()
  hideNotice()
  flight.setScene(null)
  minimap.clear()
  shownScene = null
  liveRun = null
  decisions.clear(next === 'watch' ? 'Waiting for the model…' : '')
  if (watching) watching.shown = null
  enterRoom(next === 'watch' ? 'live' : 'mine')
  describe()
  if (next === 'play') setTimeout(start, 200)
}

// ---- play

const environment = () => envOf(ui.env.value)
const task = () => environment()?.tasks.find((spec) => spec.id === ui.task.value)
const active = () => snapshot.session && !['ended', 'error'].includes(snapshot.session.state) ? snapshot.session : null

function fillEnvironments() {
  ui.env.replaceChildren(...catalog!.environments.map((env) => new Option(env.title, env.id)))
  ui.env.value = catalog!.play.env
  ui.envField.hidden = catalog!.environments.length < 2
  fillTasks(catalog!.play.task)
}

// objective id without the aircraft prefix
function objectiveOf(spec: Task) {
  const craft = aircraftOf(spec)
  return craft && spec.id.startsWith(`${craft}-`) ? spec.id.slice(craft.length + 1) : spec.id
}

// the same objective for another aircraft, if any
function sameObjective(taskId: string, aircraft: string) {
  const tasks = environment()?.tasks ?? []
  const current = tasks.find((spec) => spec.id === taskId)
  return current && tasks.find((spec) => aircraftOf(spec) === aircraft && objectiveOf(spec) === objectiveOf(current))?.id
}

// list only the selected aircraft's objectives
function fillTasks(selected?: string) {
  const env = environment()!
  const fleet = env.aircraft ?? []
  ui.aircraftField.hidden = fleet.length < 2
  const wanted = selected ? aircraftOf(env.tasks.find((spec) => spec.id === selected)) : ui.aircraft.value
  ui.aircraft.replaceChildren(...fleet.map((plane) => new Option(plane.title, plane.id)))
  ui.aircraft.value = fleet.some((plane) => plane.id === wanted) ? wanted : fleet[0]?.id ?? ''
  const tasks = env.tasks.filter((spec) => !fleet.length || aircraftOf(spec) === ui.aircraft.value)
  ui.task.replaceChildren(...tasks.map((spec) => new Option(spec.title, spec.id)))
  ui.task.value = selected && tasks.some((spec) => spec.id === selected) ? selected : tasks[0]?.id ?? ''
  const previous = ui.player.value
  // answering a model's questions is only offered locally
  const modes = [new Option('You (keyboard)', 'controls'),
    ...(catalog?.hosted ? [] : [new Option('You (answer the questions)', 'questions')])]
  ui.player.replaceChildren(...modes)
  ui.player.value = modes.some((option) => option.value === previous) ? previous : modes[0].value
  ui.playerField.hidden = modes.length < 2
  describe()
}

// in Play, the objective's tip explains how to play
function describe() {
  const spec = task()
  if (view === 'watch') renderNow(snapshot.live)
  else if (spec) {
    const how = ui.player.value === 'controls' ? controlsHint() : 'You answer the same questions a model gets, one step at a time.'
    setTip(ui.goalInfo, taskTitle(spec.env, spec.id), `${spec.goal}\n\n${how}`)
  }
  refresh()
}

async function restart() {
  exitReplay()
  describe()
  if (view !== 'play') return
  if (active()) {
    await post('/api/stop').catch(toast)
    for (let i = 0; i < 300 && active(); i++) await sleep(100)
  }
  liveRun = null
  decisions.clear()
  start()
}

async function start() {
  exitReplay()
  if (view !== 'play' || active() || !task()) return
  try {
    await post('/api/start', { mode: ui.player.value, env: ui.env.value, task: ui.task.value, seed: Number(ui.seed.value) || 0 })
    hideNotice()
  } catch (error) {
    toast(error)
  }
}

// ---- server updates

function apply(message: Partial<Snapshot> & { here?: number }) {
  const previous = snapshot.session
  snapshot = { ...snapshot, ...message }
  if (message.here !== undefined && catalog?.hosted) {
    ui.here.textContent = `${message.here} here`
    ui.here.hidden = false
  }
  if (message.live) applyLive(message.live)
  if (view === 'watch') {
    for (const [player, scene] of Object.entries(message.scenes ?? {})) arrived(player, scene)
    for (const [player, events] of Object.entries(message.decisions ?? {})) heard(player, events)
    return refresh()
  }
  const session = snapshot.session
  if (session?.run_id && session.run_id !== liveRun) {
    liveRun = session.run_id
    decisions.clear(session.mode === 'controls' ? 'You are flying.' : 'Waiting for the first decision…')
  }
  // start the controls from this run's first scene, not the last run's
  if (session?.mode === 'controls' && session.state === 'running' && !controls.active && message.scene !== undefined) {
    ;(document.activeElement as HTMLElement | null)?.blur()
    const scene = FlightView.isScene(message.scene) ? message.scene : null
    controls.start(scene ? scene.controls : null, aircraftOf(taskOf(session.env, session.task)))
  }
  if (controls.active && (session?.mode !== 'controls' || session.state !== 'running')) controls.stop()
  const started = session?.run_id && session.run_id !== previous?.run_id
  if (started && (session.env !== ui.env.value || session.task !== ui.task.value)) syncSelection(session)
  if (message.decision && !replay) decisions.add(message.decision)
  decisions.ask(session?.mode === 'questions' ? snapshot.pending : null)
  if (!replay) showScene(snapshot.scene)
  refresh()
}

// after a reload, select the running session's environment and objective
function syncSelection(session: Session) {
  ui.env.value = session.env
  fillTasks(session.task)
}

// retry, since a library record is replayed on first request
async function fetchRun(runId: string) {
  for (let attempt = 1; attempt <= 3; attempt++) {
    try {
      return await get<{ events: Event[] }>(`/api/runs/${runId}`)
    } catch (error) {
      console.warn(`Could not load run ${runId}, attempt ${attempt}`, error)
      if (attempt < 3) await sleep(attempt * 700)
    }
  }
  return null
}

// ---- watching a heat

function applyLive(live: LiveInfo) {
  const heat = live.heat
  const offset = live.now - Date.now() / 1000
  if (!heat) {
    watching = null
    syncHeat()
    return renderNow(live)
  }
  if (!watching || watching.heat.id !== heat.id) {
    // follow the viewer's pick, or else the first player (the featured model)
    const selected = heat.players.some((p) => p.player === picked) ? picked : heat.players[0]?.player ?? null
    watching = { heat, tracks: new Map(), selected, offset, shown: null, clock: new PlaybackClock<null>() }
    for (const player of heat.players) watching.tracks.set(player.player, new Track(player))
    flight.setScene(null)
    minimap.clear()
    syncHeat()
  } else {
    watching.heat = heat
    watching.offset = offset
  }
  for (const player of heat.players) {
    const track = watching.tracks.get(player.player)
    if (track) Object.assign(track.ghost, { status: player.status ?? player.state, score: player.score ?? 0, run_id: player.run_id ?? track.ghost.run_id })
    if (track && player.state === 'recorded' && player.run_id && !track.loaded) loadRecording(watching.heat.id, player, track)
  }
  renderNow(live)
}

// in Watch, always show the heat's other players
function syncHeat() {
  flight.heat = view === 'watch' && !!watching && !replay
}

// fetch a recorded player's run once (live players stream over the socket)
async function loadRecording(heatId: string, player: HeatPlayer, track: Track) {
  track.loaded = true
  const data = await fetchRun(player.run_id!)
  if (watching?.heat.id !== heatId) return
  if (data) {
    track.add(data.events)
    if (watching.selected === player.player) watching.shown = null // redraw the side panel's decisions
  }
  renderNow(snapshot.live)
}

// the followed player's scenes drive the playback clock
function arrived(player: string, scene: FlightScene) {
  const track = watching?.tracks.get(player)
  if (!track) return
  const count = track.scenes.length
  track.push(scene)
  if (player === watching!.selected && track.scenes.length > count) watching!.clock.push(scene.sim_time, null, performance.now())
}

function heard(player: string, events: RunEvent[]) {
  const track = watching?.tracks.get(player)
  if (!track) return
  track.add(events)
  if (watching!.selected === player && watching!.shown === player) decisions.add(events)
}

// a hosted stream only runs while someone watches
function sendVisibility() {
  send({ type: 'visibility', visible: document.visibilityState === 'visible' })
}
document.addEventListener('visibilitychange', sendVisibility)

const heatClock = () => (watching?.heat.started ? Date.now() / 1000 + watching.offset - watching.heat.started : null)

function select(player: string) {
  if (!watching) return
  picked = player
  watching.selected = player
  watching.shown = null
  watching.clock.reset()
  renderNow(snapshot.live)
}

function decisionAge(seconds: number | null | undefined) {
  if (typeof seconds !== 'number') return ''
  return seconds < 1 ? `${Math.round(seconds * 1000)} ms` : `${seconds.toFixed(1)} s`
}

function titleCase(text: string) {
  return text ? `${text[0].toUpperCase()}${text.slice(1)}` : text
}

// live players lag by the arrival jitter to move smoothly; recorded ones follow the heat's clock
function playhead(selected: Track): number {
  const player = watching!.heat.players.find((p) => p.player === watching!.selected)
  if (player?.state !== 'recorded') {
    watching!.clock.advance(frameTime())
    return watching!.clock.playhead ?? selected.latest
  }
  const clock = heatClock()
  return Math.min(clock === null ? 0 : Math.max(0, clock - LAG_SECONDS), selected.latest)
}

// draw the heat every frame
function showHeat() {
  if (!watching) return
  const selected = watching.selected ? watching.tracks.get(watching.selected) : null
  if (!selected) return
  if (watching.shown !== watching.selected) {
    watching.shown = watching.selected
    decisions.clear(`Following ${playerTitle(watching.selected)}`)
    decisions.add(selected.decisions as unknown as RunEvent[])
  }
  const t = playhead(selected)
  const others = [...watching.tracks.entries()].filter(([player]) => player !== watching!.selected).map(([, track]) => track.ghost)
  const around = selected.at(t)
  if (!around) return
  const { a, b, mix } = around
  flight.setGhosts(others, a.runway.elevation_m, true, a.aircraft === 'drone')
  minimap.ghosts = others
  showLegend([])
  flight.show(a, b, mix, a.instruments as Record<string, unknown> | null)
  minimap.draw(mix < 0.5 ? a : b, (a.waypoints ?? []) as Waypoint[])
  instrumentsFor(mix < 0.5 ? a : b)
  const heatPlayer = watching.heat.players.find((p) => p.player === watching!.selected)
  if (heatPlayer?.state === 'recorded' || heatPlayer?.state === 'done') decisions.at(t)
}

function renderNow(live: LiveInfo | null) {
  const heat = watching?.heat ?? null
  ui.liveBadge.textContent = catalog?.hosted ? 'LIVE' : 'NOW PLAYING'
  // update the follow buttons in place, so clicks aren't lost
  const players = heat && heat.players.length > 1 ? heat.players : []
  for (const [id, link] of modelLinks) if (!players.some((p) => p.player === id)) {
    link.button.remove()
    modelLinks.delete(id)
  }
  for (const player of players) {
    let link = modelLinks.get(player.player)
    if (!link) {
      const dot = views.el('span', 'dot'), name = views.el('span')
      const button = views.el('button', 'model-link', [dot, name])
      button.addEventListener('click', () => select(player.player))
      link = { button, dot, name }
      modelLinks.set(player.player, link)
    }
    const role = player.role || playerRole(player.player)
    link.dot.style.background = ghostColor(player.player)
    setText(link.name, player.title)
    link.button.setAttribute('aria-pressed', String(player.player === watching?.selected))
    // the bar shows names; details go in the tip
    const reason = player.reason && (player.state === 'done' || player.state === 'recorded') ? `Result: ${player.reason}` : ''
    const replayed = player.state === 'recorded' ? 'Here a replay of its own flight of this objective and seed.' : ''
    setTip(link.button, role ? `${player.title} · ${role}` : player.title,
      [playerTip(player.player), replayed, reason, 'Click to follow it.'].filter(Boolean).join('\n\n'))
  }
  const order = players.map((p) => modelLinks.get(p.player)!.button)
  if (order.some((button, i) => ui.nowModels.children[i] !== button) || ui.nowModels.children.length !== order.length) {
    ui.nowModels.replaceChildren(...order)
  }
  const next = view === 'watch' && !replay && live?.next && live.next.id !== heat?.id ? live.next : null
  ui.upNext.hidden = !next
  if (next) {
    setText(ui.upNext, `Up next: ${taskOf(next.env, next.task)?.title ?? next.task}`)
    setTip(ui.upNext, `Up next: ${taskTitle(next.env, next.task)} · seed ${next.seed}`, taskOf(next.env, next.task)?.goal ?? '')
  }
  const queued = next ? aircraftOf(taskOf(next.env, next.task)) : null
  for (const [craft, button] of queueButtons) button.setAttribute('aria-pressed', String(craft === queued))
}

// let viewers pick the aircraft of the next heat, when the stream flies several
function fillQueue() {
  const fleet = new Map<AircraftId | '', string[]>()
  for (const { env, task } of catalog!.stream) {
    const craft = aircraftOf(taskOf(env, task))
    fleet.set(craft, [...(fleet.get(craft) ?? []), task])
  }
  if (fleet.size < 2) return
  for (const [craft, tasks] of fleet) {
    const title = catalog!.environments.flatMap((env) => env.aircraft ?? []).find((plane) => plane.id === craft)?.title ?? craft
    const button = views.el('button', '', title)
    button.title = `Fly the ${title} next`
    button.addEventListener('click', () => post('/api/next', { tasks }).catch(toast))
    queueButtons.set(craft, button)
  }
  ui.queue.replaceChildren(...queueButtons.values())
}

const titles = (ids: string[]) => ids.map((id) => playerTitle(id)).join(' and ')

// who plays what, then how it's going
function watchStatus(): StatusLine {
  const live = snapshot.live
  const heat = watching?.heat
  if (!live) return { rest: 'Connecting…' }
  if (!heat) {
    if (live.loading.length) return { rest: `Loading ${titles(live.loading)}…` }
    return { rest: live.roster.length ? 'The next flight starts in a moment' : 'Nothing is flying' }
  }
  const player = heat.players.find((p) => p.player === watching!.selected)
  const named = {
    who: player?.title, whoTip: player ? playerTip(player.player) : undefined,
    what: taskTitle(heat.env, heat.task), whatTip: taskOf(heat.env, heat.task)?.goal,
  }
  const seed = ` · seed ${heat.seed}`
  if (!player) return { ...named, rest: seed }
  const clock = heatClock()
  const state = player.state
  if (state === 'error') return { ...named, rest: `${seed} · ${player.reason ?? 'error'}`, tone: 'error' }
  if (state === 'done' || state === 'recorded' && heat.ended) {
    const result = `${titleCase(player.status ?? state)} · score ${player.score ?? 0}`
    return { ...named, rest: `${seed} · ${result}${player.reason ? ` · ${player.reason}` : ''}`, tone: player.status ?? '' }
  }
  if (state === 'recorded') return { ...named, rest: `${seed} · replaying its recorded flight` }
  if (clock === null || clock < 0 || state !== 'running') return { ...named, rest: `${seed} · starting` }
  const answered = decisionAge(player.last_seconds)
  return { ...named, rest: `${seed} · step ${player.decisions ?? 0}${answered ? ` · answered in ${answered}` : ''}` }
}

// built in parts so info marks and open tips stay put while it updates
interface StatusLine { who?: string; whoTip?: string; what?: string; whatTip?: string; rest: string; tone?: string }
const line = {
  who: document.createElement('span'), whoInfo: infoButton('About this model'),
  what: document.createElement('span'), whatInfo: infoButton('About this objective'), rest: document.createElement('span'),
}
ui.status.append(line.who, line.whoInfo, line.what, line.whatInfo, line.rest)

function renderStatus(status: StatusLine) {
  setText(line.who, status.who ?? '')
  setText(line.what, status.what ? `${status.who ? ' · ' : ''}${status.what}` : '')
  setText(line.rest, status.rest)
  line.whoInfo.hidden = !status.whoTip
  line.whatInfo.hidden = !status.whatTip
  setTip(line.whoInfo, status.who ?? '', status.whoTip ?? '')
  setTip(line.whatInfo, status.what ?? '', status.whatTip ?? '')
  const tone = status.tone ?? ''
  if (ui.status.dataset.tone !== tone) ui.status.dataset.tone = tone
}

function setText(element: HTMLElement, text: string) {
  if (element.textContent !== text) element.textContent = text
}

// ---- common

function refresh() {
  const watch = view === 'watch'
  ui.modeWatch.setAttribute('aria-pressed', String(watch))
  ui.modePlay.setAttribute('aria-pressed', String(!watch))
  ui.playControls.hidden = watch
  ui.now.hidden = !watch
  if (!watch || replay) ui.upNext.hidden = true
  ui.queue.hidden = !watch || !!replay || !queueButtons.size
  ui.ghosts.hidden = watch
  const session = snapshot.session
  const running = active()
  ui.restart.disabled = !task()
  let status: StatusLine = { rest: 'Starting…' }
  if (replay) status = { rest: `Replay · ${taskTitle(replay.run.env, replay.run.task)} · ${playerTitle(replay.run.player)} · seed ${replay.run.seed}` }
  else if (watch) status = watchStatus()
  else if (running) {
    const step = snapshot.decision?.[0]?.step ?? 0
    const doing = running.state !== 'running' ? running.state : running.mode === 'controls' ? 'flying' : `step ${step}`
    status = { rest: `You · ${doing}` }
  } else if (session?.state === 'ended' && session.result) {
    const result = session.result
    status = { rest: `${titleCase(result.status)} · score ${result.score}${result.reason ? ` · ${result.reason}` : ''}`, tone: result.status }
  } else if (session?.state === 'error') {
    status = { rest: `Error: ${session.message}`, tone: 'error' }
  }
  renderStatus(status)
  renderCenterCard()
  const flying = !watch && session?.mode === 'controls' && !!running && !replay
  decisions.root.hidden = flying
  controls.root.hidden = !flying
}

function controlsHint() {
  const keys = aircraftOf(task()) === 'drone' ? DRONE_KEYS.slice(0, 3) : FLIGHT_KEYS.slice(0, 5)
  return keys.map(([key, what]) => `${key}: ${what}`).join(' · ')
}

// your run's result card, rebuilt only when it changes
let cardKey = ''
function renderCenterCard() {
  const session = snapshot.session
  const result = !replay && view === 'play' && session?.mode === 'controls' && session.state === 'ended' && session.result
  const key = result ? `ended ${session.run_id}` : ''
  if (key === cardKey) return
  cardKey = key
  ui.heatCard.hidden = !key
  ui.heatCard.classList.toggle('interactive', !!key)
  if (!result) return
  const button = (text: string, action: () => void, tone = '') => {
    const element = views.el('button', tone, text)
    element.addEventListener('click', action)
    return element
  }
  ui.heatCard.replaceChildren(views.el('strong', '', `${titleCase(result.status)} · score ${result.score}`),
    ...(result.reason ? [views.el('p', 'muted', result.reason)] : []),
    views.el('div', 'card-actions', [button('Try again', restart, 'primary'), button('Watch', () => setView('watch'))]))
}

// ---- scenes

function showScene(scene: unknown, next?: unknown, mix = 0) {
  if (scene === shownScene && next === undefined) return
  shownScene = scene
  if (FlightView.isScene(scene)) {
    const ghosts = shownGhosts()
    flight.setGhosts(ghosts, scene.runway.elevation_m, false, scene.aircraft === 'drone')
    minimap.ghosts = ghosts
    showLegend(ghosts)
    const instruments = scene.instruments as Record<string, unknown> | null
    if (replay) flight.show(scene, FlightView.isScene(next) ? next : null, mix, instruments)
    else flight.setScene(scene, instruments)
    minimap.draw(scene, (scene.waypoints ?? []) as Waypoint[])
    instrumentsFor(scene)
    controls.sync(scene.controls)
  } else if (!replay && !scene) flight.setScene(null)
}

// ghosts of other runs, never the one being shown
function shownGhosts(): Ghost[] {
  const ghosts = replay ? replayGhosts : snapshot.ghosts
  const own = replay ? replay.run.run_id : snapshot.session?.run_id
  return ghosts.filter((ghost) => ghost.run_id !== own)
}

let legendKey = ''
function showLegend(ghosts: Ghost[]) {
  const key = JSON.stringify([ghostsShown, ghosts.map((ghost) => ghost.run_id)])
  if (key === legendKey) return
  legendKey = key
  const legend = $('ghost-legend')
  legend.hidden = !ghostsShown || !ghosts.length
  legend.replaceChildren(views.el('b', '', 'Ghosts'), ...ghosts.map((ghost) => {
    const swatch = views.el('span', 'ghost-key')
    swatch.style.background = ghostColor(ghost.player)
    return views.el('div', '', [swatch, `${ghostName(ghost)} · ${ghost.status} ${ghost.score}`])
  }))
}

function toggleGhosts() {
  ghostsShown = !ghostsShown
  try { localStorage.setItem('jym-ghosts', ghostsShown ? 'on' : 'off') } catch { /* storage can be disabled */ }
  flight.ghostsShown = minimap.ghostsShown = ghostsShown
  ui.ghosts.setAttribute('aria-pressed', String(ghostsShown))
  legendKey = ''
  showLegend(shownGhosts())
}

const cells = new Map<string, HTMLElement>()
let cellLabels = ''
function instrumentsFor(scene: FlightScene) {
  const c = scene.controls
  const target = scene.target as Record<string, unknown> | null
  const values: [string, string][] = [
    ['Airspeed', `${scene.airspeed_kt.toFixed(0)} kt`],
    ['Height', `${Math.max(0, scene.position.agl_m - 1.3).toFixed(0)} m`],
    ['Climb', `${scene.vertical_speed_fpm.toFixed(0)} ft/min`],
    ['Heading', `${scene.attitude.heading_deg.toFixed(0).padStart(3, '0')}°`],
    ['Pitch · bank', `${scene.attitude.pitch_deg.toFixed(0)}° · ${scene.attitude.roll_deg.toFixed(0)}°`],
    ...(scene.aircraft === 'drone' ? [['Ground speed', `${scene.ground_speed_kt.toFixed(0)} kt`]] as [string, string][] : [
      ['Throttle', `${Math.round(c.throttle * 100)}%`],
      ['Brakes', c.brake > 0.5 ? 'ON' : 'off'],
      ['Flaps', `${Math.round(c.flaps * 30)}°`],
    ] as [string, string][]),
    ['Status', scene.status.phase],
    ['Target', target ? Object.entries(target).map(([key, value]) => readable(key, value)).join(' · ') : '—'],
  ]
  const labels = values.map(([label]) => label).join()
  if (labels !== cellLabels) { // another aircraft's instruments
    cellLabels = labels
    cells.clear()
    ui.instruments.replaceChildren()
  }
  for (const [label, value] of values) {
    let cell = cells.get(label)
    if (!cell) {
      cell = views.el('span', 'value mono')
      cells.set(label, cell)
      ui.instruments.append(views.el('div', `cell${label === 'Target' || label === 'Status' ? ' wide' : ''}`, [views.el('span', 'label', label), cell]))
    }
    if (cell.textContent !== value) cell.textContent = value
    if (label === 'Brakes') cell.dataset.alert = String(value === 'ON')
  }
}

// "runway_left_m", 1042 -> "runway left 1042 m"
function readable(key: string, value: unknown) {
  const [, name, unit] = key.match(/^(.*?)(?:_(m|kt|deg|fpm|s))?$/)!
  return `${name.replaceAll('_', ' ')} ${Array.isArray(value) ? value.join('–') : value}${unit ? ` ${unit}` : ''}`
}

function showRecording() {
  ui.record.textContent = recorder.state === 'recording' ? `Stop export (${recorder.seconds.toFixed(0)} s)` : 'Export video'
  ui.record.classList.toggle('recording', recorder.state === 'recording')
  if (recorder.message && recorder.state !== 'recording') ui.record.title = recorder.message
}

function toggleCamera() {
  flight.setView(flight.view === 'chase' ? 'cockpit' : 'chase')
  ui.camera.textContent = flight.view === 'chase' ? 'Cockpit view (C)' : 'Chase view (C)'
}

// show the side panel's decision, or a person's held keys
function updateOverlay() {
  const session = snapshot.session
  const on = replay ? replay.run : view === 'watch' ? watching?.heat : session
  const layout = aircraftOf(on ? taskOf(on.env, on.task) : task()) === 'drone' ? 'drone' : 'cessna'
  if (!replay && view === 'play' && session?.mode === 'controls' && controls.active) {
    return overlay.show(layout, InputOverlay.fromKeys(layout, controls.held, controls.values))
  }
  const decision = decisions.shown()
  const watchingSomeone = view === 'watch' && !!watching?.selected
  if (!decision || (!replay && !watchingSomeone && !session)) return overlay.show(layout, null)
  const who = replay ? playerTitle(replay.run.player) : watchingSomeone ? playerTitle(watching!.selected) : 'You'
  overlay.show(layout, InputOverlay.fromDecision(layout, who, decision, decisions.before(decision)))
}

let lastFrame = frameTime()

function frame() {
  const now = frameTime()
  // advance by the real time elapsed, so dropped frames don't slow the replay
  const seconds = Math.min(0.1, Math.max(0, (now - lastFrame) / 1000))
  lastFrame = now
  if (replay?.playing) {
    replay.time = Math.min(replay.duration, replay.time + seconds * Number(ui.replaySpeed.value))
    if (replay.time >= replay.duration) replay.playing = false
    showReplay()
  } else if (!replay && view === 'watch') showHeat()
  controls.tick()
  updateOverlay()
  flight.render()
  requestAnimationFrame(frame)
}

// ---- replay

// replays follow sim time, so slow models play back at real speed
async function openReplay(run: RunRow) {
  try {
    const data = await fetchRun(run.run_id)
    if (!data) {
      showNotice('Could not open this replay.', 'error')
      return
    }
    liveRun = null
    let clock = 0
    for (const event of data.events) {
      clock = (event.scene as FlightScene | undefined)?.sim_time ?? clock
      event.clock = clock
    }
    const frames = data.events.filter((event) => event.scene)
    replayGhosts = taskOf(run.env, run.task)?.version === run.task_version
      ? await get<Ghost[]>(`/api/ghosts?env=${run.env}&task=${run.task}&seed=${run.seed}`).catch((error) => {
        console.warn(`Could not load ghosts for ${run.env}/${run.task} seed ${run.seed}`, error)
        return []
      })
      : []
    const start = frames[0]?.clock ?? 0
    replay = { run, frames, time: 0, duration: Math.max(0, (frames.at(-1)?.clock ?? start) - start), playing: true }
    syncHeat()
    decisions.clear('Replaying a recorded run')
    decisions.add(data.events)
    ui.replayTime.max = String(replay.duration)
    ui.replay.hidden = false
    closePanel()
    showReplay()
  } catch (error) {
    console.warn(`Could not open replay ${run.run_id}`, error)
    showNotice('Could not open this replay.', 'error')
  }
}

function showReplay() {
  if (!replay) return
  const { frames } = replay
  const at = (frames[0]?.clock ?? 0) + replay.time
  let index = 0
  while (index + 1 < frames.length && frames[index + 1].clock <= at) index++
  const a = frames[index], b = frames[index + 1]
  const mix = b ? (at - a.clock) / Math.max(1e-6, b.clock - a.clock) : 0
  shownScene = null
  showScene(a?.scene ?? null, b?.scene ?? null, mix)
  decisions.at(at)
  ui.replayTime.value = String(replay.time)
  ui.replayClock.textContent = `${replay.time.toFixed(1)} / ${replay.duration.toFixed(1)} s`
  ui.replayPlay.textContent = replay.playing ? 'Pause' : 'Play'
  refresh()
}

function exitReplay() {
  if (!replay) return
  replay = null
  syncHeat()
  ui.replay.hidden = true
  shownScene = null
  flight.setScene(null)
  minimap.clear()
  decisions.clear()
  liveRun = null
  replayGhosts = []
  if (watching) watching.shown = null
  apply(snapshot)
}

// ---- side panels

async function openPanel(name: string) {
  panelFocus = document.activeElement instanceof HTMLElement ? document.activeElement : null
  ui.panel.hidden = false
  ui.rail.hidden = true
  ui.panel.setAttribute('aria-labelledby', 'panel-title')
  $('panel-close').setAttribute('aria-label', `Close ${titleCase(name)}`)
  ui.panelBody.replaceChildren(views.el('p', 'muted', 'Loading…'))
  try {
    if (name === 'runs') {
      const runs = await get<RunRow[]>('/api/runs')
      ui.panelBody.replaceChildren(heading('Recorded runs'), table(['When', 'Objective', 'Player', 'Result', ''],
        runs.map((run) => {
          const open = views.el('button', '', 'Replay')
          open.addEventListener('click', () => openReplay(run))
          return [new Date(run.created * 1000).toLocaleString([], { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' }),
            `${run.env}/${run.task} · seed ${run.seed}`, playerTitle(run.player), run.status ? `${run.status} · ${run.score}` : 'unfinished', open]
        })))
    } else if (name === 'leaderboard') {
      if (!filters.task && watching?.heat) filters.task = `${watching.heat.env}/${watching.heat.task}`
      ui.panelBody.replaceChildren(...await leaderboard())
    } else {
      ui.panelBody.replaceChildren(heading('How it works'),
        views.el('section', '', [
          views.el('h4', '', 'About this benchmark'),
          views.el('p', '', 'Models fly a Cessna 172 (in the JSBSim flight simulator) and a quadcopter by answering ' +
            'multiple-choice questions twice a second. Qwen3 0.6B flies here on a CPU server of its own, as a small model ' +
            'could on an aircraft\'s own computer. Every run is recorded and can be replayed exactly.'),
        ]),
        views.el('p', '', 'Watch: the model flies one objective and seed after another, in real time. The simulation never ' +
          'waits for it, so a slow model is a pilot with slow reactions: while it thinks, its aircraft keeps its last settings, ' +
          'and if it goes quiet the autopilot holds the wings level. The side panel shows exactly what it saw, the probability ' +
          'it gave every option, how long it took and what was applied.'),
        views.el('p', '', 'Play: fly any objective yourself with the keyboard. Ghosts of recorded flights of the same objective ' +
          'and seed fly alongside.'),
        views.el('p', '', `The leaderboard ranks every player on seeds ${(catalog?.leaderboard_seeds ?? [0, 1, 2]).join(', ')} of ` +
          'each objective by successes, then mean score, then time per decision.'),
        views.el('p', '', ['To race your own model, ', link('run jym on your machine', 'https://github.com/kashmoneygt/projectj'),
          ' and add it to models.toml.']),
        ...(catalog?.hosted ? [] : [views.el('p', 'muted', 'Every run is recorded and can be replayed from Runs.')]))
    }
  } catch (error) {
    ui.panelBody.replaceChildren(views.el('p', '', String(error)))
  }
  requestAnimationFrame(() => ui.panelBody.querySelector<HTMLElement>('#panel-title')?.focus())
}

function heading(text: string) {
  const h = views.el('h3', '', text)
  h.id = 'panel-title'
  h.tabIndex = -1
  return h
}

function link(text: string, href: string) {
  const a = views.el('a', '', text)
  a.href = href
  return a
}

interface BoardRow {
  player: string; title: string; seeds: number[]; ranked: boolean; successes: number; mean_score: number
  median_seconds: number | null; imported: boolean; role?: string | null
  runs: { seed: number; run_id: string; status: string; score: number }[]
}
interface Board { env: string; seeds: number[]; tasks: { task: string; title: string; version: string; rows: BoardRow[] }[] }

const filters = { env: '', task: '', player: '' }

async function leaderboard(): Promise<Node[]> {
  const envs = catalog?.environments.filter((env) => !filters.env || env.id === filters.env) ?? []
  const boards = await Promise.all(envs.map((env) => get<Board>(`/api/leaderboard?env=${env.id}`)))
  const select = (label: string, key: keyof typeof filters, options: [string, string][]) => {
    const input = views.el('select')
    input.replaceChildren(...options.map(([value, text]) => new Option(text, value)))
    input.value = filters[key]
    input.addEventListener('change', () => {
      filters[key] = input.value
      if (key === 'env') filters.task = ''
      openPanel('leaderboard')
    })
    return views.el('label', 'field', [views.el('span', '', label), input])
  }
  const players = new Map<string, string>()
  for (const board of boards) for (const task of board.tasks) for (const row of task.rows) players.set(row.player, row.title)
  const tasks = envs.flatMap((env) => env.tasks.map((spec): [string, string] => [`${env.id}/${spec.id}`, taskTitle(env.id, spec.id)]))
  const rule = 'Ranked on seeds 0, 1 and 2 of each objective, flown in real time: successes, then mean score, then median ' +
    'decision time. A model is ranked once it has flown all three.'
  const nodes: Node[] = [heading('Leaderboard'), views.el('p', '', rule), views.el('div', 'filters', [
    select('Environment', 'env', [['', 'All'], ...(catalog?.environments ?? []).map((env): [string, string] => [env.id, env.title])]),
    select('Objective', 'task', [['', 'All'], ...tasks]),
    select('Model', 'player', [['', 'All'], ...[...players].map(([id, title]): [string, string] => [id, title])]),
  ])]
  const shown = boards.flatMap((board) => board.tasks.map((task) => ({ env: board.env, ...task })))
    .filter((task) => !filters.task || `${task.env}/${task.task}` === filters.task)
  if (!filters.task) {
    // one row per player, one column per objective; complete players rank first
    const seeds = boards[0]?.seeds.length ?? 3
    const rows = [...players].filter(([id]) => !filters.player || id === filters.player).map(([id, title]) => {
      const cells = shown.map((task) => task.rows.find((row) => row.player === id))
      const flown = cells.filter((cell): cell is BoardRow => !!cell)
      const runs = flown.reduce((sum, cell) => sum + cell.seeds.length, 0)
      const times = flown.map((cell) => cell.median_seconds).filter((value): value is number => typeof value === 'number').sort((x, y) => x - y)
      return { id, title, cells, runs, complete: runs === shown.length * seeds,
        successes: flown.reduce((sum, cell) => sum + cell.successes, 0),
        mean: runs ? flown.reduce((sum, cell) => sum + cell.mean_score * cell.seeds.length, 0) / runs : 0,
        time: times.length ? times[Math.floor(times.length / 2)] : null }
    }).sort((x, y) => Number(y.complete) - Number(x.complete) || y.successes - x.successes || y.mean - x.mean || (x.time ?? 1e9) - (y.time ?? 1e9))
    let rank = 0
    nodes.push(rows.length ? table(['#', 'Model', ...shown.map((task) => taskTitle(task.env, task.task)), 'Runs', 'Successes', 'Mean score', 'Time per decision'],
      rows.map((row) => [row.complete ? String(++rank) : `Incomplete (${row.runs}/${shown.length * seeds})`, modelCell(row.id, row.title),
        ...row.cells.map((cell) => cell ? `${cell.successes}/${cell.seeds.length} · ${cell.mean_score}` : '—'),
        `${row.runs}/${shown.length * seeds}`, String(row.successes), row.mean.toFixed(1),
        row.time === null ? '—' : `${Math.round(row.time * 1000)} ms`]))
      : views.el('p', 'muted', 'No runs on the leaderboard seeds yet.'))
    return nodes
  }
  for (const task of shown) {
    let rank = 0
    const rows = task.rows.filter((row) => !filters.player || row.player === filters.player).map((row) => {
      const source = row.imported ? 'imported' : 'here'
      return [row.ranked ? String(++rank) : `Incomplete (${row.seeds.length}/${boards[0].seeds.length})`, modelCell(row.player, row.title, row.role),
        `${row.seeds.length}/${boards[0].seeds.length}`, String(row.successes),
        String(row.mean_score), row.median_seconds === null ? '—' : `${Math.round(row.median_seconds * 1000)} ms`, source,
        views.el('span', 'seed-runs', row.runs.map((run) => {
          const button = views.el('button', 'link', `${run.seed}:${run.score}`)
          button.title = `Replay seed ${run.seed}`
          button.addEventListener('click', () => openReplay({ run_id: run.run_id, created: 0, env: task.env, task: task.task,
            task_version: task.version, player: row.player, seed: run.seed, status: run.status, score: run.score }))
          return button
        }))]
    })
    nodes.push(views.el('h4', '', `${taskTitle(task.env, task.task)} · version ${task.version}`))
    nodes.push(rows.length ? table(['#', 'Model', 'Seeds', 'Successes', 'Mean score', 'Time per decision', 'Source', 'Score by seed'], rows)
      : views.el('p', 'muted', 'No runs on these seeds yet.'))
  }

  function modelCell(player: string, title: string, role = playerRole(player)) {
    const chip = views.el('span', 'role-chip', role ?? '')
    chip.hidden = !role
    return views.el('span', 'model-cell', [title, chip])
  }
  return nodes
}

function closePanel() {
  ui.panel.hidden = true
  ui.rail.hidden = false
  panelFocus?.focus()
  panelFocus = null
}

function table(head: string[], rows: (string | Node)[][]) {
  if (!rows.length) return views.el('p', 'muted', 'Nothing recorded yet.')
  return views.el('div', 'table-scroll', views.el('table', 'table', [views.el('tr', '', head.map((cell) => views.el('th', '', cell))),
    ...rows.map((cells) => views.el('tr', '', cells.map((cell) => views.el('td', '', cell))))]))
}

// ---- wiring

ui.modeWatch.addEventListener('click', () => setView('watch'))
ui.modePlay.addEventListener('click', () => setView('play'))
// another aircraft keeps the same objective
ui.env.addEventListener('change', () => {
  fillTasks()
  restart()
})
ui.aircraft.addEventListener('change', () => {
  fillTasks(sameObjective(ui.task.value, ui.aircraft.value))
  restart()
})
for (const field of [ui.task, ui.player, ui.seed]) field.addEventListener('change', restart)
ui.restart.addEventListener('click', restart)
ui.camera.addEventListener('click', toggleCamera)
ui.ghosts.addEventListener('click', toggleGhosts)
ui.ghosts.setAttribute('aria-pressed', String(ghostsShown))
flight.ghostsShown = minimap.ghostsShown = ghostsShown
ui.record.addEventListener('click', () => (recorder.state === 'recording' ? recorder.stop() : recorder.start()))
ui.replayPlay.addEventListener('click', () => {
  if (!replay) return
  if (replay.time >= replay.duration) replay.time = 0
  replay.playing = !replay.playing
  showReplay()
})
ui.replayTime.addEventListener('input', () => {
  if (!replay) return
  replay.time = Number(ui.replayTime.value)
  replay.playing = false
  showReplay()
})
ui.replayExit.addEventListener('click', exitReplay)
for (const button of document.querySelectorAll<HTMLButtonElement>('[data-open]')) {
  button.addEventListener('click', () => openPanel(button.dataset.open!))
}
$('panel-close').addEventListener('click', closePanel)
document.addEventListener('keydown', (event) => {
  if (event.key === 'Escape' && !ui.panel.hidden) closePanel()
})
ui.canvas.addEventListener('wheel', (event) => {
  flight.distance = Math.min(80, Math.max(8, flight.distance * (1 + Math.sign(event.deltaY) * 0.1)))
}, { passive: true })

installTips()
get<Catalog>('/api/catalog').then((data) => {
  catalog = data
  ui.runsButton.hidden = data.hosted
  fillEnvironments()
  fillQueue()
  connect()
  setView(view, true)
}).catch(toast)
requestAnimationFrame(frame)
