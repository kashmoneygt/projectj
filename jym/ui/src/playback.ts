interface Bracket<T> { a: T; b: T; mix: number }

const WINDOW = 4 // wall seconds of arrival history for the rate and the delay
const ARRIVALS = 512
const FRAMES = 1024
const TRIM = 0.1 // max speed correction, as a fraction of the rate
const GAIN = 0.5 // speed correction per wall second of timing error
const RESYNC = 2 // wall seconds behind the target before jumping forward, e.g. after a hidden tab

// plays samples back at their arrival rate, trailing the newest by an adaptive delay
export class PlaybackClock<T> {
  playhead: number | null = null
  // smoothed sim seconds per wall second
  rate = 1
  // wall seconds the playhead trails the arrival line
  delay = 0
  // true while the playhead holds the newest sample
  waiting = false
  private frames: { sim: number; value: T }[] = []
  private arrivals: { sim: number; wall: number }[] = []
  private measured: number | null = null
  private anchor = { sim: 0, wall: 0 }
  private wall: number | null = null
  private trim = 0

  get newest() { return this.frames[this.frames.length - 1] ?? null }
  get size() { return { frames: this.frames.length, arrivals: this.arrivals.length } }

  // keeps the rate estimate; everything else starts over
  reset() {
    this.frames = []
    this.arrivals = []
    this.measured = null
    this.playhead = null
    this.wall = null
    this.trim = 0
    this.delay = 0
    this.waiting = false
  }

  // wall is in milliseconds; a sample at the newest sim time replaces its value, and an older one starts over
  push(sim: number, value: T, wall: number) {
    const last = this.newest
    if (last && sim === last.sim) {
      last.value = value
      return
    }
    if (last && sim < last.sim) this.reset()
    this.frames.push({ sim, value })
    if (this.frames.length > FRAMES) {
      this.frames.shift()
      if (this.playhead !== null) this.playhead = Math.max(this.playhead, this.frames[0].sim)
    }
    const arrivals = this.arrivals
    arrivals.push({ sim, wall: wall / 1000 })
    while (arrivals.length > ARRIVALS || (arrivals.length > 2 && wall / 1000 - arrivals[0].wall > WINDOW)) arrivals.shift()
    this.measure()
  }

  private measure() {
    const list = this.arrivals
    const n = list.length
    const tN = list[n - 1].wall
    if (n >= 2 && tN - list[0].wall >= 0.25) {
      // least-squares slope, so one late arrival barely moves it
      let [mw, ms] = [0, 0]
      for (const a of list) { mw += a.wall - tN; ms += a.sim }
      mw /= n
      ms /= n
      let [num, den] = [0, 0]
      for (const a of list) { num += (a.wall - tN - mw) * (a.sim - ms); den += (a.wall - tN - mw) ** 2 }
      const slope = Math.min(20, Math.max(0.05, num / den))
      if (this.measured === null) this.rate = slope
      this.measured = slope
    }
    const r = this.measured ?? this.rate
    // the earliest arrivals define the line; later ones set how far the playhead trails
    let lead = -Infinity
    for (const a of list) lead = Math.max(lead, a.sim + r * (tN - a.wall))
    let need = 0
    for (let k = 0; k + 1 < n; k++) need = Math.max(need, list[k + 1].wall - tN - (list[k].sim - lead) / r)
    this.anchor = { sim: lead, wall: tN }
    this.delay = Math.min(3, need * 1.1 + 0.02)
  }

  // wall is in milliseconds and should be the frame timestamp
  advance(wall: number): Bracket<T> | null {
    const frames = this.frames
    if (!frames.length) return null
    const now = wall / 1000
    const dt = this.wall === null ? 0 : Math.min(0.25, Math.max(0, now - this.wall))
    this.wall = now
    if (this.measured !== null) this.rate += (this.measured - this.rate) * (1 - Math.exp(-dt))
    const newest = frames[frames.length - 1].sim
    const target = this.anchor.sim + this.rate * (now - this.anchor.wall - this.delay)
    if (this.playhead === null) {
      this.playhead = Math.max(frames[0].sim, target)
    } else {
      const behind = (target - this.playhead) / this.rate
      if (behind > RESYNC && target < newest) {
        this.playhead = target
      } else {
        // low-pass the correction so arrivals don't cause speed jumps
        this.trim += (Math.min(TRIM, Math.max(-TRIM, behind * GAIN)) - this.trim) * (1 - Math.exp(-dt / 0.5))
        this.playhead += dt * this.rate * (1 + this.trim)
      }
    }
    this.waiting = this.playhead >= newest
    this.playhead = Math.min(this.playhead, newest)
    let i = 0
    while (i + 1 < frames.length && frames[i + 1].sim <= this.playhead) i++
    if (i) frames.splice(0, i)
    const a = frames[0]
    const b = frames[1] ?? a
    const span = b.sim - a.sim
    return { a: a.value, b: b.value, mix: span > 0 ? (this.playhead - a.sim) / span : 0 }
  }
}

// use the vsync frame time (performance.now() includes earlier work in the frame)
export function frameTime() {
  const time = document.timeline?.currentTime
  return typeof time === 'number' ? time : performance.now()
}

interface Impact {
  sim_time: number
  position: { north_m: number; east_m: number; altitude_m: number }
  airspeed_kt: number
  reason: string
}

interface Impacting { sim_time: number; status?: { crashed?: boolean }; impact?: unknown }

// the recorded impact, once the blended sim time reaches it
export function impactAt(a: Impacting, b: Impacting, mix: number): Impact | null {
  const sim = a.sim_time + (b.sim_time - a.sim_time) * mix
  for (const scene of [a, b]) {
    const impact = scene.impact as Impact | undefined
    const p = impact?.position
    if (scene.status?.crashed && typeof impact?.sim_time === 'number' && typeof p?.north_m === 'number' &&
      typeof p.east_m === 'number' && typeof p.altitude_m === 'number' && sim >= impact.sim_time - 1e-9) return impact
  }
  return null
}

interface RingMark { north_m: number; east_m: number; heading_deg: number; passed?: boolean }
interface Ringed { sim_time: number; position: { north_m: number; east_m: number }; goal?: unknown }

function ringsOf(scene: Ringed): RingMark[] {
  const goal = scene.goal as { kind?: string; rings?: unknown } | undefined
  return goal?.kind === 'rings' && Array.isArray(goal.rings) ? goal.rings : []
}

// the evaluator passes rings in order, so the passed ones are a prefix
const passedCount = (rings: RingMark[]) => {
  let n = 0
  while (n < rings.length && rings[n].passed === true) n++
  return n
}

// rings passed at the blended sim time; a ring passed in b counts from the point the a-b path crosses it
export function ringsPassedAt(a: Ringed, b: Ringed, mix: number): number {
  const before = passedCount(ringsOf(a))
  const after = passedCount(ringsOf(b))
  if (after <= before) return before
  const ring = ringsOf(b)[before]
  const h = ring.heading_deg * Math.PI / 180
  const along = (s: Ringed) => (s.position.north_m - ring.north_m) * Math.cos(h) + (s.position.east_m - ring.east_m) * Math.sin(h)
  const [p, q] = [along(a), along(b)]
  const crossing = p < 0 && q >= 0 ? p / (p - q) : 1
  return mix >= crossing - 1e-9 ? after : before
}

const PASS_STEP = 2 // max forward sim step (s) still treated as continuous

// report rings passed during continuous playback; resets and seeks only resync
export class RingPasses {
  passed = 0
  // true when the last update was not continuous forward playback
  jumped = true
  private sim: number | null = null

  // index of the ring just passed, or null
  update(a: Ringed, b: Ringed, mix: number): number | null {
    const sim = a.sim_time + (b.sim_time - a.sim_time) * mix
    const passed = ringsPassedAt(a, b, mix)
    const continuous = this.sim !== null && sim >= this.sim && sim - this.sim <= PASS_STEP
    this.jumped = !continuous
    const fresh = continuous && passed > this.passed ? passed - 1 : null
    this.sim = sim
    this.passed = passed
    return fresh
  }

  reset() {
    this.sim = null
  }
}

// time since the shown impact appeared (keeps running if it's shown again)
export class ImpactClock {
  private key: string | null = null
  private start = 0

  // wall is in milliseconds
  age(impact: Impact | null, wall: number): number | null {
    if (!impact) {
      this.key = null
      return null
    }
    const { position: p } = impact
    const key = JSON.stringify([impact.sim_time, p.north_m, p.east_m, p.altitude_m, impact.reason])
    if (key !== this.key) {
      this.key = key
      this.start = wall
    }
    return (wall - this.start) / 1000
  }
}

// lower the pixel ratio while frames are late, and raise it once they're on time
export class Resolution {
  ratio: number
  readonly best: number // device pixel ratio, capped at 2
  private frames = 0
  private late = 0
  private calm = 0 // windows in a row with hardly a late frame
  private patience = 5 // calm windows before trying more pixels, doubled after each failed try
  private since = Infinity // windows since the last step up

  constructor(best: number) {
    this.best = best
    this.ratio = best
  }

  // takes a frame interval (ms); returns a new pixel ratio when it should change
  frame(interval: number): number | null {
    if (interval > 250) return null // a hidden tab, not a slow frame
    this.frames++
    if (interval > 25) this.late++
    if (this.frames < 120) return null
    const share = this.late / this.frames
    this.frames = this.late = 0
    this.since++
    if (share > 0.1 && this.ratio > 0.5) {
      if (this.since <= 3) this.patience = Math.min(60, this.patience * 2)
      this.calm = 0
      return (this.ratio = Math.max(0.5, this.ratio * 0.75))
    }
    this.calm = share < 0.01 ? this.calm + 1 : 0
    if (this.calm >= this.patience && this.ratio < this.best) {
      this.calm = this.since = 0
      return (this.ratio = Math.min(this.best, this.ratio / 0.75))
    }
    return null
  }
}
