import { ImpactClock, PlaybackClock, Resolution, RingPasses, impactAt, ringsPassedAt } from './playback.ts'

interface Arrival { wall: number; sim: number; value?: string }
interface Frame { wall: number; playhead: number; newest: number; rate: number; waiting: boolean; mix: number }

const FRAME = 1000 / 60

function random(seed: number) {
  return () => {
    seed = (seed + 0x6d2b79f5) | 0
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

// arrival walls are in milliseconds; sim times are in seconds
function play(arrivals: Arrival[], until: number) {
  const clock = new PlaybackClock<string>()
  const sorted = [...arrivals].sort((x, y) => x.wall - y.wall)
  const frames: Frame[] = []
  let next = 0
  for (let wall = sorted[0].wall; wall <= until; wall += FRAME) {
    while (next < sorted.length && sorted[next].wall <= wall) {
      const arrival = sorted[next++]
      clock.push(arrival.sim, arrival.value ?? `s${arrival.sim}`, arrival.wall)
    }
    const bracket = clock.advance(wall)
    if (!bracket) continue
    frames.push({ wall, playhead: clock.playhead!, newest: clock.newest!.sim, rate: clock.rate, waiting: clock.waiting, mix: bracket.mix })
  }
  return { frames, clock }
}

// per-frame speed is playhead seconds per wall second divided by the true stream rate
function motion(frames: Frame[], rate: number, from: number, to: number) {
  const speeds: number[] = []
  let backwards = 0
  let ahead = 0
  let held = 0
  let step = 0
  for (let i = 1; i < frames.length; i++) {
    const [p, f] = [frames[i - 1], frames[i]]
    if (f.playhead < p.playhead - 1e-9) backwards++
    if (f.playhead > f.newest + 1e-9) ahead++
    if (f.wall < from || f.wall > to) continue
    const speed = (f.playhead - p.playhead) / ((f.wall - p.wall) / 1000) / rate
    if (speed === 0) held++
    speeds.push(speed)
    step = Math.max(step, f.playhead - p.playhead)
  }
  const jerk = speeds.slice(1).reduce((m, s, i) => Math.max(m, Math.abs(s - speeds[i])), 0)
  const deviation = speeds.reduce((m, s) => Math.max(m, Math.abs(s - 1)), 0)
  const inRange = frames.filter((f) => f.wall >= from && f.wall <= to)
  const lag = inRange.map((f) => f.newest - f.playhead)
  return {
    backwards, ahead, held, frames: speeds.length, deviation: round(deviation), jerk: round(jerk), step: round(step),
    lagStart: round(lag[0]), lagEnd: round(lag[lag.length - 1]), lagMax: round(Math.max(...lag)),
  }
}

const round = (x: number) => Math.round(x * 1e4) / 1e4

function stream(seconds: number, hz: number, simPerArrival: number, jitterMs: (i: number) => number) {
  const out: Arrival[] = []
  for (let i = 0; i * (1 / hz) <= seconds; i++) out.push({ wall: 1000 + (i * 1000) / hz + jitterMs(i), sim: round(i * simPerArrival) })
  return out
}

const failures: string[] = []
const results: Record<string, unknown> = {}
function expect(name: string, ok: boolean, detail: unknown) {
  results[name] = detail
  if (!ok) failures.push(`${name}: ${JSON.stringify(detail)}`)
}

{
  const arrivals = stream(30, 5, 0.2, () => 0)
  const { frames, clock } = play(arrivals, 31000)
  const m = motion(frames, 1, 4000, 31000)
  expect('steady 5 Hz for 30 s', m.backwards === 0 && m.ahead === 0 && m.held === 0 && m.deviation < 0.02 &&
    m.jerk < 0.01 && m.lagEnd <= m.lagStart + 0.01 && clock.size.frames < 10 && clock.size.arrivals <= 21,
  { ...m, size: clock.size, delay: round(clock.delay) })
}

{
  const next = random(7)
  const arrivals = stream(30, 5, 0.2, () => 70 + 80 * next())
  const { frames } = play(arrivals, 31000)
  const m = motion(frames, 1, 5000, 31000)
  expect('5 Hz with 70-150 ms latency jitter', m.backwards === 0 && m.ahead === 0 && m.held / m.frames < 0.01 &&
    m.deviation <= 0.11 && m.jerk < 0.06 && m.step < (1.12 * FRAME) / 1000 && m.lagMax < 0.6, m)
}

{
  // a model taking about 1 s per decision, each advancing 0.2 s of sim time
  const next = random(3)
  const arrivals = stream(30, 1, 0.2, () => 200 * (next() - 0.5))
  const { frames, clock } = play(arrivals, 31000)
  const m = motion(frames, 0.2, 8000, 31000)
  expect('slow inference 0.2 sim s per second', m.backwards === 0 && m.ahead === 0 && Math.abs(clock.rate - 0.2) < 0.02 &&
    m.held / m.frames < 0.05 && m.step < (0.2 * 1.12 * FRAME) / 1000, { ...m, rate: round(clock.rate) })
}

{
  // a person's controls republish the same sim time with new values between physics ticks
  const next = random(11)
  const base = stream(20, 5, 0.2, () => 70 + 80 * next())
  const duplicates = base.flatMap((a, i) => [1, 2, 3].map((k) => ({ wall: a.wall + k * 37 + (i % 5), sim: a.sim, value: `update${i}.${k}` })))
  const plain = play(base, 21500)
  const updated = play([...base, ...duplicates], 21500)
  const same = plain.frames.length === updated.frames.length &&
    plain.frames.every((f, i) => f.playhead === updated.frames[i].playhead && f.rate === updated.frames[i].rate)
  const m = motion(updated.frames, 1, 5000, 21000)
  expect('duplicate same-time updates', same && m.backwards === 0 && m.ahead === 0 &&
    updated.clock.newest!.value === `update${base.length - 1}.3`, { same, ...m })
}

{
  const next = random(5)
  const arrivals = stream(20, 100, 0.01, () => 6 * (next() - 0.5))
  const { frames } = play(arrivals, 21000)
  const m = motion(frames, 1, 3000, 21000)
  expect('native 100 Hz', m.backwards === 0 && m.ahead === 0 && m.held === 0 && m.deviation < 0.03 && m.lagMax < 0.1, m)
}

{
  const next = random(9)
  const arrivals = stream(10, 5, 0.2, () => 70 + 80 * next())
  const last = arrivals[arrivals.length - 1]
  const { frames } = play(arrivals, last.wall + 1500)
  const end = frames[frames.length - 1]
  const reached = frames.find((f) => f.wall > last.wall && f.playhead === f.newest)
  expect('stream stop flushes to the newest sample', end.playhead === last.sim && end.waiting && !!reached &&
    reached.wall - last.wall < 1000, { settleMs: reached && Math.round(reached.wall - last.wall), end: end.playhead })
}

{
  const { clock } = play(stream(5, 5, 0.2, () => 0), 6000)
  clock.reset()
  clock.push(0, 'initial', 6100)
  const bracket = clock.advance(6110)
  expect('reset shows the initial sample', clock.playhead === 0 && bracket?.a === 'initial' && bracket.b === 'initial' &&
    bracket.mix === 0 && clock.size.frames === 1 && clock.size.arrivals === 1, { playhead: clock.playhead, bracket })
}

{
  const impact = (sim: number) => ({ sim_time: sim, position: { north_m: 40, east_m: -3, altitude_m: 101 }, airspeed_kt: 70, reason: 'terrain' })
  const flying = { sim_time: 10, status: { crashed: false } }
  const crashed = { sim_time: 10.2, status: { crashed: true }, impact: impact(10.2) }
  const midway = { sim_time: 10.2, status: { crashed: true }, impact: impact(10.08) }
  const old = { sim_time: 10.2, status: { crashed: true } }
  const unflagged = { sim_time: 10.2, status: { crashed: false }, impact: impact(10.2) }
  const selected = {
    beforeB: impactAt(flying, crashed, 0.5), atB: impactAt(flying, crashed, 1), heldA: impactAt(crashed, crashed, 0),
    beforeMidway: impactAt(flying, midway, 0.3), afterMidway: impactAt(flying, midway, 0.5),
    oldCrash: impactAt(flying, old, 1), heldOld: impactAt(old, old, 0), unflagged: impactAt(unflagged, unflagged, 0),
  }
  expect('impact only from recorded impact data once reached', !selected.beforeB && !!selected.atB && !!selected.heldA &&
    !selected.beforeMidway && !!selected.afterMidway && !selected.oldCrash && !selected.heldOld && !selected.unflagged,
  Object.fromEntries(Object.entries(selected).map(([k, v]) => [k, !!v])))

  const clock = new ImpactClock()
  const ages = {
    first: clock.age(impact(10.2), 5000),
    // repeated samples, replay redraws and frozen episodes advance in wall time
    repeated: clock.age({ ...impact(10.2) }, 5500),
    frozen: clock.age(impact(10.2), 12000),
    seekBefore: clock.age(null, 12100),
    again: clock.age(impact(10.2), 13000),
    later: clock.age(impact(10.2), 13250),
    other: clock.age(impact(33), 14000),
  }
  expect('impact age runs on wall time and restarts only for a new impact', ages.first === 0 && ages.repeated === 0.5 &&
    ages.frozen === 7 && ages.seekBefore === null && ages.again === 0 && ages.later === 0.25 && ages.other === 0, ages)
}

{
  const marks = [{ north_m: 800, east_m: 0, heading_deg: 0 }, { north_m: 1640, east_m: -650, heading_deg: 282 }]
  const at = (sim: number, north: number, passed: number) => ({ sim_time: sim, position: { north_m: north, east_m: 0 },
    goal: { kind: 'rings', next: passed, rings: marks.map((m, i) => ({ ...m, passed: i < passed })) } })
  // samples every 0.2 s at 40 m/s; ring 1 is crossed 3/4 of the way from s1 to s2
  const [s0, s1, s2, s3] = [at(10, 740, 0), at(10.2, 770, 0), at(10.4, 810, 1), at(10.6, 850, 1)]
  const passes = new RingPasses()
  const fired: (number | null)[] = []
  for (const [a, b, mix] of [[s0, s1, 0], [s0, s1, 0.5], [s1, s2, 0], [s1, s2, 0.7], [s1, s2, 0.75], [s1, s2, 0.9],
    [s2, s3, 0], [s2, s3, 0], [s3, s3, 0]] as const) fired.push(passes.update(a, b, mix))
  const attach = new RingPasses()
  const firstAttach = attach.update(s3, s3, 0)
  const seekBack = attach.update(s0, s1, 0)
  const replayed = [attach.update(s1, s2, 0.5), attach.update(s1, s2, 0.8)]
  const jump = new RingPasses()
  jump.update(at(1, 0, 0), at(1.2, 8, 0), 0)
  const seekForward = jump.update(s2, s3, 0)
  const resetOne = new RingPasses()
  resetOne.update(s1, s2, 0.5)
  resetOne.reset()
  const afterReset = resetOne.update(s1, s2, 0.9)
  const plain = { sim_time: 3, position: { north_m: 0, east_m: 0 } }
  // a burst ends on any jump, so only continuous frames may leave jumped false
  const jumps = { playing: passes.jumped, attach: new RingPasses().jumped, seekForward: jump.jumped, reset: resetOne.jumped }
  const detail = { fired, firstAttach, seekBack, replayed, seekForward, afterReset, jumps,
    missingCrossing: ringsPassedAt(at(5, 900, 0), at(5.2, 950, 1), 0.99), noRings: ringsPassedAt(plain, plain, 0) }
  expect('ring pass fires once at the interpolated crossing, never on attach, reset or seek',
    JSON.stringify(fired) === JSON.stringify([null, null, null, null, 0, null, null, null, null]) && firstAttach === null &&
    seekBack === null && replayed[0] === null && replayed[1] === 0 && seekForward === null && afterReset === null &&
    detail.missingCrossing === 0 && detail.noRings === 0 && !jumps.playing && jumps.attach && jumps.seekForward && jumps.reset, detail)
}

{
  // lower the resolution on late frames and raise it after a calm spell; each failed step up waits longer
  const resolution = new Resolution(1)
  const run = (ms: number, frames: number) => {
    let changed: number | null = null
    for (let i = 0; i < frames; i++) changed = resolution.frame(ms) ?? changed
    return changed
  }
  const lower = run(33, 120), hidden = run(1000, 240), early = run(16.7, 120 * 4), back = run(16.7, 120)
  run(33, 120) // too many pixels again right away
  const waited = run(16.7, 120 * 9), later = run(16.7, 120)
  const floor = new Resolution(1)
  for (let i = 0; i < 120 * 10; i++) floor.frame(40)
  const detail = { lower, hidden, early, back, waited, later, floor: floor.ratio }
  expect('late frames lower the resolution and on-time ones restore it, patiently', lower === 0.75 && hidden === null &&
    early === null && back === 1 && waited === null && later === 1 && floor.ratio === 0.5, detail)
}

console.log(JSON.stringify(results, null, 1))
if (failures.length) throw new Error(`${failures.length} failed:\n${failures.join('\n')}`)
console.log('all playback checks passed')
