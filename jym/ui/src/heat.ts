import type { Decision } from './decision'
import type { Ghost } from './ghosts'
import type { FlightScene } from './scene'

// a real-time race of several players on one objective and seed (see src/jym/live.py)
export interface HeatPlayer {
  player: string; title: string; state: string; run_id?: string; status?: string | null; score?: number | null
  decisions?: number; last_seconds?: number | null; reason?: string | null; role?: string | null
}
export interface HeatInfo {
  id: string; env: string; task: string; seed: number; started: number | null; ended: number | null; players: HeatPlayer[]
}
export interface LiveInfo {
  heat: HeatInfo | null; next: HeatInfo | null; roster: string[]; now: number
  loading: string[] // models still loading before the first heat
}

// one player's run, from the live stream or a recording
export class Track {
  readonly scenes: FlightScene[] = []
  readonly decisions: Decision[] = []
  readonly ghost: Ghost
  loaded = false // a recorded run has been requested

  constructor(player: HeatPlayer) {
    this.ghost = { run_id: player.run_id ?? player.player, player: player.player, title: player.title,
      status: player.status ?? 'running', score: player.score ?? 0, trajectory: [] }
  }

  push(scene: FlightScene) {
    const last = this.scenes.at(-1)
    if (last && scene.sim_time <= last.sim_time) return
    this.scenes.push(scene)
    const { position: p, attitude: a } = scene
    this.ghost.trajectory.push([scene.sim_time, p.north_m, p.east_m, p.altitude_m, a.roll_deg, a.pitch_deg, a.heading_deg])
  }

  // place decisions at the sim time they applied (the stream resends each with its step)
  add(events: Record<string, unknown>[]) {
    for (const event of events) {
      if (event.scene) this.push(event.scene as FlightScene)
      if (event.type === 'decision') {
        const decision = event as unknown as Decision
        const last = this.decisions.at(-1)
        if (last && last.step >= decision.step) continue
        decision.clock = decision.applied_at ?? decision.asked_at ?? this.scenes.at(-1)?.sim_time ?? 0
        this.decisions.push(decision)
      } else if (event.type === 'step') {
        const last = this.decisions.at(-1)
        if (last && last.step === event.step) last.action = event.action as Record<string, unknown>
      }
    }
  }

  // the two scenes around sim time t and how far between them
  at(t: number): { a: FlightScene; b: FlightScene; mix: number } | null {
    const scenes = this.scenes
    if (!scenes.length) return null
    if (t <= scenes[0].sim_time) return { a: scenes[0], b: scenes[0], mix: 0 }
    let low = 0, high = scenes.length - 1
    while (low < high) {
      const mid = (low + high + 1) >> 1
      if (scenes[mid].sim_time <= t) low = mid
      else high = mid - 1
    }
    const a = scenes[low], b = scenes[Math.min(low + 1, scenes.length - 1)]
    return { a, b, mix: b.sim_time > a.sim_time ? Math.min(1, (t - a.sim_time) / (b.sim_time - a.sim_time)) : 0 }
  }

  get latest() { return this.scenes.at(-1)?.sim_time ?? 0 }
}
