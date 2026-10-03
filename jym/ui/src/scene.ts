import * as THREE from 'three'
import { airfield, skyDome, terrain } from './airfield'
import { Cessna } from './cessna'
import { Drone } from './drone'
import { Confetti, ImpactEffect } from './effects'
import { GhostFleet, type Ghost } from './ghosts'
import { DEG, GEAR_HEIGHT } from './materials'
import { type Ring } from './minimap'
import { ImpactClock, PlaybackClock, Resolution, RingPasses, frameTime, impactAt } from './playback'

export type AircraftId = 'cessna' | 'drone'

export interface FlightScene {
  position: { north_m: number; east_m: number; altitude_m: number; agl_m: number }
  attitude: { roll_deg: number; pitch_deg: number; heading_deg: number }
  airspeed_kt: number
  ground_speed_kt: number
  vertical_speed_fpm: number
  controls: Record<string, number>
  weather: { wind_direction_deg: number; wind_speed_kt: number }
  runway: { length_m: number; width_m: number; heading_deg: number; elevation_m: number }
  sim_time: number
  aircraft: AircraftId
  status: { phase: string; on_ground: boolean; crashed: boolean; reason: string | null }
  [key: string]: unknown
}

export type ViewMode = 'cockpit' | 'chase'

export class FlightView {
  static isScene(value: unknown): value is FlightScene {
    const scene = value as FlightScene | null
    return !!scene && typeof scene === 'object' && typeof scene.position?.north_m === 'number' &&
      typeof scene.attitude?.heading_deg === 'number' && typeof scene.runway?.length_m === 'number'
  }

  renderer: THREE.WebGLRenderer
  scene = new THREE.Scene()
  camera = new THREE.PerspectiveCamera(60, 1, 0.05, 60000)
  view: ViewMode = 'chase'
  distance = 16
  private cessna = new Cessna()
  private drone = new Drone()
  private aircraft: Cessna | Drone = this.cessna
  private impact = new ImpactClock()
  private wreck = new ImpactEffect()
  private passes = new RingPasses()
  private confetti = new Confetti()
  private ghosts = new GhostFleet()
  ghostsShown = true
  heat = false // heat planes always show; ghosts can be hidden
  private field: ReturnType<typeof airfield> | null = null
  private rings = new THREE.Group()
  private ringKey = ''
  private sunDirection = new THREE.Vector3().setFromSphericalCoords(1, (90 - 38) * DEG, 150 * DEG)
  private sky = skyDome(this.sunDirection)
  private sun = new THREE.DirectionalLight(0xfff4dc, 1.9)
  private clock = new PlaybackClock<FlightScene>()
  private fixed: { a: FlightScene; b: FlightScene; mix: number } | null = null
  private instruments: Record<string, unknown> | null = null
  private groundOffset = GEAR_HEIGHT
  private last = performance.now()
  private resolution = new Resolution(Math.min(window.devicePixelRatio, 2))
  private size = { width: 0, height: 0 }
  private chase = new THREE.Vector3()
  private chaseReady = false
  private scratch = {
    start: new THREE.Vector3(), end: new THREE.Vector3(), wanted: new THREE.Vector3(),
    startRotation: new THREE.Quaternion(), endRotation: new THREE.Quaternion(), euler: new THREE.Euler(),
    drift: new THREE.Vector3(),
  }

  constructor(canvas: HTMLCanvasElement) {
    this.renderer = new THREE.WebGLRenderer({
      canvas, antialias: true, logarithmicDepthBuffer: true,
    })
    this.renderer.setPixelRatio(this.resolution.ratio)
    this.renderer.toneMapping = THREE.NoToneMapping
    this.renderer.shadowMap.enabled = true
    this.renderer.shadowMap.type = THREE.PCFShadowMap
    this.scene.fog = new THREE.Fog(0xc9f3ff, 2500, 21000)
    this.scene.add(this.sky)
    this.scene.add(new THREE.HemisphereLight(0xe6fbff, 0xf0e2c4, 1.6))
    this.sun.castShadow = true
    this.sun.shadow.mapSize.set(2048, 2048)
    Object.assign(this.sun.shadow.camera, { left: -25, right: 25, top: 25, bottom: -25, near: 1, far: 400 })
    this.scene.add(this.sun, this.sun.target)
    this.scene.add(this.rings)
    terrain(this.scene)
    this.cessna.root.visible = this.drone.root.visible = false
    this.scene.add(this.cessna.root, this.drone.root, this.wreck.root, this.confetti.root, this.ghosts.root)
    this.camera.position.set(0, 40, 120)
    this.camera.lookAt(0, 0, 0)
  }

  setView(view: ViewMode) {
    this.view = view
    this.chaseReady = false
  }

  setGhosts(ghosts: Ghost[], elevation: number, solid: boolean, drone: boolean) {
    this.ghosts.set(ghosts, elevation, solid, drone)
  }

  // play live samples back at their arrival rate, blending only between recorded samples
  setScene(scene: FlightScene | null, instruments: Record<string, unknown> | null = null) {
    if (this.fixed) this.passes.reset()
    this.fixed = null
    if (!scene) {
      this.clock.reset()
      this.aircraft.root.visible = false
      this.rings.visible = false
      this.impact.age(null, 0)
      this.wreck.root.visible = false
      this.passes.reset()
      this.confetti.stop()
      return
    }
    const last = this.clock.newest?.value
    // a reset, seek or long gap starts a new buffer and snaps the camera
    if (!last || (scene.sim_time !== last.sim_time && (scene.sim_time < last.sim_time || scene.sim_time - last.sim_time > 5 ||
      distance(last, scene) > 30 + 90 * (scene.sim_time - last.sim_time)))) {
      this.clock.reset()
      this.passes.reset()
      this.chaseReady = false
    }
    this.clock.push(scene.sim_time, scene, performance.now())
    this.instruments = instruments
    this.prepare(scene)
  }

  // for replays, the caller picks the two recorded samples around the playback time
  show(a: FlightScene, b: FlightScene | null, mix: number, instruments: Record<string, unknown> | null) {
    const before = this.fixed?.a ?? this.clock.newest?.value
    if (!before || distance(before, a) > 60) this.chaseReady = false
    if (!this.fixed) this.passes.reset()
    this.clock.reset()
    this.fixed = { a, b: b ?? a, mix: b ? Math.min(1, Math.max(0, mix)) : 0 }
    this.instruments = instruments
    this.prepare(a)
  }

  private prepare(scene: FlightScene) {
    if (!this.field || this.field.key !== JSON.stringify(scene.runway)) {
      if (this.field) this.scene.remove(this.field.group)
      this.field = airfield(scene.runway)
      this.scene.add(this.field.group)
    }
    const goal = scene.goal as { kind?: string; next?: number; rings?: Ring[] } | undefined
    const rings = goal?.kind === 'rings' && Array.isArray(goal.rings) ? goal.rings : []
    const ringKey = JSON.stringify(rings.map(({ north_m, east_m, altitude_m, heading_deg, radius_m }) =>
      [north_m, east_m, altitude_m, heading_deg, radius_m]))
    if (ringKey !== this.ringKey) {
      this.ringKey = ringKey
      for (const child of this.rings.children) {
        const mesh = child as THREE.Mesh<THREE.TorusGeometry, THREE.MeshBasicMaterial>
        mesh.geometry.dispose()
        mesh.material.dispose()
      }
      this.rings.clear()
      for (const ring of rings) {
        const geometry = new THREE.TorusGeometry(ring.radius_m, Math.max(1, ring.radius_m * 0.035), 8, 64)
        const mesh = new THREE.Mesh(geometry, new THREE.MeshBasicMaterial({ color: 0xf2a516 }))
        mesh.position.set(ring.east_m, ring.altitude_m - scene.runway.elevation_m, -ring.north_m)
        mesh.rotation.y = Math.PI - ring.heading_deg * DEG
        this.rings.add(mesh)
      }
    }
    this.rings.visible = rings.length > 0
    if (scene.status.on_ground && scene.ground_speed_kt < 1) {
      this.groundOffset = scene.position.altitude_m - scene.runway.elevation_m
    }
  }

  resize(width: number, height: number) {
    this.size = { width, height }
    this.renderer.setSize(width, height, false)
    this.camera.aspect = width / height
    this.camera.updateProjectionMatrix()
  }

  render() {
    const now = frameTime()
    const dt = Math.min(0.1, Math.max(0, (now - this.last) / 1000))
    const ratio = this.resolution.frame(now - this.last)
    if (ratio !== null) {
      this.renderer.setPixelRatio(ratio)
      if (this.size.width) this.renderer.setSize(this.size.width, this.size.height, false)
    }
    this.last = now
    const target = this.pose(now)
    const age = this.impact.age(target?.impact ?? null, now)
    this.wreck.root.visible = age !== null
    const shownTime = target ? target.blend.a.sim_time + (target.blend.b.sim_time - target.blend.a.sim_time) * target.blend.mix : null
    this.ghosts.root.visible = (this.ghostsShown || this.heat) && !!target
    this.ghosts.update(shownTime, target?.scene.runway.elevation_m ?? 0, this.groundOffset, target?.position ?? null)
    const model = target?.scene.aircraft === 'drone' ? this.drone : this.cessna
    if (model !== this.aircraft) {
      this.aircraft.root.visible = false
      this.aircraft = model
    }
    const root = this.aircraft.root
    if (target) {
      root.visible = true
      root.position.copy(target.position)
      root.quaternion.copy(target.quaternion)
      this.cheer(target.blend, now)
      this.aircraft.body.position.y = GEAR_HEIGHT - this.groundOffset
      const scene = target.scene
      this.aircraft.update(scene.controls, this.instruments, dt, age !== null)
      this.aircraft.char(age === null ? 0 : Math.min(0.7, age * 1.2))
      if (age !== null && target.impact) {
        const { position: p } = target.impact
        const wind = scene.weather
        const downwind = (wind.wind_direction_deg + 180) * DEG
        const speed = Math.min(6, wind.wind_speed_kt * 0.514 * 0.6)
        this.wreck.root.position.set(p.east_m, 0, -p.north_m)
        this.wreck.update(age, Math.max(0.6, p.altitude_m - scene.runway.elevation_m), -scene.attitude.heading_deg * DEG,
          this.scratch.drift.set(Math.sin(downwind) * speed, 0, -Math.cos(downwind) * speed))
      }
      if (this.field) {
        const wind = scene.weather
        const sock = this.field.sock
        const downwind = (wind.wind_direction_deg + 180) * DEG
        sock.rotation.set(0, 0, 0)
        sock.rotateY(-downwind + scene.runway.heading_deg * DEG)
        const droop = Math.max(0, 1 - wind.wind_speed_kt / 15)
        sock.rotateX(-droop * 70 * DEG + Math.sin(now / 180) * 0.03 * (1 - droop))
      }
      const focus = root.position
      this.sun.position.copy(focus).addScaledVector(this.sunDirection, 150)
      this.sun.target.position.copy(focus)
    }
    // after an impact, use the chase camera (the cockpit may be underground)
    const cockpit = this.view === 'cockpit' && target && age === null
    this.aircraft.cockpit.visible = !!cockpit
    for (const part of this.aircraft.exterior) part.visible = !cockpit
    if (cockpit) {
      this.aircraft.disc.visible = false
      this.camera.fov = 72
      this.aircraft.eye.getWorldPosition(this.camera.position)
      this.camera.quaternion.copy(root.quaternion)
      this.camera.rotateX(-6 * DEG)
    } else if (target) {
      this.camera.fov = 58
      const size = this.aircraft === this.drone ? 0.4 : 1 // the chase keeps the same framing for a smaller craft
      const distance = this.distance * size
      const wanted = this.scratch.wanted.set(Math.sin(target.heading) * distance, (3.5 + distance * 0.12) * size,
        Math.cos(target.heading) * distance).add(target.position)
      wanted.y = Math.max(wanted.y, 1.5)
      if (!this.chaseReady) this.chase.copy(wanted)
      this.chase.lerp(wanted, 1 - Math.exp(-dt * 5))
      this.chaseReady = true
      this.camera.position.copy(this.chase)
      this.camera.lookAt(target.position.x, target.position.y + 0.8, target.position.z)
    }
    this.confetti.update(now, this.aircraft.root.position)
    this.camera.updateProjectionMatrix()
    this.sky.position.copy(this.camera.position)
    this.renderer.render(this.scene, this.camera)
  }

  // colour passed rings, with confetti for each new one
  private cheer({ a, b, mix }: { a: FlightScene; b: FlightScene; mix: number }, now: number) {
    const index = this.passes.update(a, b, mix)
    const passed = this.passes.passed
    const rings = this.rings.children as THREE.Mesh<THREE.TorusGeometry, THREE.MeshBasicMaterial>[]
    rings.forEach((ring, i) => ring.material.color.setHex(i < passed ? 0x2fa84f : i === passed ? 0xf2a516 : 0x1b74e4))
    // a seek, reset or switch between live and replay ends the burst
    if (this.passes.jumped) this.confetti.stop()
    if (index === null || !rings[index]) return
    this.confetti.fire(index, rings[index].quaternion, now)
  }

  private pose(now: number) {
    const blend = this.fixed ?? this.clock.advance(now)
    if (!blend) return null
    const { start, end, startRotation, endRotation, euler } = this.scratch
    const place = (scene: FlightScene, position: THREE.Vector3, quaternion: THREE.Quaternion) => {
      const { position: p, attitude: a } = scene
      position.set(p.east_m, p.altitude_m - scene.runway.elevation_m, -p.north_m)
      quaternion.setFromEuler(euler.set(a.pitch_deg * DEG, -a.heading_deg * DEG, -a.roll_deg * DEG, 'YXZ'))
      return -a.heading_deg * DEG
    }
    const { a, b, mix } = blend
    const from = place(a, start, startRotation)
    const to = place(b, end, endRotation)
    start.lerp(end, mix)
    startRotation.slerp(endRotation, mix)
    const delta = Math.atan2(Math.sin(to - from), Math.cos(to - from))
    return { position: start, quaternion: startRotation, heading: from + delta * mix, scene: mix < 0.5 ? a : b,
      impact: impactAt(a, b, mix), blend }
  }
}

function distance(a: FlightScene, b: FlightScene) {
  return Math.hypot(a.position.north_m - b.position.north_m, a.position.east_m - b.position.east_m,
    a.position.altitude_m - b.position.altitude_m)
}
