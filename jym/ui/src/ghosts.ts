import * as THREE from 'three'
import { DEG, GEAR_HEIGHT, canvasTexture } from './materials'

// other players' recorded flights of the same objective and seed
export interface Ghost {
  run_id: string; player: string; title: string | null; status: string; score: number
  trajectory: number[][] // [t, north_m, east_m, altitude_m, roll_deg, pitch_deg, heading_deg]
}
export interface GhostPose { north: number; east: number; altitude: number; roll: number; pitch: number; heading: number }

const KNOWN_GHOSTS: Record<string, string> = { reference: '#6b7280', human: '#d6338c' }
const GHOST_PALETTE = ['#0ea5a4', '#7c4dff', '#ff8c1a', '#d6338c', '#2f9e44', '#1b74e4']

export function ghostColor(player: string) {
  if (KNOWN_GHOSTS[player]) return KNOWN_GHOSTS[player]
  let hash = 0
  for (const c of player) hash = (hash * 31 + c.charCodeAt(0)) >>> 0
  return GHOST_PALETTE[hash % GHOST_PALETTE.length]
}

export const ghostName = (ghost: Ghost) => ghost.title ?? ghost.player

// pose at sim time t, or null before the first sample or a few seconds after the last
export function ghostPose(ghost: Ghost, t: number): GhostPose | null {
  const rows = ghost.trajectory
  if (!rows.length || t < rows[0][0] || t > rows[rows.length - 1][0] + 3) return null
  let low = 0, high = rows.length - 1
  while (low < high) {
    const mid = (low + high + 1) >> 1
    if (rows[mid][0] <= t) low = mid
    else high = mid - 1
  }
  const a = rows[low], b = rows[Math.min(low + 1, rows.length - 1)]
  const mix = b[0] > a[0] ? Math.min(1, Math.max(0, (t - a[0]) / (b[0] - a[0]))) : 0
  const lerp = (i: number) => a[i] + (b[i] - a[i]) * mix
  const turn = ((b[6] - a[6] + 540) % 360) - 180
  return { north: lerp(1), east: lerp(2), altitude: lerp(3), roll: lerp(4), pitch: lerp(5), heading: a[6] + turn * mix }
}

function ghostLabel(text: string, colour: string) {
  const texture = canvasTexture(256, 64, (g) => {
    g.font = `600 30px "IBM Plex Sans", sans-serif`
    g.textAlign = 'center'
    g.textBaseline = 'middle'
    g.lineWidth = 8
    g.strokeStyle = 'rgba(255, 255, 255, 0.9)'
    g.strokeText(text, 128, 32, 240)
    g.fillStyle = colour
    g.fillText(text, 128, 32, 240)
  })
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: texture, transparent: true, depthWrite: false, sizeAttenuation: false }))
  sprite.scale.set(0.16, 0.04, 1)
  sprite.position.y = 4
  return sprite
}

// a translucent aircraft, trail and name tag per ghost
export class GhostFleet {
  root = new THREE.Group()
  private planes: { ghost: Ghost; group: THREE.Group; body: THREE.Group; trail: THREE.Line; written: number }[] = []
  private key = ''
  private elevation = 0
  private euler = new THREE.Euler()

  // solid planes race in the heat, translucent ones are ghosts; trajectories may keep growing
  set(ghosts: Ghost[], elevation: number, solid: boolean, drone: boolean) {
    const key = JSON.stringify(ghosts.map((ghost) => ghost.run_id)) + elevation + solid + drone
    if (key === this.key) return
    this.key = key
    this.elevation = elevation
    for (const plane of this.planes) {
      plane.group.traverse((item) => {
        if (item instanceof THREE.Mesh || item instanceof THREE.Line || item instanceof THREE.Sprite) {
          item.geometry.dispose()
          const material = item.material as THREE.Material & { map?: THREE.Texture | null }
          material.map?.dispose()
          material.dispose()
        }
      })
      plane.trail.geometry.dispose()
      ;(plane.trail.material as THREE.Material).dispose()
      this.root.remove(plane.group, plane.trail)
    }
    this.planes = ghosts.map((ghost) => {
      const colour = ghostColor(ghost.player)
      const skin = new THREE.MeshBasicMaterial({ color: colour, transparent: !solid, opacity: solid ? 1 : 0.45, depthWrite: solid })
      const part = (w: number, h: number, d: number, x: number, y: number, z: number) => {
        const item = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), skin)
        item.position.set(x, y, z)
        return item
      }
      const body = new THREE.Group()
      if (drone) {
        body.add(part(0.9, 0.45, 1.2, 0, 0, 0), part(3.4, 0.1, 0.12, 0, 0.12, 0).rotateY(Math.PI / 4),
          part(3.4, 0.1, 0.12, 0, 0.12, 0).rotateY(-Math.PI / 4))
        for (const [x, z] of [[-1.2, -1.2], [1.2, -1.2], [-1.2, 1.2], [1.2, 1.2]]) body.add(part(1.2, 0.04, 1.2, x, 0.36, z))
      } else {
        body.add(part(1.1, 1.1, 7.6, 0, 0, 1.4), part(11, 0.14, 1.45, 0, 0.82, -0.5), part(3.4, 0.08, 0.75, 0, 0.2, 4.6),
          part(0.08, 1.35, 0.9, 0, 0.9, 4.6))
      }
      const group = new THREE.Group()
      group.add(body, ghostLabel(`${ghostName(ghost)} · ${ghost.score}`, colour))
      const geometry = new THREE.BufferGeometry()
      geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(Math.max(2048, ghost.trajectory.length) * 3), 3))
      const trail = new THREE.Line(geometry, new THREE.LineBasicMaterial({ color: colour, transparent: true, opacity: 0.55 }))
      trail.frustumCulled = false
      this.root.add(group, trail)
      return { ghost, group, body, trail, written: 0 }
    })
  }

  update(t: number | null, elevation: number, groundOffset: number, followed: THREE.Vector3 | null) {
    for (const plane of this.planes) {
      const rows = plane.ghost.trajectory
      if (plane.written < rows.length) this.extend(plane, rows)
      const pose = t === null ? null : ghostPose(plane.ghost, t)
      plane.group.visible = plane.trail.visible = !!pose
      if (!pose) continue
      plane.group.position.set(pose.east, pose.altitude - elevation, -pose.north)
      // hide planes on top of the followed one (heats start everyone on the same spot)
      if (followed && plane.group.position.distanceTo(followed) < 12) plane.group.visible = false
      plane.body.quaternion.setFromEuler(this.euler.set(pose.pitch * DEG, -pose.heading * DEG, -pose.roll * DEG, 'YXZ'))
      plane.body.position.y = GEAR_HEIGHT - groundOffset
      let shown = 0
      while (shown < rows.length && rows[shown][0] <= t!) shown++
      plane.trail.geometry.setDrawRange(0, shown)
    }
  }

  // append new trajectory rows to the trail, growing its buffer when full
  private extend(plane: GhostFleet['planes'][number], rows: number[][]) {
    let attribute = plane.trail.geometry.getAttribute('position') as THREE.BufferAttribute
    if (rows.length > attribute.count) {
      const grown = new THREE.BufferAttribute(new Float32Array(rows.length * 2 * 3), 3)
      grown.array.set((attribute.array as Float32Array).subarray(0, plane.written * 3))
      plane.trail.geometry.setAttribute('position', grown)
      attribute = grown
    }
    for (let i = plane.written; i < rows.length; i++) attribute.setXYZ(i, rows[i][2], rows[i][3] - this.elevation, -rows[i][1])
    attribute.needsUpdate = true
    plane.written = rows.length
  }
}
