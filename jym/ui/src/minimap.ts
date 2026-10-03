import { DEG } from './materials'
import { ghostColor, ghostPose, type Ghost } from './ghosts'
import type { FlightScene } from './scene'

export interface Waypoint { name: string; north_m: number; east_m: number; radius_m?: number; heading_deg?: number }
export interface Ring extends Waypoint { altitude_m: number; heading_deg: number; radius_m: number; passed?: boolean }
// north-up map: centre in metres from the runway, scale in CSS px per metre
interface MapFrame { north: number; east: number; scale: number; width: number; height: number }

function mapFrame(runway: FlightScene['runway'], waypoints: Waypoint[], width: number, height: number): MapFrame {
  const h = runway.heading_deg * DEG
  const half = runway.length_m / 2
  const points: [number, number, number][] = [[Math.cos(h) * half, Math.sin(h) * half, 0], [-Math.cos(h) * half, -Math.sin(h) * half, 0],
    ...waypoints.map((w): [number, number, number] => [w.north_m, w.east_m, w.radius_m ?? 0])]
  if (!waypoints.length) points.push([2000, 2000, 0], [-2000, -2000, 0])
  const north = points.map(([n, , r]) => [n - r, n + r]).flat()
  const east = points.map(([, e, r]) => [e - r, e + r]).flat()
  const [n0, n1, e0, e1] = [Math.min(...north), Math.max(...north), Math.min(...east), Math.max(...east)]
  const scale = Math.min(width / ((e1 - e0) * 1.18), height / ((n1 - n0) * 1.18))
  return { north: (n0 + n1) / 2, east: (e0 + e1) / 2, scale, width, height }
}

function worldToMap(frame: MapFrame, north: number, east: number): [number, number] {
  return [frame.width / 2 + (east - frame.east) * frame.scale, frame.height / 2 - (north - frame.north) * frame.scale]
}

// same colours as the 3D rings
const MAP = {
  bg: '#eef3e2', ink: '#1f2430', dim: '#5f6673', route: '#a1a896', runway: '#5d5a55', red: '#e8402a',
  done: '#2fa84f', next: '#f2a516', ahead: '#1b74e4', trail: 'rgba(31, 36, 48, 0.6)',
}
const FONT = '"IBM Plex Sans", sans-serif'
const SCALES = [100, 200, 500, 1000, 2000, 5000]

// objectives reached, and the current one
function progress(scene: FlightScene, waypoints: Waypoint[]) {
  const goal = scene.goal as { kind?: string; next?: unknown } | undefined
  const next = goal?.kind === 'rings' ? goal.next : scene.next_waypoint
  const done = typeof next === 'number' ? next : null
  return { done, active: done !== null && done < waypoints.length ? done : null }
}

export class Minimap {
  ghosts: Ghost[] = []
  ghostsShown = true
  // north, east and sim time of each trail point
  private trail: [number, number, number][] = []
  private key = ''
  private last: [FlightScene, Waypoint[]] | null = null
  private readonly canvas: HTMLCanvasElement

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas
    // redraw the last scene on resize (paused or ended runs send no new ones)
    new ResizeObserver(() => this.last && this.draw(...this.last)).observe(canvas)
  }

  clear() {
    this.trail = []
    this.key = ''
    this.last = null
  }

  // draw the position, heading and trail
  draw(scene: FlightScene, waypoints: Waypoint[]) {
    const width = this.canvas.clientWidth
    const height = this.canvas.clientHeight
    if (!width || !height) return
    this.last = [scene, waypoints]
    // restart the trail only for new geometry, not when rings are passed
    const key = JSON.stringify([scene.runway, waypoints.map((w) => [w.name, w.north_m, w.east_m, w.radius_m, w.heading_deg])])
    if (key !== this.key) {
      this.key = key
      this.trail = []
    }
    const frame = mapFrame(scene.runway, waypoints, width, height)
    const ratio = Math.min(window.devicePixelRatio, 2)
    const [pixelsX, pixelsY] = [Math.round(width * ratio), Math.round(height * ratio)]
    if (this.canvas.width !== pixelsX || this.canvas.height !== pixelsY) {
      this.canvas.width = pixelsX
      this.canvas.height = pixelsY
    }
    const at = (n: number, e: number) => worldToMap(frame, n, e)
    const { north_m: n, east_m: e } = scene.position
    const sim = scene.sim_time
    while (this.trail.length && this.trail[this.trail.length - 1][2] >= sim) this.trail.pop()
    const last = this.trail[this.trail.length - 1]
    if (!last || Math.hypot(last[0] - n, last[1] - e) > 4 / frame.scale) this.trail.push([n, e, sim])
    if (this.trail.length > 2000) this.trail.splice(0, this.trail.length - 2000)
    const g = this.canvas.getContext('2d')!
    g.setTransform(pixelsX / width, 0, 0, pixelsY / height, 0, 0)
    g.fillStyle = MAP.bg
    g.fillRect(0, 0, width, height)
    g.lineCap = g.lineJoin = 'round'

    const h = scene.runway.heading_deg * DEG
    const half = scene.runway.length_m / 2
    const rings = waypoints.some((w) => typeof w.heading_deg === 'number')
    const { done, active } = progress(scene, waypoints)
    const colour = (i: number) => done === null ? MAP.dim : i < done ? MAP.done : i === active ? MAP.next : MAP.ahead
    if (waypoints.length) {
      // route from the far runway end through the objectives (back to the threshold on the circuit)
      const route = [[Math.cos(h) * half, Math.sin(h) * half], ...waypoints.map((w) => [w.north_m, w.east_m])]
      if (!rings) route.push([-Math.cos(h) * half, -Math.sin(h) * half])
      g.setLineDash([3, 3])
      g.strokeStyle = MAP.route
      g.lineWidth = 1
      g.beginPath()
      for (const [i, [rn, re]] of route.entries()) g[i ? 'lineTo' : 'moveTo'](...at(rn, re))
      g.stroke()
      g.setLineDash([])
    }
    const [rx, ry] = at(0, 0)
    g.save()
    g.translate(rx, ry)
    g.rotate(h)
    g.fillStyle = MAP.runway
    const runwayWidth = Math.max(3, scene.runway.width_m * frame.scale)
    const runwayLength = Math.max(8, scene.runway.length_m * frame.scale)
    g.fillRect(-runwayWidth / 2, -runwayLength / 2, runwayWidth, runwayLength)
    g.restore()

    // objective boxes, to keep labels clear of them
    const boxes = waypoints.map((w, i): [number, number, number, number] => {
      const [x, y] = at(w.north_m, w.east_m)
      const radius = (w.radius_m ?? 0) * frame.scale
      g.strokeStyle = g.fillStyle = colour(i)
      if (typeof w.heading_deg !== 'number') {
        const r = Math.max(4, radius)
        g.beginPath()
        g.arc(x, y, r, 0, 2 * Math.PI)
        if (i === active) {
          g.globalAlpha = 0.18
          g.fill()
          g.globalAlpha = 1
        }
        g.lineWidth = i === active ? 2 : 1.5
        g.stroke()
        return [x - r, y - r, x + r, y + r]
      }
      // a ring from above: a bar with a chevron pointing through it
      const d = w.heading_deg * DEG
      const [ax, ay, px, py] = [Math.sin(d), -Math.cos(d), Math.cos(d), Math.sin(d)]
      const l = Math.max(6, radius)
      g.lineWidth = i === active ? 3.5 : 2.5
      g.beginPath()
      g.moveTo(x - px * l, y - py * l)
      g.lineTo(x + px * l, y + py * l)
      g.stroke()
      g.lineWidth = 1.5
      g.beginPath()
      g.moveTo(x + ax * 2 - px * 3.5, y + ay * 2 - py * 3.5)
      g.lineTo(x + ax * 6, y + ay * 6)
      g.lineTo(x + ax * 2 + px * 3.5, y + ay * 2 + py * 3.5)
      g.stroke()
      const [rx, ry] = [Math.max(6, l * Math.abs(px)), Math.max(6, l * Math.abs(py))]
      return [x - rx, y - ry, x + rx, y + ry]
    })
    const metres = SCALES.findLast((m) => m * frame.scale <= width * 0.3) ?? SCALES[0]
    const bar = metres * frame.scale
    const scaleLabel = metres >= 1000 ? `${metres / 1000} km` : `${metres} m`
    g.font = `500 10px ${FONT}`
    const scaleWidth = Math.max(bar, g.measureText(scaleLabel).width)
    const crowded = (x0: number, x1: number) => boxes.some(([a, , c, d]) => a < x1 && c > x0 && d > height - 28)
    const scaleX = crowded(4, 12 + scaleWidth) && !crowded(width - 12 - scaleWidth, width - 4) ? width - 8 - scaleWidth : 8
    const taken = [...boxes, [width - 26, 2, width - 2, 32], [scaleX - 2, height - 26, scaleX + scaleWidth + 2, height - 4]]
    g.textBaseline = 'middle'
    g.textAlign = 'left'
    for (const [i, w] of waypoints.entries()) {
      const [x, y] = at(w.north_m, w.east_m)
      g.font = `${i === active ? 600 : 500} 10px ${FONT}`
      const textWidth = g.measureText(w.name).width
      const right = boxes[i][2] + 4
      const left = boxes[i][0] - 4 - textWidth
      const prefer = x > width / 2 ? left : right
      const other = prefer === left ? right : left
      const row = Math.min(height - 8, Math.max(8, y))
      const centred = x - textWidth / 2
      const spots = [[prefer, row], [other, row], [centred, boxes[i][3] + 7], [centred, boxes[i][1] - 7],
        [prefer, row - 12], [prefer, row + 12], [other, row - 12], [other, row + 12]]
      const clear = ([tx, ty]: number[]) => tx >= 3 && tx + textWidth <= width - 3 && ty >= 8 && ty <= height - 8 &&
        !taken.some(([a, b, c, d]) => tx - 1 < c && tx + textWidth + 1 > a && ty - 6 < d && ty + 6 > b)
      const [tx, ty] = spots.find(clear) ?? [Math.min(width - 3 - textWidth, Math.max(3, prefer)), row]
      taken.push([tx - 1, ty - 6, tx + textWidth + 1, ty + 6])
      g.strokeStyle = MAP.bg
      g.lineWidth = 3
      g.strokeText(w.name, tx, ty)
      g.fillStyle = i === active ? MAP.ink : MAP.dim
      g.fillText(w.name, tx, ty)
    }

    if (this.trail.length) {
      g.strokeStyle = MAP.trail
      g.lineWidth = 2
      g.beginPath()
      for (const [i, [tn, te]] of this.trail.entries()) g[i ? 'lineTo' : 'moveTo'](...at(tn, te))
      g.lineTo(...at(n, e))
      g.stroke()
    }

    g.save()
    g.translate(width - 14, 16)
    g.fillStyle = MAP.ink
    g.beginPath()
    g.moveTo(0, -9)
    g.lineTo(4.5, 2)
    g.lineTo(-4.5, 2)
    g.closePath()
    g.fill()
    g.textAlign = 'center'
    g.font = `600 10px ${FONT}`
    g.fillText('N', 0, 10)
    g.restore()
    g.strokeStyle = MAP.ink
    g.lineWidth = 1.5
    g.lineCap = 'butt'
    g.beginPath()
    g.moveTo(scaleX, height - 11)
    g.lineTo(scaleX, height - 8)
    g.lineTo(scaleX + bar, height - 8)
    g.lineTo(scaleX + bar, height - 11)
    g.stroke()
    g.font = `500 10px ${FONT}`
    g.textAlign = 'left'
    g.strokeStyle = MAP.bg
    g.lineWidth = 3
    g.strokeText(scaleLabel, scaleX, height - 19)
    g.fillStyle = MAP.ink
    g.fillText(scaleLabel, scaleX, height - 19)

    if (this.ghostsShown) {
      for (const ghost of this.ghosts) {
        const pose = ghostPose(ghost, sim)
        if (!pose) continue
        const [gx, gy] = at(pose.north, pose.east)
        g.save()
        g.translate(Math.min(width - 5, Math.max(5, gx)), Math.min(height - 5, Math.max(5, gy)))
        g.rotate(pose.heading * DEG)
        g.globalAlpha = 0.85
        g.fillStyle = ghostColor(ghost.player)
        g.beginPath()
        g.moveTo(0, -6)
        g.lineTo(4, 5)
        g.lineTo(0, 2.5)
        g.lineTo(-4, 5)
        g.closePath()
        g.fill()
        g.restore()
      }
    }

    const [ax, ay] = at(n, e)
    const inside = ax >= 0 && ay >= 0 && ax <= width && ay <= height
    const x = Math.min(width - 6, Math.max(6, ax))
    const y = Math.min(height - 6, Math.max(6, ay))
    if (active !== null) {
      const target = waypoints[active]
      g.setLineDash([2, 3])
      g.strokeStyle = MAP.next
      g.lineWidth = 1.25
      g.beginPath()
      g.moveTo(x, y)
      g.lineTo(...at(target.north_m, target.east_m))
      g.stroke()
      g.setLineDash([])
    }
    const heading = scene.attitude.heading_deg
    g.save()
    g.translate(x, y)
    g.rotate(heading * DEG)
    // off the map, pin the marker to the edge and draw it hollow
    g.fillStyle = inside ? MAP.red : '#ffffff'
    g.strokeStyle = inside ? '#ffffff' : MAP.red
    g.lineWidth = inside ? 1.5 : 2
    g.lineJoin = 'miter'
    g.beginPath()
    g.moveTo(0, -9)
    g.lineTo(6, 7)
    g.lineTo(0, 3.5)
    g.lineTo(-6, 7)
    g.closePath()
    if (inside) g.stroke()
    g.fill()
    if (!inside) g.stroke()
    g.restore()
  }
}
