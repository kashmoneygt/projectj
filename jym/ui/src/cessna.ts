import * as THREE from 'three'
import { CORAL, CYAN, DEG, GEAR_HEIGHT, INK, SOOT, WHITE, YELLOW, box, canvasTexture, hinged, ink, mesh } from './materials'

// lathe UVs: u goes around the fuselage from the belly, v from nose to tail
function liveryTexture() {
  return canvasTexture(512, 64, (g) => {
    g.fillStyle = '#ffffff'
    g.fillRect(0, 0, 512, 64)
    for (const [u, width, color] of [[0.262, 0.032, '#ff5e4d'], [0.236, 0.014, '#ffc83d']] as const) {
      g.fillStyle = color
      g.fillRect((u - width / 2) * 512, 0, width * 512, 56)
      g.fillRect((1 - u - width / 2) * 512, 0, width * 512, 56)
    }
    g.fillStyle = '#ff5e4d'
    g.fillRect(0, 56, 512, 8)
  })
}

export class Cessna {
  root = new THREE.Group()
  body = new THREE.Group()
  cockpit = new THREE.Group()
  exterior: THREE.Object3D[] = []
  prop = new THREE.Group()
  blades = new THREE.Group()
  disc: THREE.Mesh
  ailerons: THREE.Group[] = []
  flaps: THREE.Group[] = []
  elevator: THREE.Group
  rudder: THREE.Group
  eye = new THREE.Object3D()
  private paint: [THREE.MeshToonMaterial, THREE.Color][] = []
  private charred = 0

  constructor() {
    const b = this.body
    this.root.add(b)
    const profile = [
      [0, -2.45], [0.05, -2.45], [0.42, -2.3], [0.58, -1.8], [0.64, -1.0], [0.66, 0.2], [0.6, 1.2],
      [0.42, 2.8], [0.22, 4.4], [0.1, 5.3], [0, 5.3],
    ].map(([r, z]) => new THREE.Vector2(r, z))
    const fuselage = mesh(new THREE.LatheGeometry(profile, 32).rotateX(Math.PI / 2), 0xffffff, { map: liveryTexture() })
    fuselage.scale.set(0.95, 1.05, 1)
    const skin = new THREE.Group()
    skin.add(fuselage)
    skin.add(box(1.12, 0.62, 2.1, WHITE, 0, 0.45, -0.3))
    const windshield = box(1.08, 0.5, 0.05, CYAN, 0, 0.5, -1.3)
    windshield.rotation.x = -0.75
    windshield.add(box(0.08, 0.34, 0.02, 0xffffff, 0.28, 0, -0.03))
    skin.add(windshield)
    for (const side of [-1, 1]) skin.add(box(0.02, 0.34, 1.5, CYAN, side * 0.57, 0.47, -0.35))
    b.add(skin)
    this.exterior.push(skin)

    const wing = new THREE.Group()
    wing.position.set(0, 0.82, -0.55)
    wing.add(box(11, 0.14, 1.45, WHITE, 0, 0, 0.05))
    for (const side of [-1, 1]) {
      wing.add(box(0.3, 0.16, 1.45, CORAL, side * 5.4, 0, 0.05))
      const aileron = hinged(2.1, 0.07, 0.38, WHITE, side * 4.2, 0, 0.78)
      const flap = hinged(2.4, 0.07, 0.38, YELLOW, side * 1.75, -0.02, 0.78)
      this.ailerons.push(aileron)
      this.flaps.push(flap)
      wing.add(aileron, flap)
      const strut = mesh(new THREE.CylinderGeometry(0.04, 0.04, 2.9, 8), 0xe9e4da)
      strut.position.set(side * 1.8, -0.72, 0)
      strut.rotation.z = side * 1.1
      wing.add(strut)
    }
    b.add(wing)

    const tail = new THREE.Group()
    tail.position.set(0, 0.2, 4.6)
    tail.add(box(3.4, 0.08, 0.75, WHITE, 0, 0, 0))
    this.elevator = hinged(3.3, 0.06, 0.4, WHITE, 0, 0, 0.38)
    tail.add(this.elevator)
    const fin = mesh(new THREE.BoxGeometry(0.08, 1.35, 0.9), WHITE)
    fin.position.set(0, 0.72, -0.05)
    fin.rotation.x = 0.35
    tail.add(fin)
    tail.add(box(0.1, 0.25, 0.9, YELLOW, 0, 1.3, 0.2))
    this.rudder = new THREE.Group()
    this.rudder.position.set(0, 0.7, 0.42)
    this.rudder.add(box(0.06, 1.3, 0.38, CORAL, 0, 0, 0.19))
    tail.add(this.rudder)
    b.add(tail)

    const wheel = () => {
      const item = mesh(new THREE.CylinderGeometry(0.2, 0.2, 0.14, 16), 0x3a2a24)
      item.rotation.z = Math.PI / 2
      item.add(mesh(new THREE.CylinderGeometry(0.09, 0.09, 0.16, 12), YELLOW))
      return item
    }
    const nose = new THREE.Group()
    nose.position.set(0, -GEAR_HEIGHT + 0.2, -1.95)
    nose.add(wheel(), box(0.05, 0.6, 0.05, 0xd8d2c8, 0, 0.35, 0))
    b.add(nose)
    for (const side of [-1, 1]) {
      const main = new THREE.Group()
      main.position.set(side * 1.2, -GEAR_HEIGHT + 0.2, 0.2)
      main.add(wheel(), box(0.3, 0.26, 0.55, CORAL, 0, 0.12, 0))
      const leg = box(0.06, 1.3, 0.08, 0xd8d2c8, -side * 0.55, 0.45, 0)
      leg.rotation.z = side * 1.0
      main.add(leg)
      b.add(main)
    }

    this.prop.position.set(0, 0, -2.5)
    this.prop.add(mesh(new THREE.ConeGeometry(0.16, 0.35, 16).rotateX(-Math.PI / 2), YELLOW))
    for (const angle of [0, Math.PI]) {
      const hub = new THREE.Group()
      hub.rotation.z = angle
      hub.add(box(0.12, 0.95, 0.03, 0x3a2a24, 0, 0.5, 0.1), box(0.125, 0.16, 0.035, YELLOW, 0, 0.9, 0.1))
      this.blades.add(hub)
    }
    this.disc = new THREE.Mesh(new THREE.CircleGeometry(0.95, 40),
      new THREE.MeshBasicMaterial({ color: INK, transparent: true, opacity: 0.08, side: THREE.DoubleSide, depthWrite: false }))
    this.disc.position.z = 0.1
    this.prop.add(this.blades, this.disc)
    b.add(this.prop)

    const parts: THREE.Mesh[] = []
    b.traverse((item) => item instanceof THREE.Mesh && item !== this.disc && parts.push(item))
    for (const part of parts) {
      part.geometry.computeBoundingBox()
      const size = part.geometry.boundingBox!.getSize(new THREE.Vector3())
      ink(part, Math.min(0.05, 0.3 * Math.min(size.x, size.y, size.z) + 0.006))
    }
    for (const { material } of parts) if (material instanceof THREE.MeshToonMaterial) this.paint.push([material, material.color.clone()])

    const part = (w: number, h: number, d: number, x: number, y: number, z: number, color = 0x4a3129) => {
      const item = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), new THREE.MeshBasicMaterial({ color }))
      item.position.set(x, y, z)
      this.cockpit.add(ink(item, 0.003))
      return item
    }
    part(1.1, 0.26, 0.08, 0, 0.14, -0.98, 0x46b7bf)
    part(1.14, 0.04, 0.26, 0, 0.28, -1.06, CORAL)
    part(1.14, 0.03, 1.7, 0, 0.7, -0.3, WHITE)
    for (const side of [-1, 1]) {
      part(0.03, 0.46, 1.8, side * 0.58, 0.04, -0.2)
      const pillar = part(0.05, 0.05, 1, 0, 0, 0)
      const from = new THREE.Vector3(side * 0.55, 0.3, -1.08)
      const to = new THREE.Vector3(side * 0.57, 0.7, -0.7)
      pillar.position.copy(from).add(to).multiplyScalar(0.5)
      pillar.scale.z = from.distanceTo(to)
      pillar.lookAt(to)
    }
    this.cockpit.visible = false
    b.add(this.cockpit)
    this.eye.position.set(-0.27, 0.5, -0.2)
    b.add(this.eye)
  }

  // darkens the paint towards soot; 0 restores it
  char(amount: number) {
    if (amount === this.charred) return
    this.charred = amount
    for (const [material, color] of this.paint) material.color.copy(color).lerp(SOOT, amount)
  }

  update(controls: Record<string, number>, instruments: Record<string, unknown> | null, dt: number, stopped: boolean) {
    const aileron = controls.aileron ?? 0
    const actualFlaps = instruments?.flap_deg
    const flapDeg = typeof actualFlaps === 'number' ? actualFlaps : (controls.flaps ?? 0) * 30
    this.ailerons[0].rotation.x = aileron * 18 * DEG
    this.ailerons[1].rotation.x = -aileron * 18 * DEG
    for (const flap of this.flaps) flap.rotation.x = flapDeg * DEG
    const elevatorDeg = instruments?.elevator_deg
    // as in c172x: positive elevator is nose down, positive rudder yaws left
    this.elevator.rotation.x = typeof elevatorDeg === 'number' ? elevatorDeg * DEG : (controls.elevator ?? 0) * 22 * DEG
    this.rudder.rotation.y = -(controls.rudder ?? 0) * 20 * DEG
    const rpm = stopped ? 0 : instruments?.engine_rpm
    // above 400 rpm, draw a translucent disc instead of blades
    const spinning = typeof rpm === 'number' && rpm > 400
    this.blades.visible = !spinning
    this.disc.visible = spinning
    if (typeof rpm === 'number' && !spinning) this.prop.rotation.z += rpm / 60 * 2 * Math.PI * dt
  }
}
