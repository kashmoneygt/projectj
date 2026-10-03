import * as THREE from 'three'
import { CORAL, CYAN, GEAR_HEIGHT, INK, SOOT, WHITE, YELLOW, box, ink, mesh } from './materials'

export class Drone {
  root = new THREE.Group()
  body = new THREE.Group()
  cockpit = new THREE.Group()
  exterior: THREE.Object3D[] = []
  disc: THREE.Mesh
  eye = new THREE.Object3D()
  private rotors: { blades: THREE.Group; disc: THREE.Mesh }[] = []
  private paint: [THREE.MeshToonMaterial, THREE.Color][] = []
  private charred = 0

  constructor() {
    const b = this.body
    this.root.add(b)
    const frame = new THREE.Group()
    frame.add(box(0.9, 0.45, 1.2, WHITE, 0, 0, 0), box(0.92, 0.12, 0.5, CORAL, 0, 0.18, -0.2), box(0.5, 0.2, 0.3, CYAN, 0, -0.1, -0.62))
    for (const angle of [Math.PI / 4, -Math.PI / 4]) {
      const arm = box(0.12, 0.1, 3.4, 0xe9e4da, 0, 0.12, 0)
      arm.rotation.y = angle
      frame.add(arm)
    }
    for (const side of [-1, 1]) {
      frame.add(box(0.06, 0.06, 1.6, 0x3a2a24, side * 0.55, -GEAR_HEIGHT + 0.03, 0))
      for (const end of [-0.5, 0.5]) {
        const leg = box(0.05, GEAR_HEIGHT - 0.2, 0.05, 0xd8d2c8, side * 0.5, -GEAR_HEIGHT / 2 - 0.05, end)
        leg.rotation.z = side * 0.12
        frame.add(leg)
      }
    }
    for (const [x, z] of [[-1.2, -1.2], [1.2, -1.2], [-1.2, 1.2], [1.2, 1.2]]) {
      frame.add(mesh(new THREE.CylinderGeometry(0.13, 0.13, 0.22, 16), 0x3a2a24).translateX(x).translateY(0.22).translateZ(z))
      const blades = new THREE.Group()
      blades.position.set(x, 0.36, z)
      blades.add(box(1.4, 0.03, 0.12, 0x3a2a24, 0, 0, 0), box(0.14, 0.035, 0.13, YELLOW, 0.6, 0, 0))
      const disc = new THREE.Mesh(new THREE.CircleGeometry(0.75, 32).rotateX(-Math.PI / 2),
        new THREE.MeshBasicMaterial({ color: INK, transparent: true, opacity: 0.1, side: THREE.DoubleSide, depthWrite: false }))
      disc.position.set(x, 0.36, z)
      frame.add(blades, disc)
      this.rotors.push({ blades, disc })
    }
    b.add(frame)
    this.exterior.push(frame)
    const parts: THREE.Mesh[] = []
    frame.traverse((item) => item instanceof THREE.Mesh && !this.rotors.some((rotor) => rotor.disc === item) && parts.push(item))
    for (const part of parts) {
      part.geometry.computeBoundingBox()
      const size = part.geometry.boundingBox!.getSize(new THREE.Vector3())
      ink(part, Math.min(0.04, 0.3 * Math.min(size.x, size.y, size.z) + 0.006))
    }
    for (const { material } of parts) if (material instanceof THREE.MeshToonMaterial) this.paint.push([material, material.color.clone()])
    this.disc = this.rotors[0].disc
    this.cockpit.visible = false
    b.add(this.cockpit)
    this.eye.position.set(0, -0.2, -0.8) // the gimbal camera under its nose
    b.add(this.eye)
  }

  char(amount: number) {
    if (amount === this.charred) return
    this.charred = amount
    for (const [material, color] of this.paint) material.color.copy(color).lerp(SOOT, amount)
  }

  update(_controls: Record<string, number>, _instruments: Record<string, unknown> | null, dt: number, stopped: boolean) {
    for (const [i, rotor] of this.rotors.entries()) {
      rotor.disc.visible = !stopped
      rotor.blades.visible = stopped
      if (!stopped) rotor.blades.rotation.y += (i % 2 ? -1 : 1) * 60 * dt
    }
  }
}
