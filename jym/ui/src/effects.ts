import * as THREE from 'three'
import { CORAL, CYAN, INK, PISTACHIO, WHITE, YELLOW, ink, random, toon } from './materials'

// seconds until the fire and smoke are gone
const BURST = 7
const GRAVITY = 9.8
const ALONG = new THREE.Vector3(0, 0, 1)

// explosion and smoke at an impact
export class ImpactEffect {
  root = new THREE.Group()
  private burst = new THREE.Group()
  private thrown = new THREE.Group()
  private flash = new THREE.Mesh(new THREE.IcosahedronGeometry(1, 2),
    new THREE.MeshBasicMaterial({ color: 0xfffbe0, transparent: true, depthWrite: false }))
  private shock = new THREE.Mesh(new THREE.RingGeometry(0.85, 1, 48).rotateX(-Math.PI / 2),
    new THREE.MeshBasicMaterial({ color: 0xfff6d8, transparent: true, depthWrite: false, side: THREE.DoubleSide }))
  private scorch = new THREE.Mesh(new THREE.CircleGeometry(1, 28).rotateX(-Math.PI / 2), new THREE.MeshBasicMaterial({
    color: INK, transparent: true, opacity: 0.55, depthWrite: false, polygonOffset: true, polygonOffsetFactor: -2 }))
  private fire = new THREE.InstancedMesh(new THREE.IcosahedronGeometry(1, 2), new THREE.MeshBasicMaterial(), 10)
  private sparks = new THREE.InstancedMesh(new THREE.BoxGeometry(0.1, 0.1, 1.1), new THREE.MeshBasicMaterial({ color: 0xffe27a }), 28)
  private debris = new THREE.InstancedMesh(new THREE.BoxGeometry(1, 0.35, 0.8), toon(0xffffff), 22)
  private smoke = new THREE.InstancedMesh(new THREE.IcosahedronGeometry(1, 2),
    toon(0xffffff, { transparent: true, depthWrite: false }), 18)
  private blobs: { at: THREE.Vector3; size: number; delay: number; life: number }[] = []
  private embers: { velocity: THREE.Vector3; life: number }[] = []
  private pieces: { velocity: THREE.Vector3; spin: THREE.Vector3; size: number }[] = []
  private puffs: { at: THREE.Vector3; birth: number; life: number; rise: number; size: number }[] = []
  private scratch = {
    matrix: new THREE.Matrix4(), position: new THREE.Vector3(), quaternion: new THREE.Quaternion(),
    scale: new THREE.Vector3(), euler: new THREE.Euler(), direction: new THREE.Vector3(),
  }

  constructor() {
    const rand = random(29)
    const spread = () => rand() * 2 - 1
    const fireColors = [0xfff6c8, 0xfff6c8, YELLOW, YELLOW, 0xff9a3c, 0xff9a3c, CORAL, CORAL, 0xff9a3c, YELLOW]
    for (let i = 0; i < this.fire.count; i++) {
      const at = new THREE.Vector3(spread(), 0.2 + rand(), spread()).normalize().multiplyScalar(i < 2 ? 0.4 : 1.6 + rand() * 2.2)
      this.blobs.push({ at, size: i < 2 ? 3.6 : 1.8 + rand() * 1.8, delay: i < 2 ? 0 : rand() * 0.18, life: 0.9 + rand() * 0.7 })
      this.fire.setColorAt(i, new THREE.Color(fireColors[i]))
    }
    for (let i = 0; i < this.sparks.count; i++) {
      this.embers.push({ velocity: new THREE.Vector3(spread(), 0.3 + rand(), spread()).normalize().multiplyScalar(12 + rand() * 14),
        life: 0.5 + rand() * 0.6 })
    }
    const debrisColors = [WHITE, WHITE, CORAL, YELLOW, 0x5a4a44]
    for (let i = 0; i < this.debris.count; i++) {
      // mostly thrown forward (-Z)
      this.pieces.push({ velocity: new THREE.Vector3(spread() * 6, 5 + rand() * 8, -2 - rand() * 12),
        spin: new THREE.Vector3(spread(), spread(), spread()).multiplyScalar(9), size: 0.3 + rand() * 0.5 })
      this.debris.setColorAt(i, new THREE.Color(debrisColors[i % debrisColors.length]))
    }
    const smokeColors = [0x4a3f3a, 0x6b605a, 0x8a827c, 0xb3aba4]
    for (let i = 0; i < this.smoke.count; i++) {
      this.puffs.push({ at: new THREE.Vector3(spread() * 2, 0, spread() * 2), birth: 0.15 + i * 0.17 + rand() * 0.1,
        life: 3 + rand() * 1.2, rise: 2 + rand() * 2, size: 1.6 + rand() * 1.4 })
      this.smoke.setColorAt(i, new THREE.Color(smokeColors[Math.min(3, Math.floor(i / 5))]))
    }
    this.scorch.position.y = 0.08
    this.shock.position.y = 0.12
    this.burst.add(this.flash, this.shock, ink(this.fire, 0.06), this.smoke)
    this.thrown.add(this.sparks, ink(this.debris, 0.06))
    this.root.add(this.scorch, this.burst, this.thrown)
    // pieces fly outside the computed bounds, so don't cull them
    this.root.traverse((item) => (item.frustumCulled = false))
    this.root.visible = false
  }

  // height: impact height above ground (m); drift: smoke velocity (m/s)
  update(age: number, height: number, heading: number, drift: THREE.Vector3) {
    const { matrix, position, quaternion, scale, euler, direction } = this.scratch
    this.thrown.rotation.y = heading
    this.burst.visible = age < BURST
    const flash = age / 0.3
    this.flash.visible = flash < 1
    this.flash.position.y = height
    this.flash.scale.setScalar(1.5 + 10 * flash)
    this.flash.material.opacity = 0.9 * (1 - flash)
    const ring = Math.min(1, age / 0.7)
    this.shock.visible = ring < 1
    const radius = 1 + 21 * (1 - (1 - ring) ** 2)
    this.shock.scale.set(radius, 1, radius)
    this.shock.material.opacity = 0.75 * (1 - ring)
    const scorch = Math.max(0.01, 7 * Math.min(1, age / 0.4))
    this.scorch.scale.set(scorch, 1, scorch)

    quaternion.identity()
    this.blobs.forEach((blob, i) => {
      const t = age - blob.delay
      const grown = Math.min(1, Math.max(0, t) / 0.22)
      const size = t <= 0 ? 0 : t < 0.22 ? blob.size * (1 - (1 - grown) ** 3)
        : blob.size * Math.max(0, 1 - (t - 0.22) / blob.life) ** 1.5
      position.copy(blob.at).multiplyScalar(0.4 + 0.6 * grown)
      position.y += height + 1.8 * Math.max(0, t)
      this.fire.setMatrixAt(i, matrix.compose(position, quaternion, scale.setScalar(size)))
    })
    this.fire.instanceMatrix.needsUpdate = true

    this.sparks.visible = age < 1.2
    this.embers.forEach(({ velocity, life }, i) => {
      const t = Math.min(age, life)
      position.copy(velocity).multiplyScalar(t)
      position.y = Math.max(0.1, position.y + height - 0.5 * GRAVITY * t * t)
      direction.copy(velocity)
      direction.y -= GRAVITY * t
      quaternion.setFromUnitVectors(ALONG, direction.normalize())
      this.sparks.setMatrixAt(i, matrix.compose(position, quaternion, scale.setScalar(Math.max(0, 1 - age / life))))
    })
    this.sparks.instanceMatrix.needsUpdate = true

    this.pieces.forEach(({ velocity, spin, size }, i) => {
      const rest = 0.06 + size * 0.18
      const landed = (velocity.y + Math.sqrt(velocity.y ** 2 + 2 * GRAVITY * Math.max(0, height - rest))) / GRAVITY
      const t = Math.min(age, landed)
      position.set(velocity.x * t, height + velocity.y * t - 0.5 * GRAVITY * t * t, velocity.z * t)
      if (age >= landed) position.y = rest
      quaternion.setFromEuler(euler.set(spin.x * t, spin.y * t, spin.z * t))
      this.debris.setMatrixAt(i, matrix.compose(position, quaternion, scale.setScalar(size)))
    })
    this.debris.instanceMatrix.needsUpdate = true

    quaternion.identity()
    this.smoke.material.opacity = 0.8 * Math.max(0, Math.min(1, (BURST - age) / 2.5))
    this.puffs.forEach((puff, i) => {
      const t = age - puff.birth
      const k = t / puff.life
      const size = k > 0 && k < 1 ? puff.size * (0.5 + 1.3 * k) * Math.min(1, t / 0.3) * Math.min(1, (1 - k) / 0.25) : 0
      position.copy(puff.at).multiplyScalar(1 + Math.max(0, k)).addScaledVector(drift, Math.max(0, t))
      position.y = height + 0.5 + puff.rise * puff.life * (1 - (1 - Math.min(1, Math.max(0, k))) ** 2)
      this.smoke.setMatrixAt(i, matrix.compose(position, quaternion, scale.setScalar(size)))
    })
    this.smoke.instanceMatrix.needsUpdate = true
  }
}

// seconds a confetti burst lasts
const CONFETTI = 2.6

// confetti when the aircraft passes a ring; it moves with the aircraft to stay in view
// its frame has y up and z along the pass direction
export class Confetti {
  root = new THREE.Group()
  // ring index of the current burst
  ring: number | null = null
  private paper = new THREE.InstancedMesh(new THREE.PlaneGeometry(0.6, 0.35),
    new THREE.MeshBasicMaterial({ side: THREE.DoubleSide }), 72)
  private pieces: { velocity: THREE.Vector3; spin: THREE.Vector3 }[] = []
  private start = 0
  private scratch = { matrix: new THREE.Matrix4(), position: new THREE.Vector3(), quaternion: new THREE.Quaternion(),
    scale: new THREE.Vector3(), euler: new THREE.Euler() }

  constructor() {
    const rand = random(41)
    const colors = [CORAL, YELLOW, CYAN, PISTACHIO, WHITE, 0xff8fc7, 0x9d8cff]
    for (let i = 0; i < this.paper.count; i++) {
      const angle = rand() * 2 * Math.PI
      const speed = 3 + rand() * 6
      this.pieces.push({
        velocity: new THREE.Vector3(Math.cos(angle) * speed, Math.sin(angle) * speed + 2, 3 - rand() * 10),
        spin: new THREE.Vector3(rand() * 2 - 1, rand() * 2 - 1, rand() * 2 - 1).multiplyScalar(12),
      })
      this.paper.setColorAt(i, new THREE.Color(colors[i % colors.length]))
    }
    this.paper.frustumCulled = false
    this.root.add(this.paper)
    this.root.visible = false
  }

  // wall is in milliseconds
  fire(ring: number, orientation: THREE.Quaternion, wall: number) {
    this.ring = ring
    this.root.quaternion.copy(orientation)
    this.start = wall
  }

  stop() {
    this.ring = null
    this.root.visible = false
  }

  update(wall: number, aircraft: THREE.Vector3) {
    const age = (wall - this.start) / 1000
    if (this.ring !== null && age >= CONFETTI) this.stop()
    this.root.visible = this.ring !== null
    if (!this.root.visible) return
    this.root.position.copy(aircraft)
    const { matrix, position, quaternion, scale, euler } = this.scratch
    // drag slows the throw (0.8 s time constant) while the paper falls at 2 m/s
    const thrown = (1 - Math.exp(-1.25 * age)) / 1.25
    const size = Math.min(1, (CONFETTI - age) / 0.6)
    this.pieces.forEach(({ velocity, spin }, i) => {
      position.copy(velocity).multiplyScalar(thrown)
      position.y -= 2 * age
      quaternion.setFromEuler(euler.set(spin.x * age, spin.y * age, spin.z * age))
      this.paper.setMatrixAt(i, matrix.compose(position, quaternion, scale.setScalar(size)))
    })
    this.paper.instanceMatrix.needsUpdate = true
  }
}
