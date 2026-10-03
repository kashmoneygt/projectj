import * as THREE from 'three'
import { RoundedBoxGeometry } from 'three/examples/jsm/geometries/RoundedBoxGeometry.js'
import { CORAL, DEG, PISTACHIO, WHITE, YELLOW, box, canvasTexture, ink, mesh, random, toon } from './materials'
import type { FlightScene } from './scene'

function grassTexture() {
  const rand = random(7)
  const texture = canvasTexture(512, 512, (g) => {
    // near-white daubs; vertex colours add the hue
    g.fillStyle = '#eef6dc'
    g.fillRect(0, 0, 512, 512)
    const shades = ['#ffffff', '#e2eecb', '#f9ffe6', '#dbe7bf', '#f3f9e0']
    for (let i = 0; i < 1600; i++) {
      g.fillStyle = shades[Math.floor(rand() * shades.length)]
      g.beginPath()
      g.ellipse(24 + rand() * 464, 24 + rand() * 464, 4 + rand() * 16, 2 + rand() * 5, rand() * Math.PI, 0, 2 * Math.PI)
      g.fill()
    }
  })
  texture.wrapS = texture.wrapT = THREE.RepeatWrapping
  return texture
}

function runwayTexture(length: number, width: number, heading: number) {
  const W = 256
  const H = 4096
  const sx = W / width
  const sy = H / length
  const texture = canvasTexture(W, H, (g) => {
    const rand = random(3)
    g.fillStyle = '#77716b'
    g.fillRect(0, 0, W, H)
    for (let i = 0; i < 6000; i++) {
      const shade = 108 + Math.floor(rand() * 16)
      g.fillStyle = `rgb(${shade + 6},${shade},${shade - 6})`
      g.beginPath()
      g.ellipse(rand() * W, rand() * H, 1.5 + rand() * 4, 3 + rand() * 9, 0, 0, 2 * Math.PI)
      g.fill()
    }
    g.fillStyle = '#fffdf4'
    const edge = Math.max(2, 0.9 * sx)
    g.fillRect(edge, 0, edge, H)
    g.fillRect(W - 2 * edge, 0, edge, H)
    const dash = 30 * sy
    const gap = 20 * sy
    const center = Math.max(2, 0.9 * sx)
    for (let y = 90 * sy; y < H - 90 * sy; y += dash + gap) g.fillRect(W / 2 - center / 2, y, center, dash)
    const numbers = [Math.round(heading / 10) % 36 || 36, Math.round(((heading + 180) % 360) / 10) % 36 || 36]
    for (const [end, number] of numbers.entries()) {
      g.save()
      if (end === 1) {
        g.translate(W, H)
        g.rotate(Math.PI)
      }
      const keys = 8
      const keyWidth = (W - 8 * edge) / (keys * 2)
      for (let k = 0; k < keys; k++) {
        const x = 4 * edge + keyWidth * (2 * k + 0.5)
        if (Math.abs(x + keyWidth / 2 - W / 2) < keyWidth) continue
        g.fillRect(x, H - 36 * sy, keyWidth, 30 * sy)
      }
      g.font = `700 ${Math.round(18 * sy)}px "IBM Plex Sans", sans-serif`
      g.textAlign = 'center'
      g.textBaseline = 'bottom'
      g.fillText(String(number).padStart(2, '0'), W / 2, H - 45 * sy)
      g.restore()
    }
  })
  texture.anisotropy = 8
  return texture
}

function windsock() {
  const group = new THREE.Group()
  group.add(box(0.12, 6, 0.12, YELLOW, 0, 3, 0))
  const sock = new THREE.Group()
  sock.position.set(0, 5.9, 0)
  const stripes = canvasTexture(8, 64, (g) => {
    for (let i = 0; i < 5; i++) {
      g.fillStyle = i % 2 ? '#ffffff' : '#ff5e4d'
      g.fillRect(0, i * 12.8, 8, 12.8)
    }
  })
  const cone = new THREE.Mesh(
    new THREE.CylinderGeometry(0.45, 0.22, 3.2, 16, 1, true).rotateX(Math.PI / 2).translate(0, 0, -1.6),
    toon(0xffffff, { map: stripes, side: THREE.DoubleSide }),
  )
  cone.castShadow = true
  sock.add(ink(cone, 0.05))
  group.add(sock)
  return { group, sock }
}

export function airfield(runway: FlightScene['runway']) {
  const group = new THREE.Group()
  group.rotation.y = -runway.heading_deg * DEG
  const { length_m: length, width_m: width } = runway
  const strip = new THREE.Mesh(
    new THREE.PlaneGeometry(width, length).rotateX(-Math.PI / 2),
    toon(0xffffff, { map: runwayTexture(length, width, runway.heading_deg) }),
  )
  strip.position.y = 0.04
  strip.receiveShadow = true
  const shoulder = new THREE.Mesh(
    new THREE.PlaneGeometry(width + 12, length + 40).rotateX(-Math.PI / 2),
    toon(0xc4e68a),
  )
  shoulder.position.y = 0.02
  shoulder.receiveShadow = true
  group.add(shoulder, strip)
  const markers = new THREE.InstancedMesh(
    new RoundedBoxGeometry(0.34, 0.4, 0.34, 2, 0.1), toon(YELLOW),
    2 * Math.floor(length / 60) + 2,
  )
  const matrix = new THREE.Matrix4()
  let index = 0
  for (let z = -length / 2; z <= length / 2; z += 60) {
    for (const side of [-1, 1]) markers.setMatrixAt(index++, matrix.makeTranslation(side * (width / 2 + 3), 0.18, z))
  }
  markers.count = index
  group.add(ink(markers, 0.04))
  const side = width / 2 + 90
  const apron = new THREE.Mesh(
    new THREE.PlaneGeometry(120, 90).rotateX(-Math.PI / 2),
    toon(0x99928a),
  )
  apron.position.set(side, 0.03, 0)
  apron.receiveShadow = true
  const taxiway = new THREE.Mesh(
    new THREE.PlaneGeometry(side - width / 2, 12).rotateX(-Math.PI / 2),
    toon(0x857e77),
  )
  taxiway.position.set((side + width / 2) / 2 - 30, 0.035, 0)
  const centreline = new THREE.Mesh(
    new THREE.PlaneGeometry(side - width / 2 - 22, 0.35).rotateX(-Math.PI / 2), toon(YELLOW))
  centreline.position.set((side + width / 2) / 2 - 9, 0.045, 0)
  centreline.receiveShadow = taxiway.receiveShadow = true
  group.add(apron, taxiway, centreline)
  for (const [i, [walls, roofColor]] of [[WHITE, CORAL], [0xd4f0b0, YELLOW]].entries()) {
    const hangar = new THREE.Group()
    hangar.position.set(side + 35, 0, -25 + i * 45)
    const shed = box(30, 9, 26, walls, 0, 4.5, 0)
    shed.receiveShadow = true
    const roof = mesh(new THREE.CylinderGeometry(15.3, 15.3, 26.4, 24, 1, false, -Math.PI / 2, Math.PI)
      .rotateX(Math.PI / 2).scale(1, 0.25, 1), roofColor)
    roof.position.y = 9
    const door = box(18, 6.5, 0.3, 0x39b8c9, -15.1, 3.3, 0).rotateY(Math.PI / 2)
    for (const item of [shed, roof, door]) hangar.add(ink(item, 0.18))
    group.add(hangar)
  }
  const sock = windsock()
  sock.group.position.set(-(width / 2 + 45), 0, length / 4)
  group.add(sock.group)
  return { group, sock: sock.sock, key: JSON.stringify(runway) }
}

export function terrain(scene: THREE.Scene) {
  const size = 40000
  const segments = 160
  const geometry = new THREE.PlaneGeometry(size, size, segments, segments).rotateX(-Math.PI / 2)
  const colors: number[] = []
  const rand = random(11)
  const cell = new Map<string, THREE.Color>()
  const fields = [0x9fd46a, 0xb4de74, 0x8cc95d, 0xc9e27a, 0xf2d56b, 0x7fbf5c, 0xa8dc7a]
  const position = geometry.attributes.position
  for (let i = 0; i < position.count; i++) {
    const key = `${Math.floor(position.getX(i) / 500)},${Math.floor(position.getZ(i) / 500)}`
    if (!cell.has(key)) cell.set(key, new THREE.Color(fields[Math.floor(rand() * fields.length)]))
    const color = cell.get(key)!.clone().multiplyScalar(0.95 + rand() * 0.08)
    colors.push(color.r, color.g, color.b)
  }
  geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3))
  const grass = grassTexture()
  grass.repeat.set(size / 24, size / 24)
  grass.anisotropy = 8
  const ground = new THREE.Mesh(geometry, toon(0xffffff, { map: grass, vertexColors: true }))
  ground.receiveShadow = true
  scene.add(ground)

  // distant ridge and hills (scenery only; the physics ground is flat)
  const ridge = new THREE.Mesh(
    new THREE.CylinderGeometry(19000, 19500, 1, 96, 1, true),
    toon(0x7cc47a, { side: THREE.BackSide }),
  )
  const ridgePosition = ridge.geometry.attributes.position
  for (let i = 0; i < ridgePosition.count; i++) {
    if (ridgePosition.getY(i) > 0) {
      const angle = Math.atan2(ridgePosition.getZ(i), ridgePosition.getX(i))
      ridgePosition.setY(i, 260 + 180 * Math.sin(angle * 5) + 120 * Math.sin(angle * 13 + 1))
    } else ridgePosition.setY(i, 0)
  }
  ridge.geometry.computeVertexNormals()
  scene.add(ridge)

  const puff = new THREE.IcosahedronGeometry(1, 3)
  const greens = [0x7cc86a, 0x94d56e, 0x6bb85f, PISTACHIO, 0x5fae5a].map((c) => new THREE.Color(c))
  const matrix = new THREE.Matrix4()
  const scale = new THREE.Vector3()
  const quaternion = new THREE.Quaternion()
  const at = new THREE.Vector3()

  const hills = new THREE.InstancedMesh(puff, toon(0xffffff), 56)
  for (let i = 0; i < hills.count; i++) {
    const angle = (i / hills.count + rand() * 0.015) * 2 * Math.PI
    const distance = 9000 + rand() * 7000
    const s = 900 + rand() * 1400
    at.set(Math.cos(angle) * distance, 0, Math.sin(angle) * distance)
    hills.setMatrixAt(i, matrix.compose(at, quaternion, scale.set(s, s * (0.12 + rand() * 0.14), s)))
    hills.setColorAt(i, greens[i % greens.length])
  }
  scene.add(ink(hills, 0.012))

  const trees = 2400
  const trunk = new THREE.InstancedMesh(new THREE.CylinderGeometry(0.3, 0.45, 3, 6),
    toon(0x9a6440), trees)
  const crown = new THREE.InstancedMesh(new THREE.IcosahedronGeometry(1, 2).scale(2.8, 3.4, 2.8), toon(0xffffff), trees)
  const groves = Array.from({ length: 70 }, () => [(rand() - 0.5) * 12000, (rand() - 0.5) * 12000])
  let placed = 0
  while (placed < trees) {
    const [cx, cz] = groves[placed % groves.length]
    const x = cx + (rand() - 0.5) * 360
    const z = cz + (rand() - 0.5) * 360
    if (Math.hypot(x, z) < 1700) {
      groves[placed % groves.length] = [(rand() - 0.5) * 12000, (rand() - 0.5) * 12000]
      continue
    }
    const s = 0.7 + rand() * 0.8
    scale.set(s, s, s)
    trunk.setMatrixAt(placed, matrix.compose(at.set(x, 1.5 * s, z), quaternion, scale))
    crown.setMatrixAt(placed, matrix.compose(at.set(x, 5.8 * s, z), quaternion, scale))
    crown.setColorAt(placed, greens[Math.floor(rand() * greens.length)])
    placed++
  }
  scene.add(trunk, ink(crown, 0.3))

  const perCloud = 7
  const clouds = new THREE.InstancedMesh(new THREE.IcosahedronGeometry(1, 2),
    toon(0xffffff, { emissive: 0x9ad7ef, emissiveIntensity: 0.4 }), 36 * perCloud)
  for (let i = 0; i < clouds.count; i += perCloud) {
    const angle = rand() * 2 * Math.PI
    const distance = 1500 + rand() * 11000
    const radius = 70 + rand() * 90
    const height = 650 + rand() * 750
    const along = rand() * Math.PI
    for (let p = 0; p < perCloud; p++) {
      const offset = (p - 3) * radius * 0.75
      // bigger puffs in the middle give the cumulus outline
      const s = radius * (0.45 + rand() * 0.35) * (1.6 - Math.abs(p - 3) * 0.3)
      at.set(Math.cos(angle) * distance + Math.cos(along) * offset, height + s * 0.45,
        Math.sin(angle) * distance + Math.sin(along) * offset + (rand() - 0.5) * radius * 0.5)
      clouds.setMatrixAt(i + p, matrix.compose(at, quaternion, scale.set(s, s * 0.85, s)))
    }
  }
  scene.add(ink(clouds, 0.04))
}

export function skyDome(sun: THREE.Vector3) {
  const material = new THREE.ShaderMaterial({
    uniforms: {
      zenith: { value: new THREE.Color(0x1fa9e1) }, horizon: { value: new THREE.Color(0xc9f3ff) },
      glow: { value: new THREE.Color(0xfff1a6) }, sun: { value: sun },
    },
    vertexShader: `
      varying vec3 direction;
      void main() {
        direction = normalize(position);
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      }`,
    fragmentShader: `
      uniform vec3 zenith;
      uniform vec3 horizon;
      uniform vec3 glow;
      uniform vec3 sun;
      varying vec3 direction;
      void main() {
        vec3 d = normalize(direction);
        vec3 color = mix(horizon, zenith, pow(max(d.y, 0.0), 0.6));
        float s = max(dot(d, sun), 0.0);
        color = mix(color, glow, 0.7 * pow(s, 12.0));
        color = mix(color, vec3(1.0, 0.8, 0.35), smoothstep(0.9968, 0.9971, s));
        color = mix(color, vec3(1.0, 0.98, 0.82), smoothstep(0.9974, 0.9977, s));
        gl_FragColor = vec4(color, 1.0);
        #include <colorspace_fragment>
      }`,
    side: THREE.BackSide, depthTest: false, depthWrite: false,
  })
  const dome = new THREE.Mesh(new THREE.SphereGeometry(30000, 32, 16), material)
  dome.renderOrder = -1
  dome.frustumCulled = false
  return dome
}
