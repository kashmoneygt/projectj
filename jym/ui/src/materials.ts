import * as THREE from 'three'
import { RoundedBoxGeometry } from 'three/examples/jsm/geometries/RoundedBoxGeometry.js'
import * as BufferGeometryUtils from 'three/examples/jsm/utils/BufferGeometryUtils.js'

export const DEG = Math.PI / 180
// model origin height above the wheels
export const GEAR_HEIGHT = 1.25
export const WHITE = 0xfffaf2
export const CORAL = 0xff5e4d
export const YELLOW = 0xffc83d
export const CYAN = 0x6fe0ef
export const PISTACHIO = 0xa8dc7a
export const INK = 0x2a1a14
export const SOOT = new THREE.Color(0x3a2a24)

// three light bands for cel shading (DataTexture uses nearest filtering)
export const tones = new THREE.DataTexture(new Uint8Array([110, 190, 255]), 3, 1, THREE.RedFormat)
tones.needsUpdate = true

export function random(seed: number): () => number {
  return () => {
    seed = (seed + 0x6d2b79f5) | 0
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

export function canvasTexture(width: number, height: number, draw: (g: CanvasRenderingContext2D) => void) {
  const canvas = document.createElement('canvas')
  canvas.width = width
  canvas.height = height
  draw(canvas.getContext('2d')!)
  const texture = new THREE.CanvasTexture(canvas)
  texture.colorSpace = THREE.SRGBColorSpace
  return texture
}

export function toon(color: number, options: THREE.MeshToonMaterialParameters = {}) {
  return new THREE.MeshToonMaterial({ color, gradientMap: tones, ...options })
}

export function mesh(geometry: THREE.BufferGeometry, color: number, options: THREE.MeshToonMaterialParameters = {}) {
  const item = new THREE.Mesh(geometry, toon(color, options))
  item.castShadow = true
  return item
}

// ink outline: an inverted hull pushed out along smoothed normals
export function ink(item: THREE.Mesh, width: number) {
  const material = new THREE.MeshBasicMaterial({ color: INK, side: THREE.BackSide })
  material.onBeforeCompile = (shader) => {
    shader.uniforms.inkWidth = { value: width }
    shader.vertexShader = 'uniform float inkWidth;\n' + shader.vertexShader.replace('#include <begin_vertex>',
      'vec3 transformed = position + normal * inkWidth;')
  }
  material.customProgramCacheKey = () => 'ink'
  const geometry = BufferGeometryUtils.mergeVertices(item.geometry.clone().deleteAttribute('normal').deleteAttribute('uv'))
  geometry.computeVertexNormals()
  const hull = item instanceof THREE.InstancedMesh
    ? Object.assign(new THREE.InstancedMesh(geometry, material, item.count), { instanceMatrix: item.instanceMatrix })
    : new THREE.Mesh(geometry, material)
  item.add(hull)
  return item
}

export function box(w: number, h: number, d: number, color: number, x = 0, y = 0, z = 0) {
  const item = mesh(new RoundedBoxGeometry(w, h, d, 3, Math.min(0.6, 0.3 * Math.min(w, h, d))), color)
  item.position.set(x, y, z)
  return item
}

// a control surface hinged on its leading edge (the group's X axis)
export function hinged(w: number, h: number, d: number, color: number, x: number, y: number, z: number) {
  const pivot = new THREE.Group()
  pivot.position.set(x, y, z)
  pivot.add(box(w, h, d, color, 0, 0, d / 2))
  return pivot
}
