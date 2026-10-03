type RecordingState = 'idle' | 'requesting' | 'recording' | 'saving' | 'blocked' | 'error'

const TYPES = ['video/webm;codecs=vp9', 'video/webm;codecs=vp8', 'video/webm']
const FPS = 30
const BITS_PER_SECOND = 2_500_000
// at 2.5 Mbit/s the byte limit ends a recording after about 7 minutes
const LIMITS = { seconds: 600, bytes: 128 * 1024 * 1024 }

// records this tab to a local WebM file
export class TabRecorder {
  state: RecordingState = 'idle'
  message = ''
  private mimeType = ''
  private started = 0
  private bytes = 0
  private largest = 0
  private limited: 'size' | 'time' | null = null
  private stream: MediaStream | null = null
  private recorder: MediaRecorder | null = null
  private chunks: Blob[] = []
  private readonly changed: () => void

  constructor(changed: () => void) {
    this.changed = changed
  }

  get seconds() {
    return this.state === 'recording' ? (performance.now() - this.started) / 1000 : 0
  }

  // call from a click handler (getDisplayMedia needs a user gesture)
  async start() {
    if (this.state === 'requesting' || this.state === 'recording' || this.state === 'saving') return
    if (!navigator.mediaDevices?.getDisplayMedia || typeof MediaRecorder === 'undefined') {
      return this.fail('blocked', 'This browser cannot record a tab')
    }
    this.mimeType = TYPES.find((type) => MediaRecorder.isTypeSupported(type)) ?? ''
    this.set('requesting', 'Choose this tab in the browser picker')
    let stream: MediaStream
    try {
      stream = await navigator.mediaDevices.getDisplayMedia({
        video: { frameRate: { ideal: FPS, max: FPS }, displaySurface: 'browser' }, audio: false,
        preferCurrentTab: true, selfBrowserSurface: 'include', surfaceSwitching: 'exclude', monitorTypeSurfaces: 'exclude',
      } as DisplayMediaStreamOptions)
    } catch (error) {
      const name = (error as DOMException).name
      return this.fail(name === 'NotAllowedError' ? 'blocked' : 'error',
        name === 'NotAllowedError' ? 'Recording cancelled or blocked by the browser' : `Recording failed: ${(error as Error).message}`)
    }
    const surface = stream.getVideoTracks()[0]?.getSettings().displaySurface
    if (surface && surface !== 'browser') {
      stream.getTracks().forEach((track) => track.stop())
      return this.fail('blocked', 'Only a browser tab is recorded; choose this tab')
    }
    this.stream = stream
    this.chunks = []
    this.bytes = 0
    this.largest = 0
    this.limited = null
    try {
      this.recorder = new MediaRecorder(stream, { ...(this.mimeType ? { mimeType: this.mimeType } : {}), videoBitsPerSecond: BITS_PER_SECOND })
    } catch (error) {
      return this.fail('error', `Recording failed: ${(error as Error).message}`)
    }
    this.recorder.addEventListener('dataavailable', (event) => this.received(event.data))
    this.recorder.addEventListener('stop', () => this.save())
    // the browser's own "Stop sharing" ends the track; the file is still saved
    for (const track of stream.getTracks()) track.addEventListener('ended', () => this.stop())
    this.recorder.start(1000)
    this.started = performance.now()
    this.set('recording', '')
  }

  stop(limit: 'size' | 'time' | null = null) {
    if (this.recorder?.state === 'recording') {
      this.limited = limit
      this.set('saving', 'Saving the recording')
      this.recorder.stop()
    }
    this.release()
  }

  private received(data: Blob) {
    if (!data.size) return
    this.chunks.push(data)
    this.bytes += data.size
    this.largest = Math.max(this.largest, data.size)
    if (this.state !== 'recording') return
    // the last chunk arrives after stop(), so stop while two more of the largest chunks still fit
    if (this.bytes + 2 * this.largest >= LIMITS.bytes) this.stop('size')
    else if (this.seconds >= LIMITS.seconds) this.stop('time')
  }

  private save() {
    const blob = new Blob(this.chunks, { type: this.mimeType || 'video/webm' })
    this.chunks = []
    this.recorder = null
    const name = `jym-${new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19)}.webm`
    const link = document.createElement('a')
    link.href = URL.createObjectURL(blob)
    link.download = name
    link.click()
    setTimeout(() => URL.revokeObjectURL(link.href), 1000)
    const limit = this.limited === 'size' ? `${LIMITS.bytes / 2 ** 20} MiB` : `${LIMITS.seconds / 60} min`
    this.set('idle', this.limited ? `Recording reached the ${limit} limit and was saved as ${name}` : `Saved ${name}`)
  }

  private release() {
    this.stream?.getTracks().forEach((track) => track.stop())
    this.stream = null
  }

  private fail(state: RecordingState, message: string) {
    this.release()
    this.recorder = null
    this.set(state, message)
  }

  private set(state: RecordingState, message: string) {
    this.state = state
    this.message = message
    this.changed()
  }
}
