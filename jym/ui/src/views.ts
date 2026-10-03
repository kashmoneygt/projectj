export function el<K extends keyof HTMLElementTagNameMap>(tag: K, className = '', content?: string | Node | (string | Node)[]) {
  const node = document.createElement(tag)
  if (className) node.className = className
  if (content !== undefined) node.append(...(Array.isArray(content) ? content : [content]))
  return node
}
