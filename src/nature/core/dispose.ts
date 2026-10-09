import type { BufferGeometry, Material, Object3D, Texture } from "three";

type Disposable = { dispose(): void };

function isTexture(value: unknown): value is Texture {
  return typeof value === "object" && value !== null && (value as { isTexture?: boolean }).isTexture === true;
}

/** Textures referenced by a material (map, normalMap, uniforms, ...). */
export function materialTextures(material: Material): Texture[] {
  const out: Texture[] = [];
  for (const value of Object.values(material)) if (isTexture(value)) out.push(value);
  const uniforms = (material as { uniforms?: Record<string, { value: unknown }> }).uniforms;
  if (uniforms) for (const u of Object.values(uniforms)) if (isTexture(u?.value)) out.push(u.value);
  return out;
}

/**
 * Dispose geometries, materials and textures under `root`, skipping anything for
 * which `isShared(resource)` is true (e.g. resources owned by AssetRegistry).
 */
export function disposeObject(root: Object3D, isShared: (resource: object) => boolean = () => false): void {
  const seen = new Set<object>();
  const free = (r: Disposable & object) => {
    if (seen.has(r) || isShared(r)) return;
    seen.add(r);
    r.dispose();
  };
  root.traverse((obj) => {
    const mesh = obj as Object3D & { geometry?: BufferGeometry; material?: Material | Material[] };
    if (mesh.geometry) free(mesh.geometry);
    const mats = mesh.material ? (Array.isArray(mesh.material) ? mesh.material : [mesh.material]) : [];
    for (const m of mats) {
      for (const tex of materialTextures(m)) free(tex);
      free(m);
    }
    const disposable = obj as unknown as Partial<Disposable>;
    // InstancedMesh / BatchedMesh / skinned helpers expose their own dispose().
    if (typeof disposable.dispose === "function" && obj.type !== "Scene") free(obj as unknown as Disposable & object);
  });
}
