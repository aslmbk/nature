/**
 * Loading, caching and central disposal of shared assets under `public/nature/`.
 *
 * Fail-soft: scenes ask `await assets.has("models/grove.glb")` or use
 * `tryGltf(...)` / `tryTexture(...)`, which resolve to null when the file is
 * missing or broken, and fall back to their placeholder. Existence is checked
 * against `/api/nature-assets` (a directory listing, so missing files never produce
 * 404 noise); URLs carry `?v=<mtime>` so rebuilt assets are never served stale.
 *
 * Everything loaded here (GLTF geometries / materials / textures, shared
 * materials) is owned by the registry and freed in `dispose()`. Scenes dispose only
 * what they created themselves (`disposeObject(root, assets.isShared)`).
 *
 * The GLTF returned by `tryGltf` / `loadGltf` is the cached, shared original: read it,
 * never add `gltf.scene` itself to a scene or modify it. `instance(path)` hands out a
 * private copy of the scene graph (nodes and materials cloned, geometry shared).
 *
 * Textures are decoded off the main thread (`createImageBitmap`) where supported, but
 * not uploaded when they arrive: the engine's warm queue uploads every texture a set's
 * materials use before the set may be drawn, one piece per step (`uploadStep`: a large
 * image in strips of ≈ 1 MB, then its mip chain), so neither the load, a preload nor the
 * first frame that draws a texture stalls on a whole-image upload.
 */
import {
  ImageBitmapLoader,
  LoadingManager,
  NoColorSpace,
  RGBAFormat,
  RepeatWrapping,
  SRGBColorSpace,
  Texture,
  TextureLoader,
  UnsignedByteType,
  type BufferGeometry,
  type Material,
  type Mesh,
  type Object3D,
  type WebGLRenderer,
} from "three";
import { GLTFLoader, type GLTF } from "three/examples/jsm/loaders/GLTFLoader.js";
import { MeshoptDecoder } from "three/examples/jsm/libs/meshopt_decoder.module.js";
import { materialTextures } from "./core/dispose";

export const NATURE_BASE = "/nature";
export const MANIFEST_URL = "/api/nature-assets";

export type TextureRole = "color" | "data";

export interface TextureOptions {
  /** Repeat wrapping (default true). */
  repeat?: boolean;
  /** glTF UV convention (default false). */
  flipY?: boolean;
}

interface ManifestEntry {
  size: number;
  mtime: number;
}

type Disposable = { dispose(): void };

/** What one `uploadStep` did: nothing to upload / a piece went up, more to come / complete. */
export type UploadStep = "none" | "partial" | "done";

/** Image bytes per upload step (whole rows); an image up to this size goes up in one step. */
const STRIP_BYTES = 1 << 20;

/** The part of three's per-texture renderer properties read here. */
interface TextureProps {
  __version?: number;
  __webglTexture?: WebGLTexture;
}

/** A GL texture whose level 0 is being uploaded strip by strip. */
interface Fill {
  image: ImageBitmap;
  /** Textures that use the GL texture (one of them still existing keeps the fill alive). */
  owners: Set<Texture>;
  /** Next row to upload; rows per strip. */
  y: number;
  rows: number;
  mips: boolean;
  unpackAlignment: number;
}

/**
 * `createImageBitmap` with options (decode off the main thread). Safari < 17 and
 * Firefox < 98 ignore or reject the options: they keep the <img> path (same rule as
 * three's GLTFLoader).
 */
function imageBitmapsUsable(): boolean {
  if (typeof createImageBitmap === "undefined" || typeof navigator === "undefined") return false;
  const ua = navigator.userAgent;
  if (/^((?!chrome|android).)*safari/i.test(ua)) {
    const version = Number(/Version\/(\d+)/.exec(ua)?.[1] ?? 0);
    if (version < 17) return false;
  }
  const firefox = /Firefox\/(\d+)/.exec(ua);
  if (firefox && Number(firefox[1]) < 98) return false;
  return true;
}

function isImageBitmap(value: unknown): value is ImageBitmap {
  return typeof ImageBitmap !== "undefined" && value instanceof ImageBitmap;
}

/** Texture kinds that always go up in one step (their own upload paths in three). */
const STRIP_EXCLUDED = [
  "isCubeTexture",
  "isDataTexture",
  "isDataArrayTexture",
  "isData3DTexture",
  "isCompressedTexture",
  "isDepthTexture",
  "isFramebufferTexture",
  "isCanvasTexture",
] as const;

export class AssetRegistry {
  private manifest: Map<string, ManifestEntry> | null = null;
  private manifestPromise: Promise<void> | null = null;
  private readonly existence = new Map<string, Promise<boolean>>();
  private readonly gltfs = new Map<string, Promise<GLTF | null>>();
  private readonly textures = new Map<string, Promise<Texture | null>>();
  private readonly shareds = new Map<string, Promise<unknown>>();
  private readonly owned = new Set<object>();
  /** GL textures whose level 0 is being uploaded in strips. */
  private readonly fills = new Map<WebGLTexture, Fill>();
  /** GL textures holding their whole image (the image they were filled from). */
  private readonly filled = new WeakMap<WebGLTexture, ImageBitmap>();
  /** Images whose strip upload failed: three uploads them whole. */
  private readonly unstriped = new WeakSet<ImageBitmap>();
  private readonly warned = new Set<string>();
  private readonly gltfLoader: GLTFLoader;
  private readonly textureLoader: TextureLoader;
  private readonly bitmapLoaders: (ImageBitmapLoader | null)[] = [null, null];
  private readonly useBitmaps = imageBitmapsUsable();
  private readonly manager: LoadingManager;
  private inFlight = 0;
  private disposed = false;
  private anisotropy: number;

  constructor(
    private readonly renderer: WebGLRenderer,
    anisotropy: number,
  ) {
    const manager = new LoadingManager();
    this.manager = manager;
    this.gltfLoader = new GLTFLoader(manager);
    this.gltfLoader.setMeshoptDecoder(MeshoptDecoder);
    this.textureLoader = new TextureLoader(manager);
    this.anisotropy = anisotropy;
  }

  /** Number of loads in flight (capture readiness). */
  get pending(): number {
    return this.inFlight;
  }

  /** Is `resource` owned by the registry (shared, do not dispose in a scene)? */
  readonly isShared = (resource: object): boolean => this.owned.has(resource);

  private track<T>(promise: Promise<T>): Promise<T> {
    this.inFlight++;
    return promise.finally(() => {
      this.inFlight--;
    });
  }

  private warnOnce(key: string, message: string): void {
    if (this.warned.has(key)) return;
    this.warned.add(key);
    console.info(`[nature/assets] ${message}`);
  }

  /** Fetch the directory listing once. Never throws. */
  init(): Promise<void> {
    if (!this.manifestPromise) {
      this.manifestPromise = this.track(
        fetch(MANIFEST_URL, { cache: "no-store" })
          .then(async (res) => {
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            const json = (await res.json()) as { files?: Record<string, ManifestEntry> };
            this.manifest = new Map(Object.entries(json.files ?? {}));
          })
          .catch((err: unknown) => {
            this.manifest = null;
            this.warnOnce("manifest", `asset manifest unavailable (${String(err)}), falling back to HEAD probes`);
          }),
      );
    }
    return this.manifestPromise;
  }

  /** Public URL of a path relative to public/nature, with a cache-busting version when known. */
  url(path: string): string {
    const clean = path.replace(/^\/+/, "");
    const entry = this.manifest?.get(clean);
    return `${NATURE_BASE}/${clean}${entry ? `?v=${entry.mtime}` : ""}`;
  }

  /** Does `public/nature/<path>` exist (and is non-empty)? */
  has(path: string): Promise<boolean> {
    const clean = path.replace(/^\/+/, "");
    let p = this.existence.get(clean);
    if (!p) {
      p = this.init().then(async () => {
        if (this.manifest) {
          const e = this.manifest.get(clean);
          return !!e && e.size > 0;
        }
        try {
          const res = await fetch(`${NATURE_BASE}/${clean}`, { method: "HEAD", cache: "no-store" });
          return res.ok;
        } catch {
          return false;
        }
      });
      this.existence.set(clean, p);
    }
    return p;
  }

  /** Load a GLB/GLTF (cached). Resolves null when missing or unparsable. */
  tryGltf(path: string): Promise<GLTF | null> {
    const clean = path.replace(/^\/+/, "");
    let p = this.gltfs.get(clean);
    if (!p) {
      p = this.track(
        this.has(clean).then(async (exists) => {
          if (!exists) {
            this.warnOnce(clean, `${clean} not found — scenes use their placeholder`);
            return null;
          }
          try {
            const gltf = await this.gltfLoader.loadAsync(this.url(clean));
            if (this.disposed) return null;
            this.adopt(gltf.scene);
            for (const cam of gltf.cameras) this.adopt(cam);
            return gltf;
          } catch (err) {
            this.warnOnce(clean, `${clean} failed to load (${String(err)}) — using placeholder`);
            return null;
          }
        }),
      );
      this.gltfs.set(clean, p);
    }
    return p;
  }

  /** Load a GLB/GLTF or throw. */
  async loadGltf(path: string): Promise<GLTF> {
    const gltf = await this.tryGltf(path);
    if (!gltf) throw new Error(`Asset missing: ${path}`);
    return gltf;
  }

  /**
   * A private copy of a GLB's scene graph, safe to add to a scene and to modify:
   * nodes (meshes, cameras, empties, with their transforms and userData) and materials
   * are cloned; geometries and textures stay shared and registry-owned (clone a
   * geometry before editing its attributes). Every call returns a new copy. Resolves
   * null when the file is missing or broken. Skinned meshes are not supported.
   */
  async instance(path: string): Promise<Object3D | null> {
    const gltf = await this.tryGltf(path);
    if (!gltf || this.disposed) return null;
    const root = gltf.scene.clone(true);
    root.traverse((o) => {
      const mesh = o as Mesh;
      if (!mesh.material) return;
      mesh.material = Array.isArray(mesh.material) ? mesh.material.map((m) => m.clone()) : mesh.material.clone();
    });
    return root;
  }

  /**
   * Load a texture (cached per path + role). `color` → sRGB (base colour only);
   * `data` → no colour space (normal, ORM, masks). Resolves null when missing.
   */
  tryTexture(path: string, role: TextureRole, opts: TextureOptions = {}): Promise<Texture | null> {
    const clean = path.replace(/^\/+/, "");
    const key = `${clean}|${role}|${opts.repeat ?? true}|${opts.flipY ?? false}`;
    let p = this.textures.get(key);
    if (!p) {
      p = this.track(
        this.has(clean).then(async (exists) => {
          if (!exists) {
            this.warnOnce(clean, `${clean} not found — flat fallback`);
            return null;
          }
          try {
            const flipY = opts.flipY ?? false;
            const tex = await this.loadTextureFile(this.url(clean), flipY);
            if (this.disposed) {
              tex.dispose();
              if (isImageBitmap(tex.image)) tex.image.close();
              return null;
            }
            tex.name = clean;
            tex.colorSpace = role === "color" ? SRGBColorSpace : NoColorSpace;
            // ImageBitmaps are oriented at decode time (WebGL ignores UNPACK_FLIP_Y for them)
            tex.flipY = isImageBitmap(tex.image) ? false : flipY;
            if (opts.repeat ?? true) tex.wrapS = tex.wrapT = RepeatWrapping;
            tex.anisotropy = Math.min(this.anisotropy, this.renderer.capabilities.getMaxAnisotropy());
            tex.needsUpdate = true;
            this.owned.add(tex);
            // no upload here: the warm-up of the set that uses it does that (`uploadStep`)
            return tex;
          } catch (err) {
            this.warnOnce(clean, `${clean} failed to load (${String(err)})`);
            return null;
          }
        }),
      );
      this.textures.set(key, p);
    }
    return p;
  }

  /** Decoded off the main thread when possible (ImageBitmap), else through an <img>. */
  private async loadTextureFile(url: string, flipY: boolean): Promise<Texture> {
    if (!this.useBitmaps) return this.textureLoader.loadAsync(url);
    const index = flipY ? 1 : 0;
    let loader = this.bitmapLoaders[index];
    if (!loader) {
      loader = new ImageBitmapLoader(this.manager);
      // raw texels exactly like the <img> path with UNPACK_COLORSPACE_CONVERSION = NONE,
      // no premultiplication, orientation as stored (or flipped for glTF-style UVs)
      loader.setOptions({ imageOrientation: flipY ? "flipY" : "none", premultiplyAlpha: "none", colorSpaceConversion: "none" });
      this.bitmapLoaders[index] = loader;
    }
    const bitmap = await loader.loadAsync(url);
    return new Texture(bitmap);
  }

  /**
   * Texture from `public/nature/textures/<name>.webp`; the role is inferred from the
   * name (`*_basecolor` → sRGB, everything else → data).
   */
  natureTexture(name: string, opts: TextureOptions = {}): Promise<Texture | null> {
    const role: TextureRole = /(basecolor|albedo|diffuse|color)$/i.test(name) ? "color" : "data";
    return this.tryTexture(`textures/${name}.webp`, role, opts);
  }

  /**
   * One step of `texture`'s upload to the GPU. The engine's warm queue calls it (a step at
   * a time, within its frame budget) until it returns "done" or "none", before any frame
   * draws the texture:
   *  - a large decoded image (ImageBitmap, RGBA8) goes up in strips of ≈ `STRIP_BYTES`:
   *    the first step allocates storage, mip levels, sampler state and three's bookkeeping
   *    through three without data (`source.dataReady = false`), each further step uploads
   *    one strip of level 0 (a WebGL2 sub-rectangle: `UNPACK_SKIP_ROWS`), the last one
   *    also generates the mip chain — the texels end up as with a whole-image upload;
   *  - anything else goes up in one step (`renderer.initTexture`), as on first use;
   *  - a GL texture that already holds the image (e.g. a material's clone of a shared
   *    texture flagged `needsUpdate`, same sampler key) is only re-keyed, not uploaded again.
   */
  uploadStep(texture: Texture): UploadStep {
    const r = this.renderer;
    const x = texture as Texture & { isRenderTargetTexture?: boolean; isVideoTexture?: boolean; isExternalTexture?: boolean };
    if (x.isRenderTargetTexture || x.isVideoTexture || x.isExternalTexture || texture.version === 0) return "none";
    const image = texture.image as { complete?: boolean } | null | undefined;
    if (!image || image.complete === false) return "none";
    const props = r.properties.get(texture) as TextureProps;
    if (props.__version === texture.version) {
      // up to date for three; a strip upload of its GL texture may still be under way
      const glTexture = props.__webglTexture;
      const fill = glTexture ? this.fills.get(glTexture) : undefined;
      if (!glTexture || !fill) return "none";
      fill.owners.add(texture);
      return this.fillStep(glTexture, fill);
    }
    const bitmap = this.stripSource(texture);
    if (!bitmap) {
      r.initTexture(texture);
      return "done";
    }
    const source = texture.source;
    source.dataReady = false;
    try {
      r.initTexture(texture);
    } finally {
      source.dataReady = true;
    }
    const glTexture = props.__webglTexture;
    if (!glTexture || this.filled.get(glTexture) === bitmap) return "done";
    let fill = this.fills.get(glTexture);
    if (!fill || fill.image !== bitmap) {
      fill = {
        image: bitmap,
        owners: new Set(),
        y: 0,
        rows: Math.max(1, Math.floor(STRIP_BYTES / (bitmap.width * 4))),
        mips: texture.generateMipmaps,
        unpackAlignment: texture.unpackAlignment,
      };
      this.fills.set(glTexture, fill);
    }
    fill.owners.add(texture);
    return "partial";
  }

  /** The ImageBitmap of a plain RGBA8 texture worth uploading in strips, else null. */
  private stripSource(texture: Texture): ImageBitmap | null {
    const flags = texture as unknown as Record<string, unknown>;
    const image = texture.image as unknown;
    if (!isImageBitmap(image) || image.width * image.height * 4 <= STRIP_BYTES || this.unstriped.has(image)) return null;
    if (Math.max(image.width, image.height) > this.renderer.capabilities.maxTextureSize) return null;
    if (texture.format !== RGBAFormat || texture.type !== UnsignedByteType || texture.mipmaps.length > 0) return null;
    for (const kind of STRIP_EXCLUDED) if (flags[kind] === true) return null;
    return image;
  }

  /** Upload the next strip of a fill, and the mip chain after the last one. */
  private fillStep(glTexture: WebGLTexture, fill: Fill): UploadStep {
    const r = this.renderer;
    // every texture using it went away meanwhile (disposed, re-keyed): nothing to finish
    for (const owner of fill.owners) {
      if (!r.properties.has(owner) || (r.properties.get(owner) as TextureProps).__webglTexture !== glTexture) fill.owners.delete(owner);
    }
    if (fill.owners.size === 0) {
      this.fills.delete(glTexture);
      return "none";
    }
    const gl = r.getContext() as WebGL2RenderingContext;
    const state = r.state;
    const image = fill.image;
    const rows = Math.min(fill.rows, image.height - fill.y);
    const last = fill.y + rows >= image.height;
    // through three's state cache: it sees the binding, and the pixel-store values it keeps
    state.bindTexture(gl.TEXTURE_2D, glTexture, gl.TEXTURE0);
    state.pixelStorei(gl.UNPACK_ALIGNMENT, fill.unpackAlignment);
    state.pixelStorei(gl.UNPACK_SKIP_ROWS, fill.y);
    try {
      // ImageBitmaps ignore UNPACK_FLIP_Y / PREMULTIPLY / COLORSPACE (applied at decode)
      gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, fill.y, image.width, rows, gl.RGBA, gl.UNSIGNED_BYTE, image);
      if (last && fill.mips) gl.generateMipmap(gl.TEXTURE_2D);
    } catch {
      // the browser refused the sub-rectangle: three uploads the whole image instead (next step)
      this.fills.delete(glTexture);
      this.unstriped.add(image);
      for (const owner of fill.owners) owner.needsUpdate = true;
      return "partial";
    } finally {
      state.pixelStorei(gl.UNPACK_SKIP_ROWS, 0);
      state.unbindTexture();
    }
    fill.y += rows;
    if (!last) return "partial";
    this.fills.delete(glTexture);
    this.filled.set(glTexture, image);
    return "done";
  }

  /** A shared resource created once per key (e.g. a material used by several scenes). */
  shared<T extends object>(key: string, create: () => T | Promise<T>): Promise<T> {
    let p = this.shareds.get(key) as Promise<T> | undefined;
    if (!p) {
      p = this.track(
        Promise.resolve()
          .then(create)
          .then((value) => {
            this.owned.add(value);
            return value;
          }),
      );
      this.shareds.set(key, p);
    }
    return p;
  }

  /** Mark everything under an object (geometries, materials, their textures) as registry-owned. */
  adopt(root: Object3D | Material | BufferGeometry): void {
    const addMaterial = (m: Material) => {
      this.owned.add(m);
      for (const t of materialTextures(m)) this.owned.add(t);
    };
    if ((root as Object3D).isObject3D) {
      (root as Object3D).traverse((o) => {
        const mesh = o as Object3D & { geometry?: BufferGeometry; material?: Material | Material[] };
        if (mesh.geometry) this.owned.add(mesh.geometry);
        if (mesh.material) (Array.isArray(mesh.material) ? mesh.material : [mesh.material]).forEach(addMaterial);
      });
    } else if ((root as Material).isMaterial) {
      addMaterial(root as Material);
    } else {
      this.owned.add(root);
    }
  }

  setAnisotropy(value: number): void {
    this.anisotropy = value;
  }

  dispose(): void {
    this.disposed = true;
    for (const r of this.owned) {
      const d = r as Partial<Disposable>;
      if (typeof d.dispose === "function") d.dispose();
      const image = (r as Partial<Texture>).isTexture ? (r as Texture).image : null;
      if (isImageBitmap(image)) image.close();
    }
    this.owned.clear();
    this.fills.clear();
    this.gltfs.clear();
    this.textures.clear();
    this.shareds.clear();
    this.existence.clear();
  }
}
