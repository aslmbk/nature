/**
 * Listing of the runtime assets in `public/nature/{models,textures}` with size and
 * mtime. The engine (AssetRegistry) uses it to ask "is this GLB there?" without
 * producing 404s, and appends `?v=<mtime>` so rebuilt assets are never served stale.
 */
import { readdir, stat } from "node:fs/promises";
import path from "node:path";

export const dynamic = "force-dynamic";

const ROOT = path.join(process.cwd(), "public", "nature");
const DIRS = ["models", "textures"] as const;
const EXTENSIONS = /\.(glb|gltf|bin|webp|png|jpg|jpeg|ktx2|json)$/i;

export async function GET(): Promise<Response> {
  const files: Record<string, { size: number; mtime: number }> = {};
  for (const dir of DIRS) {
    let names: string[];
    try {
      names = await readdir(path.join(ROOT, dir));
    } catch {
      continue;
    }
    for (const name of names.sort()) {
      if (!EXTENSIONS.test(name)) continue;
      try {
        const info = await stat(path.join(ROOT, dir, name));
        if (info.isFile()) files[`${dir}/${name}`] = { size: info.size, mtime: Math.floor(info.mtimeMs) };
      } catch {
        // file vanished between readdir and stat (being rewritten): skip it this time
      }
    }
  }
  return Response.json({ files }, { headers: { "Cache-Control": "no-store" } });
}
