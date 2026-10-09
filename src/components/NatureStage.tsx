"use client";

/**
 * Mounts the imperative engine (src/nature) into a fixed full-viewport host.
 * The engine is imported dynamically inside the effect, so nothing WebGL-related
 * runs during SSR. StrictMode's mount → unmount → mount never leaves two renderers:
 * the first mount is cancelled before the import resolves, and every engine is
 * disposed in the effect cleanup.
 *
 * Without WebGL 2 (probed by creating a context, before any renderer exists, so the
 * browser console stays clean) or when the engine cannot load: our own hero render as a
 * still (public/nature/poster.jpg) and a short message, rendered into <body> above the
 * overlay so no card can cover it: a compact two-line chip in the top bar's row, between
 * the logo and the nav pill (the Menu button under 900 px), where it covers no copy at any
 * scroll position (the full sentence stays for screen readers).
 */
import { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import type { EngineStatus, NatureExperience } from "@/nature/NatureExperience";
import styles from "./NatureStage.module.css";

type StageStatus = EngineStatus | "unsupported";

export default function NatureStage() {
  const hostRef = useRef<HTMLDivElement>(null);
  const [status, setStatus] = useState<StageStatus>("starting");

  useEffect(() => {
    const host = hostRef.current;
    if (!host) return;
    let cancelled = false;
    let engine: NatureExperience | null = null;

    import("@/nature/NatureExperience")
      .then(({ NatureExperience }) => {
        if (cancelled) return;
        if (!NatureExperience.hasWebGL2()) {
          setStatus("unsupported");
          return;
        }
        try {
          engine = new NatureExperience({
            host,
            onStatus: (s) => {
              if (!cancelled) setStatus(s);
            },
          });
          engine.start();
        } catch (err) {
          console.error("[nature] WebGL 2 could not be started", err);
          engine?.dispose();
          engine = null;
          setStatus("unsupported");
        }
      })
      .catch((err: unknown) => {
        console.error("[nature] engine failed to load", err);
        if (!cancelled) setStatus("error");
      });

    return () => {
      cancelled = true;
      engine?.dispose();
      engine = null;
    };
  }, []);

  const fallback = status === "unsupported" || status === "error";

  return (
    <div className={styles.stage} data-status={status}>
      <div ref={hostRef} className={styles.host} />
      {fallback && (
        // eslint-disable-next-line @next/next/no-img-element -- one fixed full-bleed still, shown only without WebGL 2
        <img
          className={styles.poster}
          src="/nature/poster.jpg"
          alt="Two moss- and grass-covered mounds in soft green light (still image)"
          decoding="async"
        />
      )}
      {fallback &&
        createPortal(
          <p className={styles.message} role="status" data-nature-overlay>
            <span className={styles.messageLong}>
              {status === "unsupported"
                ? "This experience needs WebGL 2. Showing a still image instead."
                : "The forest could not be loaded. Please reload the page."}
            </span>
            <span className={styles.messageShort} aria-hidden="true">
              <b>{status === "unsupported" ? "Still image" : "Not loaded"}</b>
              {status === "unsupported" ? "no WebGL 2" : "please reload"}
            </span>
          </p>,
          document.body,
        )}
    </div>
  );
}
