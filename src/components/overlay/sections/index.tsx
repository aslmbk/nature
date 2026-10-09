/**
 * Overlay content per ScrollTrack section id (server components, rendered inside the
 * matching <section>). Gaps (e.g. "hero-arch") have no text.
 */
import type { ReactNode } from "react";
import ArchSection from "./ArchSection";
import BranchSection from "./BranchSection";
import CanopyCloseSection from "./CanopyCloseSection";
import CanopySection from "./CanopySection";
import CanyonSection from "./CanyonSection";
import FinaleSection from "./FinaleSection";
import HeroSection from "./HeroSection";
import OracleSection from "./OracleSection";
import StoneSection from "./StoneSection";
import StreamsSection from "./StreamsSection";

export const OVERLAY_SLOTS: Readonly<Partial<Record<string, ReactNode>>> = {
  hero: <HeroSection />,
  arch: <ArchSection />,
  canyon: <CanyonSection />,
  oracle: <OracleSection />,
  streams: <StreamsSection />,
  branch: <BranchSection />,
  stone: <StoneSection />,
  canopy: <CanopySection />,
  canopyClose: <CanopyCloseSection />,
  finale: <FinaleSection />,
};
