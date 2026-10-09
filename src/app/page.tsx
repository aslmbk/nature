import NatureStage from "@/components/NatureStage";
import ScrollTrack from "@/components/ScrollTrack";
import Overlay from "@/components/overlay/Overlay";
import { OVERLAY_SLOTS } from "@/components/overlay/sections";

export default function Home() {
  return (
    <main>
      <NatureStage />
      <Overlay>
        <ScrollTrack slots={OVERLAY_SLOTS} />
      </Overlay>
    </main>
  );
}
