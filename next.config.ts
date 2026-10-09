import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  reactStrictMode: true,
  // The dev indicator would show up in deterministic captures; errors are still surfaced.
  devIndicators: false,
};

export default nextConfig;
