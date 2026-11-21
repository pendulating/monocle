// The static build of the Neuronpedia share view. See scripts/f10/share_static/share_static.py.
// Static export: no server, no database, no inference host. The run files sit under public/runs.
// SHARE_BASE_PATH is the path of the site under its host, e.g. /artificial-aesthetic on GitHub Pages.
const basePath = process.env.SHARE_BASE_PATH || '';
/** @type {import('next').NextConfig} */
module.exports = {
  output: 'export',
  basePath,
  assetPrefix: basePath || undefined,
  trailingSlash: true,
  reactStrictMode: false,
  images: { unoptimized: true },
  pageExtensions: ['js', 'jsx', 'ts', 'tsx'],
  // The components and the lib come from the Neuronpedia checkout through symlinks. The webapp
  // type-checks them in its own build; this build does not repeat that check.
  typescript: { ignoreBuildErrors: true },
  eslint: { ignoreDuringBuilds: true },
  serverExternalPackages: ['mjml'],
  webpack: (config) => {
    // Keep the symlinked paths inside this project, so the Next loaders compile them.
    config.resolve.symlinks = false;
    // The same two fixes as the webapp's next.config.js.
    config.resolve.alias.handlebars = 'handlebars/dist/handlebars.js';
    config.module = { ...config.module, exprContextCritical: false };
    return config;
  },
};
