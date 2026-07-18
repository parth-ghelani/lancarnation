/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  async rewrites() {
    // Proxy /api/* through the Next.js server to the Railway backend.
    // Browser never makes cross-origin requests → CORS irrelevant.
    // Set API_URL (server-side, no NEXT_PUBLIC_) in Vercel env vars.
    const apiUrl = process.env.API_URL || 'http://localhost:8000'
    return [
      {
        source: '/api/:path*',
        destination: `${apiUrl}/api/:path*`,
      },
    ]
  },
}

module.exports = nextConfig
