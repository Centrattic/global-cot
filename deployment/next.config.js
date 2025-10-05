/** @type {import('next').NextConfig} */
const nextConfig = {
    experimental: {
        appDir: true,
    },
    async rewrites() {
        return [
            {
                source: '/data/:path*',
                destination: '/api/data/:path*',
            },
        ]
    },
}

module.exports = nextConfig
