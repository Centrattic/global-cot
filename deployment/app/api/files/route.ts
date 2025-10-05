import { NextResponse } from 'next/server'
import fs from 'fs'
import path from 'path'

export async function GET() {
    try {
        const flowchartsDir = path.join(process.cwd(), '..', 'flowcharts')

        if (!fs.existsSync(flowchartsDir)) {
            return NextResponse.json([])
        }

        const files = fs.readdirSync(flowchartsDir)
            .filter(file => file.endsWith('.json'))
            .sort()

        return NextResponse.json(files)
    } catch (error) {
        console.error('Error reading files:', error)
        return NextResponse.json([], { status: 500 })
    }
}
