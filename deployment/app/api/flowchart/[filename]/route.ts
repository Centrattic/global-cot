import { NextResponse } from 'next/server'
import fs from 'fs'
import path from 'path'

export async function GET(
    request: Request,
    { params }: { params: { filename: string } }
) {
    try {
        const filename = params.filename
        const filePath = path.join(process.cwd(), '..', 'flowcharts', filename)

        if (!fs.existsSync(filePath)) {
            return NextResponse.json({ error: 'File not found' }, { status: 404 })
        }

        const fileContent = fs.readFileSync(filePath, 'utf8')
        const data = JSON.parse(fileContent)

        return NextResponse.json(data)
    } catch (error) {
        console.error('Error reading file:', error)
        return NextResponse.json({ error: 'Failed to read file' }, { status: 500 })
    }
}
