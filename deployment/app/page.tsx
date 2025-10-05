'use client'

import { useState, useEffect } from 'react'
import { FlowchartData } from '@/types/flowchart'
import FlowchartVisualization from '@/components/FlowchartVisualization'

export default function Home() {
    const [files, setFiles] = useState<string[]>([])
    const [selectedFile, setSelectedFile] = useState<string>('')
    const [data, setData] = useState<FlowchartData | null>(null)
    const [loading, setLoading] = useState(false)
    const [rolloutInput, setRolloutInput] = useState('')
    const [selectedRollouts, setSelectedRollouts] = useState<string[]>([])

    // Load available files
    useEffect(() => {
        fetch('/api/files')
            .then(res => res.json())
            .then(setFiles)
            .catch(console.error)
    }, [])

    // Load flowchart data when file is selected
    useEffect(() => {
        if (selectedFile) {
            setLoading(true)
            fetch(`/api/flowchart/${selectedFile}`)
                .then(res => res.json())
                .then((data: FlowchartData) => {
                    setData(data)
                    setLoading(false)
                })
                .catch(err => {
                    console.error(err)
                    setLoading(false)
                })
        } else {
            setData(null)
        }
    }, [selectedFile])

    // Parse rollout selection
    const parseRolloutSelection = (input: string) => {
        if (!input.trim()) {
            setSelectedRollouts([])
            return
        }

        const ids = new Set<string>()
        const parts = input.split(',')

        parts.forEach(part => {
            part = part.trim()
            if (part.includes('-')) {
                // Handle ranges like "3-10"
                const [start, end] = part.split('-').map(n => n.trim())
                const startNum = parseInt(start)
                const endNum = parseInt(end)
                if (!isNaN(startNum) && !isNaN(endNum)) {
                    for (let i = Math.min(startNum, endNum); i <= Math.max(startNum, endNum); i++) {
                        ids.add(i.toString())
                    }
                }
            } else {
                // Handle single numbers
                const num = parseInt(part)
                if (!isNaN(num)) {
                    ids.add(num.toString())
                }
            }
        })

        setSelectedRollouts(Array.from(ids).sort((a, b) => parseInt(a) - parseInt(b)))
    }

    const handleRolloutInputChange = (value: string) => {
        setRolloutInput(value)
        parseRolloutSelection(value)
    }

    return (
        <div className="container">
            <div className="header">
                <h1>Flowchart Visualizer</h1>
                <div className="file-selector">
                    <label htmlFor="fileSelect">Select Flowchart:</label>
                    <select
                        id="fileSelect"
                        value={selectedFile}
                        onChange={(e) => setSelectedFile(e.target.value)}
                    >
                        <option value="">Choose a file...</option>
                        {files.map(file => (
                            <option key={file} value={file}>{file}</option>
                        ))}
                    </select>
                </div>
            </div>

            <div className="main-content">
                <div className="visualization-area">
                    {loading ? (
                        <div className="loading">
                            <div className="spinner"></div>
                            <span>Loading...</span>
                        </div>
                    ) : !data ? (
                        <div className="empty-state">
                            Select a flowchart file to begin
                        </div>
                    ) : (
                        <FlowchartVisualization
                            data={data}
                            selectedRollouts={selectedRollouts}
                        />
                    )}
                </div>

                <div className="rollout-controls">
                    <h2>Rollout Selection</h2>
                    <input
                        type="text"
                        className="rollout-input"
                        placeholder="e.g., 1, 2-5, 10"
                        value={rolloutInput}
                        onChange={(e) => handleRolloutInputChange(e.target.value)}
                    />

                    {data && (
                        <div className="stats">
                            <h3>Statistics</h3>
                            <p>Total Nodes: {data.nodes.length}</p>
                            <p>Total Rollouts: {Array.isArray(data.rollouts) ? data.rollouts.length : Object.keys(data.rollouts).length}</p>
                            <p>Selected Rollouts: {selectedRollouts.length}</p>
                            {selectedRollouts.length > 0 && (
                                <p>Showing: {selectedRollouts.slice(0, 5).join(', ')}{selectedRollouts.length > 5 ? '...' : ''}</p>
                            )}
                        </div>
                    )}
                </div>
            </div>
        </div>
    )
}
