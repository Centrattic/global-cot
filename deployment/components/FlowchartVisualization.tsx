'use client'

import { useEffect, useRef, useState } from 'react'
import * as d3 from 'd3'
import { FlowchartData, Node, RolloutObject } from '@/types/flowchart'

interface FlowchartVisualizationProps {
    data: FlowchartData
    selectedRollouts: string[]
}

export default function FlowchartVisualization({ data, selectedRollouts }: FlowchartVisualizationProps) {
    const svgRef = useRef<SVGSVGElement>(null)
    const [selectedNode, setSelectedNode] = useState<Node | null>(null)
    const [hoveredRollout, setHoveredRollout] = useState<string | null>(null)
    const [validRollouts, setValidRollouts] = useState<string[]>([])
    const [currentTransform, setCurrentTransform] = useState<d3.ZoomTransform | null>(null)

    // Function to animate ball along rollout path
    const animateBallAlongPath = (rolloutId: string, path: string[], nodePositions: Map<string, { x: number; y: number }>) => {
        if (path.length < 2) return

        const svg = d3.select(svgRef.current)
        const g = svg.select('g')

        // Remove any existing animation ball for this rollout
        g.selectAll(`.animation-ball-${rolloutId}`).remove()

        // Get the actual edges for this rollout
        let rolloutEdges: { node_a: string; node_b: string }[] = []
        if (Array.isArray(data.rollouts)) {
            const rollout = data.rollouts.find((r: any) => r.index.toString() === rolloutId)
            rolloutEdges = rollout?.edges || []
        } else {
            const rolloutData = data.rollouts[rolloutId]
            if (rolloutData && typeof rolloutData === 'object' && 'edges' in rolloutData) {
                rolloutEdges = (rolloutData as any).edges || []
            } else if (Array.isArray(rolloutData)) {
                rolloutEdges = rolloutData as { node_a: string; node_b: string }[]
            }
        }

        if (rolloutEdges.length === 0) return

        // Create the ball
        const ball = g.append('circle')
            .attr('class', `animation-ball-${rolloutId}`)
            .attr('r', 6)
            .attr('fill', '#ef4444')
            .attr('stroke', 'white')
            .attr('stroke-width', 2)
            .attr('opacity', 0.9)

        // Start at first edge's source node
        const firstEdge = rolloutEdges[0]
        const startPos = nodePositions.get(firstEdge.node_a)
        if (!startPos) return

        ball.attr('cx', startPos.x).attr('cy', startPos.y)

        // Animate along each edge
        let currentEdgeIndex = 0
        const animate = () => {
            if (currentEdgeIndex >= rolloutEdges.length) {
                currentEdgeIndex = 0 // Loop back to start
            }

            const edge = rolloutEdges[currentEdgeIndex]
            const sourcePos = nodePositions.get(edge.node_a)
            const targetPos = nodePositions.get(edge.node_b)

            if (sourcePos && targetPos) {
                // Move to source first (in case we're not there)
                ball.transition()
                    .duration(200)
                    .ease(d3.easeLinear)
                    .attr('cx', sourcePos.x)
                    .attr('cy', sourcePos.y)
                    .on('end', () => {
                        // Then move to target
                        ball.transition()
                            .duration(600)
                            .ease(d3.easeLinear)
                            .attr('cx', targetPos.x)
                            .attr('cy', targetPos.y)
                            .on('end', () => {
                                currentEdgeIndex++
                                animate() // Continue to next edge
                            })
                    })
            } else {
                currentEdgeIndex++
                animate()
            }
        }

        animate()
    }

    useEffect(() => {
        if (!svgRef.current || selectedRollouts.length === 0) return

        const svg = d3.select(svgRef.current)
        svg.selectAll('*').remove()

        // Initialize collections
        const edges: { node_a: string; node_b: string }[] = []
        const nodeSet = new Set<string>()
        const nodePositions = new Map<string, { x: number; y: number }>()
        const rolloutPaths = new Map<string, string[]>()

        // Build paths for each rollout, filtering out rollouts with no edges
        const newValidRollouts: string[] = []
        selectedRollouts.forEach((rolloutId, rolloutIndex) => {
            let rolloutEdges: { node_a: string; node_b: string }[] = []

            if (Array.isArray(data.rollouts)) {
                const rollout = data.rollouts.find((r: any) => r.index.toString() === rolloutId)
                rolloutEdges = rollout?.edges || []
            } else {
                const rolloutData = data.rollouts[rolloutId]
                if (rolloutData && typeof rolloutData === 'object' && 'edges' in rolloutData) {
                    rolloutEdges = (rolloutData as any).edges || []
                } else if (Array.isArray(rolloutData)) {
                    rolloutEdges = rolloutData as { node_a: string; node_b: string }[]
                }
            }

            // Only include rollouts that have edges
            if (rolloutEdges.length > 0) {
                newValidRollouts.push(rolloutId)
                edges.push(...rolloutEdges)
                rolloutEdges.forEach(edge => {
                    nodeSet.add(edge.node_a)
                    nodeSet.add(edge.node_b)
                })

                // Build path from edges
                const path = new Set<string>()
                rolloutEdges.forEach(edge => {
                    path.add(edge.node_a)
                    path.add(edge.node_b)
                })
                rolloutPaths.set(rolloutId, Array.from(path))
            }
        })

        if (edges.length === 0) {
            svg.append('text')
                .attr('x', 400)
                .attr('y', 200)
                .attr('text-anchor', 'middle')
                .attr('font-size', '18px')
                .attr('fill', '#6b7280')
                .text('No edges found for selected rollouts')
            return
        }

        // Create node lookup
        const nodeMap = new Map<string, any>()
        data.nodes.forEach(node => {
            if (nodeSet.has(node.cluster_id)) {
                nodeMap.set(node.cluster_id, node)
            }
        })

        // Update valid rollouts state
        setValidRollouts(newValidRollouts)

        // Calculate dimensions
        const nodes = Array.from(nodeSet)
        const width = 1600
        const height = 600

        svg.attr('width', width).attr('height', height)

        const g = svg.append('g')

        // Add zoom behavior
        const zoom = d3.zoom<SVGSVGElement, unknown>()
            .scaleExtent([0.1, 4])
            .on('zoom', (event) => {
                g.attr('transform', event.transform)
                setCurrentTransform(event.transform)
            })

        svg.call(zoom)

        // Apply saved transform if it exists
        if (currentTransform) {
            g.attr('transform', currentTransform.toString())
            svg.call(zoom.transform, currentTransform)
        }

        // Position nodes in tree layout
        const nodeToRollouts = new Map<string, string[]>()
        rolloutPaths.forEach((path, rolloutId) => {
            path.forEach(nodeId => {
                if (!nodeToRollouts.has(nodeId)) {
                    nodeToRollouts.set(nodeId, [])
                }
                nodeToRollouts.get(nodeId)!.push(rolloutId)
            })
        })

        // Create hierarchical layout with multiple vertical levels
        const allNodes = Array.from(nodeSet)

        // Group nodes by their position in rollout paths
        const nodeLevels = new Map<string, number>()
        const maxLevel = Math.max(...Array.from(rolloutPaths.values()).map(path => path.length - 1))

        // Assign levels based on position in rollout paths, with COT nodes first
        rolloutPaths.forEach((path, rolloutId) => {
            path.forEach((nodeId, index) => {
                // COT nodes (blue) should come before Response nodes (red/green)
                const isResponseNode = nodeId.startsWith('response-')
                const adjustedIndex = isResponseNode ? index + 1000 : index // Push Response nodes to later levels

                if (!nodeLevels.has(nodeId) || nodeLevels.get(nodeId)! > adjustedIndex) {
                    nodeLevels.set(nodeId, adjustedIndex)
                }
            })
        })

        // Position nodes in hierarchical layout
        const levelGroups = new Map<number, string[]>()
        allNodes.forEach(nodeId => {
            const level = nodeLevels.get(nodeId) || 0
            if (!levelGroups.has(level)) {
                levelGroups.set(level, [])
            }
            levelGroups.get(level)!.push(nodeId)
        })

        // Calculate positions for each level
        const sortedLevels = Array.from(levelGroups.keys()).sort((a, b) => a - b)
        const levelSpacing = Math.max(300, width / (sortedLevels.length + 1)) // Increased horizontal spacing
        const nodeSpacing = 80 // Reverted to original vertical spacing

        sortedLevels.forEach((level, levelIndex) => {
            const nodesInLevel = levelGroups.get(level)!
            const x = levelIndex * levelSpacing + 100
            const startY = (height - (nodesInLevel.length - 1) * nodeSpacing) / 2

            nodesInLevel.forEach((nodeId, index) => {
                const y = startY + index * nodeSpacing
                nodePositions.set(nodeId, { x, y })
            })
        })

        // Add arrow marker
        const defs = svg.append('defs')
        defs.append('marker')
            .attr('id', 'arrowhead')
            .attr('viewBox', '0 -5 10 10')
            .attr('refX', 15)
            .attr('refY', 0)
            .attr('markerWidth', 6)
            .attr('markerHeight', 6)
            .attr('orient', 'auto')
            .append('path')
            .attr('d', 'M0,-5L10,0L0,5')
            .attr('fill', '#000000')

        // Draw edges for each rollout
        const edgeGroup = g.append('g')

        newValidRollouts.forEach((rolloutId, index) => {
            let rolloutEdges: { node_a: string; node_b: string }[] = []

            if (Array.isArray(data.rollouts)) {
                const rollout = data.rollouts.find((r: any) => r.index.toString() === rolloutId)
                rolloutEdges = rollout?.edges || []
            } else {
                const rolloutData = data.rollouts[rolloutId]
                if (rolloutData && typeof rolloutData === 'object' && 'edges' in rolloutData) {
                    rolloutEdges = (rolloutData as any).edges || []
                } else if (Array.isArray(rolloutData)) {
                    rolloutEdges = rolloutData as { node_a: string; node_b: string }[]
                }
            }

            rolloutEdges.forEach(edge => {
                const sourcePos = nodePositions.get(edge.node_a)
                const targetPos = nodePositions.get(edge.node_b)

                if (sourcePos && targetPos) {
                    const isHighlighted = hoveredRollout === rolloutId
                    edgeGroup.append('line')
                        .attr('x1', sourcePos.x)
                        .attr('y1', sourcePos.y)
                        .attr('x2', targetPos.x)
                        .attr('y2', targetPos.y)
                        .attr('stroke', isHighlighted ? '#3b82f6' : '#d1d5db')
                        .attr('stroke-width', isHighlighted ? 4 : 2)
                        .attr('opacity', isHighlighted ? 1 : 0.6)
                        .attr('marker-end', 'url(#arrowhead)')
                        .attr('class', `rollout-${rolloutId}`)
                        .style('cursor', 'pointer')
                }
            })
        })

        // Draw nodes
        const nodeGroup = g.append('g')
        nodes.forEach(nodeId => {
            const pos = nodePositions.get(nodeId)
            if (!pos) return

            const node = nodeMap.get(nodeId)
            const displayText = node ? node.representative_sentence : nodeId

            // Set node dimensions based on type
            const isResponseNode = nodeId.startsWith('response-')
            const nodeWidth = isResponseNode ? 120 : 200
            const nodeHeight = isResponseNode ? 30 : 50

            const nodeElement = nodeGroup.append('g')
                .attr('transform', `translate(${pos.x - nodeWidth / 2}, ${pos.y - nodeHeight / 2})`)

            // Determine if this is a response node and get appropriate colors
            let isCorrectAnswer = false
            if (isResponseNode) {
                // Find the rollout that contains this response node to get its correctness
                for (const rolloutId of newValidRollouts) {
                    let rolloutEdges: { node_a: string; node_b: string }[] = []
                    let rolloutCorrectness = false

                    if (Array.isArray(data.rollouts)) {
                        const rollout = data.rollouts.find((r: any) => r.index.toString() === rolloutId)
                        rolloutEdges = rollout?.edges || []
                        rolloutCorrectness = rollout?.correctness || false
                    } else {
                        const rolloutData = data.rollouts[rolloutId] as RolloutObject
                        if (rolloutData && typeof rolloutData === 'object' && 'edges' in rolloutData) {
                            rolloutEdges = rolloutData.edges || []
                            rolloutCorrectness = rolloutData.correctness || false
                        } else if (Array.isArray(rolloutData)) {
                            rolloutEdges = rolloutData as { node_a: string; node_b: string }[]
                        }
                    }

                    // Check if this rollout contains the response node
                    if (rolloutEdges.some(edge => edge.node_b === nodeId)) {
                        isCorrectAnswer = rolloutCorrectness
                        break
                    }
                }
            }

            const fillColor = isResponseNode ? (isCorrectAnswer ? '#10b981' : '#ef4444') : '#3b82f6'
            const strokeColor = isResponseNode ? (isCorrectAnswer ? '#059669' : '#dc2626') : '#1e40af'

            // Draw rounded rectangle background
            nodeElement.append('rect')
                .attr('width', nodeWidth)
                .attr('height', nodeHeight)
                .attr('rx', 15)
                .attr('ry', 15)
                .attr('fill', fillColor)
                .attr('stroke', strokeColor)
                .attr('stroke-width', 2)

            // Function to wrap text to two lines
            const wrapText = (text: string, maxWidth: number) => {
                const words = text.split(' ')
                const lines = []
                let currentLine = ''

                // Use different character width estimates for different node types
                const charWidth = isResponseNode ? 4.5 : 5.5

                for (const word of words) {
                    const testLine = currentLine + (currentLine ? ' ' : '') + word
                    if (testLine.length * charWidth <= maxWidth) {
                        currentLine = testLine
                    } else {
                        if (currentLine) {
                            lines.push(currentLine)
                            currentLine = word
                        } else {
                            lines.push(word)
                        }
                    }
                }
                if (currentLine) {
                    lines.push(currentLine)
                }

                return lines.slice(0, 2) // Max 2 lines
            }

            const wrappedLines = wrapText(displayText, nodeWidth - 20) // 20px padding
            const fontSize = isResponseNode ? '9px' : '10px'
            const lineSpacing = isResponseNode ? 10 : 12

            // Add text lines
            wrappedLines.forEach((line, index) => {
                nodeElement.append('text')
                    .attr('x', 10) // Left padding
                    .attr('y', nodeHeight / 2 - 5 + (index * lineSpacing))
                    .attr('dy', '0.35em')
                    .attr('font-size', fontSize)
                    .attr('font-weight', '600')
                    .attr('fill', 'white')
                    .attr('text-anchor', 'start')
                    .text(line)
            })

            // Add click handler for node details
            nodeElement.on('click', () => {
                const node = nodeMap.get(nodeId)
                if (node) {
                    setSelectedNode(node)
                }
            })
        })


    }, [data, selectedRollouts, hoveredRollout])

    if (selectedRollouts.length === 0) {
        return (
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%' }}>
                <div style={{ color: '#6b7280', fontSize: '1.125rem' }}>
                    Select rollouts to see the visualization
                </div>
            </div>
        )
    }

    return (
        <div style={{ width: '100%', height: '100%', position: 'relative' }}>
            {/* Rollout slider */}
            {validRollouts.length > 0 && (
                <div style={{
                    position: 'absolute',
                    top: '20px',
                    left: '20px',
                    zIndex: 1000,
                    backgroundColor: 'white',
                    border: '1px solid #e5e7eb',
                    borderRadius: '8px',
                    padding: '12px',
                    boxShadow: '0 4px 12px rgba(0,0,0,0.1)'
                }}>
                    <div style={{ marginBottom: '8px', fontSize: '14px', fontWeight: '600' }}>
                        Rollout: {hoveredRollout || 'None'}
                    </div>
                    <input
                        type="range"
                        min="0"
                        max={validRollouts.length - 1}
                        value={validRollouts.indexOf(hoveredRollout || '')}
                        onChange={(e) => {
                            const index = parseInt(e.target.value)
                            const rolloutId = validRollouts[index]
                            setHoveredRollout(rolloutId)
                        }}
                        style={{ width: '200px' }}
                    />
                    <div style={{ fontSize: '12px', color: '#6b7280', marginTop: '4px' }}>
                        {validRollouts.length} rollouts available
                    </div>
                </div>
            )}

            <svg ref={svgRef} style={{ width: '100%', height: '100%' }} />

            {selectedNode && (
                <div style={{
                    position: 'absolute',
                    top: '20px',
                    left: '20px',
                    width: '300px',
                    maxHeight: '400px',
                    backgroundColor: 'white',
                    border: '2px solid #ccc',
                    borderRadius: '8px',
                    padding: '16px',
                    boxShadow: '0 4px 12px rgba(0,0,0,0.15)',
                    zIndex: 20000
                }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
                        <h3 style={{ margin: 0, fontSize: '16px' }}>Node {selectedNode.cluster_id}</h3>
                        <button
                            onClick={() => setSelectedNode(null)}
                            style={{
                                background: 'none',
                                border: 'none',
                                fontSize: '18px',
                                cursor: 'pointer',
                                color: '#666'
                            }}
                        >
                            ×
                        </button>
                    </div>

                    <div style={{ marginBottom: '12px' }}>
                        <strong>Representative:</strong>
                        <p style={{ margin: '4px 0', fontSize: '14px', color: '#333' }}>
                            {selectedNode.representative_sentence}
                        </p>
                    </div>

                    <div style={{ marginBottom: '12px' }}>
                        <strong>Sentences ({selectedNode.sentences.length}):</strong>
                    </div>

                    <div style={{
                        maxHeight: '250px',
                        overflowY: 'auto',
                        overflowX: 'hidden',
                        border: '1px solid #eee',
                        borderRadius: '4px',
                        padding: '8px',
                        backgroundColor: '#f9f9f9',
                        wordWrap: 'break-word',
                        wordBreak: 'break-word'
                    }}>
                        {selectedNode.sentences.map((sentence, index) => (
                            <div key={index} style={{
                                marginBottom: '8px',
                                padding: '6px',
                                backgroundColor: 'white',
                                borderRadius: '4px',
                                fontSize: '12px',
                                border: '1px solid #e5e7eb',
                                wordWrap: 'break-word',
                                wordBreak: 'break-word',
                                overflowWrap: 'break-word'
                            }}>
                                <div style={{ fontWeight: 'bold', marginBottom: '2px' }}>
                                    Count: {sentence.count}
                                </div>
                                <div style={{
                                    color: '#555',
                                    wordWrap: 'break-word',
                                    wordBreak: 'break-word',
                                    overflowWrap: 'break-word',
                                    whiteSpace: 'pre-wrap'
                                }}>
                                    {sentence.text}
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            )}
        </div>
    )
}