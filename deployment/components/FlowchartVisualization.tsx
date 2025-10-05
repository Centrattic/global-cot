'use client'

import { useEffect, useRef } from 'react'
import * as d3 from 'd3'
import { FlowchartData } from '@/types/flowchart'

interface FlowchartVisualizationProps {
    data: FlowchartData
    selectedRollouts: string[]
}

export default function FlowchartVisualization({ data, selectedRollouts }: FlowchartVisualizationProps) {
    const svgRef = useRef<SVGSVGElement>(null)

    useEffect(() => {
        if (!svgRef.current || selectedRollouts.length === 0) return

        const svg = d3.select(svgRef.current)
        svg.selectAll('*').remove()

        // Get edges for selected rollouts
        const edges: { node_a: string; node_b: string }[] = []
        const nodeSet = new Set<string>()

        selectedRollouts.forEach(rolloutId => {
            let rolloutEdges: { node_a: string; node_b: string }[] = []

            if (Array.isArray(data.rollouts)) {
                // Old format: array of rollouts
                const rollout = data.rollouts.find((r: any) => r.index === rolloutId)
                rolloutEdges = rollout?.edges || []
            } else {
                // New format: object with rollout IDs as keys
                rolloutEdges = data.rollouts[rolloutId] || []
            }

            edges.push(...rolloutEdges)
            rolloutEdges.forEach(edge => {
                nodeSet.add(edge.node_a)
                nodeSet.add(edge.node_b)
            })
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

        // Calculate dimensions
        const nodes = Array.from(nodeSet)
        const width = 800
        const height = Math.max(400, nodes.length * 80 + 100)

        svg.attr('width', width).attr('height', height)

        // Add zoom behavior
        const zoom = d3.zoom<SVGSVGElement, unknown>()
            .scaleExtent([0.1, 4])
            .on('zoom', (event) => {
                g.attr('transform', event.transform)
            })

        svg.call(zoom)

        const g = svg.append('g')

        // Position nodes vertically
        const nodePositions = new Map<string, { x: number; y: number }>()
        nodes.forEach((nodeId, index) => {
            nodePositions.set(nodeId, {
                x: width / 2,
                y: index * 80 + 50
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

        // Draw edges
        const edgeGroup = g.append('g')
        edges.forEach(edge => {
            const sourcePos = nodePositions.get(edge.node_a)
            const targetPos = nodePositions.get(edge.node_b)

            if (sourcePos && targetPos) {
                edgeGroup.append('line')
                    .attr('x1', sourcePos.x)
                    .attr('y1', sourcePos.y)
                    .attr('x2', targetPos.x)
                    .attr('y2', targetPos.y)
                    .attr('stroke', '#000000')
                    .attr('stroke-width', 2)
                    .attr('opacity', 0.7)
                    .attr('marker-end', 'url(#arrowhead)')
            }
        })

        // Draw nodes
        const nodeGroup = g.append('g')
        nodes.forEach(nodeId => {
            const pos = nodePositions.get(nodeId)
            if (!pos) return

            const nodeElement = nodeGroup.append('g')
                .attr('transform', `translate(${pos.x}, ${pos.y})`)

            nodeElement.append('circle')
                .attr('r', 15)
                .attr('fill', '#3b82f6')
                .attr('stroke', '#1e40af')
                .attr('stroke-width', 2)

            nodeElement.append('text')
                .attr('dy', '0.35em')
                .attr('font-size', '12px')
                .attr('font-weight', '600')
                .attr('fill', 'white')
                .text(nodeId)

            // Add click handler for node details
            nodeElement.on('click', () => {
                const node = nodeMap.get(nodeId)
                if (node) {
                    alert(`Node ${nodeId}\nRepresentative: ${node.representative_sentence}`)
                }
            })
        })

        // Add node labels on the left
        nodes.forEach((nodeId, index) => {
            g.append('text')
                .attr('x', 20)
                .attr('y', index * 80 + 50)
                .attr('dy', '0.35em')
                .attr('font-size', '12px')
                .attr('fill', '#6b7280')
                .text(`Node ${nodeId}`)
        })

    }, [data, selectedRollouts])

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
        <div style={{ width: '100%', height: '100%' }}>
            <svg ref={svgRef} style={{ width: '100%', height: '100%' }} />
        </div>
    )
}