// AURA Shape-Aware Memory V2 - Neo4j Schema
// Based on nowlookatthispart.md blueprint
// This schema enables sub-5ms shape similarity searches using FastRP + k-NN

// ============================================
// Core Node Types
// ============================================

// ConversationShape: The fundamental unit of topological memory
CREATE CONSTRAINT conversation_shape_id IF NOT EXISTS
ON (cs:ConversationShape) ASSERT cs.id IS UNIQUE;

CREATE INDEX conversation_shape_status IF NOT EXISTS
FOR (cs:ConversationShape) ON (cs.status);

CREATE INDEX conversation_shape_created IF NOT EXISTS
FOR (cs:ConversationShape) ON (cs.created_at);

// Shape properties as defined in nowlookatthispart.md:
// (:ConversationShape {
//     id: <uuid>,
//     created_at: <timestamp>,
//     betti_0: int,
//     betti_1: int,
//     betti_2: int,
//     ph_summary: map,        // compact stats for quick filters
//     fastrp: Vector<Float>,  // 256-D default
//     status: ["known_fail", "warning", "normal"]
// })

// ============================================
// Relationship Types
// ============================================

// SIMILAR: Pre-computed k-NN relationships
CREATE INDEX similar_score IF NOT EXISTS
FOR ()-[s:SIMILAR]->() ON (s.score);

// FOLLOWED_BY: Temporal sequence relationships
// EVOLVED_TO: Tracks shape evolution over time

// ============================================
// Graph Projections for GDS
// ============================================

// Note: These projections will be created via GDS Python API
// 1. Memory projection for FastRP computation
// 2. Similarity projection for real-time queries

// ============================================
// Performance Indexes
// ============================================

CREATE INDEX shape_status_created IF NOT EXISTS
FOR (cs:ConversationShape) ON (cs.status, cs.created_at);

// For migration from existing shape_aware_memory.py
CREATE INDEX legacy_shape_id IF NOT EXISTS
FOR (cs:ConversationShape) ON (cs.legacy_id);