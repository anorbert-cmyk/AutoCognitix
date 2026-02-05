// Neo4j Schema and Index Setup for AutoCognitix
// ================================================
// Run these commands in Neo4j Browser or cypher-shell
//
// Usage:
//   cypher-shell -f scripts/neo4j_indexes.cypher
//   OR copy/paste into Neo4j Browser
// ================================================

// ============================================
// UNIQUE CONSTRAINTS
// ============================================
// DTCNode.code must be unique (primary lookup key)

CREATE CONSTRAINT dtcnode_code_unique IF NOT EXISTS
FOR (n:DTCNode)
REQUIRE n.code IS UNIQUE;

// ============================================
// PRIMARY LOOKUP INDEXES
// ============================================
// These indexes are critical for fast node lookups

// Symptom name index - used for diagnostic matching
CREATE INDEX symptomnode_name_idx IF NOT EXISTS
FOR (n:SymptomNode)
ON (n.name);

// Component name index - used for repair path traversal
CREATE INDEX componentnode_name_idx IF NOT EXISTS
FOR (n:ComponentNode)
ON (n.name);

// Repair name index - used for finding repair actions
CREATE INDEX repairnode_name_idx IF NOT EXISTS
FOR (n:RepairNode)
ON (n.name);

// Part indexes - used for parts lookup
CREATE INDEX partnode_name_idx IF NOT EXISTS
FOR (n:PartNode)
ON (n.name);

CREATE INDEX partnode_part_number_idx IF NOT EXISTS
FOR (n:PartNode)
ON (n.part_number);

// TestPoint name index
CREATE INDEX testpointnode_name_idx IF NOT EXISTS
FOR (n:TestPointNode)
ON (n.name);

// Vehicle indexes - used for vehicle-specific lookups
CREATE INDEX vehiclenode_make_idx IF NOT EXISTS
FOR (n:VehicleNode)
ON (n.make);

CREATE INDEX vehiclenode_model_idx IF NOT EXISTS
FOR (n:VehicleNode)
ON (n.model);

// ============================================
// CATEGORY/FILTERING INDEXES
// ============================================
// These indexes support filtering queries

// DTC category index (powertrain, body, chassis, network)
CREATE INDEX dtcnode_category_idx IF NOT EXISTS
FOR (n:DTCNode)
ON (n.category);

// DTC severity index (low, medium, high, critical)
CREATE INDEX dtcnode_severity_idx IF NOT EXISTS
FOR (n:DTCNode)
ON (n.severity);

// Component system index (engine, transmission, brakes, etc.)
CREATE INDEX componentnode_system_idx IF NOT EXISTS
FOR (n:ComponentNode)
ON (n.system);

// ============================================
// COMPOSITE INDEXES
// ============================================
// For common multi-property query patterns

// Vehicle make+model composite index
CREATE INDEX vehicle_make_model_idx IF NOT EXISTS
FOR (n:VehicleNode)
ON (n.make, n.model);

// ============================================
// FULL-TEXT INDEXES
// ============================================
// For Hungarian language search support

// DTC description fulltext search (Hungarian + English)
CREATE FULLTEXT INDEX dtc_description_hu_idx IF NOT EXISTS
FOR (n:DTCNode)
ON EACH [n.description_hu, n.description_en];

// Symptom description fulltext search
CREATE FULLTEXT INDEX symptom_description_hu_idx IF NOT EXISTS
FOR (n:SymptomNode)
ON EACH [n.description, n.description_hu];

// Component name fulltext search (Hungarian + English)
CREATE FULLTEXT INDEX component_name_hu_idx IF NOT EXISTS
FOR (n:ComponentNode)
ON EACH [n.name, n.name_hu];

// Repair description fulltext search
CREATE FULLTEXT INDEX repair_description_hu_idx IF NOT EXISTS
FOR (n:RepairNode)
ON EACH [n.name, n.description_hu];

// ============================================
// VERIFICATION QUERIES
// ============================================
// Run these to verify indexes were created

// Show all indexes
// SHOW INDEXES;

// Show all constraints
// SHOW CONSTRAINTS;

// Test DTC lookup performance
// PROFILE MATCH (d:DTCNode {code: 'P0420'}) RETURN d;

// Test symptom search
// PROFILE MATCH (s:SymptomNode {name: 'Check Engine Light'}) RETURN s;

// Test fulltext search
// CALL db.index.fulltext.queryNodes('dtc_description_hu_idx', 'oxigen') YIELD node, score
// RETURN node.code, node.description_hu, score
// ORDER BY score DESC LIMIT 10;

// ============================================
// USEFUL DIAGNOSTIC QUERIES
// ============================================

// Get DTC with all related symptoms and components
// MATCH (d:DTCNode {code: 'P0420'})
// OPTIONAL MATCH (d)-[r1:CAUSES]->(s:SymptomNode)
// OPTIONAL MATCH (d)-[r2:INDICATES_FAILURE_OF]->(c:ComponentNode)
// RETURN d, collect(DISTINCT s) as symptoms, collect(DISTINCT c) as components;

// Find repair path for a DTC
// MATCH (d:DTCNode {code: 'P0420'})-[:INDICATES_FAILURE_OF]->(c:ComponentNode)
//       -[:REPAIRED_BY]->(r:RepairNode)
// OPTIONAL MATCH (r)-[:USES_PART]->(p:PartNode)
// RETURN d.code, c.name, r.name, collect(p.name) as parts;

// Get statistics
// MATCH (n) RETURN labels(n)[0] as label, count(*) as count;
