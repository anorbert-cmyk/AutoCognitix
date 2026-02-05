// =============================================================================
// AutoCognitix Neo4j Schema Definition
// =============================================================================
// This file contains all Cypher commands needed to set up the Neo4j database
// schema for the AutoCognitix diagnostic system.
//
// Usage:
//   1. Connect to Neo4j with cypher-shell or Neo4j Browser
//   2. Execute this file: cypher-shell -u neo4j -p <password> -f neo4j_schema.cypher
//   Or run via Python: python scripts/setup_neo4j_indexes.py
//
// Node Labels:
//   - DTCNode: Diagnostic Trouble Codes
//   - SymptomNode: Vehicle symptoms
//   - ComponentNode: Vehicle components
//   - RepairNode: Repair actions
//   - PartNode: Replacement parts
//   - TestPointNode: Diagnostic test points
//   - VehicleNode: Vehicle types (make/model combinations)
//
// Relationship Types:
//   - CAUSES: DTC -> Symptom
//   - INDICATES_FAILURE_OF: DTC -> Component
//   - RELATED_TO: DTC -> DTC
//   - REQUIRES_CHECK: Symptom -> TestPoint
//   - REPAIRED_BY: Component -> Repair
//   - CONTAINS: Component -> Component (hierarchy)
//   - USES_PART: Repair -> Part
//   - LEADS_TO: TestPoint -> Repair
//   - HAS_COMMON_ISSUE: Vehicle -> DTC
//   - USES_COMPONENT: Vehicle -> Component
// =============================================================================

// =============================================================================
// SECTION 1: UNIQUE CONSTRAINTS
// =============================================================================
// Unique constraints also create an index automatically

// DTCNode - code must be unique (e.g., P0300, B1234)
CREATE CONSTRAINT dtcnode_code_unique IF NOT EXISTS
FOR (n:DTCNode)
REQUIRE n.code IS UNIQUE;

// =============================================================================
// SECTION 2: PROPERTY INDEXES (B-tree)
// =============================================================================
// Indexes for fast lookups on commonly queried properties

// --- SymptomNode indexes ---
CREATE INDEX symptomnode_name_idx IF NOT EXISTS
FOR (n:SymptomNode)
ON (n.name);

// --- ComponentNode indexes ---
CREATE INDEX componentnode_name_idx IF NOT EXISTS
FOR (n:ComponentNode)
ON (n.name);

CREATE INDEX componentnode_system_idx IF NOT EXISTS
FOR (n:ComponentNode)
ON (n.system);

// --- RepairNode indexes ---
CREATE INDEX repairnode_name_idx IF NOT EXISTS
FOR (n:RepairNode)
ON (n.name);

// --- PartNode indexes ---
CREATE INDEX partnode_name_idx IF NOT EXISTS
FOR (n:PartNode)
ON (n.name);

CREATE INDEX partnode_part_number_idx IF NOT EXISTS
FOR (n:PartNode)
ON (n.part_number);

// --- TestPointNode indexes ---
CREATE INDEX testpointnode_name_idx IF NOT EXISTS
FOR (n:TestPointNode)
ON (n.name);

// --- VehicleNode indexes ---
CREATE INDEX vehiclenode_make_idx IF NOT EXISTS
FOR (n:VehicleNode)
ON (n.make);

CREATE INDEX vehiclenode_model_idx IF NOT EXISTS
FOR (n:VehicleNode)
ON (n.model);

// --- DTCNode category/severity indexes ---
CREATE INDEX dtcnode_category_idx IF NOT EXISTS
FOR (n:DTCNode)
ON (n.category);

CREATE INDEX dtcnode_severity_idx IF NOT EXISTS
FOR (n:DTCNode)
ON (n.severity);

// =============================================================================
// SECTION 3: COMPOSITE INDEXES
// =============================================================================
// Composite indexes for common multi-property queries

// Vehicle lookup by make and model together
CREATE INDEX vehicle_make_model_idx IF NOT EXISTS
FOR (n:VehicleNode)
ON (n.make, n.model);

// =============================================================================
// SECTION 4: FULL-TEXT SEARCH INDEXES
// =============================================================================
// Full-text indexes for natural language search (Hungarian + English)

// DTC descriptions - for searching by symptom description
CREATE FULLTEXT INDEX dtc_description_hu_idx IF NOT EXISTS
FOR (n:DTCNode)
ON EACH [n.description_hu, n.description_en];

// Symptom descriptions - for user symptom input matching
CREATE FULLTEXT INDEX symptom_description_hu_idx IF NOT EXISTS
FOR (n:SymptomNode)
ON EACH [n.description, n.description_hu];

// Component names - for part/component search
CREATE FULLTEXT INDEX component_name_hu_idx IF NOT EXISTS
FOR (n:ComponentNode)
ON EACH [n.name, n.name_hu];

// Repair descriptions - for repair procedure search
CREATE FULLTEXT INDEX repair_description_hu_idx IF NOT EXISTS
FOR (n:RepairNode)
ON EACH [n.name, n.description_hu];

// =============================================================================
// SECTION 5: RELATIONSHIP PROPERTY INDEXES (Neo4j 5.x+)
// =============================================================================
// These require Neo4j 5.x or later

// Index on CAUSES relationship confidence for filtering high-confidence associations
// CREATE INDEX causes_confidence_idx IF NOT EXISTS
// FOR ()-[r:CAUSES]-()
// ON (r.confidence);

// Index on HAS_COMMON_ISSUE frequency for filtering common issues
// CREATE INDEX has_common_issue_frequency_idx IF NOT EXISTS
// FOR ()-[r:HAS_COMMON_ISSUE]-()
// ON (r.frequency);

// =============================================================================
// SECTION 6: VERIFICATION QUERIES
// =============================================================================
// Run these to verify the schema was created correctly

// Show all indexes
// SHOW INDEXES;

// Show all constraints
// SHOW CONSTRAINTS;

// Count nodes by label
// CALL db.labels() YIELD label
// CALL apoc.cypher.run('MATCH (n:' + label + ') RETURN count(n) as count', {}) YIELD value
// RETURN label, value.count as count;

// =============================================================================
// SECTION 7: SAMPLE DATA QUERIES (for testing)
// =============================================================================
// These are example queries - uncomment to test the schema

// Create a sample DTC node
// CREATE (d:DTCNode {
//   code: 'P0300',
//   description_en: 'Random/Multiple Cylinder Misfire Detected',
//   description_hu: 'Veletlenszeru/Tobbszoros henger kihagyas eszlelve',
//   category: 'powertrain',
//   severity: 'high',
//   is_generic: 'true',
//   system: 'Engine'
// });

// Create a sample symptom and relationship
// MATCH (d:DTCNode {code: 'P0300'})
// CREATE (s:SymptomNode {
//   name: 'Engine rough idle',
//   description: 'Engine runs rough at idle',
//   description_hu: 'A motor durvan jar alapjaron'
// })
// CREATE (d)-[:CAUSES {confidence: 0.8, data_source: 'generic'}]->(s);

// Query: Find all symptoms for a DTC
// MATCH (d:DTCNode {code: 'P0300'})-[r:CAUSES]->(s:SymptomNode)
// RETURN d.code, s.name, r.confidence;

// Query: Full-text search for Hungarian symptoms
// CALL db.index.fulltext.queryNodes('symptom_description_hu_idx', 'motor durva')
// YIELD node, score
// RETURN node.name, node.description_hu, score;

// Query: Find diagnostic path for a DTC
// MATCH path = (d:DTCNode {code: 'P0300'})-[:INDICATES_FAILURE_OF]->(c:ComponentNode)
//              -[:REPAIRED_BY]->(r:RepairNode)-[:USES_PART]->(p:PartNode)
// RETURN path;

// =============================================================================
// END OF SCHEMA DEFINITION
// =============================================================================
