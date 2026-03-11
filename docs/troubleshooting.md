# Troubleshooting Guide

## 1. Neo4j Connection Failed

**Symptom**: `ServiceUnavailable: Unable to retrieve routing information`

**Solutions**:
```bash
# Check if Neo4j is running
docker-compose ps

# View logs
docker-compose logs -f neo4j

# Restart
docker-compose restart neo4j

# Verify connectivity
curl http://localhost:7474
```

If running Neo4j natively (not Docker), verify the bolt port:
```bash
lsof -i:7687
```

## 2. Neo4j Property Type Error

**Symptom**: `Property values can only be of primitive types or arrays of primitive types`

**Cause**: graphiti-core sometimes produces nested dicts/lists as node properties, which Neo4j cannot store.

**Solution**: This is handled automatically by the `flatten_value()` and `fix_field_names()` utilities in `graphiti_zep/utils.py`. They convert nested objects to JSON strings before Neo4j write. If you still see this error, it may be a new edge case — please file an issue.

Related: [graphiti-core Issue #683](https://github.com/getzep/graphiti/issues/683)

## 3. LLM Request Timeout

**Symptom**: `Request timed out` during episode ingestion

**Causes**:
- LLM provider is slow (each episode triggers ~15 LLM calls for entity/edge extraction)
- Network issues

**Solutions**:
- The server retries timeout errors automatically (up to 6 attempts with exponential backoff)
- For Anthropic-style APIs, the default timeout is 900s
- For OpenAI-style APIs, the default timeout is 300s
- If timeouts persist, check your provider's status page

## 4. Rate Limiting (429)

**Symptom**: `429 Too Many Requests` or `rate_limit_error`

**Cause**: Too many LLM calls in a short period. Large documents with many batches can exhaust API rate limits.

**Solutions**:
- The server retries rate-limited requests automatically (up to 6 attempts, max 180s backoff)
- Reduce batch concurrency by sending fewer episodes per batch
- Use a higher-tier API plan
- Add a delay between batches on the client side

## 5. DashScope Embedding Batch Error

**Symptom**: `400 - batch size is invalid, it should not be larger than 10`

**Cause**: DashScope API limits embedding requests to 10 texts per batch.

**Solution**: Set `EMBEDDING_BATCH_SIZE=10` in your `.env` file. The server will automatically split large batches.

## 6. Empty Search Results

**Symptom**: `search()` returns no results

**Possible causes**:
1. Wrong `group_id` — verify with `GET /v1/groups/{group_id}/nodes`
2. No episodes ingested yet
3. Query doesn't match any extracted entities/edges

**Debug**:
```bash
# Check if nodes exist
curl http://localhost:8000/v1/groups/YOUR_GROUP_ID/nodes \
  -H "Authorization: Bearer local-graphiti"

# Check edges
curl http://localhost:8000/v1/groups/YOUR_GROUP_ID/edges \
  -H "Authorization: Bearer local-graphiti"
```

Or query Neo4j directly:
```cypher
MATCH (n:Entity) WHERE n.group_id = "your_group_id" RETURN n LIMIT 10
```

## 7. Slow Startup

**Symptom**: Server takes a long time to start (especially first time)

**Cause**: `build_indices_and_constraints()` creates Neo4j indexes on first run.

**Solution**: This is normal. Subsequent starts will be fast. If it hangs indefinitely, check Neo4j connectivity.

## 8. LLM Returns Schema Instead of Data

**Symptom**: Extracted entities contain schema definitions instead of actual data (e.g., `{"properties": {"extracted_entities": []}}`)

**Cause**: Some OpenAI-compatible providers (e.g., Gemini) return the JSON schema structure instead of conforming to it.

**Solution**: The `unwrap_structured_payload()` utility handles this automatically by detecting and unwrapping provider wrapper objects. If you see a new wrapper pattern, please file an issue.

## 9. Field Name Mismatch

**Symptom**: Validation errors like `missing field 'extracted_entities'` when the LLM used a different name like `entities` or `nodes`

**Cause**: LLMs sometimes use synonyms for field names.

**Solution**: The `fix_field_names()` utility handles this automatically with:
- Substring matching (`entity_name` → `name`)
- Forced 1-to-1 mapping when exactly one field is missing and one is extra

## 10. Authentication Error (401)

**Symptom**: `401 Unauthorized`

**Solution**: Ensure the `Authorization` header matches the `GRAPHITI_API_KEY` in `.env`:
```bash
curl -H "Authorization: Bearer local-graphiti" http://localhost:8000/healthcheck
```

## 11. Neo4j Memory Issues

**Symptom**: Neo4j OOM or slow queries on large graphs

**Solution**: Tune Neo4j memory in `docker-compose.yml`:
```yaml
environment:
  NEO4J_dbms_memory_heap_initial__size: 1G
  NEO4J_dbms_memory_heap_max__size: 4G
deploy:
  resources:
    limits:
      memory: 6G
```

## Diagnostic Commands

```bash
# Check all services
curl http://localhost:8000/healthcheck | python3 -m json.tool

# Count entities in a graph
curl http://localhost:8000/v1/groups/YOUR_GROUP/nodes?limit=1 \
  -H "Authorization: Bearer local-graphiti"

# Neo4j Browser (if using Docker)
open http://localhost:7474

# View server logs
tail -f /tmp/graphiti-zep.log
```
