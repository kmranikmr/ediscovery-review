# Debug Mode Configuration

## Overview

The FastAPI server supports a debug mode that provides detailed logging for QA operations, including request details, OpenSearch queries, and response information.

## Environment Variable

**`DEBUG_MODE`** - Controls whether detailed debug logging is enabled

- **Default**: `false` (disabled)
- **Values**: `true` or `false` (case-insensitive)

## Usage Examples

### Enable Debug Mode

```bash
# For current session
export DEBUG_MODE=true
python main.py

# Or inline
DEBUG_MODE=true python main.py

# For Docker
docker run -e DEBUG_MODE=true your-image

# For docker-compose
environment:
  - DEBUG_MODE=true
```

### Disable Debug Mode (Default)

```bash
# For current session
export DEBUG_MODE=false
python main.py

# Or simply run without setting (defaults to false)
python main.py
```

## Debug Output

When `DEBUG_MODE=true`, you'll see detailed logging for:

### Family QA (`/qa/family`)
```
=== FAMILY QA REQUEST DEBUG ===
Query: What are the main discussion points?
Index: deephousedeephouse_ediscovery_docs_chunks
Top K: 10
Filters: {}
Documents provided: 0
Request object: QARequest(...)
==============================

=== FAMILY QA OPENSEARCH QUERY ===
Index: deephousedeephouse_ediscovery_docs_chunks
Search Body: {
  "query": {
    "bool": {
      "must": [...]
    }
  },
  "size": 10,
  "_source": ["content", "meta"]
}
================================

=== FAMILY QA FINAL OPENSEARCH REQUEST ===
Index: deephousedeephouse_ediscovery_docs_chunks
Final Search Body: { ... }
=========================================
```

### Direct Index QA (`/qa-direct-index`)
```
=== DIRECT INDEX QA REQUEST DEBUG ===
Query: What meetings were scheduled?
Index: deephousedeephouse_ediscovery_docs_chunks
Top K: 10
Direct Access: true
Filters: {"author": "scheduler@company.com"}
Raw Query Body: null
Request object: DirectIndexQARequest(...)
====================================

=== REGULAR QUERY BODY (BUILT) ===
Built Search Body: { ... }
=================================

=== FINAL OPENSEARCH REQUEST ===
Index: deephousedeephouse_ediscovery_docs_chunks
Final Search Body: { ... }
===============================

=== DIRECT INDEX QA OPENSEARCH RESPONSE ===
Total hits: 25
Returned hits: 10
Hit IDs: ["doc1", "doc2", ...]
==========================================
```

## Benefits

- **Request Debugging**: See exactly what parameters are being received
- **Query Construction**: View how OpenSearch queries are built from your filters
- **Raw Query Support**: Monitor raw query body processing
- **Response Analysis**: Check what documents are being returned
- **Performance Tracking**: Identify slow queries or large result sets

## Production Use

For production environments, keep `DEBUG_MODE=false` to:
- Reduce log volume
- Improve performance
- Avoid exposing sensitive query details
- Keep logs clean and focused

## Integration with Logging

The debug output uses the `debug_print()` function which only prints when debug mode is enabled. This allows you to:
- Keep debug statements in code without performance impact
- Easily toggle detailed logging on/off
- Maintain clean production logs

## Troubleshooting

If you're experiencing issues with QA endpoints:

1. **Enable debug mode**: `DEBUG_MODE=true`
2. **Make your request** to see detailed logging
3. **Check the logs** for:
   - Request parameters being received correctly
   - OpenSearch query construction
   - Filter application
   - Response data structure
   - Any error messages

This helps identify whether issues are in:
- Request formatting
- Query construction
- OpenSearch connectivity
- Data retrieval
- Response processing
