# Router Backend Integration

This document describes how the SpringBoot backend should consume the Router service response. It is an integration note only; it does not require changing the Router workflow taxonomy.

## Response Fields

The backend should mainly use `workflow`, not `route`.

`route` is retained for backward compatibility and currently equals the final `workflow`.

`toolHints` are tool-selection hints. They are not the final answer and should not be shown directly as answer content.

Additional post-processing fields:

- `correctedByRule`: `true` when Router post-processing rules corrected or confirmed a strong boundary case.
- `ruleName`: the rule that fired, such as `DIRECT_AI_WRITING_RULE`, `SYSTEM_ACTION_RULE`, `UNKNOWN_CONTEXT_RULE`, `FILE_QA_RULE`, `AGENT_TASK_SCHEDULE_RULE`, or `USER_KNOWLEDGE_RULE`.
- `rawWorkflow`: the original model-predicted workflow before post-processing.
- `rawConfidence`: the original model confidence before post-processing.

When `correctedByRule=true`, the backend should execute according to the final `workflow`.

When `confidence < 0.65` and `correctedByRule=false`, prefer routing to `UNKNOWN` or another safe fallback.

## Workflow Handling

`SYSTEM_ACTION`

Must pass permission checks and secondary confirmation before mutating user data. Examples include deleting chats, removing files, changing theme, updating profile data, and saving content into the global knowledge base.

`DIRECT_AI`

Does not use RAG. It can be sent directly to the LLM service.

`KNOWLEDGE_QA`

Use the public campus knowledge base RAG.

`FILE_QA`

Use the current `conversationId` file store RAG.

`USER_KNOWLEDGE_QA`

Use the user's global knowledge base RAG.

`AGENT_TASK`

For the first version, collect single-turn tool data such as schedule, todo, time, or global knowledge. Do not start multi-turn Agent planning yet.

`UNKNOWN`

Read current `conversationId` context first, then pass the enriched request to the LLM service.

## Tool Hint Examples

- `GET_CURRENT_TIME`: current time is needed.
- `SEARCH_PUBLIC_KNOWLEDGE`: public campus knowledge retrieval.
- `SEARCH_FILE_CONTENT`: current conversation file retrieval.
- `SEARCH_GLOBAL_KNOWLEDGE`: user global knowledge retrieval.
- `SEARCH_CONVERSATION_HISTORY`: conversation context retrieval.
- `WRITE_USER_DATA`: user-data mutation; requires permission and confirmation.
- `DIRECT_LLM`: direct LLM response without RAG.

## Router Unavailable Fallback

If the Router service is unavailable, the backend may use:

```json
{
  "workflow": "KNOWLEDGE_QA",
  "route": "KNOWLEDGE_QA",
  "toolHints": ["SEARCH_PUBLIC_KNOWLEDGE"],
  "available": false,
  "message": "Router service unavailable, fallback to public knowledge QA."
}
```

This fallback is conservative and should not trigger `SYSTEM_ACTION`.
