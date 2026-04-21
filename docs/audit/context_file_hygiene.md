# Context File Hygiene Audit

Date: 2026-04-21

| Section | Verdict | Reasoning |
|---|---|---|
| `CLAUDE.md`, `agent.md`, `.cursorrules` | KEEP (none) | No project context file exists. Best hygiene outcome: zero token overhead and zero stale instruction risk. |
| Discoverable project details | REMOVE | Stack, structure, commands, dependencies, and architecture are discoverable from codebase (`pyproject.toml`, directory tree, source files). They should not be duplicated in context files. |

Trimmed file:

No context file required right now.
