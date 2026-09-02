# Agent Session Summaries

Directory for storing summaries from chats with coding agents for future context.

## Session File Format

Based on Claude Code's session memory structure.

```markdown
# Session: YYYY-MM-DD-HHMM

**Authoring agent:** <agent-name>
**Model:** <model name or id>

## Objective
Brief 1-2 sentence overview of what was discussed/accomplished.

## Completed
- Completed item 1
- Completed item 2

## Errors and Lessons
- Lesson learned (from error)

## Open Questions
- Question 1

## Notes
Any other relevant context for future agents

## References
- [Title](URL) - brief description
```

## Naming Convention

Session files should be named with the date and start time (24h):
```
YYYY-MM-DD-HHMM.md
```

The timestamp is the time the session started. Multiple sessions on the same
day get separate files; a later session can merge earlier ones for that day
into a single `YYYY-MM-DD.md` file if desired, keeping each session's log
separate until then. Existing combined logs remain named `YYYY-MM-DD.md`.
