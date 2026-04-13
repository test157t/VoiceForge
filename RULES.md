# VoiceForge Engineering Rules

## Zero Fallback Policy (Mandatory)

Fallback logic is explicitly banned in this codebase.

- Do not silently switch to a different backend, route, model, voice, prompt, cache source, or processing mode.
- Do not auto-retry with altered parameters to "make it work".
- Do not infer alternate defaults when a required input is missing for the selected path.

## Fail-Fast Requirement

If the selected path cannot run exactly as requested, fail immediately and surface a clear error.

- Return explicit errors with enough context to fix configuration or code.
- Never swallow exceptions.
- Never convert a hard failure into a soft success.

## Strict Per-Backend Payloads

For every request, send only fields required by the selected backend/path.

- Remove irrelevant fields instead of tolerating them.
- If required fields are missing/invalid, fail with a direct error.

## Cache Rules

Caching must be explicit and deterministic.

- No hidden fallback cache layers.
- No alternate cache read/write paths that mask primary-path failures.
- Cache miss is not a reason to route to another implementation.

## Exception Handling Rules

Exceptions may be handled only to report/propagate failure with context.

- `try/catch` that suppresses errors is banned.
- Logging-only catch blocks that continue execution as success are banned.

## Retrofit Requirement

These rules apply retroactively.

- Existing fallback and exception-swallowing behavior must be removed.
- Existing masked-failure paths must be replaced with explicit fail-fast behavior.
