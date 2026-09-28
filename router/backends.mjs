// Pure parsing of the backend catalogue — what a server advertises about its
// models, and what a container is called — kept apart from the server so it can
// be tested without starting one.
//
// Two shapes: TEI answers /info with the single model it serves, while
// OpenAI-shaped servers (vLLM) list their served names under /v1/models.

/** The single model id TEI advertises via /info, or null when absent. */
export function teiModelId(info) {
  return typeof info?.model_id === "string" ? info.model_id : null;
}

/** Every id in an OpenAI-style /v1/models body; [] for anything malformed. */
export function openAiModelIds(models) {
  const list = Array.isArray(models?.data) ? models.data : [];
  return list.map((entry) => entry?.id).filter((id) => typeof id === "string" && id.length > 0);
}

// The container's compose label: it is a service name and a container's network
// alias. The Docker API spells this label out, so the router never has to guess.
const COMPOSE_SERVICE_LABEL = "com.docker.compose.service";

/** The name a container is routable by, and therefore its identity: one
 * container is one entry in the rotation.
 *
 * Compose calls the container `<project>-<service>-<index>` but gives it a
 * network alias of the bare service name, and labels it with the service. So the
 * label is the name that a discovered container and a hand-written BACKENDS
 * entry agree on, while the container name disagrees with both — which is how
 * one container came to sit in the rotation twice under two names and take two
 * of every three requests for its model.
 *
 * A container from a plain `docker run` carries no compose label and is routable
 * by its own name. */
export function backendName(container) {
  const service = container?.Labels?.[COMPOSE_SERVICE_LABEL];
  if (typeof service === "string" && service.length > 0) return service;
  return String(container?.Names?.[0] ?? "").replace(/^\//, "");
}

/** Records that `base` serves `id`, keeping every replica of a model.
 *
 * A model can be served by more than one backend — two GPUs, one model each — and
 * the routing map has to hold all of them. Discovery order is the order they are
 * tried in, so a replica added later joins the rotation rather than replacing it.
 */
export function addBackend(found, id, base) {
  const urls = found.get(id);
  if (urls === undefined) found.set(id, [base]);
  else if (!urls.includes(base)) urls.push(base);
}

/** POST paths a backend advertises in /info (`paths`), or [] for anything
 * malformed. Entries must be absolute paths; anything else is ignored, never
 * an error — a misbehaving backend degrades to unroutable paths, not a broken
 * catalogue. */
export function infoPaths(info) {
  const list = Array.isArray(info?.paths) ? info.paths : [];
  const seen = new Set();
  for (const entry of list) {
    if (typeof entry === "string" && entry.startsWith("/") && !seen.has(entry)) seen.add(entry);
  }
  return [...seen];
}

/** Records that the backends serving `id` also answer `paths`. Empty adds
 * nothing, so backends that omit `paths` leave no trace. */
export function addPaths(found, id, paths) {
  if (paths.length === 0) return;
  const known = found.get(id);
  if (known === undefined) found.set(id, new Set(paths));
  else for (const path of paths) known.add(path);
}

/** Round-robin over a model's backends. Pure: the caller keeps the turn.
 *
 * Round-robin is enough here because every request is seconds long and evenly
 * sized, so there is nothing for a smarter policy to exploit. */
export function pickBackend(urls, turn) {
  if (urls === undefined || urls.length === 0) return undefined;
  return urls[turn % urls.length];
}

/** One access-log line per forwarded request. Pure, so it can be tested.
///
/// Bodies are never logged: states may be sensitive, and the catalogue fields
/// are enough to debug rotation skew (TUNING.md, "One container, one entry").
/// Newlines are stripped so a client-supplied model id cannot forge lines. */
export function accessLine({ method, path, model, backend, status, ms }) {
  const clean = (value) => String(value ?? "-").replace(/[\r\n]/g, "_");
  return `router: ${clean(method)} ${clean(path)} model=${clean(model)} backend=${clean(backend)} status=${clean(status)} ${clean(ms)}ms`;
}
