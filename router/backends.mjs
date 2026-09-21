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

/** Round-robin over a model's backends. Pure: the caller keeps the turn.
 *
 * Round-robin is enough here because every request is seconds long and evenly
 * sized, so there is nothing for a smarter policy to exploit. */
export function pickBackend(urls, turn) {
  if (urls === undefined || urls.length === 0) return undefined;
  return urls[turn % urls.length];
}
