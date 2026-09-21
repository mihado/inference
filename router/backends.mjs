// Pure parsing of what a backend advertises, kept apart from the server so it
// can be tested without starting one.
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
