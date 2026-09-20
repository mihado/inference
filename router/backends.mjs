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
