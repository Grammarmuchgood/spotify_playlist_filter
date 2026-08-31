const form = document.getElementById("search-form");
const input = document.getElementById("search-input");
const statusEl = document.getElementById("status");
const resultsEl = document.getElementById("results");
const submitButton = form.querySelector("button");

function setStatus(text, isError = false) {
  statusEl.textContent = text;
  statusEl.hidden = !text;
  statusEl.classList.toggle("error", isError);
}

function matchTypeLabel(matchType) {
  // Mirrors backend/search/hybrid.py's match_type values - shown as a
  // small badge so it's visible *why* a result is here, matching how
  // match_type was used for debugging all through this project rather
  // than hiding it once there was finally a UI. artist_only_backfill /
  // genre_only_backfill are the two tiers tried before giving up on a
  // combined genre+artist request entirely - each still honors half of
  // what was asked, so they get their own label rather than blending
  // into the fully-generic "related".
  const labels = {
    rrf: "vibe match",
    backfill: "related",
    genre_locked: "genre match",
    artist_locked: "artist match",
    "genre+artist_locked": "genre + artist match",
    artist_only_backfill: "same artist, different genre",
    genre_only_backfill: "same genre, different artist",
  };
  return labels[matchType] || matchType;
}

// Builds a human label for what the query was actually understood as -
// e.g. "Rock + Tame Impala" - so the "no exact matches" message can name
// what wasn't found instead of just reporting a raw count.
function detectedLabel(detected) {
  const parts = [];
  if (detected.genre) parts.push(detected.genre);
  if (detected.artist) parts.push(detected.artist);
  if (detected.reference_track) parts.push(`"${detected.reference_track.name}"`);
  return parts.join(" + ");
}

// Honest about what actually happened, not just a raw count - a genre +
// artist lock silently falling through to 100% backfill (confirmed
// common: 94.8% of artist x genre combos in this library have zero
// overlap) used to look identical to a fully successful search, since
// every result still just said "20 results for ...".
function buildStatusText(data) {
  const { results, query, detected, exact_match_count } = data;
  const label = detectedLabel(detected);
  const base = `${results.length} results for "${query}"`;
  if (!label || exact_match_count === results.length) {
    return base;
  }
  if (exact_match_count === 0) {
    return `No exact matches for ${label} - showing related songs instead.`;
  }
  const relatedCount = results.length - exact_match_count;
  const matchWord = exact_match_count === 1 ? "match" : "matches";
  return `${exact_match_count} exact ${matchWord} for ${label}, plus ${relatedCount} related.`;
}

function renderResults(results) {
  resultsEl.innerHTML = "";

  if (results.length === 0) {
    const empty = document.createElement("li");
    empty.className = "empty-state";
    empty.textContent = "No matches found.";
    resultsEl.appendChild(empty);
    return;
  }

  for (const track of results) {
    const item = document.createElement("li");
    item.className = "result";

    const top = document.createElement("div");
    top.className = "result-top";

    const titleBlock = document.createElement("div");
    const title = document.createElement("div");
    title.className = "result-title";
    title.textContent = track.name;
    const artist = document.createElement("div");
    artist.className = "result-artist";
    artist.textContent = track.artist;
    titleBlock.appendChild(title);
    titleBlock.appendChild(artist);

    const badges = document.createElement("div");
    badges.className = "result-badges";
    if (track.genre_bucket) {
      const genreBadge = document.createElement("span");
      genreBadge.className = "badge";
      genreBadge.textContent = track.genre_bucket;
      badges.appendChild(genreBadge);
    }
    if (track.match_type) {
      const matchBadge = document.createElement("span");
      matchBadge.className = `badge match-${track.match_type}`;
      matchBadge.textContent = matchTypeLabel(track.match_type);
      badges.appendChild(matchBadge);
    }

    top.appendChild(titleBlock);
    top.appendChild(badges);
    item.appendChild(top);

    if (track.description) {
      const description = document.createElement("div");
      description.className = "result-description";
      description.textContent = track.description;
      item.appendChild(description);
    }

    resultsEl.appendChild(item);
  }
}

async function runSearch(query) {
  submitButton.disabled = true;
  resultsEl.innerHTML = "";
  // The reranker is the dominant cost of every search (~4s, confirmed
  // by profiling) - a plain "Searching..." with no fake progress bar is
  // honest about that rather than pretending it's instant.
  setStatus("Searching...");

  try {
    const response = await fetch(`/search?q=${encodeURIComponent(query)}`);
    if (!response.ok) {
      throw new Error(`Search failed (${response.status})`);
    }
    const data = await response.json();
    setStatus(buildStatusText(data));
    renderResults(data.results);
  } catch (err) {
    setStatus(err.message || "Something went wrong.", true);
  } finally {
    submitButton.disabled = false;
  }
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  const query = input.value.trim();
  if (query) {
    runSearch(query);
  }
});
