function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function formatNumber(value, digits = 4) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "-";
  }
  return Number(value).toFixed(digits);
}

function formatInteger(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "-";
  }
  return new Intl.NumberFormat("en-US", { maximumFractionDigits: 0 }).format(Number(value));
}

function formatUsd(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "-";
  }
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 0,
  }).format(Number(value));
}

function sanitizeAnswerText(value) {
  return escapeHtml(String(value ?? "").replaceAll("### ", "").replaceAll("**", ""));
}

function renderParagraphs(text) {
  const paragraphs = String(text ?? "")
    .split(/\n\s*\n/)
    .map((part) => part.trim())
    .filter(Boolean);

  if (!paragraphs.length) {
    return '<p class="empty-state">No response available yet.</p>';
  }

  return paragraphs.map((part) => `<p class="response-text">${sanitizeAnswerText(part)}</p>`).join("");
}

function toPayload(formElement) {
  const formData = new FormData(formElement);
  const payload = {};

  formData.forEach((value, key) => {
    const trimmed = typeof value === "string" ? value.trim() : value;
    if (trimmed === "") {
      return;
    }

    if (key === "question" || key === "query") {
      payload[key] = trimmed;
      return;
    }

    payload[key] = Number(trimmed);
  });

  return payload;
}

function setContent(elementId, html) {
  const element = document.getElementById(elementId);
  if (!element) {
    return;
  }
  element.innerHTML = html;
}

function setLoading(elementId, text) {
  setContent(
    elementId,
    `<span class="response-title">Working</span><p class="response-text">${escapeHtml(text)}</p>`,
  );
}

function renderPredictResponse(data) {
  const predictedUsd = formatUsd(Number(data.predicted_price) * 100000);
  return `
    <span class="response-title">Prediction Ready</span>
    <p class="response-lead">Estimated property value: ${predictedUsd}</p>
    <div class="response-meta">
      <div class="response-grid">
        <div>
          <strong>Model used</strong>
          <span>${escapeHtml(data.model_name)}</span>
        </div>
        <div>
          <strong>Saved prediction ID</strong>
          <span>${escapeHtml(data.prediction_id ?? "Not stored")}</span>
        </div>
      </div>
      <p class="response-kicker">The model output is stored in units of 100,000 USD internally and shown here as an easy-to-read dollar estimate.</p>
    </div>
  `;
}

function renderMarketResponse(data) {
  return `
    <span class="response-title">Market Answer</span>
    ${renderParagraphs(data.answer)}
    <div class="response-meta">
      <div class="response-grid">
        <div>
          <strong>Answer model</strong>
          <span>${escapeHtml(data.model_name)}</span>
        </div>
        <div>
          <strong>Sources used</strong>
          <span>${formatInteger(data.sources?.length ?? 0)}</span>
        </div>
      </div>
    </div>
  `;
}

function renderAdviceResponse(data) {
  return `
    <span class="response-title">Property Advice</span>
    <p class="response-lead">Estimated value: ${formatUsd(data.predicted_price_usd)}</p>
    ${renderParagraphs(data.answer)}
    <div class="response-meta">
      <div class="response-grid">
        <div>
          <strong>Answer model</strong>
          <span>${escapeHtml(data.model_name)}</span>
        </div>
        <div>
          <strong>Context sources</strong>
          <span>${formatInteger(data.sources?.length ?? 0)}</span>
        </div>
      </div>
      <p class="response-kicker">This combines the price model with retrieved market context. If the supporting evidence is weak, the advisor stays conservative instead of guessing.</p>
    </div>
  `;
}

function renderListingItems(items) {
  if (!items?.length) {
    return '<p class="empty-state">No property listings are available for the current search.</p>';
  }

  return items
    .map(
      (item) => `
        <div class="response-list-item">
          <strong>${escapeHtml(item.title)} (${escapeHtml(item.listing_code)})</strong>
          <p class="response-text">${escapeHtml(item.city)}, ${escapeHtml(item.locality)} • ${escapeHtml(item.property_type)} • ${formatInteger(item.bedrooms)} bed • ${formatNumber(item.bathrooms, 1)} bath • ${formatInteger(item.area_sqft)} sqft</p>
          <p class="response-text">${formatUsd(item.asking_price_usd)}</p>
          <p class="response-kicker">${escapeHtml(item.description)}</p>
        </div>
      `,
    )
    .join("");
}

function renderSearchLikeResponse(data, includeAnswer = false) {
  const heading =
    data.match_strategy === "closest_match"
      ? "No exact match was found, so the closest alternatives are shown below."
      : `Found ${formatInteger(data.count)} matching listing${Number(data.count) === 1 ? "" : "s"}.`;

  const detectedPreferences = Array.isArray(data.detected_preferences) && data.detected_preferences.length
    ? data.detected_preferences.map((item) => escapeHtml(item.replaceAll("_", " "))).join(", ")
    : "None detected";

  return `
    <span class="response-title">${includeAnswer ? "Recommended Listings" : "Search Results"}</span>
    <p class="response-lead">${escapeHtml(heading)}</p>
    ${includeAnswer ? renderParagraphs(data.answer) : ""}
    <div class="response-meta">
      <div class="response-grid">
        <div>
          <strong>Search strategy</strong>
          <span>${escapeHtml(data.match_strategy || "exact")}</span>
        </div>
        <div>
          <strong>Detected preferences</strong>
          <span>${detectedPreferences}</span>
        </div>
      </div>
      ${
        data.advisory_note
          ? `<p class="response-kicker">${escapeHtml(data.advisory_note)}</p>`
          : ""
      }
    </div>
    <div class="response-list">${renderListingItems(data.items)}</div>
  `;
}

function renderMonitoringResponse(data) {
  const runtime = data.runtime || {};
  const database = data.database || {};
  const model = data.model || {};
  const ragIndex = data.rag_index || {};
  const metrics = model.metrics || {};

  return `
    <span class="response-title">Platform Health</span>
    <p class="response-lead">The app is responding, the database is reachable, and the current model and retrieval assets are available for use.</p>
    <div class="response-meta">
      <div class="response-grid">
        <div>
          <strong>Prediction records saved</strong>
          <span>${formatInteger(database.prediction_record_count)}</span>
        </div>
        <div>
          <strong>Listings in local inventory</strong>
          <span>${formatInteger(database.property_listing_count)}</span>
        </div>
        <div>
          <strong>Current model RMSE</strong>
          <span>${formatNumber(metrics.rmse)}</span>
        </div>
        <div>
          <strong>Knowledge documents indexed</strong>
          <span>${formatInteger(ragIndex.document_count)}</span>
        </div>
      </div>
      <p class="response-kicker">Runtime counters reflect live API traffic. Lower RMSE means the model's average prediction error is smaller.</p>
      <p class="response-kicker">Current request totals: ${formatInteger(runtime.total_requests)} requests, ${formatInteger(runtime.error_requests)} errors, ${formatNumber(runtime.average_duration_ms, 2)} ms average response time.</p>
    </div>
  `;
}

function renderEvaluationResponse(data) {
  const modelEvaluation = data.model_evaluation || {};
  const ragEvaluation = data.rag_evaluation || {};
  const inventoryEvaluation = data.inventory_evaluation || {};
  const metrics = modelEvaluation.metrics || {};

  return `
    <span class="response-title">What Evaluation Means</span>
    <p class="response-lead">This is the quality-check panel for the full system.</p>
    <p class="response-text">It tells you whether the price model is reasonably accurate, whether the knowledge index is built and ready, and whether the demo inventory covers enough cities to make search and advisory features useful.</p>
    <div class="response-meta">
      <div class="response-grid">
        <div>
          <strong>RMSE</strong>
          <span>${formatNumber(metrics.rmse)}</span>
        </div>
        <div>
          <strong>MAE</strong>
          <span>${formatNumber(metrics.mae)}</span>
        </div>
        <div>
          <strong>R-squared</strong>
          <span>${formatNumber(metrics.r2)}</span>
        </div>
        <div>
          <strong>Inventory coverage</strong>
          <span>${formatInteger(inventoryEvaluation.total_cities)} cities, ${formatInteger(ragEvaluation.chunk_count)} RAG chunks</span>
        </div>
      </div>
      <p class="response-kicker">In plain language: lower RMSE and MAE are better, higher R-squared is better, and more well-structured inventory plus RAG content makes the assistant more useful and safer.</p>
    </div>
  `;
}

function updateMonitoringCards(data) {
  const runtime = data.runtime || {};
  const database = data.database || {};
  const model = data.model || {};
  const ragIndex = data.rag_index || {};
  const metrics = model.metrics || {};

  const cardValues = {
    "metric-total-requests": formatInteger(runtime.total_requests),
    "metric-average-duration": `${formatNumber(runtime.average_duration_ms, 2)} ms`,
    "metric-prediction-records": formatInteger(database.prediction_record_count),
    "metric-property-listings": formatInteger(database.property_listing_count),
    "metric-rmse": formatNumber(metrics.rmse, 4),
    "metric-rag-documents": formatInteger(ragIndex.document_count),
  };

  Object.entries(cardValues).forEach(([elementId, value]) => {
    const element = document.getElementById(elementId);
    if (element) {
      element.textContent = value;
    }
  });
}

function updateEvaluationCards(data) {
  const modelEvaluation = data.model_evaluation || {};
  const ragEvaluation = data.rag_evaluation || {};
  const inventoryEvaluation = data.inventory_evaluation || {};
  const metrics = modelEvaluation.metrics || {};

  const cardValues = {
    "metric-mae": formatNumber(metrics.mae, 4),
    "metric-r2": formatNumber(metrics.r2, 4),
    "metric-cities": formatInteger(inventoryEvaluation.total_cities),
    "metric-rag-chunks": formatInteger(ragEvaluation.chunk_count),
  };

  Object.entries(cardValues).forEach(([elementId, value]) => {
    const element = document.getElementById(elementId);
    if (element) {
      element.textContent = value;
    }
  });
}

async function fetchJson(url, options = {}) {
  const response = await fetch(url, options);
  const data = await response.json().catch(() => ({}));

  if (!response.ok) {
    const message = data?.detail || `Request failed with status ${response.status}`;
    throw new Error(message);
  }

  return data;
}

async function loadMonitoring() {
  setLoading("monitoring-output", "Loading monitoring summary...");
  try {
    const data = await fetchJson("/monitoring/summary");
    updateMonitoringCards(data);
    setContent("monitoring-output", renderMonitoringResponse(data));
  } catch (error) {
    setContent(
      "monitoring-output",
      `<span class="response-title">Monitoring Error</span><p class="response-text">${escapeHtml(error.message)}</p>`,
    );
  }
}

async function loadEvaluation() {
  setLoading("evaluation-output", "Loading evaluation summary...");
  try {
    const data = await fetchJson("/evaluation/summary");
    updateEvaluationCards(data);
    setContent("evaluation-output", renderEvaluationResponse(data));
  } catch (error) {
    setContent(
      "evaluation-output",
      `<span class="response-title">Evaluation Error</span><p class="response-text">${escapeHtml(error.message)}</p>`,
    );
  }
}

function attachFormHandler(formId, endpoint, outputId, renderer) {
  const form = document.getElementById(formId);
  if (!form) {
    return;
  }

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    setLoading(outputId, "Working on your request...");

    try {
      const payload = toPayload(form);
      const data = await fetchJson(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      setContent(outputId, renderer(data));
      await Promise.all([loadMonitoring(), loadEvaluation()]);
    } catch (error) {
      setContent(
        outputId,
        `<span class="response-title">Request Error</span><p class="response-text">${escapeHtml(error.message)}</p>`,
      );
    }
  });
}

document.addEventListener("DOMContentLoaded", () => {
  document.getElementById("refresh-monitoring")?.addEventListener("click", async () => {
    await Promise.all([loadMonitoring(), loadEvaluation()]);
  });

  attachFormHandler("predict-form", "/predict-price", "predict-output", renderPredictResponse);
  attachFormHandler("market-form", "/ask-market", "market-output", renderMarketResponse);
  attachFormHandler("advice-form", "/advise-property", "advice-output", renderAdviceResponse);
  attachFormHandler("search-form", "/search-properties/query", "search-output", (data) =>
    renderSearchLikeResponse(data, false),
  );
  attachFormHandler("recommend-form", "/search-properties/recommend", "recommend-output", (data) =>
    renderSearchLikeResponse(data, true),
  );

  void loadMonitoring();
  void loadEvaluation();
});
