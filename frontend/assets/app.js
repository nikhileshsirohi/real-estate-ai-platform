const endpoints = {
  monitoring: "/monitoring/summary",
  evaluation: "/evaluation/summary",
  predict: "/predict-price",
  market: "/ask-market",
  advice: "/advise-property",
  search: "/search-properties/query",
  recommend: "/search-properties/recommend",
};

function prettyPrint(targetId, payload) {
  const el = document.getElementById(targetId);
  el.textContent = typeof payload === "string" ? payload : JSON.stringify(payload, null, 2);
}

function formToJson(form) {
  const formData = new FormData(form);
  const payload = {};
  for (const [key, value] of formData.entries()) {
    if (value === "") {
      payload[key] = null;
      continue;
    }
    const numericKeys = new Set([
      "median_income",
      "house_age",
      "average_rooms",
      "average_bedrooms",
      "population",
      "average_occupancy",
      "latitude",
      "longitude",
      "limit",
    ]);
    payload[key] = numericKeys.has(key) ? Number(value) : value;
  }
  return payload;
}

async function callJson(endpoint, options = {}) {
  const response = await fetch(endpoint, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const data = await response.json();
  if (!response.ok) {
    throw new Error(JSON.stringify(data));
  }
  return data;
}

function updateMonitoringCards(summary) {
  document.getElementById("metric-total-requests").textContent = summary.runtime.total_requests;
  document.getElementById("metric-average-duration").textContent = `${summary.runtime.average_duration_ms} ms`;
  document.getElementById("metric-prediction-records").textContent = summary.database.prediction_record_count;
  document.getElementById("metric-property-listings").textContent = summary.database.property_listing_count;
  document.getElementById("metric-rmse").textContent = summary.model.metrics.rmse ?? "-";
  document.getElementById("metric-rag-documents").textContent = summary.rag_index.document_count ?? "-";
}

async function loadMonitoring() {
  try {
    const monitoring = await callJson(endpoints.monitoring);
    updateMonitoringCards(monitoring);
    prettyPrint("monitoring-json", monitoring);
  } catch (error) {
    prettyPrint("monitoring-json", { error: String(error) });
  }
}

async function loadEvaluation() {
  try {
    const evaluation = await callJson(endpoints.evaluation);
    prettyPrint("evaluation-json", evaluation);
  } catch (error) {
    prettyPrint("evaluation-json", { error: String(error) });
  }
}

function attachFormHandler(formId, outputId, endpoint) {
  const form = document.getElementById(formId);
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const payload = formToJson(form);
    prettyPrint(outputId, { loading: true, endpoint, payload });
    try {
      const response = await callJson(endpoint, {
        method: "POST",
        body: JSON.stringify(payload),
      });
      prettyPrint(outputId, response);
      await loadMonitoring();
      await loadEvaluation();
    } catch (error) {
      prettyPrint(outputId, { error: String(error) });
    }
  });
}

document.getElementById("refresh-monitoring").addEventListener("click", async () => {
  await loadMonitoring();
  await loadEvaluation();
});

attachFormHandler("predict-form", "predict-output", endpoints.predict);
attachFormHandler("market-form", "market-output", endpoints.market);
attachFormHandler("advice-form", "advice-output", endpoints.advice);
attachFormHandler("search-form", "search-output", endpoints.search);
attachFormHandler("recommend-form", "recommend-output", endpoints.recommend);

loadMonitoring();
loadEvaluation();
