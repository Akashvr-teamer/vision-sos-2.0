// ============================================================
// VISIONSOS — UNIFIED MULTI-HAZARD FRONTEND CONTROLLER
// ============================================================

let currentSelectedEventId = null;
let isStreamActive = false;
let cameraErrorCount = 0;
let lastStateString = "";

// ============================================================
// SYSTEM & HAZARD MATRIX STATUS UPDATES
// ============================================================

async function updateStatus() {
    try {
        const response = await fetch("/api/status?t=" + Date.now());
        if (!response.ok) return;
        const data = await response.json();

        // 1. Source display & Stream state
        const sourceDisplay = document.getElementById("source-display");
        const streamStateText = document.getElementById("stream-state-text");
        const streamLivePill = document.getElementById("stream-live-pill");
        const standbyOverlay = document.getElementById("camera-standby-state");
        const cameraImg = document.getElementById("live-camera");
        const btnCamToggle = document.getElementById("btn-cam-toggle");

        isStreamActive = Boolean(data.is_processing);

        if (sourceDisplay) {
            sourceDisplay.textContent = isStreamActive ? (data.source_name || "ONLINE") : "STANDBY";
        }

        if (streamStateText && streamLivePill) {
            if (isStreamActive) {
                streamStateText.textContent = "LIVE FEED";
                streamLivePill.className = "stream-pill active";
                if (btnCamToggle) btnCamToggle.textContent = "Webcam Running";
            } else {
                streamStateText.textContent = "STANDBY";
                streamLivePill.className = "stream-pill";
                if (btnCamToggle) btnCamToggle.innerHTML = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M23 7l-7 5 7 5V7z"/><rect x="1" y="5" width="15" height="14" rx="2" ry="2"/></svg> Start Live Webcam`;
            }
        }

        if (standbyOverlay && cameraImg) {
            if (isStreamActive) {
                standbyOverlay.style.display = "none";
                cameraImg.style.display = "block";
            } else {
                standbyOverlay.style.display = "flex";
                cameraImg.style.display = "none";
            }
        }

        // 2. Multi-Hazard Matrix Real-Time Updating
        const sub = data.sub_states || {};
        updateHazardCard("fall", sub.fall_state || "NORMAL");
        updateHazardCard("car", sub.car_state || "NORMAL");
        updateHazardCard("robbery", sub.robbery_state || "NORMAL");
        updateHazardCard("fire", sub.fire_state || "NORMAL");

        // 3. Overall Stream HUD Overlay
        updateStreamHUD(data.state || "NORMAL", data.countdown);

    } catch (err) {
        console.error("Status fetch error:", err);
    }
}

function updateHazardCard(hazardKey, state) {
    const card = document.getElementById(`hazard-${hazardKey}`);
    const chip = document.getElementById(`chip-${hazardKey}`);
    const metric = document.getElementById(`metric-${hazardKey}`);
    if (!card || !chip || !metric) return;

    card.classList.remove("state-warning", "state-danger");
    chip.classList.remove("normal", "warning", "danger");

    if (state.includes("EMERGENCY")) {
        card.classList.add("state-danger");
        chip.classList.add("danger");
        chip.textContent = "EMERGENCY";
        metric.textContent = "🚨 CRITICAL HAZARD";
        metric.style.color = "#ff2d55";
    } else if (state.includes("SUSPECTED") || state.includes("SUSPICIOUS")) {
        card.classList.add("state-warning");
        chip.classList.add("warning");
        chip.textContent = "ALERT";
        metric.textContent = "⚠️ Anomaly Detected";
        metric.style.color = "#ffb800";
    } else {
        chip.classList.add("normal");
        chip.textContent = "NORMAL";
        metric.textContent = isStreamActive ? "All Clear • Scanning" : "Standby";
        metric.style.color = "#90a0be";
    }
}

function updateStreamHUD(state, countdown) {
    const banner = document.getElementById("stream-overlay-banner");
    const icon = document.getElementById("overlay-icon");
    const title = document.getElementById("overlay-title");
    const subtitle = document.getElementById("overlay-subtitle");
    const cdElem = document.getElementById("overlay-countdown");
    if (!banner || !title || !subtitle) return;

    banner.className = "stream-overlay-banner";

    if (state === "STANDBY" || !isStreamActive) {
        icon.textContent = "⏸";
        icon.style.color = "#90a0be";
        title.textContent = "SYSTEM STANDBY";
        subtitle.textContent = "Camera idle. Ready to test video or start webcam.";
        cdElem.style.display = "none";
    } else if (state.includes("EMERGENCY")) {
        banner.classList.add("danger");
        icon.textContent = "🚨";
        icon.style.color = "#ff2d55";
        title.textContent = `${state.replace("_", " ")} CONFIRMED`;
        subtitle.textContent = "Emergency response protocol initiated • Evidence saved";
        cdElem.style.display = "none";
    } else if (state.includes("SUSPECTED") || state.includes("SUSPICIOUS")) {
        banner.classList.add("warning");
        icon.textContent = "⚠️";
        icon.style.color = "#ffb800";
        title.textContent = `ALERT: ${state.replace("_", " ")}`;
        subtitle.textContent = "Analyzing hazard pattern confirmation window";
        if (countdown !== null && countdown !== undefined) {
            cdElem.style.display = "block";
            cdElem.textContent = `${Number(countdown).toFixed(1)}s`;
        } else {
            cdElem.style.display = "none";
        }
    } else {
        icon.textContent = "✓";
        icon.style.color = "#00f084";
        title.textContent = "ALL HAZARDS NORMAL";
        subtitle.textContent = "AI running 4 concurrent neural & computer vision pipelines";
        cdElem.style.display = "none";
    }
}

// ============================================================
// LIVE CAMERA FRAME STREAMER
// ============================================================

let isFetchingFrame = false;

function updateCameraFrame() {
    if (!isStreamActive) return;
    const camera = document.getElementById("live-camera");
    if (!camera || isFetchingFrame) return;

    isFetchingFrame = true;
    const nextFrame = new Image();
    nextFrame.src = "/api/frame?t=" + Date.now();

    nextFrame.onload = () => {
        camera.src = nextFrame.src;
        cameraErrorCount = 0;
        isFetchingFrame = false;
    };

    nextFrame.onerror = () => {
        cameraErrorCount++;
        isFetchingFrame = false;
    };
}

// ============================================================
// INCIDENTS LOG & SPOTLIGHT
// ============================================================

async function updateEvents() {
    try {
        const response = await fetch("/api/events?t=" + Date.now());
        if (!response.ok) return;
        const events = await response.json();

        const historyList = document.getElementById("history-list");
        const historyCount = document.getElementById("history-count");
        if (historyCount) {
            historyCount.textContent = `${events.length} ${events.length === 1 ? "incident" : "incidents"}`;
        }

        if (!events || events.length === 0) {
            if (historyList) {
                historyList.innerHTML = `<div class="empty-history">No prior incidents recorded in database.</div>`;
            }
            updateSpotlight(null);
            return;
        }

        // Update spotlight with newest event
        updateSpotlight(events[0]);

        // Render history list
        if (historyList) {
            historyList.innerHTML = events.map(ev => {
                const icon = getEmergencyIcon(ev.event_type);
                const tagClass = `tag-${(ev.severity_label || "low").toLowerCase()}`;
                const title = (ev.event_type || "Emergency").replace("_", " ").toUpperCase();
                return `
                    <div class="history-card-row" onclick="viewHistoricalEvent(${ev.id})">
                        <div class="row-left">
                            <span class="row-icon">${icon}</span>
                            <div>
                                <div class="row-title">${title}</div>
                                <div class="row-sub">${ev.timestamp || "—"}</div>
                            </div>
                        </div>
                        <div class="row-right">
                            <span class="row-tag ${tagClass}">${ev.severity_label || "ALERT"}</span>
                            <span class="row-sub">${ev.severity || 0}/100</span>
                        </div>
                    </div>
                `;
            }).join("");
        }

    } catch (err) {
        console.error("Events fetch error:", err);
    }
}

function getEmergencyIcon(type) {
    if (!type) return "🚨";
    const t = type.toLowerCase();
    if (t.includes("fire")) return "🔥";
    if (t.includes("accident") || t.includes("car")) return "🚗";
    if (t.includes("robbery")) return "🚨";
    if (t.includes("fall") || t.includes("collapse")) return "💥";
    return "🚨";
}

function updateSpotlight(event) {
    const emptyState = document.getElementById("no-incident-state");
    const activeState = document.getElementById("active-incident-details");
    const badge = document.getElementById("spotlight-severity-badge");
    if (!emptyState || !activeState) return;

    if (!event) {
        emptyState.style.display = "flex";
        activeState.classList.add("hidden");
        if (badge) {
            badge.textContent = "MONITORING";
            badge.className = "severity-badge";
        }
        return;
    }

    currentSelectedEventId = event.id;
    emptyState.style.display = "none";
    activeState.classList.remove("hidden");

    if (badge) {
        const sev = event.severity_label || "HIGH";
        badge.textContent = sev;
        badge.className = `severity-badge ${sev}`;
    }

    // Populate details
    document.getElementById("incident-icon").textContent = getEmergencyIcon(event.event_type);
    document.getElementById("incident-title").textContent = `${(event.event_type || "Emergency").replace("_", " ").toUpperCase()} DETECTED`;
    document.getElementById("incident-trigger").textContent = `Category: ${event.category || "Auto-Detected"} • ID #${event.id}`;
    document.getElementById("incident-score").textContent = `${event.severity || 0}/100`;
    document.getElementById("score-bar-fill").style.width = `${Math.min(100, event.severity || 0)}%`;
    document.getElementById("incident-confidence").textContent = `${Math.round((event.confidence || 0.9) * 100)}%`;
    document.getElementById("incident-motionless").textContent = `${Number(event.motionless_sec || 0).toFixed(1)}s`;
    document.getElementById("incident-time").textContent = event.timestamp || "—";

    // Snapshot
    const snapContainer = document.getElementById("snapshot-container");
    const snapImg = document.getElementById("snapshot-img");
    if (event.snapshot_url && snapContainer && snapImg) {
        snapImg.src = event.snapshot_url;
        snapContainer.style.display = "block";
    } else if (snapContainer) {
        snapContainer.style.display = "none";
    }
}

// ============================================================
// USER CONTROLS & STREAM ACTIONS
// ============================================================

async function startWebcam() {
    const formData = new FormData();
    formData.append("source_type", "camera");

    showToast("Starting live webcam feed with Unified AI...", "info");

    try {
        const response = await fetch("/api/start_test", {
            method: "POST",
            body: formData
        });
        const data = await response.json();
        if (response.ok) {
            showToast(data.message || "Webcam active", "success");
            isStreamActive = true;
            await updateStatus();
        } else {
            showToast(data.error || "Could not launch webcam", "error");
        }
    } catch (err) {
        showToast("Network error starting webcam", "error");
    }
}

async function stopStream() {
    try {
        const response = await fetch("/api/stop_test", { method: "POST" });
        const data = await response.json();
        if (response.ok) {
            showToast("Stream stopped. System in Standby.", "info");
            isStreamActive = false;
            await updateStatus();
        }
    } catch (err) {
        showToast("Error stopping stream", "error");
    }
}

async function resetAlerts() {
    try {
        const response = await fetch("/api/reset", { method: "POST" });
        const data = await response.json();
        if (response.ok) {
            showToast("All hazard states reset to NORMAL", "success");
            await updateStatus();
        }
    } catch (err) {
        showToast("Error resetting system", "error");
    }
}

function handleFileSelected(input) {
    const dropTitle = document.getElementById("drop-title");
    const dropHint = document.getElementById("drop-hint");
    const submitBtn = document.getElementById("upload-submit-btn");

    if (input.files && input.files.length > 0) {
        const file = input.files[0];
        dropTitle.textContent = `Selected: ${file.name}`;
        dropHint.textContent = `Size: ${(file.size / (1024 * 1024)).toFixed(1)} MB • Click Start Unified Test below`;
        submitBtn.disabled = false;
    }
}

async function handleVideoUpload(event) {
    event.preventDefault();
    const fileInput = document.getElementById("video-file-input");
    const submitBtn = document.getElementById("upload-submit-btn");
    const feedback = document.getElementById("upload-feedback");

    if (!fileInput.files || fileInput.files.length === 0) {
        showToast("Please select a video file first", "error");
        return;
    }

    const formData = new FormData();
    formData.append("source_type", "video");
    formData.append("video", fileInput.files[0]);

    submitBtn.disabled = true;
    submitBtn.textContent = "Uploading & Launching...";
    feedback.textContent = "Uploading test video...";
    feedback.style.color = "#90a0be";

    try {
        const response = await fetch("/api/start_test", {
            method: "POST",
            body: formData
        });
        const data = await response.json();

        if (response.ok) {
            showToast("Video loaded! Unified detection running...", "success");
            feedback.textContent = data.message;
            feedback.style.color = "#00f084";
            isStreamActive = true;
            await updateStatus();
        } else {
            showToast(data.error || "Failed to start test", "error");
            feedback.textContent = data.error || "Failed";
            feedback.style.color = "#ff2d55";
        }
    } catch (err) {
        showToast("Network upload error", "error");
        feedback.textContent = "Network error connecting to backend.";
        feedback.style.color = "#ff2d55";
    } finally {
        submitBtn.disabled = false;
        submitBtn.textContent = "Start Unified Test";
    }
}

async function classifyCurrentIncident(category) {
    if (!currentSelectedEventId) return;

    try {
        const response = await fetch(`/api/events/${currentSelectedEventId}/category`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ category })
        });

        if (response.ok) {
            showToast(`Incident #${currentSelectedEventId} classified as: ${category}`, "success");
            await updateEvents();
        }
    } catch (err) {
        showToast("Classification update failed", "error");
    }
}

// ============================================================
// TOAST NOTIFICATIONS
// ============================================================

function showToast(message, type = "info") {
    const container = document.getElementById("toast-container");
    if (!container) return;

    const toast = document.createElement("div");
    toast.className = `toast ${type}`;
    toast.textContent = message;

    container.appendChild(toast);
    setTimeout(() => {
        toast.style.opacity = "0";
        setTimeout(() => toast.remove(), 300);
    }, 3200);
}

// ============================================================
// CYCLIC POLLERS
// ============================================================

updateStatus();
updateEvents();

setInterval(updateStatus, 600);
setInterval(updateEvents, 1800);
setInterval(updateCameraFrame, 120);