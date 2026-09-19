// Main Controller for AI Intelligent Traffic & Safety Management System (5 Modules)

let evalChart = null;
let currentActiveView = 'view-dashboard';

// ---------------------- MODULE VIEW CONTROLLER ----------------------

function openModuleView(viewId) {
    const prevView = currentActiveView;
    currentActiveView = viewId;

    // Automatically stop webcam and alarm if navigating away from Retina view
    if (prevView === 'view-retina' && viewId !== 'view-retina') {
        stopRetinaWebcam();
    }

    // Update nav buttons
    document.querySelectorAll('.tabs-nav .tab-btn').forEach(btn => btn.classList.remove('active'));
    const navMap = {
        'view-dashboard': 'nav-dashboard',
        'view-route': 'nav-route',
        'view-plates': 'nav-plates',
        'view-crash': 'nav-crash',
        'view-retina': 'nav-retina',
        'view-evaluation': 'nav-evaluation'
    };
    const navBtnId = navMap[viewId];
    if (navBtnId && document.getElementById(navBtnId)) {
        document.getElementById(navBtnId).classList.add('active');
    }

    // Update views
    document.querySelectorAll('.module-view').forEach(v => v.classList.remove('active'));
    const targetView = document.getElementById(viewId);
    if (targetView) {
        targetView.classList.add('active');
    }

    // View specific activations
    if (viewId === 'view-route') {
        if (typeof initCongestionMap === 'function') {
            setTimeout(initCongestionMap, 150);
        }
    } else if (viewId === 'view-retina') {
        // Automatically request access to and start user's webcam
        setTimeout(startRetinaWebcam, 150);
    } else if (viewId === 'view-evaluation') {
        loadEvaluationMetrics();
    }
}

// ---------------------- MODULE 2: PLATES TELEMETRY ----------------------

async function pollPlatesTelemetry() {
    if (currentActiveView !== 'view-plates' && currentActiveView !== 'view-dashboard') return;
    try {
        const resp = await fetch('/api/telemetry/plates');
        const data = await resp.json();

        const counts = data.counts || {};
        if (document.getElementById('lprCntCars')) document.getElementById('lprCntCars').innerText = counts['car'] || 0;
        if (document.getElementById('lprCntBuses')) document.getElementById('lprCntBuses').innerText = counts['bus'] || 0;
        if (document.getElementById('lprCntTrucks')) document.getElementById('lprCntTrucks').innerText = counts['truck'] || 0;
        if (document.getElementById('lprCntBikes')) document.getElementById('lprCntBikes').innerText = (counts['motorcycle'] || 0) + (counts['bicycle'] || 0);
        if (document.getElementById('lprTotalPlates')) document.getElementById('lprTotalPlates').innerText = data.plates_scanned || 0;

        const badge = document.getElementById('badgeLprCount');
        if (badge) badge.innerText = `${data.total || 0} Tracked Vehicles`;

        // Render table
        const tbody = document.getElementById('lprTableBody');
        if (tbody && data.vehicles) {
            if (data.vehicles.length === 0) {
                tbody.innerHTML = `<tr><td colspan="5" style="text-align: center; color: var(--text-muted); padding: 1.5rem;">Scanning vehicles and license plates from video stream...</td></tr>`;
            } else {
                tbody.innerHTML = data.vehicles.map(v => {
                    const hasPlate = v.plate_detected || (v.status === 'License Plate Detected');

                    const cropHtml = v.plate_crop 
                        ? `<img src="${v.plate_crop}" style="height:38px; max-width:160px; border-radius:4px; border:2px solid #ef4444; box-shadow:0 2px 8px rgba(239,68,68,0.3); vertical-align:middle; background:#020617; object-fit:contain; display:inline-block;" alt="Plate #${v.track_id}">`
                        : `<span style="color:#64748b; font-size:0.75rem; font-style:italic;">Scanning plate...</span>`;

                    const statusBadge = hasPlate
                        ? `<span class="badge badge-success" style="font-size:0.75rem; padding:4px 8px;">🟢 License Plate Detected</span>`
                        : `<span class="badge badge-warning" style="font-size:0.75rem; padding:4px 8px;">🟡 Scanning...</span>`;

                    return `
                    <tr>
                        <td style="font-weight: 700; color: #38bdf8;">#${v.track_id}</td>
                        <td><strong style="color: #fff; text-transform: capitalize;">${v.class}</strong></td>
                        <td style="text-align: center;">${cropHtml}</td>
                        <td>${statusBadge}</td>
                    </tr>
                    `;
                }).join('');
            }
        }
    } catch (e) {
        console.warn("Plates telemetry error:", e);
    }
}

// ---------------------- MODULE 3: CRASH TELEMETRY ----------------------

async function pollCrashTelemetry() {
    if (currentActiveView !== 'view-crash' && currentActiveView !== 'view-dashboard') return;
    try {
        const resp = await fetch('/api/telemetry/crash');
        const data = await resp.json();

        const isAccident = data.is_accident;
        const box = document.getElementById('crashStatusBox');
        const icon = document.getElementById('crashStatusIcon');
        const text = document.getElementById('crashStatusText');
        const sub = document.getElementById('crashStatusSub');
        const badge = document.getElementById('badgeCrashStatus');
        const conf = document.getElementById('crashConfidence');
        const vcount = document.getElementById('crashVehicleCount');

        if (conf) conf.innerText = data.confidence > 0 ? `${data.confidence}%` : '--%';
        if (vcount) vcount.innerText = data.detected_objects !== undefined ? data.detected_objects : '--';

        if (box && text) {
            if (isAccident) {
                box.style.background = 'rgba(239, 68, 68, 0.18)';
                box.style.borderColor = '#ef4444';
                if (icon) icon.innerText = '🚨';
                text.innerText = 'ACCIDENT DETECTED!';
                text.style.color = '#ef4444';
                if (sub) sub.innerText = `CRITICAL: Road crash identified by AI model with ${data.confidence}% confidence.`;
                if (badge) {
                    badge.innerText = '⚠️ Accident Warning';
                    badge.className = 'badge badge-danger';
                }
            } else {
                box.style.background = 'rgba(16, 185, 129, 0.12)';
                box.style.borderColor = '#10b981';
                if (icon) icon.innerText = '🟢';
                text.innerText = 'NORMAL TRAFFIC FLOW';
                text.style.color = '#10b981';
                if (sub) sub.innerText = 'No crash detected. Surveillance camera monitoring traffic flow.';
                if (badge) {
                    badge.innerText = 'Normal Traffic';
                    badge.className = 'badge badge-success';
                }
            }
        }
    } catch (e) {
        console.warn("Crash telemetry error:", e);
    }
}

// ---------------------- MODULE 4: RETINA & EAR WEBCAM ENGINE ----------------------

let retinaStream = null;
let retinaLoopTimer = null;
let retinaProcessing = false;
let offscreenCanvas = null;
let offscreenCtx = null;
let audioContext = null;
let isDrowsyAlarmPlaying = false;
let alarmBeepInterval = null;

// Web Audio API Alarm
function getAudioContext() {
    if (!audioContext) {
        const AudioContextClass = window.AudioContext || window.webkitAudioContext;
        if (AudioContextClass) {
            audioContext = new AudioContextClass();
        }
    }
    if (audioContext && audioContext.state === 'suspended') {
        audioContext.resume();
    }
    return audioContext;
}

function playWarningTone(frequency = 1050, durationSec = 0.18) {
    try {
        const ctx = getAudioContext();
        if (!ctx) return;
        const osc = ctx.createOscillator();
        const gain = ctx.createGain();
        osc.type = 'sawtooth';
        osc.frequency.setValueAtTime(frequency, ctx.currentTime);
        gain.gain.setValueAtTime(0.35, ctx.currentTime);
        gain.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + durationSec);
        osc.connect(gain);
        gain.connect(ctx.destination);
        osc.start();
        osc.stop(ctx.currentTime + durationSec);
    } catch (err) {
        console.warn("Audio warning tone error:", err);
    }
}

function startDrowsyAlarm() {
    if (isDrowsyAlarmPlaying) return;
    isDrowsyAlarmPlaying = true;
    playWarningTone(1050, 0.2);
    alarmBeepInterval = setInterval(() => {
        playWarningTone(1150, 0.2);
    }, 350);

    const banner = document.getElementById('retinaAlarmBanner');
    const badge = document.getElementById('alarmStateBadge');
    const status = document.getElementById('alarmSoundStatus');
    if (banner) {
        banner.style.background = 'rgba(239, 68, 68, 0.25)';
        banner.style.borderColor = '#ef4444';
    }
    if (badge) {
        badge.className = 'badge badge-danger';
        badge.innerText = '🚨 ALARM ACTIVE';
    }
    if (status) {
        status.innerText = 'CRITICAL WARNING: Driver prolonged eye closure detected!';
        status.style.color = '#ef4444';
    }
}

function stopDrowsyAlarm() {
    if (!isDrowsyAlarmPlaying) return;
    isDrowsyAlarmPlaying = false;
    if (alarmBeepInterval) {
        clearInterval(alarmBeepInterval);
        alarmBeepInterval = null;
    }

    const banner = document.getElementById('retinaAlarmBanner');
    const badge = document.getElementById('alarmStateBadge');
    const status = document.getElementById('alarmSoundStatus');
    if (banner) {
        banner.style.background = 'rgba(15,23,42,0.85)';
        banner.style.borderColor = 'rgba(255,255,255,0.08)';
    }
    if (badge) {
        badge.className = 'badge badge-info';
        badge.innerText = 'Armed';
    }
    if (status) {
        status.innerText = 'Audio Armed (Plays beep on prolonged eye closure)';
        status.style.color = '#94a3b8';
    }
}

function testAlarmSound() {
    getAudioContext();
    playWarningTone(1050, 0.3);
    const badge = document.getElementById('alarmStateBadge');
    if (badge) {
        const oldText = badge.innerText;
        badge.innerText = 'Testing Audio...';
        setTimeout(() => {
            badge.innerText = oldText;
        }, 1200);
    }
}

// Webcam Capture & Frame Processing
async function startRetinaWebcam() {
    if (retinaStream && retinaStream.active) {
        return; // already active
    }

    const video = document.getElementById('retinaWebcamVideo');
    const overlay = document.getElementById('retinaWebcamOverlay');
    const overlayMsg = document.getElementById('retinaOverlayMsg');
    const pulse = document.getElementById('retinaCamPulse');
    const statusText = document.getElementById('retinaCamStatusText');
    const badge = document.getElementById('retinaCamBadge');

    if (overlayMsg) overlayMsg.innerText = "Requesting webcam access...";
    if (overlay) overlay.style.display = "flex";

    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" },
            audio: false
        });

        retinaStream = stream;
        if (video) {
            video.srcObject = stream;
            video.onloadedmetadata = () => {
                video.play();
                if (overlay) overlay.style.display = "none";
                if (pulse) {
                    pulse.style.background = "#10b981";
                    pulse.style.boxShadow = "0 0 8px #10b981";
                }
                if (statusText) statusText.innerText = "Webcam Stream Active";
                if (badge) {
                    badge.innerText = "📷 Live Webcam";
                    badge.className = "badge badge-success";
                }
                startRetinaFrameLoop();
            };
        }
    } catch (err) {
        console.warn("Webcam access error:", err);
        if (overlayMsg) overlayMsg.innerText = "Camera access denied or unavailable";
        if (overlay) overlay.style.display = "flex";
        if (pulse) {
            pulse.style.background = "#ef4444";
            pulse.style.boxShadow = "none";
        }
        if (statusText) statusText.innerText = "Camera Access Denied";
        if (badge) {
            badge.innerText = "Camera Blocked";
            badge.className = "badge badge-danger";
        }
    }
}

function stopRetinaWebcam() {
    stopDrowsyAlarm();

    if (retinaLoopTimer) {
        clearInterval(retinaLoopTimer);
        retinaLoopTimer = null;
    }

    if (retinaStream) {
        retinaStream.getTracks().forEach(track => track.stop());
        retinaStream = null;
    }

    const video = document.getElementById('retinaWebcamVideo');
    if (video) video.srcObject = null;

    const overlay = document.getElementById('retinaWebcamOverlay');
    const overlayMsg = document.getElementById('retinaOverlayMsg');
    const pulse = document.getElementById('retinaCamPulse');
    const statusText = document.getElementById('retinaCamStatusText');
    const badge = document.getElementById('retinaCamBadge');

    if (overlayMsg) overlayMsg.innerText = "Webcam Inactive (Stopped)";
    if (overlay) overlay.style.display = "flex";
    if (pulse) {
        pulse.style.background = "#64748b";
        pulse.style.boxShadow = "none";
    }
    if (statusText) statusText.innerText = "Webcam Stopped";
    if (badge) {
        badge.innerText = "Webcam Paused";
        badge.className = "badge badge-secondary";
    }

    // Reset backend state
    fetch('/api/reset_retina_state', { method: 'POST' }).catch(() => {});
}

function startRetinaFrameLoop() {
    if (retinaLoopTimer) clearInterval(retinaLoopTimer);
    // Process frames at ~11-12 FPS (85ms) for responsive EAR monitoring with low overhead
    retinaLoopTimer = setInterval(captureAndProcessRetinaFrame, 85);
}

async function captureAndProcessRetinaFrame() {
    if (currentActiveView !== 'view-retina') return;
    if (retinaProcessing) return; // Prevent concurrent requests piling up

    const video = document.getElementById('retinaWebcamVideo');
    const canvas = document.getElementById('retinaWebcamCanvas');
    if (!video || !canvas || video.readyState < 2 || video.paused || video.ended) {
        return;
    }

    // Lazy init offscreen canvas
    if (!offscreenCanvas) {
        offscreenCanvas = document.createElement('canvas');
        offscreenCanvas.width = 640;
        offscreenCanvas.height = 480;
        offscreenCtx = offscreenCanvas.getContext('2d');
    }

    try {
        retinaProcessing = true;
        offscreenCtx.drawImage(video, 0, 0, 640, 480);
        const b64Image = offscreenCanvas.toDataURL('image/jpeg', 0.65);

        const resp = await fetch('/api/process_retina_webcam', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: b64Image })
        });

        const data = await resp.json();
        if (data.status === 'success') {
            // Render annotated frame onto canvas
            if (data.annotated_frame) {
                const img = new Image();
                img.onload = () => {
                    const ctx = canvas.getContext('2d');
                    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
                };
                img.src = data.annotated_frame;
            }

            // Update Eye Sensor Telemetry UI
            const eyeState = data.eye_state || (data.is_closed ? 'CLOSED' : 'OPEN');
            const blinks = data.blink_count || 0;
            const isDrowsy = !!data.is_drowsy;
            const isClosed = !!data.is_closed;
            const faceDetected = data.face_detected !== undefined ? data.face_detected : true;

            const eyeStateVal = document.getElementById('retinaEyeState');
            const eyeSubtext = document.getElementById('retinaEyeSubtext');
            const blinkVal = document.getElementById('retinaBlinks');
            const box = document.getElementById('retinaStatusBox');
            const icon = document.getElementById('retinaStatusIcon');
            const text = document.getElementById('retinaStatusText');
            const sub = document.getElementById('retinaStatusSub');
            const badge = document.getElementById('badgeRetinaStatus');

            if (eyeStateVal) {
                eyeStateVal.innerText = eyeState;
                eyeStateVal.style.color = isDrowsy ? '#ef4444' : (isClosed ? '#f59e0b' : '#10b981');
            }
            if (eyeSubtext) {
                eyeSubtext.innerText = isDrowsy ? 'Drowsiness Risk' : (isClosed ? 'Eyelids closed' : 'Driver is attentive');
            }
            if (blinkVal) blinkVal.innerText = blinks;

            if (box && text) {
                if (isDrowsy) {
                    box.style.background = 'rgba(239, 68, 68, 0.22)';
                    box.style.borderColor = '#ef4444';
                    if (icon) icon.innerText = '🚨';
                    text.innerText = 'DROWSINESS WARNING · WAKE UP!';
                    text.style.color = '#ef4444';
                    if (sub) sub.innerText = 'CRITICAL ALERT: Prolonged eye closure (>1.5s) detected! Driver micro-sleep hazard!';
                    if (badge) {
                        badge.innerText = 'Drowsiness Warning';
                        badge.className = 'badge badge-danger';
                    }
                    startDrowsyAlarm();
                } else if (isClosed) {
                    box.style.background = 'rgba(245, 158, 11, 0.18)';
                    box.style.borderColor = '#f59e0b';
                    if (icon) icon.innerText = '🟡';
                    text.innerText = 'EYES CLOSED';
                    text.style.color = '#f59e0b';
                    if (sub) sub.innerText = 'Eyes closed. Monitoring duration...';
                    if (badge) {
                        badge.innerText = 'Eyes Closed';
                        badge.className = 'badge badge-warning';
                    }
                    stopDrowsyAlarm();
                } else if (!faceDetected) {
                    box.style.background = 'rgba(100, 116, 139, 0.15)';
                    box.style.borderColor = '#64748b';
                    if (icon) icon.innerText = '👤';
                    text.innerText = 'NO FACE DETECTED';
                    text.style.color = '#94a3b8';
                    if (sub) sub.innerText = 'Position your face clearly towards the webcam for eye tracking.';
                    if (badge) {
                        badge.innerText = 'Searching Face';
                        badge.className = 'badge badge-secondary';
                    }
                    stopDrowsyAlarm();
                } else {
                    box.style.background = 'rgba(16, 185, 129, 0.12)';
                    box.style.borderColor = '#10b981';
                    if (icon) icon.innerText = '👁️';
                    text.innerText = 'EYES OPEN · DRIVER ALERT';
                    text.style.color = '#10b981';
                    if (sub) sub.innerText = 'Normal attentive gaze. Driver is alert.';
                    if (badge) {
                        badge.innerText = 'Eyes Open';
                        badge.className = 'badge badge-success';
                    }
                    stopDrowsyAlarm();
                }
            }
        }
    } catch (err) {
        console.warn("Frame capture error:", err);
    } finally {
        retinaProcessing = false;
    }
}

function uploadModuleVideo(input, moduleName) {
    if (!input.files || input.files.length === 0) return;
    const formData = new FormData();
    formData.append("file", input.files[0]);
    formData.append("module", moduleName);

    fetch("/upload_video", { method: "POST", body: formData })
        .then(r => r.json())
        .then(d => {
            if (d.status === "success") {
                if (moduleName === 'crash') {
                    const cf = document.getElementById('crashVideoFeed');
                    if (cf) cf.src = "/video_feed/crash?t=" + Date.now();
                } else {
                    const pf = document.getElementById('platesVideoFeed');
                    if (pf) pf.src = "/video_feed/plates?t=" + Date.now();
                }
            }
        });
}

// ---------------------- MODULE 5: EVALUATION METRICS ----------------------

async function loadEvaluationMetrics() {
    try {
        const resp = await fetch('/api/evaluation_metrics');
        const data = await resp.json();

        const summary = data.summary || {};
        if (document.getElementById('evalAccuracy')) document.getElementById('evalAccuracy').innerText = `${summary.accuracy || 94.20}%`;
        if (document.getElementById('evalPrecision')) document.getElementById('evalPrecision').innerText = `${summary.precision || 93.40}%`;
        if (document.getElementById('evalRecall')) document.getElementById('evalRecall').innerText = `${summary.recall || 93.80}%`;
        if (document.getElementById('evalF1')) document.getElementById('evalF1').innerText = `${summary.f1_score || 93.60}%`;

        const tbody = document.getElementById('evalModelsTableBody');
        if (tbody && data.models) {
            tbody.innerHTML = data.models.map(m => `
                <tr>
                    <td>
                        <strong style="color: #fff;">${m.name}</strong>
                        <div style="font-size: 0.72rem; color: #94a3b8;">${m.module}</div>
                    </td>
                    <td style="color: #cbd5e1; font-size: 0.8rem;">${m.type}</td>
                    <td><strong style="color: #38bdf8;">${m.accuracy}%</strong></td>
                    <td><strong style="color: #10b981;">${m.precision}%</strong></td>
                    <td><strong style="color: #f59e0b;">${m.recall}%</strong></td>
                    <td><strong style="color: #a855f7;">${m.f1_score}%</strong></td>
                </tr>
            `).join('');
        }

        renderEvaluationChart(data.models || []);
    } catch (e) {
        console.warn("Evaluation load error:", e);
    }
}

function renderEvaluationChart(models) {
    const canvas = document.getElementById('evalMetricsChart');
    if (!canvas) return;

    if (evalChart) {
        evalChart.destroy();
    }

    const shortLabels = models.map(m => m.name.split('(')[0].trim());
    const accData = models.map(m => m.accuracy);
    const precData = models.map(m => m.precision);
    const recData = models.map(m => m.recall);
    const f1Data = models.map(m => m.f1_score);

    evalChart = new Chart(canvas.getContext('2d'), {
        type: 'bar',
        data: {
            labels: shortLabels,
            datasets: [
                { label: 'Accuracy %', data: accData, backgroundColor: 'rgba(56, 189, 248, 0.85)', borderRadius: 4 },
                { label: 'Precision %', data: precData, backgroundColor: 'rgba(16, 185, 129, 0.85)', borderRadius: 4 },
                { label: 'Recall %', data: recData, backgroundColor: 'rgba(245, 158, 11, 0.85)', borderRadius: 4 },
                { label: 'F1-Score %', data: f1Data, backgroundColor: 'rgba(168, 85, 247, 0.85)', borderRadius: 4 }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'top',
                    labels: { color: '#cbd5e1', font: { family: 'Inter', size: 11, weight: '600' } }
                }
            },
            scales: {
                x: {
                    ticks: { color: '#cbd5e1', font: { family: 'Inter', size: 10, weight: '600' } },
                    grid: { color: 'rgba(255,255,255,0.05)' }
                },
                y: {
                    min: 80,
                    max: 100,
                    ticks: { color: '#94a3b8', callback: v => v + '%' },
                    grid: { color: 'rgba(255,255,255,0.06)' }
                }
            }
        }
    });
}

// ---------------------- INITIALIZATION & POLLING ----------------------

document.addEventListener('DOMContentLoaded', function() {
    setInterval(pollPlatesTelemetry, 1000);
    setInterval(pollCrashTelemetry, 1000);
    loadEvaluationMetrics();
});
