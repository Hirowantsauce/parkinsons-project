/**
 * NeuroVoice AI - Clinical Voice Biomarker Studio Application Logic
 */

document.addEventListener("DOMContentLoaded", () => {
    // --- Application State ---
    let samplesList = [];
    let currentAudioBlob = null;
    let currentSampleName = null;
    let selectedSex = "M";
    let mediaRecorder = null;
    let audioChunks = [];
    let recordInterval = null;
    let recordSeconds = 0;
    
    // Web Audio API Visualizer
    let audioCtx = null;
    let analyser = null;
    let visualizerSource = null;
    let animFrameId = null;

    // --- DOM Elements ---
    const tabBtns = document.querySelectorAll(".tab-btn");
    const tabPanels = document.querySelectorAll(".tab-panel");

    const sampleSelect = document.getElementById("sample-select");
    const quickPdBtn = document.getElementById("quick-sample-pd");
    const quickHcBtn = document.getElementById("quick-sample-hc");

    const dropzone = document.getElementById("audio-dropzone");
    const fileInput = document.getElementById("file-input");
    const uploadedFileName = document.getElementById("uploaded-file-name");

    const btnModeSamples = document.getElementById("btn-mode-samples");
    const btnModeUpload = document.getElementById("btn-mode-upload");
    const btnModeMic = document.getElementById("btn-mode-mic");
    const viewModeSamples = document.getElementById("view-mode-samples");
    const viewModeUpload = document.getElementById("view-mode-upload");
    const viewModeMic = document.getElementById("view-mode-mic");

    const micBtn = document.getElementById("btn-mic-toggle");
    const micStatusText = document.getElementById("mic-status-text");
    const micTimer = document.getElementById("mic-timer");

    const ageSlider = document.getElementById("patient-age");
    const ageDisplay = document.getElementById("age-display");
    const sexBtnM = document.getElementById("sex-btn-m");
    const sexBtnF = document.getElementById("sex-btn-f");

    const audioPlayer = document.getElementById("audio-player");
    const canvas = document.getElementById("waveform-canvas");
    const canvasCtx = canvas.getContext("2d");
    const btnRunAnalysis = document.getElementById("btn-run-analysis");

    const resultsIdle = document.getElementById("results-idle");
    const resultsActive = document.getElementById("results-active");

    const gaugeCircle = document.getElementById("gauge-circle");
    const gaugePct = document.getElementById("gauge-pct");
    const riskBadge = document.getElementById("risk-badge");
    const diagTitle = document.getElementById("diag-title");
    const diagRecommendation = document.getElementById("diag-recommendation");
    const sampleMetadataTag = document.getElementById("sample-metadata-tag");
    const flagsContainer = document.getElementById("flags-container");
    const flagsList = document.getElementById("flags-list");

    const cohortTbody = document.getElementById("cohort-tbody");
    const tableSearch = document.getElementById("table-search");
    const filterPills = document.querySelectorAll(".filter-pill");

    // --- 1. Tab Navigation ---
    tabBtns.forEach(btn => {
        btn.addEventListener("click", () => {
            const targetTab = btn.getAttribute("data-tab");
            tabBtns.forEach(b => {
                b.classList.remove("active");
                b.setAttribute("aria-selected", "false");
            });
            tabPanels.forEach(p => p.classList.remove("active"));

            btn.classList.add("active");
            btn.setAttribute("aria-selected", "true");
            const panel = document.getElementById(`panel-${targetTab}`);
            if (panel) panel.classList.add("active");
        });
    });

    // --- 2. Input Mode Switcher ---
    function switchMode(mode) {
        [btnModeSamples, btnModeUpload, btnModeMic].forEach(b => b.classList.remove("active"));
        [viewModeSamples, viewModeUpload, viewModeMic].forEach(v => v.classList.remove("active"));

        if (mode === "samples") {
            btnModeSamples.classList.add("active");
            viewModeSamples.classList.add("active");
        } else if (mode === "upload") {
            btnModeUpload.classList.add("active");
            viewModeUpload.classList.add("active");
        } else if (mode === "mic") {
            btnModeMic.classList.add("active");
            viewModeMic.classList.add("active");
        }
    }

    btnModeSamples.addEventListener("click", () => switchMode("samples"));
    btnModeUpload.addEventListener("click", () => switchMode("upload"));
    btnModeMic.addEventListener("click", () => switchMode("mic"));

    // --- 3. Age & Sex Controls ---
    ageSlider.addEventListener("input", (e) => {
        ageDisplay.textContent = e.target.value;
    });

    sexBtnM.addEventListener("click", () => {
        selectedSex = "M";
        sexBtnM.classList.add("active");
        sexBtnF.classList.remove("active");
    });

    sexBtnF.addEventListener("click", () => {
        selectedSex = "F";
        sexBtnF.classList.add("active");
        sexBtnM.classList.remove("active");
    });

    // --- 4. Fetch Sample Cohort Recordings ---
    async function fetchSamples() {
        try {
            const res = await fetch("/api/samples");
            const data = await res.json();
            samplesList = data.samples || [];

            sampleSelect.innerHTML = `<option value="" disabled selected>-- Select a Phonation Recording (${samplesList.length} Available) --</option>`;
            samplesList.forEach(s => {
                const opt = document.createElement("option");
                opt.value = s.filename;
                const statusTag = s.status === 1 ? "[PD Patient]" : "[Healthy Control]";
                const ageText = s.age ? `${Math.round(s.age)}y` : "";
                opt.textContent = `${s.name.substring(0, 32)}... ${statusTag} (${s.sex}, ${ageText})`;
                sampleSelect.appendChild(opt);
            });

            // Pre-select first PD sample for convenience
            const firstPd = samplesList.find(s => s.status === 1);
            if (firstPd) {
                loadSampleAudio(firstPd.filename);
                sampleSelect.value = firstPd.filename;
            }
        } catch (e) {
            console.error("Error fetching samples:", e);
            sampleSelect.innerHTML = `<option disabled>Error loading samples</option>`;
        }
    }

    function loadSampleAudio(filename) {
        currentSampleName = filename;
        currentAudioBlob = null;
        uploadedFileName.style.display = "none";
        
        audioPlayer.src = `/api/audio/${filename}`;
        audioPlayer.load();

        // Sync metadata
        const matched = samplesList.find(s => s.filename === filename);
        if (matched) {
            if (matched.age) {
                ageSlider.value = Math.round(matched.age);
                ageDisplay.textContent = Math.round(matched.age);
            }
            if (matched.sex === "F") {
                selectedSex = "F";
                sexBtnF.classList.add("active");
                sexBtnM.classList.remove("active");
            } else {
                selectedSex = "M";
                sexBtnM.classList.add("active");
                sexBtnF.classList.remove("active");
            }
        }
    }

    sampleSelect.addEventListener("change", (e) => {
        loadSampleAudio(e.target.value);
    });

    quickPdBtn.addEventListener("click", () => {
        switchMode("samples");
        const pdSamples = samplesList.filter(s => s.status === 1);
        if (pdSamples.length > 0) {
            const chosen = pdSamples[Math.floor(Math.random() * pdSamples.length)];
            sampleSelect.value = chosen.filename;
            loadSampleAudio(chosen.filename);
        }
    });

    quickHcBtn.addEventListener("click", () => {
        switchMode("samples");
        const hcSamples = samplesList.filter(s => s.status === 0);
        if (hcSamples.length > 0) {
            const chosen = hcSamples[Math.floor(Math.random() * hcSamples.length)];
            sampleSelect.value = chosen.filename;
            loadSampleAudio(chosen.filename);
        }
    });

    // --- 5. Custom WAV Drag-and-Drop / Upload ---
    dropzone.addEventListener("click", () => fileInput.click());

    dropzone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropzone.classList.add("dragover");
    });

    dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));

    dropzone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropzone.classList.remove("dragover");
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            handleUploadedFile(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener("change", (e) => {
        if (e.target.files && e.target.files[0]) {
            handleUploadedFile(e.target.files[0]);
        }
    });

    function handleUploadedFile(file) {
        if (!file.name.toLowerCase().endsWith(".wav")) {
            alert("Please upload an uncompressed PCM .wav audio file.");
            return;
        }
        currentAudioBlob = file;
        currentSampleName = null;
        uploadedFileName.textContent = `Selected: ${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
        uploadedFileName.style.display = "block";

        const objectUrl = URL.createObjectURL(file);
        audioPlayer.src = objectUrl;
        audioPlayer.load();
    }

    // --- 6. Live Microphone Recording ---
    micBtn.addEventListener("click", async () => {
        if (mediaRecorder && mediaRecorder.state === "recording") {
            stopRecording();
        } else {
            startRecording();
        }
    });

    async function startRecording() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            mediaRecorder = new MediaRecorder(stream);
            audioChunks = [];

            mediaRecorder.ondataavailable = (e) => audioChunks.push(e.data);
            mediaRecorder.onstop = () => {
                const blob = new Blob(audioChunks, { type: "audio/wav" });
                currentAudioBlob = blob;
                currentSampleName = null;
                const url = URL.createObjectURL(blob);
                audioPlayer.src = url;
                audioPlayer.load();
                micStatusText.textContent = "Recording captured successfully!";
            };

            mediaRecorder.start();
            micBtn.classList.add("recording");
            micStatusText.textContent = "Recording... Please sustain vowel 'AHHH' for 3-4 seconds";
            micTimer.style.display = "block";
            recordSeconds = 0;
            recordInterval = setInterval(() => {
                recordSeconds++;
                const mins = String(Math.floor(recordSeconds / 60)).padStart(2, "0");
                const secs = String(recordSeconds % 60).padStart(2, "0");
                micTimer.textContent = `${mins}:${secs}`;
                if (recordSeconds >= 5) {
                    stopRecording();
                }
            }, 1000);

        } catch (err) {
            console.error("Mic access denied:", err);
            alert("Microphone access was denied. Please allow audio permissions in your browser.");
        }
    }

    function stopRecording() {
        if (mediaRecorder && mediaRecorder.state === "recording") {
            mediaRecorder.stop();
            mediaRecorder.stream.getTracks().forEach(t => t.stop());
        }
        micBtn.classList.remove("recording");
        clearInterval(recordInterval);
        micTimer.style.display = "none";
    }

    // --- 7. Web Audio API Canvas Visualizer ---
    function initVisualizer() {
        if (audioCtx) return;
        try {
            const AudioContext = window.AudioContext || window.webkitAudioContext;
            audioCtx = new AudioContext();
            analyser = audioCtx.createAnalyser();
            analyser.fftSize = 256;
            
            visualizerSource = audioCtx.createMediaElementSource(audioPlayer);
            visualizerSource.connect(analyser);
            analyser.connect(audioCtx.destination);
            drawWaveform();
        } catch (e) {
            console.warn("AudioContext init error:", e);
        }
    }

    audioPlayer.addEventListener("play", () => {
        initVisualizer();
        if (audioCtx && audioCtx.state === "suspended") {
            audioCtx.resume();
        }
    });

    function drawWaveform() {
        animFrameId = requestAnimationFrame(drawWaveform);
        if (!analyser) return;

        const bufferLength = analyser.frequencyBinCount;
        const dataArray = new Uint8Array(bufferLength);
        analyser.getByteFrequencyData(dataArray);

        canvasCtx.fillStyle = "#030508";
        canvasCtx.fillRect(0, 0, canvas.width, canvas.height);

        const barWidth = (canvas.width / bufferLength) * 2.2;
        let x = 0;

        for (let i = 0; i < bufferLength; i++) {
            const barHeight = (dataArray[i] / 255) * canvas.height * 0.9;
            const grad = canvasCtx.createLinearGradient(0, canvas.height, 0, 0);
            grad.addColorStop(0, "#06b6d4");
            grad.addColorStop(0.7, "#3b82f6");
            grad.addColorStop(1, "#8b5cf6");

            canvasCtx.fillStyle = grad;
            canvasCtx.fillRect(x, canvas.height - barHeight, barWidth - 1, barHeight);
            x += barWidth;
        }
    }

    // --- 8. Run Deep Biomarker Analysis ---
    btnRunAnalysis.addEventListener("click", async () => {
        if (!currentAudioBlob && !currentSampleName) {
            alert("Please select a sample phonation, upload a WAV file, or record your voice.");
            return;
        }

        btnRunAnalysis.disabled = true;
        btnRunAnalysis.innerHTML = `<span class="status-pulse" style="display:inline-block;margin-right:8px;"></span> Analyzing Acoustic Biomarkers...`;

        try {
            let res;
            if (currentSampleName) {
                res = await fetch("/api/predict", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        sample_name: currentSampleName,
                        age: parseFloat(ageSlider.value),
                        sex: selectedSex
                    })
                });
            } else if (currentAudioBlob) {
                res = await fetch("/api/predict", {
                    method: "POST",
                    headers: { "Content-Type": "audio/wav" },
                    body: currentAudioBlob
                });
            }

            const data = await res.json();
            if (data.error) {
                alert(`Analysis Error: ${data.error}`);
                return;
            }

            displayResults(data);

        } catch (err) {
            console.error("Prediction error:", err);
            alert("Error communicating with biomarker diagnostic server.");
        } finally {
            btnRunAnalysis.disabled = false;
            btnRunAnalysis.innerHTML = `<svg class="cta-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"/></svg><span>Run Deep Biomarker Diagnostic</span>`;
        }
    });

    function displayResults(data) {
        resultsIdle.style.display = "none";
        resultsActive.style.display = "block";

        // Animate circular gauge
        const pct = data.probability_pd;
        const circumference = 2 * Math.PI * 50; // 314.16
        const offset = circumference - (pct / 100) * circumference;
        gaugeCircle.style.strokeDashoffset = offset;
        gaugeCircle.style.stroke = data.risk_color;
        gaugePct.textContent = `${pct}%`;
        gaugePct.style.color = data.risk_color;

        // Risk badge
        riskBadge.textContent = data.risk_tier;
        riskBadge.style.backgroundColor = `${data.risk_color}25`;
        riskBadge.style.color = data.risk_color;
        riskBadge.style.border = `1px solid ${data.risk_color}60`;

        diagTitle.textContent = data.label;
        diagRecommendation.textContent = data.recommendation;

        // Sample metadata tag if loaded
        if (data.sample_meta && data.sample_meta.sample_id) {
            sampleMetadataTag.textContent = `Patient File: ${data.sample_meta.sample_id} | Clinical Cohort Label: ${data.sample_meta.ground_truth}`;
            sampleMetadataTag.style.display = "block";
        } else {
            sampleMetadataTag.style.display = "none";
        }

        // Abnormal indicator flags
        if (data.abnormal_indicators && data.abnormal_indicators.length > 0) {
            flagsContainer.style.display = "block";
            flagsList.innerHTML = data.abnormal_indicators
                .map(flag => `<div class="flag-chip">${flag}</div>`)
                .join("");
        } else {
            flagsContainer.style.display = "none";
        }

        // Fill Biomarker Metric Cards
        const b = data.biomarkers;
        updateMetricCard("f0", `${b.f0} Hz`, b.f0, 85, 255, b.f0 >= 85 && b.f0 <= 255);
        updateMetricCard("hnr", `${b.hnr} dB`, b.hnr, 5, 25, b.hnr >= 15.0);
        updateMetricCard("jitter", `${b.jitter_percent} %`, b.jitter_percent, 0.1, 1.5, b.jitter_percent <= 0.60);
        updateMetricCard("shimmer", `${b.shimmer}`, b.shimmer, 0.01, 0.15, b.shimmer <= 0.05);
        updateMetricCard("rpde", `${b.rpde}`, b.rpde, 0.1, 1.0, b.rpde <= 0.65);
        updateMetricCard("dfa", `${b.dfa}`, b.dfa, 0.3, 1.2, b.dfa >= 0.5 && b.dfa <= 0.85);

        // Smooth scroll on mobile
        if (window.innerWidth < 980) {
            resultsActive.scrollIntoView({ behavior: "smooth" });
        }
    }

    function updateMetricCard(id, displayVal, rawVal, min, max, isNormal) {
        const valElem = document.getElementById(`val-${id}`);
        const badgeElem = document.getElementById(`badge-${id}`);
        const barElem = document.getElementById(`bar-${id}`);

        if (valElem) valElem.textContent = displayVal;
        
        if (badgeElem) {
            badgeElem.textContent = isNormal ? "Normal" : "Deviant";
            if (isNormal) {
                badgeElem.classList.remove("deviant");
            } else {
                badgeElem.classList.add("deviant");
            }
        }

        if (barElem) {
            const pct = Math.min(100, Math.max(5, ((rawVal - min) / (max - min)) * 100));
            barElem.style.width = `${pct}%`;
            barElem.style.backgroundColor = isNormal ? "var(--accent-cyan)" : "var(--accent-rose)";
        }
    }

    // --- 9. Cohort Explorer Table & Audio Audition ---
    let datasetRecords = [];

    async function fetchDatasetExplorer() {
        try {
            const res = await fetch("/api/dataset");
            const data = await res.json();
            datasetRecords = data.records || [];
            renderCohortTable(datasetRecords);
        } catch (e) {
            console.error("Error fetching dataset table:", e);
        }
    }

    function renderCohortTable(records) {
        if (!records || records.length === 0) {
            cohortTbody.innerHTML = `<tr><td colspan="10" style="text-align:center;">No records found matching filter.</td></tr>`;
            return;
        }

        cohortTbody.innerHTML = records.map(r => {
            const isPd = r.status === 1;
            const statusBadge = isPd
                ? `<span class="pill-pill pill-amber">Parkinson's</span>`
                : `<span class="pill-pill pill-emerald">Healthy</span>`;
            
            const f0Val = r["MDVP:Fo(Hz)"] ? `${Number(r["MDVP:Fo(Hz)"]).toFixed(1)} Hz` : "--";
            const jitterVal = r["MDVP:Jitter(%)"] ? `${(Number(r["MDVP:Jitter(%)"]) * 100).toFixed(2)}%` : "--";
            const shimmerVal = r["MDVP:Shimmer"] ? `${Number(r["MDVP:Shimmer"]).toFixed(3)}` : "--";
            const hnrVal = r["HNR"] ? `${Number(r["HNR"]).toFixed(1)} dB` : "--";
            const rpdeVal = r["RPDE"] ? `${Number(r["RPDE"]).toFixed(3)}` : "--";
            const ageVal = r["Age"] ? Math.round(r["Age"]) : "--";
            const sexVal = r["Sex"] || "--";

            const wavName = `${r.name}.wav`;

            return `
                <tr>
                    <td><strong>${r.name.substring(0, 24)}...</strong></td>
                    <td>${statusBadge}</td>
                    <td>${ageVal}</td>
                    <td>${sexVal}</td>
                    <td>${f0Val}</td>
                    <td>${jitterVal}</td>
                    <td>${shimmerVal}</td>
                    <td>${hnrVal}</td>
                    <td>${rpdeVal}</td>
                    <td>
                        <button class="audition-btn" data-file="${wavName}">Audition</button>
                    </td>
                </tr>
            `;
        }).join("");

        // Attach audition handlers
        document.querySelectorAll(".audition-btn").forEach(btn => {
            btn.addEventListener("click", () => {
                const file = btn.getAttribute("data-file");
                switchMode("samples");
                sampleSelect.value = file;
                loadSampleAudio(file);
                // Switch tab back to studio
                document.getElementById("tab-btn-studio").click();
                audioPlayer.play();
            });
        });
    }

    // Filter by Search Query
    tableSearch.addEventListener("input", (e) => {
        const query = e.target.value.toLowerCase();
        const filtered = datasetRecords.filter(r => r.name.toLowerCase().includes(query));
        renderCohortTable(filtered);
    });

    // Filter by Cohort Pills
    filterPills.forEach(pill => {
        pill.addEventListener("click", () => {
            filterPills.forEach(p => p.classList.remove("active"));
            pill.classList.add("active");
            const filterType = pill.getAttribute("data-filter");

            if (filterType === "pd") {
                renderCohortTable(datasetRecords.filter(r => r.status === 1));
            } else if (filterType === "hc") {
                renderCohortTable(datasetRecords.filter(r => r.status === 0));
            } else {
                renderCohortTable(datasetRecords);
            }
        });
    });

    // Initialize data
    fetchSamples();
    fetchDatasetExplorer();
});
