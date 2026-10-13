const state = {
  dir: "",
  video: null,
  info: null,
  player: null,
  offset: 0,
  cues: [],
  subtitleCues: [],
  subtitleCache: {},
  socket: null,
  jobId: null,
  restarting: false,
  lastPositionSent: 0,
  detailId: null,
  detailRunning: false,
  dragging: false,
};

const filesEl = document.querySelector("#files");
const crumbsEl = document.querySelector("#crumbs");
const trackEl = document.querySelector("#track");
const langEl = document.querySelector("#lang");
const videoEl = document.querySelector("#video");
const captionEl = document.querySelector("#caption");
const hintEl = document.querySelector("#hint");
const subsEl = document.querySelector("#subtitle");
const seekEl = document.querySelector("#seek");
const timeNowEl = document.querySelector("#time-now");
const timeEndEl = document.querySelector("#time-end");
const togglePlayEl = document.querySelector("#toggle-play");

const LANG_LABEL = {
  chi: "中文",
  zho: "中文",
  zh: "中文",
  cmn: "中文",
  yue: "粤语",
  eng: "英文",
  en: "英文",
  jpn: "日语",
  ja: "日语",
  kor: "韩语",
  ko: "韩语",
  spa: "西班牙语",
  es: "西班牙语",
  fra: "法语",
  fre: "法语",
  fr: "法语",
  deu: "德语",
  ger: "德语",
  de: "德语",
  rus: "俄语",
  ru: "俄语",
  tur: "土耳其语",
  tr: "土耳其语",
  ita: "意大利语",
  it: "意大利语",
  und: "未知语言",
};

document.querySelector("#tab-jobs").onclick = openJobs;
document.querySelector("#jobs-close").onclick = closeJobs;
document.querySelector("#job-back").onclick = () => {
  showJobList();
  loadJobs();
};
document.querySelector("#jobs-view").onclick = (event) => {
  if (event.target.id === "jobs-view") closeJobs();
};
document.addEventListener("keydown", (event) => {
  if (event.key !== "Escape") return;
  const translate = document.querySelector("#translate-view");
  if (translate && !translate.hidden) {
    closeTranslate();
    return;
  }
  if (document.querySelector("#jobs-view").hidden) return;
  if (state.detailId) {
    showJobList();
    loadJobs();
    return;
  }
  closeJobs();
});
document.querySelector("#transcribe").onclick = () => startJob("transcribe");
document.querySelector("#translate").onclick = openTranslate;
document.querySelector("#translate-close").onclick = closeTranslate;
document.querySelector("#translate-confirm").onclick = () => {
  closeTranslate();
  startJob("translate");
};
document.querySelector("#translate-view").onclick = (event) => {
  if (event.target.id === "translate-view") closeTranslate();
};
langEl.onchange = updateTranslateNote;
trackEl.onchange = renderTrack;
subsEl.onchange = () => loadSubtitle();
document.querySelector("#play-gate").onclick = () => play(state.offset || 0);
togglePlayEl.onclick = togglePlay;
document.querySelector("#full").onclick = toggleFull;
seekEl.addEventListener("pointerdown", () => {
  state.dragging = true;
});
seekEl.addEventListener("input", () => {
  timeNowEl.textContent = formatClock(Number(seekEl.value));
});
seekEl.addEventListener("change", () => {
  state.dragging = false;
  const target = Number(seekEl.value) || 0;
  if (state.player) play(target);
  else {
    state.offset = target;
    timeNowEl.textContent = formatClock(target);
  }
});
videoEl.ontimeupdate = paintCue;

loadLanguages();
loadDir("");
setInterval(paintCue, 200);

function openJobs() {
  const view = document.querySelector("#jobs-view");
  view.hidden = false;
  document.querySelector("#tab-jobs").classList.add("active");
  loadJobs();
}

function closeJobs() {
  showJobList();
  document.querySelector("#jobs-view").hidden = true;
  document.querySelector("#tab-jobs").classList.remove("active");
}

function showJobList() {
  state.detailId = null;
  state.detailRunning = false;
  document.querySelector("#jobs-list").hidden = false;
  document.querySelector("#job-detail").hidden = true;
}

async function loadLanguages() {
  const rows = await fetch("/api/languages").then((res) => res.json());
  langEl.innerHTML = rows.map((row) => `<option value="${row.id}">${row.label}</option>`).join("");
}

async function loadDir(path) {
  state.dir = path;
  let data;
  try {
    data = await fetch("/api/files?path=" + encodeURIComponent(path)).then(readJson);
  } catch (err) {
    filesEl.innerHTML = "";
    const item = document.createElement("li");
    item.textContent = err.message;
    filesEl.append(item);
    return;
  }
  crumbsEl.innerHTML = "";
  crumbsEl.hidden = !path;
  if (path) {
    const back = document.createElement("button");
    back.type = "button";
    back.textContent = "上一级";
    const parent = path.split("/").slice(0, -1).join("/");
    back.onclick = () => loadDir(parent);
    crumbsEl.append(back);
  }
  filesEl.innerHTML = "";
  if (!data.entries.length) {
    const item = document.createElement("li");
    item.className = "note";
    item.textContent = "这个目录里没有影片";
    filesEl.append(item);
    return;
  }
  for (const entry of data.entries) {
    const item = document.createElement("li");
    item.textContent = entry.name;
    item.title = entry.name;
    item.onclick = () => (entry.kind === "dir" ? loadDir(entry.path) : openVideo(entry.path));
    if (state.video && state.video.path === entry.path) item.classList.add("selected");
    filesEl.append(item);
  }
}

async function openVideo(path) {
  state.info = await fetch("/api/files/info?path=" + encodeURIComponent(path)).then(readJson);
  state.video = { path };
  document.querySelector("#now").textContent = state.info.name;
  trackEl.replaceChildren();
  for (const track of state.info.tracks) {
    const option = document.createElement("option");
    option.value = String(track.streamIndex);
    option.textContent = trackLabel(track);
    trackEl.append(option);
  }
  const preferred = state.info.tracks.find((track) => track.default) || state.info.tracks[0];
  if (preferred) trackEl.value = String(preferred.streamIndex);
  renderTrack();
  stopPlayer();
  state.offset = 0;
  state.dragging = false;
  const duration = state.info.durationSec || 0;
  seekEl.max = String(Math.max(0, Math.floor(duration)));
  seekEl.value = "0";
  timeNowEl.textContent = "0:00";
  timeEndEl.textContent = formatClock(duration);
  document.querySelector("#transport").hidden = false;
  document.querySelector("#play-gate").hidden = false;
  loadSubtitle();
  loadDir(state.dir);
}

function currentTrack() {
  if (!state.info) return null;
  return state.info.tracks.find((track) => String(track.streamIndex) === trackEl.value) || null;
}

function languageName(tag) {
  return LANG_LABEL[tag] || tag || "未知语言";
}

function codecName(codec) {
  const key = (codec || "").toLowerCase();
  const names = {
    aac: "AAC",
    ac3: "AC3",
    eac3: "E-AC3",
    dts: "DTS",
    truehd: "TrueHD",
    flac: "FLAC",
    mp3: "MP3",
    opus: "Opus",
    vorbis: "Vorbis",
    pcm_s16le: "PCM",
    pcm_s24le: "PCM",
  };
  if (!key) return "";
  return names[key] || key.toUpperCase();
}

function trackLabel(track) {
  const lang = languageName(track.language);
  const codec = codecName(track.codec);
  return codec ? `${lang} ${codec}` : lang;
}

function renderSubtitles() {
  const previous = subsEl.value;
  const track = currentTrack();
  subsEl.replaceChildren();
  const rows = (state.info && state.info.aiSubtitles) || [];
  const mine = track ? rows.filter((row) => row.streamIndex === track.streamIndex) : rows;
  const off = document.createElement("option");
  off.value = "";
  off.textContent = mine.length ? "关闭" : state.info ? "没有字幕" : "选择影片后显示";
  subsEl.append(off);
  for (const row of mine) {
    const option = document.createElement("option");
    option.value = row.id;
    option.textContent = row.label;
    subsEl.append(option);
  }
  if ([...subsEl.options].some((option) => option.value === previous)) subsEl.value = previous;
}

function selectedSubtitle() {
  if (!state.info || !subsEl.value) return null;
  return (state.info.aiSubtitles || []).find((row) => row.id === subsEl.value) || null;
}

function clearTracks() {
  for (const node of [...videoEl.querySelectorAll("track")]) node.remove();
  for (const track of videoEl.textTracks) track.mode = "disabled";
}

function clockToSec(value) {
  const [hms, fraction = "0"] = value.trim().split(".");
  const parts = hms.split(":").map(Number);
  const seconds = parts.pop() || 0;
  const minutes = parts.pop() || 0;
  const hours = parts.pop() || 0;
  const millis = Number(fraction.padEnd(3, "0").slice(0, 3));
  return hours * 3600 + minutes * 60 + seconds + millis / 1000;
}

function parseVtt(text) {
  const cues = [];
  for (const block of text.replace(/\r/g, "").split(/\n\n+/)) {
    const lines = block.split("\n").filter((line) => line.trim() && !line.startsWith("WEBVTT") && !line.startsWith("NOTE"));
    const timing = lines.findIndex((line) => line.includes("-->"));
    if (timing < 0) continue;
    const [start, end] = lines[timing].split("-->").map((part) => part.trim().split(/\s+/)[0]);
    cues.push({ startSec: clockToSec(start), endSec: clockToSec(end), text: lines.slice(timing + 1).join("\n") });
  }
  return cues;
}

async function loadSubtitle() {
  clearTracks();
  const row = selectedSubtitle();
  state.subtitleCues = [];
  if (!row) return;
  if (state.subtitleCache[cacheKey(row)]) {
    state.subtitleCues = state.subtitleCache[cacheKey(row)];
    return;
  }
  const response = await fetch(row.url);
  const cues = parseVtt(await response.text());
  state.subtitleCache[cacheKey(row)] = cues;
  if (selectedSubtitle() && selectedSubtitle().id === row.id) state.subtitleCues = cues;
}

function cacheKey(row) {
  return (state.video ? state.video.path : "") + "|" + row.id;
}

function renderTrack() {
  renderSubtitles();
  const track = currentTrack();
  const transcribe = document.querySelector("#transcribe");
  const translate = document.querySelector("#translate");
  if (!track) {
    transcribe.disabled = true;
    translate.disabled = true;
    return;
  }
  transcribe.disabled = false;
  translate.disabled = !track.srtReady;
  const names = track.translations.map((id) => languageName(id));
  const ready = names.length ? `已有翻译：${names.join("、")}` : "";
  hintEl.textContent = track.srtReady
    ? ready
      ? `原语言字幕已就绪。${ready}`
      : "原语言字幕已就绪。"
    : track.wavReady
      ? "音频已抽出，还没有原语言字幕。"
      : "还没有抽出这条音轨。";
}

function openTranslate() {
  const track = currentTrack();
  if (!state.video || !track || !track.srtReady) return;
  const from = languageName(track.language);
  document.querySelector("#translate-from").textContent = `当前音轨是${from}，原语言字幕会翻译成你选的语言。`;
  updateTranslateNote();
  document.querySelector("#translate-view").hidden = false;
}

function closeTranslate() {
  document.querySelector("#translate-view").hidden = true;
}

function updateTranslateNote() {
  const track = currentTrack();
  const note = document.querySelector("#translate-note");
  const confirm = document.querySelector("#translate-confirm");
  if (!track || !note) return;
  const target = langEl.value;
  const from = languageName(track.language);
  const to = languageName(target);
  const sameLang = track.languageShort === target || from === to;
  const already = (track.translations || []).includes(target);
  if (confirm) confirm.disabled = sameLang || already;
  if (sameLang) {
    note.textContent = `音轨已经是${from}，和目标相同，无需翻译。`;
    return;
  }
  if (already) {
    note.textContent = `已经有${to}字幕，无需再翻译。`;
    return;
  }
  note.textContent = `把${from}字幕翻译成${to}。`;
}

function play(start) {
  if (!state.video) return;
  state.restarting = true;
  stopPlayer();
  state.offset = start;
  const track = currentTrack();
  const index = track ? track.streamIndex : "";
  const url =
    "/api/files/stream?path=" +
    encodeURIComponent(state.video.path) +
    "&start=" +
    start +
    "&streamIndex=" +
    index;
  if (!window.mpegts || !mpegts.getFeatureList().mseLivePlayback) {
    hintEl.textContent = "这个浏览器不能播 MPEG-TS";
    state.restarting = false;
    document.querySelector("#play-gate").hidden = false;
    return;
  }
  state.player = mpegts.createPlayer({ type: "mpegts", isLive: true, url });
  state.player.attachMediaElement(videoEl);
  state.player.load();
  state.player.play();
  state.restarting = false;
  document.querySelector("#play-gate").hidden = true;
}

function stopPlayer() {
  if (state.player) {
    state.player.pause();
    state.player.unload();
    state.player.detachMediaElement();
    state.player.destroy();
    state.player = null;
  }
  videoEl.removeAttribute("src");
  videoEl.load();
  clearTracks();
}

videoEl.addEventListener("play", () => {
  togglePlayEl.textContent = "❚❚";
  togglePlayEl.setAttribute("aria-label", "暂停");
});
videoEl.addEventListener("pause", () => {
  togglePlayEl.textContent = "▶";
  togglePlayEl.setAttribute("aria-label", "播放");
});

function togglePlay() {
  if (!state.video) return;
  if (!state.player) {
    play(Number(seekEl.value) || 0);
    return;
  }
  if (videoEl.paused) videoEl.play();
  else videoEl.pause();
}

function toggleFull() {
  const stage = document.querySelector("#stage");
  if (document.fullscreenElement) document.exitFullscreen();
  else stage.requestFullscreen();
}

function formatClock(seconds) {
  const total = Math.max(0, Math.floor(seconds || 0));
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const secs = total % 60;
  const mm = String(minutes).padStart(2, "0");
  const ss = String(secs).padStart(2, "0");
  return hours ? `${hours}:${mm}:${ss}` : `${minutes}:${ss}`;
}

function movieTime() {
  const media = videoEl.currentTime || 0;
  if (media + 1 >= state.offset) return media;
  return state.offset + media;
}

function paintCue() {
  const position = movieTime();
  const now = performance.now();
  const cues = state.subtitleCues.length ? state.subtitleCues : state.cues;
  const cue = cues.find((item) => position >= item.startSec && position < item.endSec);
  captionEl.textContent = cue ? cue.text : "";
  if (!state.dragging && state.info && state.player) {
    const duration = Number(seekEl.max) || position;
    seekEl.value = String(Math.min(Math.floor(position), duration));
    timeNowEl.textContent = formatClock(position);
  }
  if (state.socket && state.socket.readyState === WebSocket.OPEN && now - state.lastPositionSent > 1000) {
    state.lastPositionSent = now;
    state.socket.send(JSON.stringify({ type: "position", positionSec: position }));
  }
}

async function startJob(mode) {
  const track = currentTrack();
  if (!state.video || !track) return;
  const body = {
    path: state.video.path,
    mode,
    streamIndex: track.streamIndex,
    targetLang: langEl.value,
  };
  const response = await fetch("/api/jobs", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const payload = await response.json();
  if (!response.ok) {
    hintEl.textContent = payload.detail || "任务没有开始";
    return;
  }
  state.jobId = payload.id;
  state.cues = [];
  const targetName = languageName(langEl.value);
  const fromName = languageName(track.language);
  hintEl.textContent =
    mode === "transcribe"
      ? "正在把音频转成文字，打开右上角的任务可看进度"
      : `正在把${fromName}翻译成${targetName}，打开右上角的任务可看进度`;
  if (mode === "translate") openSocket(payload.id);
}

function openSocket(jobId) {
  if (state.socket) state.socket.close();
  const protocol = location.protocol === "https:" ? "wss" : "ws";
  const socket = new WebSocket(`${protocol}://${location.host}/api/jobs/${jobId}`);
  state.socket = socket;
  socket.onopen = () => socket.send(JSON.stringify({ type: "resume", after: 0 }));
  socket.onmessage = (event) => {
    const message = JSON.parse(event.data);
    if (message.type === "cue") {
      const existing = state.cues.findIndex((cue) => cue.seq === message.seq);
      if (existing >= 0) state.cues[existing] = message;
      else state.cues.push(message);
    }
    if (message.type === "done") {
      hintEl.textContent = "翻译完成";
      if (state.video) refreshInfo();
    }
    if (message.type === "error") hintEl.textContent = message.message;
  };
}

async function refreshInfo() {
  if (!state.video) return;
  state.info = await fetch("/api/files/info?path=" + encodeURIComponent(state.video.path)).then(readJson);
  renderTrack();
}

function jobCell(text) {
  const cell = document.createElement("td");
  cell.textContent = text;
  return cell;
}

async function loadJobs() {
  const data = await fetch("/api/jobs").then((res) => res.json());
  const body = document.querySelector("#job-rows");
  const empty = document.querySelector("#jobs-empty");
  body.innerHTML = "";
  empty.hidden = data.jobs.length > 0;
  body.closest("table").hidden = data.jobs.length === 0;
  for (const job of data.jobs) {
    const row = document.createElement("tr");
    const name = (job.videoPath || "").split("/").pop();
    const progress =
      job.totalCount != null ? `${job.doneCount} / ${job.totalCount}` : String(job.doneCount ?? "");
    const lang =
      job.kind === "translate"
        ? `${languageName(job.language)} → ${languageName(job.targetLang)}`
        : languageName(job.language);
    const film = jobCell(name);
    film.title = job.videoPath || "";
    const status = document.createElement("td");
    const badge = document.createElement("span");
    badge.className = "status " + job.status;
    badge.textContent = statusText(job);
    status.append(badge);
    if (job.error) status.title = job.error;
    const actions = document.createElement("td");
    const wrap = document.createElement("div");
    wrap.className = "row-actions";
    if (job.status === "running") {
      const stop = document.createElement("button");
      stop.type = "button";
      stop.textContent = "停止";
      stop.onclick = (event) => {
        event.stopPropagation();
        fetch("/api/jobs/" + job.id, { method: "DELETE" }).then(loadJobs);
      };
      wrap.append(stop);
    }
    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "danger";
    remove.textContent = "删除";
    remove.onclick = (event) => {
      event.stopPropagation();
      const running = job.status === "running" ? "这个任务还在跑，删除会先停掉它。 " : "";
      if (!window.confirm(`${running}删除这条任务记录？`)) return;
      fetch("/api/jobs/" + job.id + "?remove=1", { method: "DELETE" }).then(() => {
        if (state.detailId === job.id) showJobList();
        loadJobs();
      });
    };
    wrap.append(remove);
    actions.append(wrap);
    row.append(
      film,
      jobCell(kindText(job.kind)),
      jobCell(lang),
      status,
      jobCell(progress),
      jobCell(elapsedText(job)),
      actions,
    );
    row.onclick = () => openDetail(job.id);
    body.append(row);
  }
}

async function openDetail(id) {
  state.detailId = id;
  document.querySelector("#jobs-list").hidden = true;
  document.querySelector("#job-detail").hidden = false;
  await loadDetail();
}

async function loadDetail() {
  if (!state.detailId) return;
  const job = await fetch("/api/jobs/" + state.detailId).then(readJson);
  state.detailRunning = job.status === "running";
  const name = (job.videoPath || "").split("/").pop();
  const lang =
    job.kind === "translate"
      ? `${languageName(job.language)} → ${languageName(job.targetLang)}`
      : languageName(job.language);
  const progress =
    job.totalCount != null ? `${job.doneCount} / ${job.totalCount}` : String(job.doneCount ?? "");
  document.querySelector("#job-detail-title").textContent = name;
  document.querySelector("#job-detail-sub").textContent =
    `${kindText(job.kind)} · ${lang} · ${statusText(job)} · ${elapsedText(job)} · ${progress}`;
  document.querySelector("#job-log").textContent = job.log || "还没有日志";
}

function kindText(kind) {
  return kind === "translate" ? "翻译" : "音频转文字";
}

function elapsedText(job) {
  if (!job.createdAt) return "";
  const start = Date.parse(job.createdAt);
  const end = job.finishedAt ? Date.parse(job.finishedAt) : Date.now();
  if (Number.isNaN(start) || Number.isNaN(end)) return "";
  const total = Math.max(0, Math.round((end - start) / 1000));
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const seconds = total % 60;
  if (hours) return `${hours} 时 ${minutes} 分`;
  if (minutes) return `${minutes} 分 ${seconds} 秒`;
  return `${seconds} 秒`;
}

function statusText(job) {
  if (job.status === "running" && job.stage === "extract") return "抽音频";
  if (job.status === "running" && job.stage === "transcribe") return "转文字";
  if (job.status === "running" && job.stage === "translate") return "翻译";
  return { running: "进行中", done: "已完成", stopped: "已停止", failed: "失败" }[job.status] || job.status;
}

async function readJson(response) {
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.detail || response.statusText);
  return payload;
}

setInterval(() => {
  const view = document.querySelector("#jobs-view");
  if (!view || view.hidden) return;
  if (state.detailId) {
    if (state.detailRunning) loadDetail();
    return;
  }
  loadJobs();
}, 1000);
