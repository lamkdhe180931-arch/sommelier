const assert = require("assert");
const fs = require("fs");
const path = require("path");
const { JSDOM } = require("jsdom");

const ROOT = path.resolve(__dirname, "..");
const TEMPLATE = fs.readFileSync(path.join(ROOT, "tools", "sortformer_tuning_viewer.html"), "utf8");

function fixtureData() {
  return {
    schema_version: 1,
    audio_name: "fixture audio.wav",
    audio_path: "/tmp/fixture audio.wav",
    audio_candidates: ["/tmp/fixture audio.wav"],
    audio_duration_seconds: 3,
    defaults: {
      onset: 0.5,
      offset: 0.4,
      min_duration_on: 0.1,
      min_duration_off: 0.2,
      sortformer_pad_onset: 0,
      sortformer_pad_offset: 0,
    },
    chunks: [
      {
        chunk_index: 0,
        offset: 0,
        duration: 3,
        frame_count: 6,
        speaker_count: 3,
        frame_shift: 0.5,
        probs: [
          [0.1, 0.6, 0.1],
          [0.6, 0.7, 0.1],
          [0.55, 0.2, 0.9],
          [0.1, 0.1, 0.8],
          [0.58, 0.1, 0.1],
          [0.2, 0.1, 0.1],
        ],
      },
    ],
    summary: {
      chunk_count: 1,
      total_frame_count: 6,
      max_speaker_count: 3,
    },
  };
}

function buildDom() {
  const html = TEMPLATE.replace(
    "window.SORTFORMER_TUNING_DATA = null;",
    `window.SORTFORMER_TUNING_DATA = ${JSON.stringify(fixtureData())};`
  );

  return new JSDOM(html, {
    runScripts: "dangerously",
    url: "file:///tmp/sortformer_tuning.html",
    beforeParse(window) {
      window.Element.prototype.scrollIntoView = function () {};
      Object.defineProperty(window.HTMLMediaElement.prototype, "paused", {
        configurable: true,
        get() {
          return !this.__playing;
        },
      });
      window.HTMLMediaElement.prototype.play = function () {
        this.__playing = true;
        this.__playCalls = (this.__playCalls || 0) + 1;
        return Promise.resolve();
      };
      window.HTMLMediaElement.prototype.pause = function () {
        this.__playing = false;
        this.__pauseCalls = (this.__pauseCalls || 0) + 1;
      };
    },
  });
}

function rows(document) {
  return Array.from(document.querySelectorAll("#segments tr"));
}

function click(window, el) {
  el.dispatchEvent(new window.MouseEvent("click", { bubbles: true, cancelable: true }));
}

function pressSpace(window, target = window.document) {
  const ev = new window.KeyboardEvent("keydown", {
    key: " ",
    code: "Space",
    bubbles: true,
    cancelable: true,
  });
  target.dispatchEvent(ev);
  return ev;
}

{
  const dom = buildDom();
  const { document, Event } = dom.window;
  const player = document.getElementById("player");

  assert.strictEqual(rows(document).length, 4, "fixture should binarize into four rows");
  assert.ok(player.getAttribute("src").includes("fixture%20audio.wav"), "audio path is URL encoded");

  click(dom.window, rows(document)[0]);

  assert.strictEqual(player.currentTime, 0, "clicking a segment seeks to its start");
  assert.strictEqual(player.__playCalls, 1, "clicking a segment starts playback");
  assert.strictEqual(rows(document)[0].classList.contains("active-row"), true, "clicked row is highlighted");

  player.currentTime = 1.01;
  player.dispatchEvent(new Event("timeupdate"));

  assert.strictEqual(player.paused, true, "segment playback pauses when it reaches the segment end");
  assert.strictEqual(player.currentTime, 1, "segment playback clamps to the segment end");
}

{
  const dom = buildDom();
  const { document, Event } = dom.window;
  const onset = document.querySelector('[data-range="onset"]');

  onset.value = "0.95";
  onset.dispatchEvent(new Event("input", { bubbles: true }));

  assert.strictEqual(rows(document).length, 0, "raising onset recomputes and removes weak segments");
  assert.ok(document.getElementById("command").value.includes("--onset 0.95"), "command mirrors tuned params");
}

{
  const dom = buildDom();
  const { document } = dom.window;
  const player = document.getElementById("player");

  const startEvent = pressSpace(dom.window);

  assert.strictEqual(startEvent.defaultPrevented, true, "Space prevents scroll while toggling audio");
  assert.strictEqual(player.paused, false, "Space starts sticky audio");

  pressSpace(dom.window);

  assert.strictEqual(player.paused, true, "Space pauses sticky audio");

  const callsBeforeInputSpace = player.__playCalls || 0;
  pressSpace(dom.window, document.querySelector('[data-range="onset"]'));

  assert.strictEqual(player.__playCalls || 0, callsBeforeInputSpace, "Space inside controls is ignored");
}

console.log("sortformer tuning viewer interactions ok");
