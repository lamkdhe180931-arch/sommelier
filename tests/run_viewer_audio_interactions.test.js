const assert = require("assert");
const fs = require("fs");
const path = require("path");
const { JSDOM } = require("jsdom");

const ROOT = path.resolve(__dirname, "..");

function buildViewer() {
  const segments = [
    {
      index: "00001",
      start: 1,
      end: 2.5,
      text: "first segment",
      speaker: "SPEAKER_04",
      audio_file: "data_audio/00001.mp3",
    },
    {
      index: "00002",
      start: 10,
      end: 12,
      text: "second segment",
      speaker: "SPEAKER_05",
      audio_file: "data_audio/00002.mp3",
    },
  ];
  const html = fs
    .readFileSync(path.join(ROOT, "index.html"), "utf8")
    .replace(
      "<script>\nconst COLORS",
      `<script>
const STAGE_DATA = {"05_export": ${JSON.stringify(segments)}};
const VAD_CHUNKS = [];
let current_stage = "05_export";
let SEGMENTS = STAGE_DATA[current_stage];
</script>
<script>
const COLORS`
    );

  return new JSDOM(html, {
    runScripts: "dangerously",
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
  const dom = buildViewer();
  const { document, Event } = dom.window;
  const player = document.getElementById("player-full");
  const rows = document.querySelectorAll("#segments tr");

  assert.strictEqual(rows.length, 2, "fixture should render two segment rows");

  click(dom.window, rows[1].querySelector(".seg-text"));

  assert.strictEqual(player.currentTime, 10, "clicking anywhere in a row seeks to segment start");
  assert.strictEqual(player.__playCalls, 1, "clicking a row starts sticky player playback");

  player.currentTime = 12.01;
  player.dispatchEvent(new Event("timeupdate"));

  assert.strictEqual(player.paused, true, "segment playback pauses after the segment end");
}

{
  const dom = buildViewer();
  const { document } = dom.window;
  const player = document.getElementById("player-full");

  const startEvent = pressSpace(dom.window);

  assert.strictEqual(startEvent.defaultPrevented, true, "Space prevents page scroll while toggling audio");
  assert.strictEqual(player.paused, false, "Space starts playback when the sticky player is paused");

  pressSpace(dom.window);

  assert.strictEqual(player.paused, true, "Space pauses playback when the sticky player is playing");

  const callsBeforeInputSpace = player.__playCalls || 0;
  pressSpace(dom.window, document.getElementById("search"));

  assert.strictEqual(player.__playCalls || 0, callsBeforeInputSpace, "Space in search input is ignored");
}

console.log("run viewer audio interactions ok");
