const { JSDOM } = require('jsdom');
const dom = new JSDOM('<!DOCTYPE html><html><body><div id="speaker-stats"></div><svg id="timeline"></svg><span id="now-label"></span><audio id="player-full"></audio><select id="filter-speaker"></select><select id="filter-flag"></select><input id="search" /><span id="visible-count"></span><tbody id="segments"></tbody></body></html>');
global.document = dom.window.document;
global.window = dom.window;

require('./test_js.js');
