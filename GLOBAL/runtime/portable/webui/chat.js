// Offline tutor chat. Talks to the llama-server that serves this page (same origin).
// Applies the model profile like the CoreCS desktop app: preamble folded into the FIRST user
// turn (several chat templates reject a system role), sampling passed through, and the oldest
// turns trimmed to fit the context window.
(function () {
  "use strict";
  var CHARS_PER_TOKEN = 3;
  var STORE_KEY = "corecs-portable-chat";
  var profile = { sampling: {}, context_length: 4096 };
  var messages = [];
  var controller = null;

  var $ = function (id) { return document.getElementById(id); };
  var esc = function (s) {
    return String(s).replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
  };

  // Right-to-left when RTL words make up a real share of the text. Counting words (not the
  // first character, like dir="auto") keeps "TCP is ..." in Hebrew right-to-left.
  function textDir(text) {
    var plain = String(text).replace(/```[\s\S]*?```/g, " ").replace(/`[^`]*`/g, " ");
    var rtl = (plain.match(/[֐-׿؀-ۿ]+/g) || []).length;
    var ltr = (plain.match(/[A-Za-z]+/g) || []).length;
    return rtl > 0 && rtl * 3 >= ltr ? "rtl" : "ltr";
  }

  function markdown(src) {
    var parts = String(src).split(/```/);
    var html = "";
    for (var i = 0; i < parts.length; i++) {
      if (i % 2 === 1) {
        var body = parts[i].replace(/^[\w+-]*\n/, "");
        html += "<pre><code>" + esc(body) + "</code></pre>";
        continue;
      }
      var blocks = parts[i].split(/\n{2,}/);
      for (var b = 0; b < blocks.length; b++) {
        var block = blocks[b].trim();
        if (!block) continue;
        var lines = block.split("\n");
        var inline = function (s) {
          return esc(s).replace(/`([^`]+)`/g, "<code>$1</code>").replace(/\*\*([^*]+)\*\*/g, "<b>$1</b>");
        };
        if (lines.every(function (l) { return /^\s*([-*]|\d+[.)])\s+/.test(l); })) {
          var ordered = /^\s*\d/.test(lines[0]);
          html += (ordered ? "<ol>" : "<ul>") + lines.map(function (l) {
            return "<li>" + inline(l.replace(/^\s*([-*]|\d+[.)])\s+/, "")) + "</li>";
          }).join("") + (ordered ? "</ol>" : "</ul>");
        } else {
          html += "<p>" + lines.map(inline).join("<br>") + "</p>";
        }
      }
    }
    return html;
  }

  function save() { try { localStorage.setItem(STORE_KEY, JSON.stringify(messages)); } catch (e) {} }
  function load() { try { return JSON.parse(localStorage.getItem(STORE_KEY) || "[]"); } catch (e) { return []; } }

  function render() {
    $("empty").hidden = messages.length > 0;
    $("messages").innerHTML = messages.map(function (m, i) {
      var streaming = controller && i === messages.length - 1 && m.role === "assistant";
      return '<div class="msg ' + m.role + (streaming ? " cursor" : "") + '" dir="' + textDir(m.content) + '">' +
        (m.role === "user" ? "<p>" + esc(m.content).replace(/\n/g, "<br>") + "</p>" : markdown(m.content)) + "</div>";
    }).join("");
    var main = $("chat");
    main.scrollTop = main.scrollHeight;
  }

  function buildRequestMessages() {
    var budget = Math.max(512, (profile.context_length || 4096) - (profile.sampling.max_tokens || 1024)) * CHARS_PER_TOKEN;
    var pre = profile.prompt && profile.prompt.preamble ? profile.prompt.preamble + (profile.prompt.separator || "\n\n") : "";
    budget -= pre.length;
    var turns = [];
    var used = 0;
    for (var i = messages.length - 1; i >= 0; i--) {
      var m = messages[i];
      if (used + m.content.length > budget && turns.length) break;
      used += m.content.length;
      turns.unshift({ role: m.role, content: m.content });
    }
    while (turns.length && turns[0].role !== "user") turns.shift();
    if (turns.length && pre) turns[0].content = pre + turns[0].content;
    return turns;
  }

  async function send(text) {
    if (!text.trim() || controller) return;
    messages.push({ role: "user", content: text.trim() });
    var reply = { role: "assistant", content: "" };
    var body = Object.assign({ stream: true }, profile.sampling, { messages: buildRequestMessages() });
    messages.push(reply);
    controller = new AbortController();
    setBusy(true);
    render();
    try {
      var res = await fetch("/v1/chat/completions", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body), signal: controller.signal,
      });
      if (!res.ok) throw new Error("The tutor engine returned " + res.status);
      var reader = res.body.getReader();
      var decoder = new TextDecoder();
      var buf = "";
      for (;;) {
        var chunk = await reader.read();
        if (chunk.done) break;
        buf += decoder.decode(chunk.value, { stream: true });   // keeps split UTF-8 intact
        var lines = buf.split("\n");
        buf = lines.pop();
        for (var i = 0; i < lines.length; i++) {
          var line = lines[i].trim();
          if (!line.startsWith("data:")) continue;
          var data = line.slice(5).trim();
          if (data === "[DONE]") continue;
          try {
            var delta = JSON.parse(data).choices[0].delta;
            if (delta && delta.content) { reply.content += delta.content; render(); }
          } catch (e) {}
        }
      }
    } catch (e) {
      if (e.name !== "AbortError") reply.content += (reply.content ? "\n\n" : "") + "⚠ " + e.message;
    } finally {
      controller = null;
      setBusy(false);
      if (!reply.content) messages.pop();
      save();
      render();
    }
  }

  function setBusy(busy) {
    $("send").textContent = busy ? "Stop" : "Send";
    $("send").classList.toggle("stop", busy);
  }

  async function health() {
    try {
      var r = await fetch("/health");
      $("status").textContent = r.ok ? "Ready · offline" : "Loading model…";
      $("status").className = "status " + (r.ok ? "ok" : "");
    } catch (e) {
      $("status").textContent = "Engine stopped";
      $("status").className = "status bad";
    }
  }

  async function init() {
    try {
      var r = await fetch("profile.json", { cache: "no-store" });
      if (r.ok) profile = Object.assign(profile, await r.json());
    } catch (e) {}
    profile.sampling = profile.sampling || {};
    if (profile.display_name) { $("name").textContent = profile.display_name; document.title = profile.display_name; }
    if (profile.courses && profile.courses.length) {
      $("courses").textContent = "Your tutor for " + profile.courses.join(" · ");
      $("courses").dir = textDir(profile.courses.join(" "));
    }
    $("examples").innerHTML = (profile.example_questions || []).map(function (q) {
      return '<button type="button" dir="' + textDir(q) + '">' + esc(q) + "</button>";
    }).join("");
    $("examples").onclick = function (e) { if (e.target.tagName === "BUTTON") send(e.target.textContent); };
    messages = load();
    render();
    health();
    setInterval(health, 5000);
  }

  $("composer").onsubmit = function (e) {
    e.preventDefault();
    if (controller) { controller.abort(); return; }
    var text = $("input").value;
    $("input").value = "";
    $("input").style.height = "";
    send(text);
  };
  $("input").onkeydown = function (e) {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); $("composer").requestSubmit(); }
  };
  $("input").oninput = function () {
    this.style.height = "";
    this.style.height = Math.min(this.scrollHeight, 200) + "px";
  };
  $("new").onclick = function () {
    if (controller) controller.abort();
    messages = [];
    save();
    render();
  };
  init();
})();
