const express = require('express');
const cors = require('cors');
const fs = require('fs');
const path = require('path');

const app = express();
app.use(cors());
app.use(express.json());

const PORT = process.env.PORT || 6142;
const configPath = process.env.EMII_CONFIG || path.join(process.env.APPDATA || '.', 'EmiiPilot', 'config.json');

function loadConfig() {
  // 1. Try env vars first (Render, cloud)
  if (process.env.OPENAI_API_KEY || process.env.ANTHROPIC_API_KEY || process.env.GEMINI_API_KEY) {
    const key = process.env.OPENAI_API_KEY || '';
    const baseUrl = process.env.OPENAI_BASE_URL || (key.startsWith('sk-or-') ? 'https://openrouter.ai/api/v1' : 'https://api.openai.com/v1');
    return {
      emiiApiKey: process.env.EMII_API_KEY || '',
      openai: key, openaiBaseUrl: baseUrl,
      anthropic: process.env.ANTHROPIC_API_KEY || '',
      gemini: process.env.GEMINI_API_KEY || '',
      emiiUrl: process.env.EMII_API_URL || '',
    };
  }
  // 2. Fallback: local config.json
  try {
    const raw = fs.readFileSync(configPath, 'utf8');
    const cfg = JSON.parse(raw);
    const keys = cfg.ApiKeys || {};
    const oKey = keys.OpenAI || '';
    const oBase = oKey.startsWith('sk-or-') ? 'https://openrouter.ai/api/v1' : 'https://api.openai.com/v1';
    return {
      emiiApiKey: keys.EmiiAPI || '',
      openai: oKey, openaiBaseUrl: oBase,
      anthropic: keys.Anthropic || '',
      gemini: keys.Gemini || '',
      emiiUrl: cfg.EmiiApiUrl || '',
    };
  } catch {
    return { emiiApiKey: '', openai: '', anthropic: '', gemini: '', emiiUrl: '', openaiBaseUrl: 'https://api.openai.com/v1' };
  }
}

function detectProvider(model) {
  if (!model) return null;
  const m = model.toLowerCase();
  if (m.startsWith('gpt') || m.startsWith('o1') || m.startsWith('o3')) return 'openai';
  if (m.startsWith('claude')) return 'anthropic';
  if (m.startsWith('gemini')) return 'gemini';
  if (m.startsWith('ollama') || m.includes('/')) return 'ollama';
  return 'openai';
}

async function* streamOpenAI(body, apiKey, baseUrl = 'https://api.openai.com/v1') {
  const res = await fetch(`${baseUrl}/chat/completions`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${apiKey}` },
    body: JSON.stringify({ ...body, stream: true }),
  });
  if (!res.ok) throw new Error(`OpenAI ${res.status}: ${await res.text()}`);
  const reader = res.body.getReader();
  const dec = new TextDecoder();
  let buf = '';
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    const lines = buf.split('\n');
    buf = lines.pop() || '';
    for (const line of lines) {
      if (line.startsWith('data: ')) yield line.slice(6) + '\n';
    }
  }
}

async function* streamAnthropic(body, apiKey) {
  const messages = body.messages || [];
  const sys = messages.filter(m => m.role === 'system').pop();
  const chat = messages.filter(m => m.role !== 'system');
  const res = await fetch('https://api.anthropic.com/v1/messages', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'x-api-key': apiKey, 'anthropic-version': '2023-06-01' },
    body: JSON.stringify({
      model: body.model, max_tokens: body.max_tokens || 4096, temperature: body.temperature ?? 0.7,
      system: sys?.content, messages: chat.map(m => ({ role: m.role, content: m.content })), stream: true,
    }),
  });
  if (!res.ok) throw new Error(`Anthropic ${res.status}: ${await res.text()}`);
  const reader = res.body.getReader();
  const dec = new TextDecoder();
  let buf = '';
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    for (const line of buf.split('\n')) {
      buf = '';
      if (!line.startsWith('data: ')) continue;
      const data = line.slice(6).trim();
      if (data === '[DONE]') { yield 'data: [DONE]\n\n'; continue; }
      try {
        const p = JSON.parse(data);
        if (p.type === 'content_block_delta')
          yield `data: ${JSON.stringify({ choices: [{ delta: { content: p.delta?.text || '' }, index: 0 }] })}\n\n`;
        if (p.type === 'message_stop') yield 'data: [DONE]\n\n';
      } catch {}
    }
  }
}

async function* streamGemini(body, apiKey) {
  const model = body.model || 'gemini-2.0-flash';
  const messages = body.messages || [];
  const contents = messages.filter(m => m.role !== 'system').map(m => ({
    role: m.role === 'assistant' ? 'model' : 'user', parts: [{ text: m.content }],
  }));
  const sys = messages.filter(m => m.role === 'system').pop();
  const res = await fetch(`https://generativelanguage.googleapis.com/v1beta/models/${model}:streamGenerateContent?key=${apiKey}&alt=sse`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      contents, systemInstruction: sys ? { parts: [{ text: sys.content }] } : undefined,
      generationConfig: { temperature: body.temperature ?? 0.7, maxOutputTokens: body.max_tokens || 4096 },
    }),
  });
  if (!res.ok) throw new Error(`Gemini ${res.status}: ${await res.text()}`);
  const reader = res.body.getReader();
  const dec = new TextDecoder();
  let buf = '';
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    for (const line of buf.split('\n')) {
      buf = '';
      if (!line.startsWith('data: ')) continue;
      try {
        const p = JSON.parse(line.slice(6));
        const text = p.candidates?.[0]?.content?.parts?.[0]?.text || '';
        if (text) yield `data: ${JSON.stringify({ choices: [{ delta: { content: text }, index: 0 }] })}\n\n`;
      } catch {}
    }
  }
  yield 'data: [DONE]\n\n';
}

async function* streamOllama(body) {
  const model = body.model?.replace('ollama/', '') || 'llama3';
  const messages = (body.messages || []).map(m => ({ role: m.role, content: m.content }));
  const res = await fetch('http://localhost:11434/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model, messages, stream: true, options: { temperature: body.temperature ?? 0.7 } }),
  });
  if (!res.ok) throw new Error(`Ollama ${res.status}: ${await res.text()}`);
  const reader = res.body.getReader();
  const dec = new TextDecoder();
  let buf = '';
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    for (const line of buf.split('\n')) {
      buf = '';
      if (!line.trim()) continue;
      try {
        const p = JSON.parse(line);
        if (p.done) { yield 'data: [DONE]\n\n'; continue; }
        if (p.message?.content)
          yield `data: ${JSON.stringify({ choices: [{ delta: { content: p.message.content }, index: 0 }] })}\n\n`;
      } catch {}
    }
  }
}

app.post('/v1/chat/completions', async (req, res) => {
  const cfg = loadConfig();
  const stream = req.body.stream !== false;
  const provider = detectProvider(req.body.model);

  if (stream) {
    res.setHeader('Content-Type', 'text/event-stream');
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('Connection', 'keep-alive');
  }

  try {
    let gen;
    switch (provider) {
      case 'anthropic':
        if (!cfg.anthropic) throw new Error('Anthropic API anahtari yok (Settings > Anthropic)');
        gen = streamAnthropic(req.body, cfg.anthropic);
        break;
      case 'gemini':
        if (!cfg.gemini) throw new Error('Gemini API anahtari yok (Settings > Gemini)');
        gen = streamGemini(req.body, cfg.gemini);
        break;
      case 'ollama':
        gen = streamOllama(req.body);
        break;
      default:
        if (!cfg.openai) throw new Error('OpenAI/OpenRouter API anahtari yok (Settings > OpenAI)');
        gen = streamOpenAI(req.body, cfg.openai, cfg.openaiBaseUrl);
        break;
    }

    if (stream) {
      for await (const chunk of gen) res.write(chunk);
      res.end();
    } else {
      const result = await (async () => {
        const res2 = await fetch(`${cfg.openaiBaseUrl}/chat/completions`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${cfg.openai}` },
          body: JSON.stringify(req.body),
        });
        return res2.json();
      })();
      res.json(result);
    }
  } catch (err) {
    if (stream) {
      res.write(`data: ${JSON.stringify({ error: err.message })}\n\n`);
      res.end();
    } else res.status(500).json({ error: err.message });
  }
});

app.get('/v1/models', (req, res) => {
  const cfg = loadConfig();
  const models = [];
  if (cfg.openai) models.push({ id: 'gpt-4o-mini', provider: 'openai' }, { id: 'gpt-4o', provider: 'openai' });
  if (cfg.anthropic) models.push({ id: 'claude-sonnet-4-20250514', provider: 'anthropic' });
  if (cfg.gemini) models.push({ id: 'gemini-2.0-flash', provider: 'gemini' });
  models.push({ id: 'emii-model', provider: 'auto' }, { id: 'ollama/llama3', provider: 'ollama' });
  res.json({ object: 'list', data: models });
});

app.get('/', (req, res) => {
  const cfg = loadConfig();
  res.json({
    name: 'EmiiAPI', version: '1.0.0', status: 'running',
    openai_base: cfg.openaiBaseUrl || 'https://api.openai.com/v1',
    providers: {
      openai: !!cfg.openai, anthropic: !!cfg.anthropic, gemini: !!cfg.gemini, ollama: true,
    }
  });
});

app.listen(PORT, () => {
  const cfg = loadConfig();
  console.log(`\n  â—† EmiiAPI running at http://localhost:${PORT}`);
  console.log(`  â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€`);
  console.log(`  Endpoint:  http://localhost:${PORT}/v1/chat/completions`);
  console.log(`  Config:    ${configPath}`);
  console.log(`  Keys:`);
  console.log(`    OpenAI:    ${cfg.openai ? 'âœ“' : 'âœ—'}`);
  console.log(`    Anthropic: ${cfg.anthropic ? 'âœ“' : 'âœ—'}`);
  console.log(`    Gemini:    ${cfg.gemini ? 'âœ“' : 'âœ—'}`);
  console.log(`    Ollama:    localhost:11434`);
  if (!cfg.openai && !cfg.anthropic && !cfg.gemini) {
    console.log(`\n  âš  Hicbir API anahtari bulunamadi!`);
    console.log(`  Desktop uygulamadan Settings > API Keys kismina gir`);
    console.log(`  ve en az bir saglayiciya anahtar ekle.\n`);
  } else {
    console.log(`\n  âœ“ Kullanima hazir. En az bir saglayici aktif.\n`);
  }
});
