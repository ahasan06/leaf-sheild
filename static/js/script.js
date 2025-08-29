document.addEventListener('DOMContentLoaded', () => {
  const body = document.body;
  const MODEL_NAME = body.dataset.model;        // "watermelon" or "guava"
  const fileEl = document.getElementById('file');
  const previewEl = document.getElementById('preview');
  const formEl = document.getElementById('form');
  const btnEl = document.getElementById('btn');
  const btnText = document.getElementById('btnText');
  const spinner = document.getElementById('spinner');
  const resultCard = document.getElementById('resultCard');
  const summaryEl = document.getElementById('summary');
  const badgeEl = document.getElementById('badge');
  const probsBody = document.getElementById('probsBody');
  const rawEl = document.getElementById('raw');

  if (!formEl) return; // we're probably on index.html

  // Image preview
  fileEl.addEventListener('change', () => {
    const f = fileEl.files && fileEl.files[0];
    if (!f) return;
    const url = URL.createObjectURL(f);
    previewEl.src = url;
    previewEl.onload = () => URL.revokeObjectURL(url);
  });

  function setLoading(v) {
    if (!btnEl) return;
    btnEl.disabled = v;
    if (spinner) spinner.classList.toggle('hidden', !v);
    if (btnText) btnText.textContent = v ? 'Running…' : 'Predict';
  }

  function setBadge(prob) {
    const pct = (prob * 100).toFixed(2);
    badgeEl.textContent = `Confidence ${pct}%`;
    badgeEl.className = 'inline-flex items-center rounded-full px-3 py-1 text-xs font-medium';
    if (prob >= 0.8) {
      badgeEl.classList.add('bg-green-100','text-green-800');
    } else if (prob >= 0.6) {
      badgeEl.classList.add('bg-yellow-100','text-yellow-800');
    } else {
      badgeEl.classList.add('bg-red-100','text-red-800');
    }
  }

  formEl.addEventListener('submit', async (e) => {
    e.preventDefault();
    const f = fileEl.files && fileEl.files[0];
    if (!f) return;

    setLoading(true);
    resultCard.classList.add('hidden');
    summaryEl.textContent = '';
    probsBody.innerHTML = '';
    rawEl.textContent = '';

    try {
      const fd = new FormData();
      fd.append('image', f);
      fd.append('model', MODEL_NAME);
      const r = await fetch('/predict', { method: 'POST', body: fd });
      const data = await r.json();

      if (!r.ok) {
        summaryEl.innerHTML = `<span class="text-red-600 font-medium">Error:</span> ${data.error || 'Prediction failed'}`;
        resultCard.classList.remove('hidden');
        return;
      }

      const pct = (data.confidence * 100).toFixed(2);
      summaryEl.innerHTML = `Top class: <span class="font-semibold">${data.top_class}</span> <span class="text-gray-500">(${pct}% confidence)</span>`;
      setBadge(data.confidence);

      data.all_probs.forEach(row => {
        const tr = document.createElement('tr');
        tr.innerHTML = `
          <td class="px-4 py-2 whitespace-nowrap">${row.label}</td>
          <td class="px-4 py-2 whitespace-nowrap">${(row.prob * 100).toFixed(2)}%</td>
        `;
        probsBody.appendChild(tr);
      });

      rawEl.textContent = JSON.stringify(data, null, 2);
      resultCard.classList.remove('hidden');
    } catch (err) {
      summaryEl.innerHTML = `<span class="text-red-600 font-medium">Error:</span> ${err.message || err}`;
      resultCard.classList.remove('hidden');
    } finally {
      setLoading(false);
    }
  });
});
