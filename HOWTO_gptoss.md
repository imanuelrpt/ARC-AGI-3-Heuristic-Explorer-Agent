# HOWTO — LLM agent (GPT-OSS-120B) untuk ARC Prize 2026 / ARC-AGI-3

Berkas ini menjelaskan cara memakai agen LLM berbasis template resmi ARC Prize.

## Berkas terkait

| Berkas | Isi |
|---|---|
| `template_gptoss.ipynb` | Salinan **template resmi** `gregkamradt/arc-agi-3-gpt-oss-120b` (20 sel), apa adanya. Sumber kebenaran. |
| `gptoss_my_agent_cell.py` | Agen versi kita (v3). Ini isi **sel 6** yang menggantikan agen bawaan. |
| `arc3_gptoss_improved.ipynb` | Notebook lengkap = template + sel 6 kita. |
| `build_gptoss_notebook.py` | Regenerasi `arc3_gptoss_improved.ipynb` dari dua berkas di atas. |
| `fetch_template.py` | Mengunduh ulang template dari Kaggle API (`kernels/pull`). |
| `test_gptoss_agent.py` | Uji offline logika agen (fungsi prompt/diff/JSON). |

## Cara kerja template resminya

1. **Sel 1** — pasang `vllm` dari *wheel* offline (dataset Kaggle `philipvonderlind/vllm-deps`).
2. **Sel 3** — pasang `arc-agi` dari wheel kompetisi.
3. **Sel 6** — menulis `/kaggle/working/my_agent.py` (agennya).
4. **Sel 8–10** — menyalin framework `ARC-AGI-3-Agents` ke `/kaggle/working` dan menulis ulang `agents/__init__.py`.
5. **Sel 12** — menulis `.env` (mode kompetisi vs lokal).
6. **Sel 16–18** — menjalankan server vLLM + `main.py --agent myagent`.
   - Mode kompetisi (`KAGGLE_IS_COMPETITION_RERUN` aktif) → melawan gateway `gateway:8001`.
   - Mode lokal → menjalankan server ARC lokal dan bermain game `bp35`.
7. **Sel 19** — menulis `submission.parquet` dummy (wajib agar bisa di-submit).

Agen memanggil vLLM lewat protokol OpenAI (`http://127.0.0.1:8000/v1`) dengan
**guided JSON**: model dipaksa menjawab `{"action":"ACTION4"}` atau
`{"action":"ACTION6","x":12,"y":30}`.

Input yang harus terpasang di notebook:

- Kompetisi: `arc-prize-2026-arc-agi-3`
- Dataset: `philipvonderlind/vllm-deps`
- Model: `danielhanchen/gpt-oss-120b` (Transformers)

## Perubahan kita (v3) vs template

| Aspek | Template | Kita | Alasan |
|---|---|---|---|
| Representasi frame | literal list Python (`[[0, 0, ...]]`) | baris heksadesimal dengan nomor baris | ~12.4k → ~4.3k karakter (≈3× lebih murah), posisi tetap mudah dibaca |
| Memori aksi | tidak ada | 8 aksi terakhir + ringkasan diff (`ACTION3 -> grid 0: NO CHANGE`) | model tahu langkah yang sudah sia-sia, mengurangi loop |
| `ACTION_MAX_TOKENS` | 8192 | 4096 | jawaban JSON sangat pendek; token reasoning berlebih = waktu terbuang di jam permainan |
| `ACTION_RETRY_LIMIT` | 1 | 2 | lebih tahan saat model gagal format |
| `MAX_ACTIONS` | 50 | 60 | sedikit lebih banyak kesempatan menyelesaikan level |

Semua parameter vLLM (`MAX_MODEL_LEN`, `MAX_NUM_SEQS`, `ENFORCE_EAGER`, …)
sengaja **tidak** diubah supaya startup GPU tetap seperti yang sudah terbukti.

## Langkah di Kaggle

1. Buka <https://www.kaggle.com/code/gregkamradt/arc-agi-3-gpt-oss-120b> → **Copy & Edit** (fork ke akun Anda).
2. Cek panel **Data** (kanan): pastikan ketiga input di atas terpasang. Kalau hilang, **Add Input**.
3. Panel **Settings**: **Accelerator = GPU RTX Pro 6000**, **Internet = Off**.
4. Buka `gptoss_my_agent_cell.py` di laptop, **copy semua isinya**, lalu tempel menggantikan seluruh **sel 6** di notebook (sel yang diawali `%%writefile /kaggle/working/my_agent.py`).
5. **Run All** dulu untuk uji cepat (mode lokal, main game `bp35`). Perhatikan log: vLLM harus "ready", lalu agen bergerak.
6. Kalau lancar → **Save & Run All (Commit)**.
7. Setelah commit selesai → **Submit to Competition**.

## Perkiraan waktu

- Startup vLLM: ~10–20 menit.
- Semua game dijalankan **paralel** (satu thread per game), dibatasi
  `GAME_TIME_LIMIT_S = 20 menit`, jadi total run ≈ **30–45 menit**.
- Batas platform: 9 jam/notebook. Jauh dari limit.

## Catatan penting

- `MAX_NUM_SEQS = 12` sedangkan jumlah game ±25 → sebagian agen mengantre.
  Menaikkan nilai ini (atau menurunkan `MAX_MODEL_LEN`) bisa menambah
  throughput, **tapi** berisiko kehabisan VRAM. Uji satu per satu.
- Aturan kompetisi: notebook ≤ 9 jam, **internet mati**. Jangan menambah
  dependensi dari internet.
- Kalau run gagal dengan "scorecard not produced in time", turunkan
  `GAME_TIME_LIMIT_S` atau `MAX_ACTIONS`.
