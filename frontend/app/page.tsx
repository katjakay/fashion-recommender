'use client';

import { useState } from 'react';

type Outfit = {
  id: number;
  title: string;
  image_url: string;
  color: string;
  tags: string;
  score: number;
};

export default function Home() {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<Outfit[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!query.trim()) return;

    setLoading(true);
    setError(null);

    try {
      const res = await fetch('http://127.0.0.1:8000/recommend', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, top_k: 5 }),
      });

      if (!res.ok) {
        throw new Error(`API error: ${res.status}`);
      }

      const data = await res.json();
      setResults(data.results ?? []);
    } catch (err: any) {
      console.error(err);
      setError(err.message ?? 'Unknown error');
      setResults([]);
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="min-h-screen flex flex-col items-center px-4 py-8 bg-slate-50">
      <div className="w-full max-w-2xl">
        <h1 className="text-3xl font-semibold mb-2">
          Fashion Recommender <span className="text-sm font-normal">v1</span>
        </h1>
        <p className="text-sm text-slate-600 mb-6">
          Beschreib deinen Look, z. B.{' '}
          <span className="italic">"black minimal streetwear blazer"</span> und
          ich suche dir passende Outfits aus dem Mini-Katalog.
        </p>

        <form onSubmit={handleSubmit} className="flex gap-2 mb-6">
          <input
            className="flex-1 border border-slate-300 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-black/60"
            placeholder='z.B. "red cozy knit" oder "blue denim streetwear"'
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
          <button
            type="submit"
            className="px-4 py-2 rounded-lg bg-black text-white text-sm font-medium disabled:opacity-50"
            disabled={loading}
          >
            {loading ? 'Suche…' : 'Go'}
          </button>
        </form>

        {error && <p className="text-sm text-red-500 mb-4">{error}</p>}

        <section className="space-y-4">
          {results.map((item) => (
            <article
              key={item.id}
              className="bg-white border border-slate-200 rounded-2xl p-4 flex gap-4 items-start shadow-sm"
            >
              {item.image_url && (
                <img
                  src={item.image_url}
                  alt={item.title}
                  className="w-24 h-24 rounded-xl object-cover flex-shrink-0 bg-slate-100"
                />
              )}
              <div>
                <h2 className="font-semibold text-base mb-1">{item.title}</h2>
                <p className="text-xs text-slate-600 mb-1">
                  Color: <span className="font-mono">{item.color}</span>
                  {' · '}
                  Tags: <span className="font-mono">{item.tags}</span>
                </p>
                <p className="text-[11px] text-slate-400">
                  Score: {item.score.toFixed(3)}
                </p>
              </div>
            </article>
          ))}

          {!loading && !error && results.length === 0 && (
            <p className="text-sm text-slate-500">
              Noch keine Ergebnisse – gib oben eine Beschreibung ein ✨
            </p>
          )}
        </section>
      </div>
    </main>
  );
}
