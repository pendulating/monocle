// The index of the static share site: one row for each group, one column for each model.
import shares from '@/data/shares.json';
// Plain links with the base path: a next/link navigation fetches an RSC payload, which the
// sandbox of Anonymous GitHub blocks. A full page load works on every host.
const BASE = process.env.NEXT_PUBLIC_SHARE_BASE_PATH ?? '';

const MODELS = Array.from(new Set(shares.map((s) => s.modelId)));
const NAMES: Record<string, string> = { 'gemma-4-31b': 'Gemma-4-31B', 'qwen3.5-9b': 'Qwen3.5-9B', 'qwen3.6-27b': 'Qwen3.6-27B' };

// The group is the name of the run file (runs/{model}/{group}.json). The url can be a data: URL.
function groupOf(s: { file?: string; url: string }) {
  return (s.file ?? s.url).split('/').pop()!.replace(/\.json$/, '');
}

export default function Index() {
  const groups = Array.from(new Set(shares.map((s) => groupOf(s))));
  const byKey = new Map(shares.map((s) => [`${s.modelId}/${groupOf(s)}`, s]));
  return (
    <div className="mx-auto w-full max-w-screen-lg px-4 py-8 text-slate-800">
      <h1 className="text-xl font-semibold">J++ Lens readouts</h1>
      <p className="mt-2 text-sm text-slate-600">
        Each page is a Neuronpedia share view of one run: the prompt as a row of tokens, and at each token the top
        words of the J++ Lens at each layer. The second column, Contrast, holds the difference named in the text of the
        run. Click a token to read it. Inference is off, so Steer, Swap and Send do nothing.
      </p>
      <table className="mt-6 w-full border-collapse text-sm">
        <thead>
          <tr className="border-b border-slate-300 text-left">
            <th className="py-2 pr-4">Group</th>
            {MODELS.map((m) => (
              <th key={m} className="py-2 pr-4">
                {NAMES[m] ?? m}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {groups.map((g) => (
            <tr key={g} className="border-b border-slate-200 align-top">
              <td className="py-2 pr-4 font-medium">{g}</td>
              {MODELS.map((m) => {
                const s = byKey.get(`${m}/${g}`);
                return (
                  <td key={m} className="py-2 pr-4">
                    {s ? (
                      <a href={`${BASE}/${m}/jlens/${s.id}/`} className="text-sky-700 underline">
                        {s.description}
                      </a>
                    ) : (
                      <span className="text-slate-400">—</span>
                    )}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
      <p className="mt-6 text-xs text-slate-500">
        MANIFEST.json beside this page lists the sha256 of each run file, the model, the lens, the layers and the
        prompt. README.txt says how to serve the folder.
      </p>
    </div>
  );
}
