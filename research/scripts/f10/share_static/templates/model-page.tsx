// The landing page of one model, at /{modelId}/jlens/. The webapp shows free chat here; this site
// has no inference, so the page lists the runs of the model instead. The model tabs and the
// "Free Chat" button of the share view lead here.
import shares from '@/data/shares.json';
// Plain links with the base path: a next/link navigation fetches an RSC payload, which the
// sandbox of Anonymous GitHub blocks. A full page load works on every host.
const BASE = process.env.NEXT_PUBLIC_SHARE_BASE_PATH ?? '';

export const dynamicParams = false;

export function generateStaticParams() {
  return Array.from(new Set(shares.map((s) => s.modelId))).map((modelId) => ({ modelId }));
}

export default async function ModelPage({ params }: { params: Promise<{ modelId: string }> }) {
  const { modelId } = await params;
  const rows = shares.filter((s) => s.modelId === modelId);
  return (
    <div className="mx-auto w-full max-w-screen-md px-4 py-8 text-slate-800">
      <h1 className="text-xl font-semibold">{modelId}</h1>
      <p className="mt-2 text-sm text-slate-600">
        This site holds shared runs only. There is no inference host, so there is no free chat. Pick a run.
      </p>
      <ul className="mt-4 list-disc pl-5 text-sm">
        {rows.map((s) => (
          <li key={s.id} className="py-1">
            <a href={`${BASE}/${modelId}/jlens/${s.id}/`} className="text-sky-700 underline">
              {s.description}
            </a>
          </li>
        ))}
      </ul>
      <p className="mt-6 text-sm">
        <a href={`${BASE}/`} className="text-sky-700 underline">
          All models
        </a>
      </p>
    </div>
  );
}
