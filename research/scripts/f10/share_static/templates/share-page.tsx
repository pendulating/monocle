// One static share page for each row of data/shares.json, at /{modelId}/jlens/{shareId}/.
// The webapp's page reads the row from the database at request time. This page reads it from
// the JSON file at build time and renders the same client component with inference off.
import shares from '@/data/shares.json';
import { shareLensColumns } from '@/lib/utils/lens';
import { notFound } from 'next/navigation';
import { Suspense } from 'react';
import JlensPageClient, { JlensShareData } from '../jlens-page-client';

export const dynamicParams = false;

export function generateStaticParams() {
  return shares.map((s) => ({ modelId: s.modelId, shareId: s.id }));
}

export async function generateMetadata({ params }: { params: Promise<{ modelId: string; shareId: string }> }) {
  const { modelId, shareId } = await params;
  const row = shares.find((s) => s.id === shareId && s.modelId === modelId);
  return { title: `${row?.description ?? shareId} – ${modelId}` };
}

export default async function Page({ params }: { params: Promise<{ modelId: string; shareId: string }> }) {
  const { modelId, shareId } = await params;
  const row = shares.find((s) => s.id === shareId && s.modelId === modelId);
  if (!row) {
    notFound();
  }
  const share: JlensShareData = {
    url: row.url,
    description: row.description,
    descriptionAttribution: null,
    lockedTokens: [],
    selectedPositions: row.selectedPositions ?? [],
    lensColumns: shareLensColumns(row.lensColumns, row.activeLensModeTab),
    topN: row.topN,
    hideNonWordTokens: row.hideNonWordTokens,
    temperature: row.temperature,
    numCompletionTokens: row.numCompletionTokens,
    numPromptTokens: row.numPromptTokens ?? null,
  };
  return (
    <Suspense fallback={null}>
      <JlensPageClient modelId={modelId} share={share} inferenceAvailable={false} />
    </Suspense>
  );
}
