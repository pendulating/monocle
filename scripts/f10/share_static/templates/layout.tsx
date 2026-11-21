// The root layout of the static share site. It replaces the webapp's layout, which reads the
// session and the database. The providers get the 3 models from data/models.json, with
// inference off, and nothing else.
import { Providers } from '@/components/provider/providers';
import Toast from '@/components/toast';
import models from '@/data/models.json';
import { inter } from './fonts';
import './globals.css';

export const metadata = {
  title: 'J++ Lens readouts',
  description: 'Static share pages of the J++ Lens readouts that the paper reports.',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body
        className={`flex min-h-[100dvh] flex-col bg-slate-50 bg-fixed overscroll-none ${inter.className}`}
        // The site has no navbar, so the share view gets the full height.
        style={{ ['--site-chrome-height' as string]: '0px' } as React.CSSProperties}
      >
        <Providers
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          initialModels={models as any}
          initialExplanationTypes={[]}
          initialExplanationModels={[]}
          initialReleases={[]}
          initialExplanationScoreTypes={[]}
          initialExplanationScoreModelTypes={[]}
        >
          <Toast />
          <main className="flex w-full flex-1 flex-col items-center gap-0 pt-0">{children}</main>
        </Providers>
      </body>
    </html>
  );
}
