import { Wrench, Search, Zap } from 'lucide-react';

interface BlogPost {
  date: string;
  title: string;
  description: string;
  icon: React.ElementType;
  gradient: string;
}

const blogPosts: BlogPost[] = [
  {
    date: '2026. január 5.',
    title: 'Hogyan Építs Hatékony Diagnosztikai Folyamatot',
    description:
      'Egy jól felépített diagnosztikai folyamat átalakítja a szerviz munkáját. Optimalizáld minden lépést a jobb eredményekért.',
    icon: Wrench,
    gradient: 'from-orange-400 to-amber-500',
  },
  {
    date: '2026. február 12.',
    title: 'Az OBD-II Hibakódok Megértésének Művészete',
    description:
      'A hibakódok helyes értelmezése kulcsfontosságú a pontos diagnózishoz. Fedezd fel a leggyakoribb kódok rejtett jelentéseit.',
    icon: Search,
    gradient: 'from-amber-400 to-orange-500',
  },
  {
    date: '2026. március 20.',
    title: 'AI Stratégiák a Modern Járműdiagnosztikában',
    description:
      'A mesterséges intelligencia új szintre emeli a járműdiagnosztikát. Ismerd meg a legújabb AI-alapú megoldásokat.',
    icon: Zap,
    gradient: 'from-orange-500 to-red-400',
  },
];

function BlogCard({ post }: { post: BlogPost }) {
  const Icon = post.icon;

  return (
    <div className="rounded-xl border border-gray-200 bg-white shadow-sm overflow-hidden flex flex-col">
      <div className={`h-48 bg-gradient-to-br ${post.gradient} flex items-center justify-center`}>
        <Icon className="h-16 w-16 text-white/80" strokeWidth={1.5} />
      </div>
      <div className="p-6 flex flex-col flex-1">
        <p className="text-sm text-gray-500 mb-2">{post.date}</p>
        <h3 className="text-lg font-semibold text-gray-900 mb-3">{post.title}</h3>
        <p className="text-gray-600 text-sm leading-relaxed mb-6 flex-1">{post.description}</p>
        <div>
          <span className="inline-flex items-center gap-2 bg-gray-300 text-gray-500 text-sm font-medium px-5 py-2.5 rounded-full cursor-not-allowed select-none">
            Hamarosan...
          </span>
        </div>
      </div>
    </div>
  );
}

export default function BlogPage() {
  return (
    <div className="min-h-screen bg-white">
      <div className="max-w-6xl mx-auto px-4 py-16 sm:px-6 lg:px-8">
        <div className="text-center mb-12">
          <h1 className="text-4xl font-serif text-gray-900 mb-4">Blog &amp; Hírek</h1>
          <p className="text-gray-600 text-lg">
            Olvasd el a legújabb cikkeinket az autódiagnosztika világából!
          </p>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
          {blogPosts.map((post) => (
            <BlogCard key={post.title} post={post} />
          ))}
        </div>
      </div>
    </div>
  );
}
