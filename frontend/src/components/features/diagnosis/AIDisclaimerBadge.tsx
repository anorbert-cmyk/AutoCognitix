import { AlertTriangle } from "lucide-react";

interface AIDisclaimerBadgeProps {
  compact?: boolean;
}

export default function AIDisclaimerBadge({ compact = false }: AIDisclaimerBadgeProps) {
  if (compact) {
    return (
      <div className="inline-flex items-center gap-1.5 px-3 py-1.5 bg-amber-50 border border-amber-200 rounded-full text-xs text-amber-700">
        <AlertTriangle className="h-3.5 w-3.5" aria-hidden="true" />
        <span>AI-alapú becslés – nem helyettesíti a szakszervizi vizsgálatot</span>
      </div>
    );
  }

  return (
    <div className="p-4 bg-amber-50 border border-amber-300 rounded-lg" role="alert">
      <div className="flex items-start gap-3">
        <AlertTriangle className="h-5 w-5 text-amber-600 mt-0.5 flex-shrink-0" aria-hidden="true" />
        <div>
          <p className="font-medium text-amber-800">AI Diagnosztikai Eszköz – Felelősségi nyilatkozat</p>
          <p className="text-sm text-amber-700 mt-1">
            Ez az eszköz mesterséges intelligencia alapú becslést nyújt, amely nem helyettesíti
            a képzett szerelő vagy hivatalos szakszerviz vizsgálatát. A javasolt diagnózis és
            költségbecslés tájékoztató jellegű. A tényleges hiba és javítási költség eltérhet.
          </p>
        </div>
      </div>
    </div>
  );
}
