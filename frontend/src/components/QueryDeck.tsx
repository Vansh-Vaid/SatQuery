// PANEL // 03 // QUERY-DECK - bottom-docked natural-language command input (spec §5.2.3)
interface Props {
  query: string;
  onQuery: (q: string) => void;
  onRun: () => void;
  chips: string[];
  busy: boolean;
  disabled: boolean;
}

export default function QueryDeck({ query, onQuery, onRun, chips, busy, disabled }: Props) {
  return (
    <section className="hud-panel deck" aria-label="Natural-language query deck">
      <div className="deck-row">
        <span className="panel-code deck-code">03 · Ask</span>
        <div className="chips" role="list">
          {chips.map((c) => (
            <button key={c} role="listitem" className="chip" onClick={() => onQuery(c)} title={c}>{c}</button>
          ))}
        </div>
      </div>
      <div className="deck-input">
                <textarea value={query} rows={2} placeholder="Ask about the imagery… e.g. What changed between these two dates, and where?"
          aria-label="Query" onChange={(e) => onQuery(e.target.value)}
          onKeyDown={(e) => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey || !e.shiftKey)) { e.preventDefault(); onRun(); } }} />
        <button className="execute" onClick={onRun} disabled={busy || disabled || !query.trim()}>
          {busy ? "Analysing…" : "Analyse →"}
        </button>
      </div>
    </section>
  );
}
