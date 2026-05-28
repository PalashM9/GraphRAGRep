/**
 * ChapterSidebar.tsx – Chapter/section tree for navigation.
 */

import React, { useEffect, useState } from "react";
import { ChapterNode, fetchChapters } from "../api/client";

interface Props {
  onSelectNode: (id: string) => void;
  selectedNodeId?: string;
}

const ChapterSidebar: React.FC<Props> = ({ onSelectNode, selectedNodeId }) => {
  const [chapters, setChapters] = useState<ChapterNode[]>([]);
  const [expanded, setExpanded] = useState<Set<string>>(new Set());
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchChapters()
      .then((data) => {
        setChapters(data);
        // Auto-expand first chapter
        if (data.length > 0) setExpanded(new Set([data[0].id]));
      })
      .finally(() => setLoading(false));
  }, []);

  const toggleExpand = (id: string) => {
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  if (loading) {
    return (
      <div style={{ padding: "var(--space-4)" }}>
        {[140, 100, 120, 90].map((w, i) => (
          <div key={i} className="loading-shimmer"
            style={{ height: 14, width: w, marginBottom: 12 }} />
        ))}
      </div>
    );
  }

  if (chapters.length === 0) {
    return (
      <div className="empty-state" style={{ padding: "var(--space-4)" }}>
        <p style={{ fontSize: "var(--text-xs)" }}>
          No chapters found. Place thesis.pdf in backend/data/ and restart.
        </p>
      </div>
    );
  }

  return (
    <div style={{ padding: "var(--space-3)" }}>
      <p className="section-heading" style={{ padding: "0 var(--space-1)" }}>
        Contents
      </p>
      {chapters.map((ch) => (
        <div key={ch.id} style={{ marginBottom: "var(--space-1)" }}>
          {/* Chapter row */}
          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: "var(--space-2)",
              padding: "var(--space-2) var(--space-2)",
              borderRadius: "var(--radius-sm)",
              cursor: "pointer",
              background: selectedNodeId === ch.id
                ? "var(--color-surface-offset)" : "transparent",
              transition: "background var(--transition)",
            }}
            onClick={() => {
              onSelectNode(ch.id);
              toggleExpand(ch.id);
            }}
          >
            <span style={{
              fontSize: 10, color: "var(--color-text-faint)",
              transform: expanded.has(ch.id) ? "rotate(90deg)" : "rotate(0deg)",
              transition: "transform var(--transition)",
              display: "inline-block",
            }}>▶</span>
            <span style={{
              fontSize: "var(--text-xs)", fontWeight: 600,
              color: selectedNodeId === ch.id
                ? "var(--color-primary)" : "var(--color-text)",
              lineHeight: 1.4,
            }}>
              {ch.label}
            </span>
          </div>

          {/* Sections */}
          {expanded.has(ch.id) && ch.sections.map((sec) => (
            <div
              key={sec.id}
              style={{
                padding: "var(--space-1) var(--space-2) var(--space-1) var(--space-8)",
                cursor: "pointer",
                borderRadius: "var(--radius-sm)",
                background: selectedNodeId === sec.id
                  ? "var(--color-surface-offset)" : "transparent",
                transition: "background var(--transition)",
              }}
              onClick={() => onSelectNode(sec.id)}
            >
              <span style={{
                fontSize: "var(--text-xs)",
                color: selectedNodeId === sec.id
                  ? "var(--color-primary)" : "var(--color-text-muted)",
                lineHeight: 1.4,
              }}>
                {sec.label}
              </span>
            </div>
          ))}
        </div>
      ))}
    </div>
  );
};

export default ChapterSidebar;