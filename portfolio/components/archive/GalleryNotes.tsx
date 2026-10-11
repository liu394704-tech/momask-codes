import type { CSSProperties } from 'react';
import type { GalleryNotesLayout } from './galleryNotesLayout';

export default function GalleryNotes({ layout }: { layout: GalleryNotesLayout }) {
  if (layout.level === 'hidden') return null;
  const rectStyle = (rect: GalleryNotesLayout['left']): CSSProperties => ({
    left: rect.left, top: rect.top, width: rect.width, height: rect.height,
    '--note-width': `${rect.width}px`,
  } as CSSProperties);
  const screenHeight = layout.left.height / layout.fontUnit;
  const tight = screenHeight < 320;
  const headingSize = Math.min(layout.headingSize,
    layout.level === 'full' && tight ? 30 : layout.level === 'compact' && screenHeight < 160 ? 22 : layout.headingSize,
  );

  return (
    <div className="archive-gallery-notes" data-density={layout.level} data-tight={tight} aria-hidden="true" style={{
      opacity: layout.opacity,
      '--note-unit': layout.fontUnit,
      '--note-heading': `${headingSize}px`,
    } as CSSProperties}>
      <div className="archive-gallery-note archive-gallery-note-left" style={rectStyle(layout.left)}>
        <span className="archive-gallery-note-label">{layout.level === 'minimal' ? '05' : 'ARCHIVE / 05'}</span>
        <h2>{layout.level === 'minimal' ? '作品' : <>选中的<br /><em>作品.</em></>}</h2>
        {layout.level === 'full' && <div className="archive-gallery-note-copy">
          <p>一件是支付宝 AI 版「阿宝」的资产页，一件是仍在进行的机器人情感动作研究。完整履历在桌上的简历纸条里。</p>
          <p>研究方案按简历里的写法保留：动作生成、物理安全层，以及仿真里的端到端闭环。</p>
        </div>}
        {layout.level === 'compact' && <p className="archive-gallery-note-copy">资产页，<br />和仍在做的研究。</p>}
        {layout.level === 'full' && <span className="archive-gallery-note-foot">再靠近一点 ↗</span>}
      </div>
    </div>
  );
}
