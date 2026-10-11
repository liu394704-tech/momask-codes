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
          <p>陪伴机器人、城中村社区服务，和一盏靠触感说话的灯。每件都从一个人的情绪或一个社区的不便开始。</p>
          <p>点开一张，可以看它怎样走到原型。完整履历在桌上的简历纸条里。</p>
        </div>}
        {layout.level === 'compact' && <p className="archive-gallery-note-copy">三件作品，<br />都还连着人。</p>}
        {layout.level === 'full' && <span className="archive-gallery-note-foot">再靠近一点 ↗</span>}
      </div>
    </div>
  );
}
