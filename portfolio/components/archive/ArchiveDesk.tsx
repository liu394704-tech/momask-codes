import React from 'react';
import { ArrowUpRight, Circle, Box, Pencil } from 'lucide-react';
import { contactEmail, contactPhone, templateProfile } from './content';
import ResumeSlip from './ResumeSlip';
import type { ModuleId } from './timeline';

type ArchiveDeskProps = {
  onSelect: (id: ModuleId) => void;
  onResume: () => void;
};

function CardArrow() {
  return (
    <span className="archive-desk-card-arrow" aria-hidden="true">
      <ArrowUpRight size={28} />
    </span>
  );
}

export default function ArchiveDesk({ onSelect, onResume }: ArchiveDeskProps) {
  return (
    <div className="archive-desk">
      <div className="archive-desk-module archive-desk-about" data-module="about">
        <button className="archive-desk-about-main" type="button" onClick={() => onSelect('about')} aria-label="Read about this archive">
          <span className="archive-desk-about-sheet">
            <span className="archive-desk-eyebrow">01 / {templateProfile.brand}</span>
            <strong className="archive-desk-about-title">刘淑雯<br /><span className="archive-desk-title-accent">AI 产品.</span></strong>
            <span className="archive-desk-about-description">{templateProfile.introduction}</span>
            <span className="archive-desk-about-copy">香港科技大学（广州）<br />人工智能硕士在读。</span>
          </span>
          <CardArrow />
        </button>
      </div>
      <ResumeSlip onOpen={onResume} />

      <button type="button" className="archive-desk-module archive-desk-fragments" data-module="fragments" onClick={() => onSelect('fragments')} aria-label="Open collected fragments">
        <span className="archive-desk-eyebrow">02</span>
        <strong className="archive-desk-stat">三件.</strong>
        <span className="archive-desk-stat-label">都和情绪有关。</span>
        <span className="archive-desk-fragment-facts">机器人、社区，和一盏灯。</span>
        <CardArrow />
      </button>

      <button type="button" className="archive-desk-module archive-desk-craft" data-module="craft" onClick={() => onSelect('craft')} aria-label="Explore the creative practice">
        <span className="archive-desk-eyebrow">03</span>
        <strong className="archive-desk-module-title">先问清<br />再定义。</strong>
        <span className="archive-desk-craft-marks" aria-hidden="true">
          <Pencil size={34} /><Circle size={34} /><Box size={34} />
        </span>
        <span className="archive-desk-craft-copy">调研 / 定义 / 评测</span>
        <span className="archive-desk-platforms">韶音 · 面壁 · 字节</span>
        <CardArrow />
      </button>

      <button type="button" className="archive-desk-module archive-desk-perspective" data-module="perspective" onClick={() => onSelect('perspective')} aria-label="Read notes on perspective">
        <span className="archive-desk-eyebrow">04</span>
        <strong className="archive-desk-module-title">硬件里的<br />情感。</strong>
        <span className="archive-desk-user-tag">INTJ</span>
        <span className="archive-desk-user-fact">情绪要能被看见，也要能被做成动作。</span>
        <CardArrow />
      </button>

      <button type="button" className="archive-desk-module archive-desk-collaboration" data-module="collaboration" onClick={() => onSelect('collaboration')} aria-label="Contact and collaboration">
        <span className="archive-desk-eyebrow">06</span>
        <strong className="archive-desk-module-title">想一起<br />做点什么？</strong>
        <span className="archive-desk-email"><span>{contactEmail}</span><span>{contactPhone}</span></span>
        <CardArrow />
      </button>
    </div>
  );
}
