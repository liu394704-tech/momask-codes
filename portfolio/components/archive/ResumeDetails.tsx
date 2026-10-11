import {
  contactEmail,
  contactGithub,
  contactPhone,
  otherPortfolioUrl,
  templateProfile,
  templateResume,
  type ResumeEntry,
} from './content';

function ResumeEntries({ entries }: { entries: ResumeEntry[] }) {
  return (
    <>
      {entries.map((entry) => (
        <article className="archive-resume-entry" key={`${entry.title}-${entry.period}`}>
          <div className="archive-resume-entry-heading">
            <h4>{entry.title}</h4>
            {entry.period ? <span className="archive-resume-entry-date">{entry.period}</span> : null}
          </div>
          {entry.subtitle && <p className="archive-resume-entry-subtitle">{entry.subtitle}</p>}
          {entry.summary && <p className="archive-resume-summary">{entry.summary}</p>}
          {entry.bullets && entry.bullets.length > 0 && (
            <ul className="archive-resume-bullets">
              {entry.bullets.map((item) => <li key={item}>{item}</li>)}
            </ul>
          )}
        </article>
      ))}
    </>
  );
}

export default function ResumeDetails() {
  return (
    <article className="archive-resume-details" aria-label="刘淑雯的简历">
      <header className="archive-resume-details-header">
        <h2 className="archive-resume-details-name">{templateProfile.name}</h2>
        <p className="archive-resume-subline">{templateProfile.disciplines}</p>
        <div className="archive-resume-contact">
          <a href={`mailto:${contactEmail}`}>{contactEmail}</a>
          <span aria-hidden="true">/</span>
          <a href={`tel:${contactPhone.replace(/[^\d+]/g, '')}`}>{contactPhone}</a>
          <span aria-hidden="true">/</span>
          <span>{templateProfile.location}</span>
          <span aria-hidden="true">/</span>
          <a href={contactGithub} target="_blank" rel="noopener noreferrer">GitHub</a>
          <span aria-hidden="true">/</span>
          <a href={otherPortfolioUrl} target="_blank" rel="noopener noreferrer">主页</a>
        </div>
      </header>

      <div className="archive-resume-details-body">
        <section className="archive-resume-section">
          <h3>关于</h3>
          <p>{templateProfile.biography}</p>
        </section>

        <section className="archive-resume-section">
          <h3>教育背景</h3>
          <ResumeEntries entries={templateResume.education} />
        </section>

        <section className="archive-resume-section">
          <h3>实习经历</h3>
          <ResumeEntries entries={templateResume.experience} />
        </section>

        <section className="archive-resume-section">
          <h3>研究方向</h3>
          <ResumeEntries entries={templateResume.projects} />
        </section>

        {templateResume.awards.length > 0 && (
          <section className="archive-resume-section">
            <h3>竞赛</h3>
            <ul className="archive-resume-bullets">
              {templateResume.awards.map((item) => <li key={item}>{item}</li>)}
            </ul>
          </section>
        )}

        <section className="archive-resume-section">
          <h3>个人总结</h3>
          <p>{templateResume.practice}</p>
        </section>

        <section className="archive-resume-section">
          <h3>技能</h3>
          <div className="archive-resume-skills">
            {templateResume.tools.map((item) => <p key={item}>{item}</p>)}
          </div>
        </section>

        <section className="archive-resume-section">
          <h3>语言</h3>
          <div className="archive-resume-skills">
            {templateResume.languages.map((item) => <p key={item}>{item}</p>)}
          </div>
        </section>
      </div>
    </article>
  );
}
