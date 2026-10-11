import type { ModuleId } from './timeline';

export const portfolioModules: Array<{ id: ModuleId; label: string; reasonIds: string[] }> = [
  { id: 'about', label: '关于', reasonIds: [] },
  { id: 'fragments', label: '片段', reasonIds: ['01'] },
  { id: 'craft', label: '实践', reasonIds: ['02', '04', '05'] },
  { id: 'perspective', label: '视角', reasonIds: ['09', '03', '07', '08'] },
  { id: 'works', label: '作品', reasonIds: [] },
  { id: 'collaboration', label: '联系', reasonIds: ['06'] },
];
