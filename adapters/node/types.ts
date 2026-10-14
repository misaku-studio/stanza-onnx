/** Public JSON output shared by the native runtime and JavaScript adapter. */
export interface Word {
  text: string;
  upos?: string | null;
  xpos?: string | null;
  feats?: string | null;
  lemma?: string | null;
}

export interface Token {
  text: string;
  start: number;
  end: number;
  words: Word[];
}

export interface Sentence {
  tokens: Token[];
}

export interface Capabilities {
  tokenize: boolean;
  sentenceSplit: boolean;
  multiwordExpansion: boolean;
  pos: boolean;
  morphology: boolean;
  lemma: 'trained' | 'identity' | 'none';
}
