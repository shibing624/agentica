import remarkGfm from "remark-gfm";
import type { PluggableList } from "unified";

/** Single `~` is a numeric range in Chinese prose; GFM strikethrough needs `~~`. */
export const GFM_PLUGINS: PluggableList = [[remarkGfm, { singleTilde: false }]];