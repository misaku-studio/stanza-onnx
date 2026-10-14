import { parseArgs } from 'node:util';
import { createNativePipeline } from '../dist/native.js';

// Build the adapter with npm run build, then run this file with Node.js.
const { values } = parseArgs({
  options: {
    binary: { type: 'string' },
    'ort-library': { type: 'string' },
    bundle: { type: 'string' },
    text: { type: 'string', default: 'The children were reading books.' },
  },
});
if (!values.binary || !values['ort-library'] || !values.bundle) {
  throw new Error('Usage: analyze.js --binary PATH --ort-library PATH --bundle PATH [--text TEXT]');
}
const pipeline = await createNativePipeline({
  executable: values.binary,
  modelDir: values.bundle,
  ortLibrary: values['ort-library'],
  offsetUnit: 'utf16',
});
try {
  console.log(JSON.stringify(await pipeline.analyze(values.text), null, 2));
} finally {
  await pipeline.close();
}
