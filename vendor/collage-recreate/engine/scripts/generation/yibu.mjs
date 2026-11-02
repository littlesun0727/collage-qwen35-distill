/** One Gemini reference-image request through the local audit gateway. */
import fs from 'node:fs/promises';
import path from 'node:path';
import {parseArgs} from 'node:util';
import {CREDENTIALS,auditBase,credential,hash,saveJSON,redact,imageMime,safePayload,post} from './transport.mjs';
const {values:a}=parseArgs({options:{
  image:{type:'string'},prompt:{type:'string'},output:{type:'string'},credentials:{type:'string',default:CREDENTIALS},
  'base-url':{type:'string'},model:{type:'string',default:'gemini-3-pro-image'},
  ratio:{type:'string',default:'1:1'},'timeout-seconds':{type:'string',default:'300'},'dry-run':{type:'boolean',default:false}
}});
if(!a.image || !a.prompt || !a.output) throw new Error('Require --image, --prompt and --output');
await fs.mkdir(path.dirname(path.resolve(a.output)),{recursive:true});await fs.mkdir(a.output);
const write=(name,value)=>saveJSON(path.join(a.output,name),value);
const start=performance.now();
const audit={status:'preparing',provider:'yibuapi',requested_model:a.model,http_dispatches:0,automatic_retries:0,
  dry_run:a['dry-run'],started:new Date().toISOString()};
let key='';
try {
  const base=auditBase(a['base-url']);
  if(!/^gemini-[a-zA-Z0-9.-]+$/.test(a.model)) throw new Error('Expected explicit Gemini model ID');
  const timeout=Number(a['timeout-seconds'])*1000;
  if(!Number.isInteger(timeout) || timeout<=0) throw new Error('Invalid timeout');
  if(!['1:1','2:3','3:2','3:4','4:3','4:5','5:4','9:16','16:9','21:9'].includes(a.ratio)) throw new Error('Unsupported aspect ratio');
  if(!a['dry-run']) key=await credential(a.credentials);
  const bytes=await fs.readFile(a.image),mime=imageMime(bytes);
  const prompt=(await fs.readFile(a.prompt,'utf8')).replace(/^\uFEFF/,'');
  const payload={contents:[{role:'user',parts:[{text:prompt},{inlineData:{mimeType:mime,data:bytes.toString('base64')}}]}],
    generationConfig:{responseModalities:['TEXT','IMAGE'],imageConfig:{aspectRatio:a.ratio,imageSize:'1K'}}};
  const route='/v1beta/models/'+a.model+':generateContent';
  Object.assign(audit,{endpoint:base+route,reference_sha256:hash(bytes),prompt_sha256:hash(prompt),aspect_ratio:a.ratio});
  await write('request-body.redacted.json',safePayload(payload));
  if(a['dry-run']) audit.status='dry_run_verified';
  else {
    audit.status='request_in_flight';audit.http_dispatches=1;await write('call.json',audit);
    const response=await post(route,payload,key,{base,timeout});
    audit.http_status=response.status;audit.request_id=response.requestId;
    const data=JSON.parse(response.text);
    audit.request_id=response.requestId || data.responseId || null;
    // Exclude hidden thought parts and inline bytes from the on-disk response log.
    const safe=structuredClone(data);
    for(const candidate of safe.candidates || []) if(candidate.content?.parts) candidate.content.parts=candidate.content.parts.filter(p=>!p.thought).map(p=>{delete p.thoughtSignature;return p;});
    await write('response.redacted.json',safePayload(safe));
    const candidates=data.candidates || [];
    const candidate=candidates[0];
    const parts=(candidate?.content?.parts || []).filter(p=>!p.thought);
    const images=parts.map(p=>p.inlineData || p.inline_data).filter(p=>p && (p.mimeType || p.mime_type || '').startsWith('image/'));
    audit.usage=data.usageMetadata;audit.returned_model=data.modelVersion || null;audit.finish_reason=candidate?.finishReason;
    if(candidates.length!==1 || candidate.finishReason!=='STOP' || images.length!==1) throw new Error('Expected one completed candidate containing exactly one image');
    const encoded=images[0].data;
    if(typeof encoded!=='string' || !/^[A-Za-z0-9+/]*={0,2}$/.test(encoded)) throw new Error('Invalid image base64');
    const generated=Buffer.from(encoded,'base64');imageMime(generated);
    await fs.writeFile(path.join(a.output,'original-image.bin'),generated);
    await fs.writeFile(path.join(a.output,'raw-response.txt'),parts.filter(p=>p.text).map(p=>p.text).join('\n'),'utf8');
    audit.image_sha256=hash(generated);audit.image_bytes=generated.length;audit.status='completed';
  }
} catch(error) {audit.status='failed';audit.error=redact(error.message,key);if(error.status) audit.http_status=error.status;}
audit.elapsed_seconds=Number(((performance.now()-start)/1000).toFixed(3));audit.finished=new Date().toISOString();
await write('call.json',audit);
console.log(JSON.stringify({status:audit.status,http_dispatches:audit.http_dispatches,elapsed_seconds:audit.elapsed_seconds,output:a.output}));
process.exitCode=['completed','dry_run_verified'].includes(audit.status)?0:2;
