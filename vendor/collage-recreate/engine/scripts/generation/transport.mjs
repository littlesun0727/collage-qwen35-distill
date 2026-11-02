/** Audited yibu transport. No SDK, retries, redirects, or direct-upstream fallback. */
import fs from 'node:fs/promises';
import http from 'node:http';
import {createHash} from 'node:crypto';
import {fileURLToPath} from 'node:url';
import {extname} from 'node:path';
export const BASE_URL = 'http://127.0.0.1:17860';
export const CREDENTIALS = process.env.COLLAGE_YIBU_CREDENTIALS || fileURLToPath(new URL('../../../.env', import.meta.url));
export const hash = value => createHash('sha256').update(value).digest('hex');
export const readJSON = async file => JSON.parse((await fs.readFile(file,'utf8')).replace(/^\uFEFF/,''));
export const saveJSON = async (file,value) => fs.writeFile(file,JSON.stringify(value,null,2)+'\n');
export function auditBase(value=process.env.YIBU_BASE_URL || BASE_URL) {
  if (value.replace(/\/$/,'')!==BASE_URL) throw new Error('Model requests must use the audit gateway '+BASE_URL);
  return BASE_URL;
}
export async function credential(file=CREDENTIALS) {
  if(process.env.YIBU_API_KEY) return process.env.YIBU_API_KEY;
  let key;
  if(extname(file).toLowerCase()==='.json') {
    const cfg=await readJSON(file);
    key=cfg.api_keys?.find(k=>typeof k==='string' && k.trim())?.trim();
  } else {
    const content=(await fs.readFile(file,'utf8')).replace(/^\uFEFF/,'');
    for(const line of content.split(/\r?\n/)) {
      const match=line.match(/^\s*(?:export\s+)?YIBU_API_KEY\s*=\s*(.*?)\s*$/);
      if(!match) continue;
      const value=match[1];
      if(value.startsWith('"') || value.startsWith("'")) {
        const end=value.indexOf(value[0],1);
        if(end<1) throw new Error('Unclosed quote for YIBU_API_KEY');
        key=value.slice(1,end);
      } else key=value.split(' #',1)[0].trim();
      break;
    }
  }
  if (!key) throw new Error('YIBU_API_KEY missing in environment or configured credential file');
  return key;
}
export const redact = (value,key) => String(value).replaceAll(key || '\u0000','[REDACTED]');
export function imageMime(bytes) {
  if(bytes.subarray(0,3).equals(Buffer.from([255,216,255]))) return 'image/jpeg';
  if(bytes.subarray(0,8).equals(Buffer.from([137,80,78,71,13,10,26,10]))) return 'image/png';
  throw new Error('Prepared images must be PNG or JPEG');
}
export function safePayload(payload) {
  return JSON.parse(JSON.stringify(payload, (key,value)=>{
    if(key==='url' && typeof value==='string' && value.startsWith('data:')) {
      const b=Buffer.from(value.slice(value.indexOf(',')+1),'base64');
      return {data_omitted:true,bytes:b.length,sha256:hash(b)};
    }
    if(key==='inlineData' || key==='inline_data') {
      const b=Buffer.from(value.data || '', 'base64');
      return {mimeType:value.mimeType || value.mime_type,data_omitted:true,bytes:b.length,sha256:hash(b)};
    }
    return value;
  }));
}
export function post(route,payload,key,{base=BASE_URL,timeout=180000}={}) {
  auditBase(base);
  if(!route.startsWith('/') || route.startsWith('//')) throw new Error('Invalid API route');
  const body=Buffer.from(JSON.stringify(payload));
  return new Promise((resolve,reject)=>{
    let timer;
    const req=http.request(new URL(route,base),{method:'POST',agent:false,headers:{
      'content-type':'application/json','content-length':body.length,'authorization':'Bearer '+key
    }},res=>{
      const chunks=[];let size=0;
      res.on('data',chunk=>{
        size+=chunk.length;
        if(size>100*1024*1024) req.destroy(new Error('Response exceeds 100 MiB limit'));
        else chunks.push(chunk);
      });
      res.on('error',error=>{clearTimeout(timer);reject(error);});
      res.on('end',()=>{
        clearTimeout(timer);
        const text=Buffer.concat(chunks).toString('utf8');
        if(res.statusCode<200 || res.statusCode>=300) {
          const error=new Error('HTTP '+res.statusCode+': '+redact(text.slice(0,1500),key));
          error.status=res.statusCode;reject(error);return;
        }
        resolve({text,status:res.statusCode,requestId:res.headers['x-request-id'] || res.headers['request-id'] || null});
      });
    });
    req.on('error',error=>{clearTimeout(timer);reject(error);});
    timer=setTimeout(()=>req.destroy(new Error('Request deadline exceeded; outcome may be unknown, no retry')),timeout);
    req.end(body);
  });
}
export function chatResult(text) {
  const documents=[];
  if(text.trimStart().startsWith('{')) documents.push(JSON.parse(text));
  else {
    for(const block of text.split(/\r?\n\r?\n/)) {
      const data=block.split(/\r?\n/).filter(l=>l.startsWith('data:')).map(l=>l.slice(5).trimStart()).join('\n');
      if(data && data!=='[DONE]') documents.push(JSON.parse(data));
    }
  }
  let content='',finish=null,model=null,usage=null,id=null;
  for(const row of documents) {
    if(row.id) id=row.id;
    if(row.error) throw new Error('Model error: '+JSON.stringify(row.error));
    if(row.model) model=row.model;
    if(row.usage) usage=row.usage;
    const choice=row.choices?.[0];
    const part=choice?.delta || choice?.message;
    if(typeof part?.content==='string') content+=part.content;
    if(choice?.finish_reason) finish=choice.finish_reason;
    // reasoning_content is deliberately not persisted.
  }
  return {content,finish,model,usage,id};
}
