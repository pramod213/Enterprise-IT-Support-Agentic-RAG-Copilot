const chat = document.getElementById('chat');
const form = document.getElementById('chatForm');
const question = document.getElementById('question');
const trace = document.getElementById('trace');
const sourceUsed = document.getElementById('sourceUsed');
const themeToggle = document.getElementById('themeToggle');
const themeIcon = document.getElementById('themeIcon');

function updateThemeIcon() {
  const isLight = document.documentElement.getAttribute('data-theme') === 'light';
  if (themeIcon) {
    themeIcon.textContent = isLight ? '🌙' : '☀';
  }
}

if (themeToggle) {
  themeToggle.addEventListener('click', function() {
    const isLight = document.documentElement.getAttribute('data-theme') === 'light';
    if (isLight) {
      document.documentElement.removeAttribute('data-theme');
      localStorage.removeItem('theme');
    } else {
      document.documentElement.setAttribute('data-theme', 'light');
      localStorage.setItem('theme', 'light');
    }
    updateThemeIcon();
  });
}

updateThemeIcon();

function escapeHtml(s=''){return s.replace(/[&<>'"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]));}
function formatText(s=''){return escapeHtml(s).replace(/\n/g,'<br>');}
function addMessage(role, text, source='', citations=[]){
  const wrap=document.createElement('div'); wrap.className=`message ${role}`;
  const citeHtml=citations.length?`<div class="citations"><strong>Sources</strong><br>${citations.map(c=>c.url?`<a href="${escapeHtml(c.url)}" target="_blank" rel="noopener">${escapeHtml(c.title)}</a>`:escapeHtml(c.title)).join('<br>')}</div>`:'';
  wrap.innerHTML=`<div class="avatar">AI</div><div class="bubble">${formatText(text)}${source?`<div class="answer-source">Source: ${escapeHtml(source)}</div>`:''}${citeHtml}</div>`;
  chat.appendChild(wrap); chat.scrollTop=chat.scrollHeight;
}
function renderTrace(items=[]){trace.innerHTML=items.length?items.map(x=>`<div class="trace-item">${escapeHtml(x)}</div>`).join(''):'<div class="empty">No trace.</div>';}

let currentStreamingMessage = null;

async function askAgent(q){
  addMessage('user',q); question.value=''; renderTrace(['Running LangGraph workflow...']); sourceUsed.textContent='Running';
  const btn=form.querySelector('button'); btn.disabled=true;
  
  try{
    const res=await fetch('/api/chat/stream',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question:q})});
    if(!res.ok) throw new Error('Request failed');
    
    const reader=res.body.getReader();
    const decoder=new TextDecoder();
    let buffer='';
    let answerBuffer='';
    let citations=[];
    let traceData=[];
    let sourceUsedValue='';
    
    currentStreamingMessage = createStreamingMessage();
    
    while(true){
      const {value,done}=await reader.read();
      if(done) break;
      
      buffer+=decoder.decode(value,{stream:true});
      const lines=buffer.split('\n');
      buffer=lines.pop()||'';
      
      for(const line of lines){
        if(line.startsWith('data: ')){
          try{
            const event=JSON.parse(line.slice(6));
            if(event.type==='token'){
              answerBuffer+=event.content;
              appendToStreamingMessage(event.content);
            }else if(event.type==='metadata'){
              sourceUsedValue=event.source_used;
              citations=event.citations||[];
              traceData=event.trace||[];
            }else if(event.type==='error'){
              throw new Error(event.message||'Streaming error');
            }
          }catch(e){
            console.error('Parse error:',e,line);
          }
        }
      }
    }
    
    finalizeStreamingMessage(citations, sourceUsedValue);
    renderTrace(traceData);
    sourceUsed.textContent=sourceUsedValue||'Unknown';
    
  }catch(e){
    if(currentStreamingMessage){
      removeStreamingMessage();
      currentStreamingMessage=null;
    }
    addMessage('assistant',`Error: ${e.message}`);
    renderTrace(['Request failed']);
    sourceUsed.textContent='Error';
  }finally{
    btn.disabled=false;
  }
}

function createStreamingMessage(){
  const wrap=document.createElement('div');
  wrap.className='message assistant streaming';
  wrap.innerHTML=`<div class="avatar">AI</div><div class="bubble"></div>`;
  chat.appendChild(wrap);
  chat.scrollTop=chat.scrollHeight;
  return wrap.querySelector('.bubble');
}

function appendToStreamingMessage(content){
  if(!currentStreamingMessage) return;
  const escaped=escapeHtml(content);
  currentStreamingMessage.innerHTML+=escaped.replace(/\n/g,'<br>');
  chat.scrollTop=chat.scrollHeight;
}

function finalizeStreamingMessage(citations, source){
  if(!currentStreamingMessage) return;
  
  const wrap=currentStreamingMessage.closest('.message');
  const citeHtml=citations.length?`<div class="citations"><strong>Sources</strong><br>${citations.map(c=>c.url?`<a href="${escapeHtml(c.url)}" target="_blank" rel="noopener">${escapeHtml(c.title)}</a>`:escapeHtml(c.title)).join('<br>')}</div>`:'';
  const sourceHtml=source?`<div class="answer-source">Source: ${escapeHtml(source)}</div>`:'';
  
  currentStreamingMessage.innerHTML+=`${sourceHtml}${citeHtml}`;
  wrap.classList.remove('streaming');
  chat.scrollTop=chat.scrollHeight;
  
  currentStreamingMessage=null;
}

function removeStreamingMessage(){
  if(!currentStreamingMessage) return;
  const wrap=currentStreamingMessage.closest('.message');
  if(wrap) wrap.remove();
  currentStreamingMessage=null;
}
form.addEventListener('submit',e=>{e.preventDefault(); const q=question.value.trim(); if(q) askAgent(q);});
document.querySelectorAll('.example').forEach(b=>b.addEventListener('click',()=>askAgent(b.textContent.trim())));

const modal=document.getElementById('uploadModal');
document.getElementById('openUpload').onclick=()=>modal.classList.remove('hidden');
document.getElementById('closeUpload').onclick=()=>modal.classList.add('hidden');
document.getElementById('uploadBtn').onclick=async()=>{
  const file=document.getElementById('fileInput').files[0]; const key=document.getElementById('adminKey').value; const status=document.getElementById('uploadStatus');
  if(!file){status.textContent='Choose a file first.'; return;}
  status.textContent='Indexing document...'; const fd=new FormData(); fd.append('file',file);
  try{const r=await fetch('/api/ingest',{method:'POST',headers:{'X-Admin-Key':key},body:fd}); const d=await r.json(); if(!r.ok) throw new Error(d.detail||'Upload failed'); status.textContent=`Indexed ${d.file}: ${d.chunks} chunks.`;}catch(e){status.textContent=`Error: ${e.message}`;}
};