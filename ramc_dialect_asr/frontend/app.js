const $ = (id) => document.getElementById(id);

function clearUI(){
  $("dialect").textContent = "-";
  $("conf").textContent = "-";
  $("time").textContent = "-";
  $("committed").textContent = "";
  $("partial").textContent = "";
  $("final").textContent = "";
}

async function stream(){
  const file = $("file").files[0];
  if(!file){ alert("Please select a .wav file."); return; }

  $("btnStream").disabled = true;
  $("btnOffline").disabled = true;

  const fd = new FormData();
  fd.append("audio", file);
  fd.append("simulate_realtime", $("simulate").checked ? "true" : "false");
  fd.append("chunk_sec", $("chunkSec").value);
  fd.append("overlap_sec", $("overlapSec").value);

  const resp = await fetch("/api/stream", { method:"POST", body: fd });
  if(!resp.ok){
    alert("Request failed: " + resp.status);
    $("btnStream").disabled = false;
    $("btnOffline").disabled = false;
    return;
  }

  const reader = resp.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buf = "";

  while(true){
    const {value, done} = await reader.read();
    if(done) break;
    buf += decoder.decode(value, {stream:true});
    let idx;
    while((idx = buf.indexOf("\n\n")) >= 0){
      const block = buf.slice(0, idx);
      buf = buf.slice(idx+2);
      handleSSEBlock(block);
    }
  }

  $("btnStream").disabled = false;
  $("btnOffline").disabled = false;
}

function handleSSEBlock(block){
  const lines = block.split("\n").filter(Boolean);
  let ev = "message";
  let dataLine = "";
  for(const ln of lines){
    if(ln.startsWith("event:")) ev = ln.slice(6).trim();
    if(ln.startsWith("data:")) dataLine += ln.slice(5).trim();
  }
  if(!dataLine) return;
  let data;
  try{ data = JSON.parse(dataLine); }catch(e){ return; }

  if(ev === "meta"){
    $("engine").textContent = data.engine || "unknown";
  }else if(ev === "partial"){
    if(data.dialect){
      $("dialect").textContent = data.dialect.label ?? "-";
      $("conf").textContent = (data.dialect.confidence ?? 0).toFixed(2);
    }
    $("time").textContent = `${data.t0.toFixed(2)}s - ${data.t1.toFixed(2)}s`;
    $("committed").textContent = data.committed ?? "";
    $("partial").textContent = data.partial ?? "";
  }else if(ev === "final"){
    $("final").textContent = data.text ?? "";
  }
}

async function offline(){
  const file = $("file").files[0];
  if(!file){ alert("Please select a .wav file."); return; }

  $("btnStream").disabled = true;
  $("btnOffline").disabled = true;

  const fd = new FormData();
  fd.append("audio", file);

  const resp = await fetch("/api/offline", { method:"POST", body: fd });
  const j = await resp.json();

  $("engine").textContent = j.engine || "unknown";
  $("dialect").textContent = j.dialect?.label ?? "-";
  $("conf").textContent = (j.dialect?.confidence ?? 0).toFixed(2);
  $("final").textContent = j.text ?? "";

  $("btnStream").disabled = false;
  $("btnOffline").disabled = false;
}

$("btnStream").addEventListener("click", stream);
$("btnOffline").addEventListener("click", offline);
$("btnClear").addEventListener("click", clearUI);
