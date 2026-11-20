import asyncio, audioop, wave, websockets

def load(path):
    with wave.open(path,'rb') as w:
        ch,rate,fr = w.getnchannels(),w.getframerate(),w.readframes(w.getnframes())
    if ch==2: fr=audioop.tomono(fr,2,0.5,0.5)
    if rate!=16000: fr,_=audioop.ratecv(fr,2,1,rate,16000,None)
    return fr

async def turn(ws,pcm,tag):
    for i in range(0,len(pcm),1024):
        await ws.send(pcm[i:i+1024])
    await ws.send("END_OF_SPEECH")
    while True:
        m=await asyncio.wait_for(ws.recv(),timeout=90)
        if isinstance(m,bytes): continue
        if m=="END_OF_AUDIO": print(f"[{tag}] done"); return
        print(f"[{tag}] {m}")

async def main():
    pcm=load('voice_assistant/test_speech.wav')
    async with websockets.connect('ws://127.0.0.1:8000/ws') as ws:
        # 3 реплики в одном соединении: история должна копиться
        await turn(ws,pcm,'t1')
        await turn(ws,pcm,'t2')
        await turn(ws,pcm,'t3')

asyncio.run(main())
