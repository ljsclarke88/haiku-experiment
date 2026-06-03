// AudioWorklet processor — batches 128-sample blocks into 4096-sample chunks
// and posts them to the main thread for onset/offset detection.
class CatProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this._buf = [];
    this._chunkSize = 4096;
  }

  process(inputs) {
    const ch = inputs[0]?.[0];
    if (!ch) return true;
    for (let i = 0; i < ch.length; i++) this._buf.push(ch[i]);
    while (this._buf.length >= this._chunkSize) {
      this.port.postMessage(new Float32Array(this._buf.splice(0, this._chunkSize)));
    }
    return true;
  }
}

registerProcessor('cat-processor', CatProcessor);
