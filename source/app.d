import std.stdio;
import std.process;
import std.string;

import nv = rtc;
import cudriver;

string nvidia_smi() {
  auto p = executeShell("nvidia-smi");
  if (p.status != 0) throw new Exception("Failed to execute nvidia-smi");
  return p.output;
}

void main() {
  writeln(">>> finished <<<");
  nvidia_smi().writeln;
}
