import std.conv : to;
import std.string : format, split;
import std.algorithm : findSplitAfter;
import std.range : retro;


auto findLast(R1, R2)(R1 haystack, R2 needle) {
  return haystack.retro
    .findSplitAfter(needle.retro)[0]
    .to!R1.retro;
}

auto findBetween(R1, R2)(R1 haystack, R2 start, R2 end) {
  // NOTE: this extract first found `end` to backward nearest `start`
  //       including both corner ranges.
  return haystack.findSplitAfter(end)[0]
    .findLast(start);
}

unittest {
  assert(findLast("abcdefg abc", "ab").to!string == "abc");
  assert(findBetween("a abcdefg bc", "a", "bc").to!string == "abc");
}


enum CImportEnum(string name, string headerPath) = function(){
  enum start = "typedef enum";
  enum header = import(headerPath);
  enum end = "} " ~ name ~ ";";
  enum cdef = findBetween(header, start, end).to!string;
  enum content = cdef.findSplitAfter("{")[1].findSplitAfter("}")[0];
  return "enum %s {%s".format(name, content);
}();

enum enumCUresultMixin = CImportEnum!("CUresult", "cuda.h");
enum enumNvrtcResultMixin = CImportEnum!("nvrtcResult", "nvrtc.h");

unittest {
  import cudriver : CUresult, nvrtcResult;
  static assert(CUresult.CUDA_ERROR_UNKNOWN == 999);
  static assert(nvrtcResult.NVRTC_SUCCESS == 0);
}

