module dnv.error;

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
  return "extern (C++) enum %s : int {%s".format(name, content);
}();

mixin(CImportEnum!("CUresult", "cuda.h"));
mixin(CImportEnum!("nvrtcResult", "nvrtc.h"));


unittest {
  static assert(CUresult.CUDA_ERROR_UNKNOWN == 999);
  static assert(nvrtcResult.NVRTC_SUCCESS == 0);
}


class CudaError(Result) : Exception {
  const Result result;

  this(Result r, string file = __FILE__, int line = __LINE__) {
    result = r;
    super(toString, file, line);
  }

  override string toString() {
    return Result.stringof ~ "." ~ result.to!string;
  }
}

void check(Result, string file = __FILE__, int line = __LINE__)(Result result) {
  if (result) {
    throw new CudaError!Result(result, file, line);
  }
}

void cuCheck(string file = __FILE__, int line = __LINE__)(int result) {
  if (result) {
      check!(CUresult, file, line)(cast(CUresult) result);
  }
}

void nvrtcCheck(string file = __FILE__, int line = __LINE__)(int result) {
  if (result) {
      check!(nvrtcResult, file, line)(cast(nvrtcResult) result);
  }
}


unittest {
  import std.exception;
  import dnv.storage;
  import dnv.driver;

  try {
    check(CUresult.CUDA_ERROR_UNKNOWN);
  } catch(CudaError!CUresult e) {
    auto s = e.toString();
    assert(s == "CUresult.CUDA_ERROR_UNKNOWN");
  }
  assertThrown(new Array!float(3, 100));
}
