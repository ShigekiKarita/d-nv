import std.conv : to;


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

unittest {
  import std.exception;
  import storage;
  import driver;

  try {
    check(CUresult.CUDA_ERROR_UNKNOWN);
  } catch(CudaError!CUresult e) {
    auto s = e.toString();
    assert(s == "CUresult.CUDA_ERROR_UNKNOWN");
  }
  assertThrown(new Array!float(3, -1));
}
