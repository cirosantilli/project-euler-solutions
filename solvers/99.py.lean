import solvers.«99.py.lean.d».Rounding

namespace solvers.«99».round_power_bounds
open FormalSW.PyAst HeapCalls ProjectEulerSolutions.PowerIntervals

theorem «Returns» (bounds : Bounds) (precision : Nat) :
    (HeapCalls.call program "round_power_bounds"
      [number bounds.lower, number bounds.upper, number bounds.shift, number precision] []).Returns :=
  (result bounds precision []).returns

theorem «Supported» (bounds : Bounds) (precision : Nat) :
    (HeapCalls.call program "round_power_bounds"
      [number bounds.lower, number bounds.upper, number bounds.shift, number precision] []).Supported :=
  («Returns» bounds precision).supported

/-- The tuple returned by the actual source encloses every value admitted by the incoming bounds. -/
theorem encloses (bounds : Bounds) (precision value : Nat) (inside : Encloses bounds value)
    (heap : Mutable.Heap) (globals : Mutable.Globals := []) :
    ReturnsValue program "round_power_bounds"
      [number bounds.lower, number bounds.upper, number bounds.shift, number precision] heap
      (.ref heap.length) (heap ++ [boundsObject (trim bounds precision)]) globals ∧
      Encloses (trim bounds precision) value :=
  ⟨result bounds precision heap globals, trim_encloses bounds precision value inside⟩

end solvers.«99».round_power_bounds
