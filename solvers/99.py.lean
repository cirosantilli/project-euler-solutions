import solvers.«99.py.lean.d».Refinement

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

namespace solvers.«99».power_bounds
open FormalSW.PyAst HeapCalls ProjectEulerSolutions.PowerIntervals

theorem «Returns» (base exponent precision : Nat) :
    (HeapCalls.call program "power_bounds" [number base, number exponent, number precision] []).Returns := by
  obtain ⟨extra, returned⟩ := result base exponent precision []
  exact returned.returns

theorem «Supported» (base exponent precision : Nat) :
    (HeapCalls.call program "power_bounds" [number base, number exponent, number precision] []).Supported :=
  («Returns» base exponent precision).supported

theorem encloses (base exponent precision : Nat) (heap : Mutable.Heap) (globals : Mutable.Globals := []) :
    ∃ extra, ReturnsValue program "power_bounds" [number base, number exponent, number precision] heap
      (.ref (heap ++ extra).length) ((heap ++ extra) ++ [boundsObject (power base exponent precision)]) globals ∧
      Encloses (power base exponent precision) (base ^ exponent) := by
  obtain ⟨extra, returned⟩ := result base exponent precision heap globals
  exact ⟨extra, returned, power_encloses base exponent precision⟩

end solvers.«99».power_bounds

namespace solvers.«99».compare_scaled
open FormalSW.PyAst HeapCalls

theorem «Returns» (left leftShift right rightShift : Nat) :
    (HeapCalls.call program "compare_scaled"
      [number left, number leftShift, number right, number rightShift] []).Returns :=
  (result left leftShift right rightShift []).returns

theorem «Supported» (left leftShift right rightShift : Nat) :
    (HeapCalls.call program "compare_scaled"
      [number left, number leftShift, number right, number rightShift] []).Supported :=
  («Returns» left leftShift right rightShift).supported

end solvers.«99».compare_scaled
