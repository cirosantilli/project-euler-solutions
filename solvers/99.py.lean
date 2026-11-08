import solvers.«99.py.lean.d».Search
import data.«project-euler-statements».data.py.project_euler_statements.«p99.py»

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

namespace solvers.«99».compare_powers
open FormalSW.PyAst HeapCalls

theorem «Returns» (a : Nat) (m : Int) (b : Nat) (n : Int) (positiveA : 0 < a) (positiveB : 0 < b) :
    (HeapCalls.call program "compare_powers" [number a, number m, number b, number n] []).Returns := by
  obtain ⟨extra, returned⟩ := result a m b n positiveA positiveB []
  exact returned.returns

theorem «Supported» (a : Nat) (m : Int) (b : Nat) (n : Int) (positiveA : 0 < a) (positiveB : 0 < b) :
    (HeapCalls.call program "compare_powers" [number a, number m, number b, number n] []).Supported :=
  («Returns» a m b n positiveA positiveB).supported

theorem reference_agreement (a : Nat) (m : Int) (b : Nat) (n : Int)
    (positiveA : 0 < a) (positiveB : 0 < b) (left right : Nat)
    (heap : Mutable.Heap) (globals : Mutable.Globals)
    (foundLeft : heap[left]? = some
      (data.«project-euler-statements».data.py.project_euler_statements.p99.pairObject a m))
    (foundRight : heap[right]? = some
      (data.«project-euler-statements».data.py.project_euler_statements.p99.pairObject b n)) :
    ∃ extra value, ReturnsValue program "compare_powers" [number a, number m, number b, number n] heap
      (number value) (heap ++ extra) globals ∧
      ReturnsValue P99Ast.module "greater_power" [.ref left, .ref right] heap
        (.scalar (.bool (decide (0 < value)))) heap globals := by
  obtain ⟨extra, ran⟩ := result a m b n positiveA positiveB heap globals
  exact ⟨extra, _, ran,
    data.«project-euler-statements».data.py.project_euler_statements.p99.greater_power.result
      a m b n left right heap globals foundLeft foundRight⟩

end solvers.«99».compare_powers

namespace solvers.«99».solve
open FormalSW.PyAst HeapCalls ProjectEulerSolutions.PowerMaximum

theorem «Returns» (entries : List Entry) (address : Nat) (heap : Mutable.Heap)
    (stored : heap[address]? = some (.list (entries.map entryValue)))
    (valid : ∀ entry ∈ entries, 0 < entry.2.1 ∧ heap[entry.1]? = some (entryObject entry)) :
    (HeapCalls.call program "solve" [.ref address] heap).Returns := by
  obtain ⟨extra, returned⟩ := result entries address heap stored valid
  exact returned.returns

theorem «Supported» (entries : List Entry) (address : Nat) (heap : Mutable.Heap)
    (stored : heap[address]? = some (.list (entries.map entryValue)))
    (valid : ∀ entry ∈ entries, 0 < entry.2.1 ∧ heap[entry.1]? = some (entryObject entry)) :
    (HeapCalls.call program "solve" [.ref address] heap).Supported :=
  («Returns» entries address heap stored valid).supported

/-- Both complete source calls agree, including the original heap's aliases and cycles. -/
theorem «Equiv(naive)» (entries : List Entry) (address : Nat) (heap : Mutable.Heap)
    (globals : Mutable.Globals)
    (closed : heap.Valid) (globalsWithin : ∀ entry ∈ globals, entry.2.edge.Within heap.length)
    (stored : heap[address]? = some (.list (entries.map entryValue)))
    (valid : ∀ entry ∈ entries, 0 < entry.2.1 ∧ heap[entry.1]? = some (entryObject entry)) :
    (HeapCalls.call program "solve" [.ref address] heap globals).Equiv
      (HeapCalls.call P99Ast.module "naive" [.ref address] heap globals) := by
  obtain ⟨extra, left⟩ := result entries address heap stored valid globals
  have right := data.«project-euler-statements».data.py.project_euler_statements.p99.naive.result
    entries address heap stored valid globals
  apply Mutable.Call.equiv_of_finishes left.finishes right.finishes
  simpa only [List.append_nil] using
    Mutable.Completion.equivalent_allocations heap extra []
      (number (maximumIndex (entries.map Prod.snd))) globals closed trivial globalsWithin

end solvers.«99».solve
