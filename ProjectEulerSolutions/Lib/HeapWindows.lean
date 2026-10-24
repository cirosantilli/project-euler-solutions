import lang.py.FormalSW.HeapCalls
import Mathlib.Tactic.IntervalCases

/-! Exact three-element slices of a ten-element list, retaining references and allocating a copy. -/
namespace ProjectEulerSolutions.HeapWindows
open FormalSW FormalSW.PyAst

set_option Elab.async false

private theorem get_prefix (values : List α) (count : Nat) :
    (List.range count).filterMap (fun index => values[index]?) = values.take count := by
  induction count generalizing values with
  | zero => simp
  | succ count ih =>
      cases values with
      | nil => simp
      | cons value values =>
          simpa [List.range_succ_eq_map, List.filterMap_map, Function.comp_def] using
            congrArg (List.cons value) (ih values)

theorem get_window (values : List α) (start count : Nat) :
    (List.range' start count).filterMap (fun index => values[index]?) = (values.drop start).take count := by
  rw [List.range'_eq_map_range, List.filterMap_map]
  simpa [List.getElem?_drop, Function.comp_def] using get_prefix (values.drop start) count

theorem slice_three (values : List α) (size : values.length = 10) (start : Nat) (bound : start ≤ 7) :
    Py.Builtin.Sequence.slice values { start := some start, stop := some ((start : Int)+3) } =
      .ok ((values.drop start).take 3) := by
  have indices : Py.Builtin.Slice.indexList
      { start := some start, stop := some ((start : Int)+3) } 10 = .ok (List.range' start 3) := by
    interval_cases start <;> decide +kernel
  unfold Py.Builtin.Sequence.slice
  rw [size, indices]
  change Except.ok ((List.range' start 3).filterMap (fun index => values[index]?)) = _
  rw [get_window]

theorem getSlice_three (heap : Mutable.Heap) (address : Nat) (values : List Mutable.Value)
    (found : heap[address]? = some (.list values)) (size : values.length = 10)
    (start : Nat) (bound : start ≤ 7) :
    Mutable.getSlice (.ref address) (.scalar (.int start)) (.scalar (.int ((start : Int)+3)))
      (.scalar .none) heap = some (.ok (.ref heap.length), heap ++ [.list ((values.drop start).take 3)]) := by
  simp! [Mutable.getSlice, found, Mutable.sliceBounds,
    slice_three values size start bound, Mutable.allocate, Except.map]

end ProjectEulerSolutions.HeapWindows
